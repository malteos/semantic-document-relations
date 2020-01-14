import json
import logging
import math
import multiprocessing
import os
import pickle
import random

import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler

from urllib.parse import unquote

import pandas as pd

from experiments.data_helpers import DataHelper, BERTDataHelper, DocRelDataHelper
from experiments.data_loaders import DefaultXYDataLoader

logger = logging.getLogger(__name__)


class BaseWikiDataHelper(BERTDataHelper, DocRelDataHelper):
    docs = None
    token_ids_map_path = None

    wiki_relations_path = './wiki/relations.csv'
    wiki_articles_path = './wiki/enwiki-dump.jsonl'

    train_dataframe_path = None
    test_dataframe_path = None

    doc_a_col = 'subject_enwiki'
    doc_b_col = 'object_enwiki'

    max_side_length = None
    df_limit = 0
    train_df = None
    test_df = None
    include_section_title = False
    sections_limit = 0

    workers = 1  # set to None for auto

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_pairs_from_df(self, df, neg_ratio=1.):
        pairs = []
        # subject_col = 'subject_enwiki_name'
        # object_col = 'object_enwiki_name'
        pos_titles = set()

        a_set = list(set(df[self.doc_a_col].values))
        b_set = list(set(df[self.doc_b_col].values))

        if self.df_limit > 0:
            df = df[:self.df_limit]
            logger.warning('You have limited the input data frame (model label count might not match anymore)')

        # positive pairs
        for idx, row in df.iterrows():
            # positive
            pairs.append((
                row[self.doc_a_col],
                row[self.doc_b_col],
                row[self.label_col] if len(self.labels) > 1 else self.labels[0],
            ))

            # log for negative sampling
            pos_titles.add((row[self.doc_a_col], row[self.doc_b_col]))

        # negative pairs
        neg_count = int(len(pairs) * neg_ratio)
        neg_pairs = []
        neg_titles = set()

        logger.info(f'Generating {neg_count} negative pairs... ratio={neg_ratio}')

        while len(neg_pairs) < neg_count:
            rand_a = random.choice(a_set)
            rand_b = random.choice(b_set)

            # Candidate should not be already used as positive or negative
            if (rand_a, rand_b) not in pos_titles and (rand_a, rand_b) not in neg_titles:
                neg_pairs.append((
                    rand_a,
                    rand_b,
                    self.none_label
                ))
                neg_titles.add((rand_a, rand_b))

        pairs += neg_pairs

        logger.debug(f'Generated {len(pairs)} pairs from data frame')

        return pd.DataFrame(pairs, columns=[self.doc_a_col, self.doc_b_col, self.label_col])

    def get_docs(self):
        if self.docs is None:
            self.docs = []

            with open(self.wiki_articles_path, 'r') as f:
                for line in f:
                    doc = json.loads(line)
                    self.docs.append(doc)

        return self.docs

    @staticmethod
    def get_text_from_doc(doc, sections_limit=0, title_text_sep='\n\n', sect_sep='\n\n', include_section_title=True):
        """

        Get plain text from Wikipedia document (as in gensim segments)

        :param include_section_title: Set to false to ignore section titles
        :param doc: Article dict in gensim format
        :param sections_limit: Limit number of sections included
        :param title_text_sep:
        :param sect_sep:
        :return: str
        """
        texts = []
        for i, sect_title in enumerate(doc['section_titles']):
            if i < len(doc['section_texts']):

                sect_text = doc['section_texts'][i]

                if len(sect_text) > 0:
                    if include_section_title:
                        texts.append(sect_title + title_text_sep + sect_text)
                    else:
                        texts.append(sect_text)

                    if 0 < sections_limit <= i+1:
                        break

        return sect_sep.join(texts)

    def get_df(self):
        with open(self.wiki_relations_path, 'r') as f:
            df = pd.read_csv(f, index_col=0)

        logger.debug(f'Loaded gold data from: {self.wiki_relations_path}')
            
        #df[self.doc_b_col] = list(map(lambda v: unquote(v.split('/')[-1]).replace('_', ' '), df['object_enwiki']))
        # df[self.doc_a_col] = list(
        #     map(lambda v: unquote(v.split('/')[-1]).replace('_', ' '), df['subject_enwiki']))

        return df

    @staticmethod
    def tokenize_doc(title, doc, tokenizer, sections_limit, doc_token_limit):
        # tokenize title + text (only first 3 sections)
        tokens = tokenizer.tokenize(title) + tokenizer.tokenize(
            JointBERTWikiDataHelper.get_text_from_doc(doc, sections_limit=sections_limit,
                                   include_section_title=False))  # TODO maybe more sections

        # enforce max length
        if doc_token_limit > 0:
            tokens = tokens[:doc_token_limit]

        return title, tokenizer.convert_tokens_to_ids(tokens)

    @staticmethod
    def tokenize_doc_pool(args):
            return JointBERTWikiDataHelper.tokenize_doc(*args)

    def tokenize_docs(self, title2doc, title_needles: set = None, doc_token_limit=0):
        if title_needles is None:
            title_needles = set()

        if self.workers != 1:
            return self.tokenize_docs_in_parallel(title2doc, title_needles, doc_token_limit=doc_token_limit, workers=self.workers)

        token_ids_map = {}
        doc_iterator = title2doc.items()

        logger.debug(f'Documents available: {len(title2doc)}')
        logger.debug(f'Documents needed: {len(title_needles)}')

        # from cache (if path is set and file exist)
        if self.token_ids_map_path and os.path.exists(self.token_ids_map_path):
            with open(self.token_ids_map_path, 'rb') as f:
                token_ids_map = pickle.load(f)

                return token_ids_map

        if self.tqdm_cls:
            doc_iterator = self.tqdm_cls(doc_iterator, total=len(title2doc), desc='Tokenize docs')

        # Tokenize text (once for all)
        for title, doc in doc_iterator:
            # Skip if title is not needed
            if title_needles and title not in title_needles:
                continue
            """
            # tokenize title + text (only first 3 sections)
            tokens = self.get_tokenizer().tokenize(title) + self.get_tokenizer().tokenize(
                self.get_text_from_doc(doc, sections_limit=self.sections_limit, include_section_title=False))  # TODO maybe more sections

            # enforce max length
            if doc_token_limit > 0:
                tokens = tokens[:doc_token_limit]

            # self.get_tokenizer().convert_tokens_to_ids(tokens)
            """
            title, token_ids = self.tokenize_doc(title, doc, self.get_tokenizer(), self.sections_limit, doc_token_limit)
            token_ids_map[title] = token_ids


        # save to cache
        # (if path is set and file not exist)
        if self.token_ids_map_path and not os.path.exists(self.token_ids_map_path):
            with open(self.token_ids_map_path, 'wb') as f:
                pickle.dump(token_ids_map, f)

        return token_ids_map

    def tokenize_docs_in_parallel(self, title2doc, title_needles: set = None, doc_token_limit=0, workers=1):
        logger.info(f'Tokenize with {workers} workers')

        if workers is None:  # Auto-define number of works based on CPU count
            workers = max(1, multiprocessing.cpu_count() - 1)

        pool = multiprocessing.Pool(workers)

        doc_iterator = title2doc.items()

        docs = [(title, doc, self.get_tokenizer(), self.sections_limit, doc_token_limit) for title, doc in doc_iterator if not title_needles or title in title_needles]

        pool_out = pool.map(JointBERTWikiDataHelper.tokenize_doc_pool, docs)

        # stop threads when done
        pool.close()
        pool.join()

        token_ids_map = {title: token_ids for title, token_ids in pool_out}

        return token_ids_map


    def get_data_loaders(self):
        # self.label_col = 'y'
        title2doc = {doc['title']: doc for doc in self.get_docs()}

        if self.wiki_relations_path:
            df = self.get_df()

            selector = (df[self.label_col].isin(self.labels)) \
                       & (df[self.doc_a_col].isin(title2doc)) \
                       & (df[self.doc_b_col].isin(title2doc))


            pair_df = self.get_pairs_from_df(df[selector].copy(), neg_ratio=self.negative_sampling_ratio)

            self.train_df, self.test_df = self.get_train_test_split(pair_df)

            # Tokenize only need documents
            title_needles = set(pair_df[self.doc_a_col].values).union(pair_df[self.doc_b_col].values)

            token_ids_map = self.tokenize_docs(title2doc, title_needles, doc_token_limit=512)  # BERT can handle only 512 tokens
            logger.info('Tokenizing completed')

        elif self.train_dataframe_path and self.test_dataframe_path:
            # Load CSVs
            with open(self.train_dataframe_path, 'r') as f:
                self.train_df = pd.read_csv(f, index_col=0)
            with open(self.test_dataframe_path, 'r') as f:
                self.test_df = pd.read_csv(f, index_col=0)

            # Tokenize only need documents
            title_needles = set(list(self.train_df[self.doc_a_col].values) + list(self.train_df[self.doc_b_col].values) +
                                list(self.test_df[self.doc_a_col].values) + list(self.test_df[self.doc_b_col].values))

            token_ids_map = self.tokenize_docs(title2doc, title_needles,
                                               doc_token_limit=512)  # BERT can handle only 512 tokens
            logger.info('Tokenizing completed')

        else:
            raise ValueError('Either `wiki_relations_path` or `train_dataframe_path` and `test_datframe_path` must be set!')

        # Labels
        self.label_encoder = LabelEncoder()
        self.labels_integer_encoded = self.label_encoder.fit_transform(
            list(self.train_df[self.label_col].values) + list(self.test_df[self.label_col].values))

        self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        self.labels_onehot_encoded = self.onehot_encoder.fit_transform(
            self.labels_integer_encoded.reshape(len(self.labels_integer_encoded), 1))

        train_dl = self.to_data_loader(self.train_df, token_ids_map, self.train_batch_size, sampler_cls=RandomSampler)
        test_dl = self.to_data_loader(self.test_df, token_ids_map, self.test_batch_size, sampler_cls=SequentialSampler)
        #
        return train_dl, test_dl

    def to_data_loader(self, df, token_ids_map, batch_size, sampler_cls=None, sampler=None):
        raise NotImplementedError()


class JointBERTWikiDataHelper(BaseWikiDataHelper):
    """

    Joint BERT data helper: documents are joined as single sequence

    """
    def to_data_loader(self, df, token_ids_map, batch_size, sampler_cls=None, sampler=None):
        ys = self.get_ys_as_tensor(df)

        titles = df[[self.doc_a_col, self.doc_b_col]].values
        joint_ids, masks, token_types = self.get_joint_token_ids_and_types(titles, token_ids_map)

        # build dataset
        dataset = TensorDataset(
            joint_ids,
            masks,
            token_types,
            ys)

        return DefaultXYDataLoader(dataset, sampler=self.get_data_sampler(sampler, dataset, sampler_cls), batch_size=batch_size)


class SiameseBERTWikiDataHelper(BaseWikiDataHelper):

    def to_data_loader(self, df, token_ids_map, batch_size, sampler_cls=None, sampler=None):

        ys = self.get_ys_as_tensor(df)

        titles = df[[self.doc_a_col, self.doc_b_col]].values

        if self.tqdm_cls:
            titles = self.tqdm_cls(titles, total=len(titles), desc='Building tensor data set')

        token_ids_a = [torch.tensor([self.get_tokenizer().cls_token_id] + token_ids_map[a][:self.max_seq_length - 2] + [self.get_tokenizer().sep_token_id]) for a, b in titles]
        token_ids_b = [torch.tensor([self.get_tokenizer().cls_token_id] + token_ids_map[b][:self.max_seq_length - 2] + [self.get_tokenizer().sep_token_id]) for a, b in titles]

        token_ids_a = pad_sequence(token_ids_a, batch_first=True, padding_value=self.get_tokenizer().pad_token_id)
        token_ids_b = pad_sequence(token_ids_b, batch_first=True, padding_value=self.get_tokenizer().pad_token_id)

        masks_a = torch.tensor([[float(i > 0) for i in ii] for ii in token_ids_a])
        masks_b = torch.tensor([[float(i > 0) for i in ii] for ii in token_ids_b])

        # build dataset
        dataset = TensorDataset(
            token_ids_a,
            masks_a,
            token_ids_b,
            masks_b,
            ys)

        return DefaultXYDataLoader(dataset, sampler=self.get_data_sampler(sampler, dataset, sampler_cls), batch_size=batch_size)


class SiameseLongBERTWikiDataHelper(BaseWikiDataHelper):
    max_chunks_per_doc = 20

    def get_tokens_and_masks(self, titles, title2token_ids, cut_long_documents=True):
        """
        Returns token ids and masks: len(titles) * max_chunks_per_doc * max_seq_length.

        chunks_per_doc holds the actual length of the document (in chunks) that is needed to pack the sequence.

        :return: token_ids, masks, chunk_per_doc (all as tensors)
        """
        reserved_token_count = 2  # <CLS> + <SEP>
        hard_limit = self.max_chunks_per_doc * (self.max_seq_length - reserved_token_count)

        if cut_long_documents:
            tokenized_documents = [title2token_ids[t][:hard_limit] for t in titles]
        else:
            tokenized_documents = [title2token_ids[t] for t in titles]

        chunks_per_doc = [math.ceil(len(x)/(self.max_seq_length-reserved_token_count)) for x in tokenized_documents]

        max_sequences_per_document = max(chunks_per_doc)

        assert max_sequences_per_document <= self.max_chunks_per_doc, f"Your document is too large, arbitrary size when writing: {max_sequences_per_document}"

        token_ids = torch.zeros((len(titles), max_sequences_per_document, 512), dtype=torch.long)
        masks = torch.zeros((len(titles), max_sequences_per_document, 512), dtype=torch.long)

        # Iterate over docs
        for doc_idx, tokenized_document in enumerate(tokenized_documents):
            # Iterate over chunks
            for chunk_idx, i in enumerate(range(0, len(tokenized_document), (self.max_seq_length - reserved_token_count))):
                chunk_token_ids = [self.get_tokenizer().cls_token_id] + tokenized_document[i:i + (self.max_seq_length - reserved_token_count)] + [self.get_tokenizer().sep_token_id]

                # Set values for tokens and masks (at specific offset)
                token_ids[doc_idx, chunk_idx, 0:len(chunk_token_ids)] = torch.tensor(chunk_token_ids)
                masks[doc_idx, chunk_idx, 0:len(chunk_token_ids)] = torch.tensor([1.] * len(chunk_token_ids))

        return token_ids, masks, torch.LongTensor(chunks_per_doc)

    def to_data_loader(self, df, token_ids_map, batch_size, sampler_cls=None, sampler=None):
        ys = self.get_ys_as_tensor(df)

        token_ids_a, masks_a, chunks_per_doc_a = self.get_tokens_and_masks(df[self.doc_a_col].values, token_ids_map)
        token_ids_b, masks_b, chunks_per_doc_b = self.get_tokens_and_masks(df[self.doc_b_col].values, token_ids_map)

        dataset = TensorDataset(
            token_ids_a,
            masks_a,
            chunks_per_doc_a,
            token_ids_b,
            masks_b,
            chunks_per_doc_b,
            ys)

        return DefaultXYDataLoader(dataset, sampler=self.get_data_sampler(sampler, dataset, sampler_cls), batch_size=batch_size)

