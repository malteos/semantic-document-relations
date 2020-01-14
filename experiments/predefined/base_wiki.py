import os

from tqdm import tqdm
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer, XLNetModel, XLNetTokenizer

from experiments.environment import get_env
from experiments.predefined import update

env = get_env()
split_dir = './output/splits10k/'
#'has_cause', # drop has cause (P828)
labels = ['different_from', 'employer', 'facet_of',
       'country_of_citizenship', 'opposite_of', 'has_quality', 'symptoms',
       'has_effect', 'educated_at']
dh_params = dict(
    train_dataframe_path = split_dir + 'train.csv',
    test_dataframe_path = split_dir + 'test.csv',
    wiki_relations_path = None,
    wiki_articles_path = os.path.join(env['datasets_dir'], '/wikipedia_en/dumps/enwiki-20191101-pages-articles.weighted.10k.jsonl'),
    train_batch_size = 4,
    test_batch_size = 5,
    include_section_title = False,
    labels = labels,
    label_col = 'relation_name',
    max_seq_length=512,
    workers=env['workers'],
)
base_config = dict(
    optimizer_cls='torch.optim.Adam',
    # lr': 1e-6
    epochs=4,
    test_every_n_epoch=1,
    loss_func_cls='torch.nn.BCELoss', # MSELoss, BCELoss
    data_loader_to_loss_input=lambda ys: ys.float(),

    data_helper_cls='wiki.data_helpers.SiameseBERTWikiDataHelper',
    data_helper_params=dh_params,

    output_dir='./output',
    tqdm_cls=tqdm,
    classification_by_max=True,
    tensorboard_params={
        'auto': True,
        'dir': './runs',
    },
    model_params=dict(
        prob='sigmoid',  # sigmoid, softmax, none  # sigmoid >> softmax for wikirel
        labels_count=len(labels) + 1,
        mlp_dim=512,
        mlp_layers_count=1,
    )
)


# Architectures & Data

siamese_config = dict(
    model_cls='models.transformers.SiameseBERT',
    data_helper_cls='wiki.data_helpers.SiameseBERTWikiDataHelper',
    model_params=dict(
        mlp_layers_count=2,
        mlp_dim=512,
        # 'concat': '4d-prod-dif'
    ),
    data_helper_params=dict(
        train_batch_size=3,  # Siamese requires more GPU memory, use smaller batch size
    ),
)
joint_config = dict(
    model_cls='models.transformers.JointBERT',
    data_helper_cls='wiki.data_helpers.JointBERTWikiDataHelper',
    model_params=dict(
        mlp_layers_count=1,
        mlp_dim=512,
    )
)

# Transformer models
bert_base_config = dict(
    model_params=dict(
        bert_cls=BertModel,
        bert_model_path=env['bert_dir'] + '/' + 'bert-base-cased',
    ),
    data_helper_params=dict(
        bert_tokenizer_cls=BertTokenizer,
        bert_model_path=env['bert_dir'] + '/' + 'bert-base-cased',
        bert_tokenizer_params={'do_lower_case': False,}
    )
)
roberta_base_config = dict(
    model_params=dict(
        bert_cls=RobertaModel,
        bert_model_path=env['bert_dir'] + '/' + 'roberta-base',
    ),
    data_helper_params=dict(
        bert_tokenizer_cls=RobertaTokenizer,
        bert_model_path=env['bert_dir'] + '/' + 'roberta-base',
        bert_tokenizer_params={
            'vocab_file': env['bert_dir'] + '/' + 'roberta-base' + '/vocab.json',
            'merges_file': env['bert_dir'] + '/' + 'roberta-base' + '/merges.txt',
        }
    )
)
xlnet_base_config = dict(
    model_params=dict(
        bert_cls=XLNetModel,
        bert_model_path=env['bert_dir'] + '/' + 'xlnet-base-cased',
    ),
    data_helper_params=dict(
        bert_tokenizer_cls=XLNetTokenizer,
        bert_model_path=env['bert_dir'] + '/' + 'xlnet-base-cased',
        bert_tokenizer_params={
            'do_lower_case': False,
            'vocab_file': env['bert_dir'] + '/' + 'xlnet-base-cased' + '/spiece.model',
        }
    )
)

#######

bert_base__joint_config = update(bert_base_config, joint_config)
bert_base__siamese_config = update(bert_base_config, siamese_config)
roberta_base__joint_config = update(roberta_base_config, joint_config)
roberta_base__siamese_config = update(roberta_base_config, siamese_config)
xlnet_base__joint_config = update(xlnet_base_config, joint_config)
xlnet_base__siamese_config = update(xlnet_base_config, siamese_config)

#####


seq512_config = dict(
    data_helper_params=dict(
        max_seq_length=512,
    )
)
seq256_config = dict(
    data_helper_params=dict(
        max_seq_length=256,
    )
)
seq128_config = dict(
    data_helper_params=dict(
        max_seq_length=128,
    )
)
seq64_config = dict(
    data_helper_params=dict(
        max_seq_length=64,
    )
)

###

concat2d_config = dict(
    model_params=dict(
        concat='simple',
    )
)
concat3d_config = dict(
    model_params=dict(
        concat='3d-dif',
    )
)
concat4d_config = dict(
    model_params=dict(
        concat='4d-prod-dif',
    ),
    data_helper_params=dict(
        train_batch_size=3,  # 4D requires more GPU memory, use smaller batch size
    ),
)