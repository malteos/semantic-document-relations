# Semantic Relations between Wikipedia Articles

Implementation, trained models and result data for the paper. 
The supplemental material is available for download under [GitHub Releases](https://github.com/malteos/semantic-document-relations/releases).

## Getting started

**Requirements:**
- Python >= 3.6 (Conda)
- Jupyter notebook (for evaluation)
- GPU with CUDA-support (for training Transformer models)

At first we advise to create a new virtual environment for Python 3.6 with Conda:
```bash
conda create -n docrel python=3.6
conda activate docrel
```

Install all Python dependencies:
```bash
pip install -r requirements.txt
```

Download dataset (and pretrained models):

```bash
# Wikipedia corpus
wget https://github.com/malteos/semantic-document-relations/releases/download/untagged-c09ebf56a398180d57d7/enwiki-20191101-pages-articles.weighted.10k.jsonl.bz2 -O data/
bzip2 data/enwiki-20191101-pages-articles.weighted.10k.jsonl.bz2

# Train and test data
wget https://github.com/malteos/semantic-document-relations/releases/download/untagged-c09ebf56a398180d57d7/train_testdata__4folds.tar.gz -O data/
tar -xzf train_testdata__4folds.tar.gz

# Models

```

## Evaluation 

TODO

## Demo

TODO Notebook on Google Colab

## License

MIT