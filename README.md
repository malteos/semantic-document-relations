# Semantic Relations between Wikipedia Articles

<a href="https://colab.research.google.com/github/malteos/semantic-document-relations/blob/master/demo_wikidocrel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3713183.svg)](https://doi.org/10.5281/zenodo.3713183)


Implementation, trained models and result data for the paper **Pairwise Multi-Class Document Classification for Semantic Relations between Wikipedia Articles** [(PDF on Arxiv)](https://arxiv.org/abs/2003.09881). 
The supplemental material is available for download under [GitHub Releases](https://github.com/malteos/semantic-document-relations/releases) or [Zenodo](https://doi.org/10.5281/zenodo.3713183).

![Wikipedia Relations](https://github.com/malteos/semantic-document-relations/raw/master/wikipedia_example.png)

## Getting started

**Requirements:**
- Python >= 3.7 (Conda)
- Jupyter notebook (for evaluation)
- GPU with CUDA-support (for training Transformer models)

At first we advise to create a new virtual environment for Python 3.7 with Conda:
```bash
conda create -n docrel python=3.7
conda activate docrel
```

Install all Python dependencies:
```bash
pip install -r requirements.txt
```

Download dataset (and pretrained models):

```bash
# Navigate to data directory
cd data

# Wikipedia corpus
wget https://github.com/malteos/semantic-document-relations/releases/download/1.0/enwiki-20191101-pages-articles.weighted.10k.jsonl.bz2
bzip2 enwiki-20191101-pages-articles.weighted.10k.jsonl.bz2

# Train and test data
wget https://github.com/malteos/semantic-document-relations/releases/download/1.0/train_testdata__4folds.tar.gz
tar -xzf train_testdata__4folds.tar.gz

# Models
wget https://github.com/malteos/semantic-document-relations/releases/download/1.0/model_wiki.bert_base__joint__seq512.tar.gz
tar -xzf model_wiki.bert_base__joint__seq512.tar.gz
```


## Experiments 


Run predefined experiment (settings can be found in `experiments/predefined/wiki`)
```bash
# Config: wiki.bert_base__joint__seq128
# GPU ID: 1 (set via CUDA_VISIBLE_DEVICES=1)
# Output dir: ./output
python cli.py run ./output 1 wiki.bert_base__joint__seq512
```


## Demo

You can run a Jupyter notebook on Google Colab:

<a href="https://colab.research.google.com/github/malteos/semantic-document-relations/blob/master/demo_wikidocrel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## How to cite

If you are using our code, please cite [our paper](https://arxiv.org/abs/2003.09881):

```bibtex
@InProceedings{Ostendorff2020,
  title = {Pairwise Multi-Class Document Classification for Semantic Relations between Wikipedia Articles},
  booktitle = {Proceedings of the {ACM}/{IEEE} {Joint} {Conference} on {Digital} {Libraries} ({JCDL})},
  author = {Ostendorff, Malte and Ruas, Terry and Schubotz, Moritz and Gipp, Bela},
  year = {2020},
  month = {Aug.},
}
```

## License

MIT
