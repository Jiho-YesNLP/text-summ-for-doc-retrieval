# Document Reranking for Precision Medicine with Neural Matching and Faceted Summarization

Neural document matching model and neural text summarization model for scoring documents in reranking purpose; `REL` is a document-relevance classification model. `EXT` is an extractive summarization model which identifies relevant words to the trained theme from the source document. `ABS` is an abstractive model which generates query-like sentences by given topic signal.

## Requirements

- Stanford CoreNLP library
- Python3.7+
- PyTorch 1.4
- Solr


## Data Preparation

We use the PubMed abstracts and the TREC-Precision Medicine document retrieval reference datasets. Followings are the links to the resources.

- [TREC PM/CDS HOME](http://www.trec-cds.org/)
- [PubMed: Medline Annual Base](ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/)

### Download data files

**PubMed**

Run [download script](data/pubmed/download.py); it will download the compressed PubMed files up to the specified numeric argument given. (e.g., pubmed20n[0001--1015].xml.gz)

```python
mkdir -p data/pubmed/
cd data/pubmed/
python download.py 1015
```

**TREC relevance judgment files**

TREC relevance judgment files are in [data/trec_ref/](data/trec_ref].

**BMET embeddings**

We use the BMET embeddings which are pretrained embeddings for the PubMed articles annotated with the MeSH (Medical Subject Headings) entities. This embeddings contain MeSH codes in this format: `𝛆mesh_D00001`. You can loaded pretrained the BMET embeddings from [here](https://drive.google.com/file/d/1s1Axugg--VoKq-pWzI95wGIWe0tPs1q0/view?usp=sharing)

### Install Stanford CoreNLP

We use the Stanford CoreNLP to pre-tokenize documents. You can download it from [here](https://stanfordnlp.github.io/CoreNLP/) or use a programming language-specific library.

After client installation, let the shell environment know where the library is located:

```shell script
export CORENLP_HOME="[path_to]/stanford-corenlp-full-2018-10-05"
export CLASSPATH="$CORENLP_HOME/stanford-corenlp-3.9.2.jar:$CLASSPATH"
```

Replace `[path_to]` with the path you saved the Stanford CoreNLP. `CORENLP_HOME` is needed for the Python `stanfordnlp` package which runs the installed program as a server.

### Run preprocessing.py

In order to run proprocessing script, PubMed corpus needs to be imported in Solr database. Instruction to install and import PubMed corpus into a Solr core is available in this [documentation](https://tinyurl.com/yyxwx8wv).

`preprocessing.py` will generate training datasets split into train/valid sets. Preprocessed datsets are also available for downloading [[here](https://drive.google.com/file/d/1bHRhDV-SYJysr_w7mdGg7Yc4ep__S0gj/view?usp=sharing)]

```shell script
python preprocessing.py --dir_out [PATH_OUT] --dir_pubmed [PATH_PUBMED] --dir_trec [PATH_TREC] --file_emb [PATH_EMB]
```

e.g.
```
python preprocessing.py --dir_out data/tasumm --dir_pubmed data/pubmed --dir_trec data/trec_ref --file_emb data/wbmet-1211.vec
```

Specify the paths to TREC reference files (`PATH_TREC`), PubMed documents (`PATH_PUBMED`), and where to store processed data files (`PATH_OUT`).


## Train

We use a pretrained BERT under the hood, and it has some issues of running in CPU mode with the pretrained parameters. So, we assume that you train this model in GPU mode. 

### REL

REl model is trained for document-query matching which predicts how probable the given document is relevant to a query (patient case). Following command starts training:

```
% python train.py --model_type rel 
Jun01 11:42 __main__ INFO: [ === Experiment exp06011142 =============================================================== ]
Jun01 11:42 __main__ INFO: [ Start training rel model  ]
Jun01 11:42 __main__ INFO: [ *** Epoch 1 *** ]
Jun01 11:43 utils INFO: [ steps: 100 loss: 0.4340 recall: 0.6704 prec.: 0.4450 lr 0.000010 time: 59.22s ]
Jun01 11:44 utils INFO: [ steps: 200 loss: 0.3455 recall: 0.2976 prec.: 0.6117 lr 0.000010 time: 118.37s ]
...
```

### EXT

EXT model is trained for keyword extraction from the source document. Following command starts training:

```
% python train.py --model_type ext
Jun01 11:55 __main__ INFO: [ === Experiment exp06011155 =============================================================== ]
Jun01 11:55 __main__ INFO: [ Start training ext model  ]
Jun01 11:55 __main__ INFO: [ *** Epoch 1 *** ]
Jun01 11:56 utils INFO: [ steps: 100 loss: 0.4564 recall: 0.0109 prec.: 0.8920 lr 0.000010 time: 54.62s ]
Jun01 11:57 utils INFO: [ steps: 200 loss: 0.3747 recall: 0.1938 prec.: 0.8701 lr 0.000010 time: 108.54s ]
...
```

### ABS

ABS model is trained for text summarization for pseudo-query generation. It requires a pretrained EXT model and custom embeddings. Running command should be like:

```
% python train.py --model_type abs --file_trained_ext [path to pretrained ext model] --file_dec_emb [path to wbmet embeddings] --batch_size 6
Jun01 12:07 __main__ INFO: [ === Experiment exp06011207 =============================================================== ]
Jun01 12:08 model INFO: [ Loading a pre-trained extractive model from data/models/ext_26000000_exp05251635.pt... ]
Jun01 12:08 __main__ INFO: [ Start training abs model  ]
Jun01 12:08 __main__ INFO: [ *** Epoch 1 *** ]
Jun01 12:09 utils INFO: [ steps: 100 loss: 16.8745 lr p0/0.000010, p1/0.001000 time: 34.38s ]
Jun01 12:09 utils INFO: [ steps: 200 loss: 6.5567 lr p0/0.000010, p1/0.001000 time: 68.20s ]
...
```

