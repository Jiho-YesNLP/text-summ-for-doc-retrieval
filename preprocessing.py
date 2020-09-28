"""
preprocessing.py

This script generates model data and split into training datasets by processing
the TREC-PM reference data (topics and qrels) and associated PubMed documents.

Data formats in source files
============================

[qrels]

filename: qrels-abstracts-YYYY.txt  (/data/trec_ref/)
record format: [topic_id] [iteration] [document_id] [relevance]

```
1 0 10065107 0
1 0 10101594 1
1 0 10220412 0
...
```

[topics]

filename: topicsYYYY.xml  (/data/trec_ref/)

```
<topics task="2017 TREC Precision Medicine">
  <topic number="1">
    <disease>Liposarcoma</disease>
    <gene>CDK4</gene>
    <demographic>38-year-old male</demographic>
  </topic>
  <topic number="2">
    <disease>cholangiocarcinoma</disease>
    <gene>BRAF (V600E)</gene>
    <demographic>64-year-old male</demographic>
  </topic>
    ...
```

[PubMed documents]

filename: pubmedYYnNNNN.xml.gz  (/data/pubmed/)
description of XML data elements can be accessed from https://tinyurl.com/ya7blzp4

Training Examples
=================

Following elements consists one example:

- qid: topic id (e.g., t2018-05)
- did: document id (e.g., 27197542)
- src: source text in BERT indicies (i.e., document title + body)
- src_sent_lens: list of sentence lengths in source text
- tgtB: target text in BMET embedding indicies (i.e., concatenated field values
         of a patient topic, MeSH terms and keywords from a document)*
- tgtC: target text in BMET indicies
- tgtB_sent_lens: list of tgtB field value lengths
- tgtC_sent_lens: list of tgtC field value lengths
- token_labels: token-level relevances
- doc_label: document-level relevance

* Note, we consider five fields: (1) disease, (2) gene, (3) demographic,
(4) mesh terms, (5) keywords. Hence tgt_sent_lens should be a list of five
positive integers.
"""

import argparse
import logging
import os
from os.path import join as pjoin
import pickle
import re
from multiprocessing import cpu_count
import time
import gzip
from pathlib import Path
import random

from lxml import etree as et
from tqdm import tqdm, trange
import pysolr

import torch
from transformers import BertTokenizer
from stanfordnlp.server import CoreNLPClient

from utils import Tokenizer
from uts_api_client import UtsClient

logger = logging.getLogger('preprocessing')
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('pysolr').setLevel(logging.WARNING)

SOLR_URI = 'http://localhost:8983/solr/pubmed20'
CPU_CNT = max(1, cpu_count() - 2)


class ExsBuilder:
    """ExsBuilder produces a list of examples given a document set"""

    def __init__(self, bert_model='bert-base-uncased', file_emb='',
                 vocab_size=150000, min_src_nsents=1, max_src_nsents=50,
                 min_src_ntokens_per_sent=3, max_src_ntokens_per_sent=100):
        logger.info('=== Initializing a example builder'.ljust(80, '='))
        self.min_src_nsents = min_src_nsents
        self.max_src_nsents = max_src_nsents
        self.min_src_ntokens_per_sent = min_src_ntokens_per_sent
        self.max_src_ntokens_per_sent = max_src_ntokens_per_sent

        logger.debug(f'Loading BERT pre-trained model [{bert_model}]')
        self.tokB = BertTokenizer.from_pretrained(bert_model)
        self.tokC = None
        if file_emb != '':
            logger.debug('Loading the WBMET dictionary for custom tokenizer')
            self.tokC = Tokenizer(vocab_size=vocab_size)
            self.tokC.from_pretrained(file_emb)
        self.doc_lbl_freq = [0, 0]  # document-level [irrel, rel]
        self.ext_lbl_freq = [0, 0]  # token-level [irrel, rel]

    @staticmethod
    def tokenize(data, src_keys=['title', 'body'], tgt_key='text'):
        """Use Stanford CoreNLP tokenizer to tokenize all the documents."""
        REMAP = {"-LRB-": "(", "-RRB-": ")", "-LCB-": "{", "-RCB-": "}",
                 "-LSB-": "[", "-RSB-": "]", "``": '"', "''": '"'}
        with CoreNLPClient(annotators=['tokenize', 'ssplit'], threads=CPU_CNT)\
                as client:
            for did, d in tqdm(data.items()):
                text = ''
                for k in src_keys:
                    text += d[k] + ' '
                ann = client.annotate(text.strip())
                tokens = []  # list of tokenized sentences
                for sent in ann.sentence:
                    tokens.append([REMAP[t.word]
                                   if t.word in REMAP else t.word.lower()
                                   for t in sent.token])
                d[tgt_key] = tokens

    def encode(self, exs):
        """Convert sequences into indicies and create data entries for
        model inputs"""
        rtn = []
        logger.info('Encoding examples...')
        for qid, did, rel, doc, flds, mesh, keywords in tqdm(exs):
            entry = {
                'qid': qid, 'did': did,
                'src': [], 'src_sent_lens': [],
                'tgtB': [], 'tgtB_sent_lens': [],
                'tgtC': [], 'tgtC_sent_lens': []
            }

            # src
            for s in doc:  # CoreNLP tokenized sequences (list of sentences)
                if len(s) <= self.min_src_ntokens_per_sent:
                    continue
                src_str = ' '.join(s[:self.max_src_ntokens_per_sent])
                entry['src'] += self.tokB.convert_tokens_to_ids(
                    self.tokB.tokenize(src_str)
                )
                entry['src_sent_lens'].append(len(entry['src']))
            if len(entry['src']) == 0:
                continue

            # tgt - fields
            tgt_tokens = set()  # Used in identifying token-level labels
            for seq in flds:  # flds (disease, gene, demo)
                # BERT
                ids = self.tokB.convert_tokens_to_ids(self.tokB.tokenize(seq))
                tgt_tokens.update(ids)
                entry['tgtB'] += ids
                entry['tgtB_sent_lens'].append(len(entry['tgtB']))
                # BMET
                ids = self.tokC.convert_tokens_to_ids(self.tokC.tokenize(seq))
                ids = list(filter(lambda x: x > 1, ids))  # Remove UNKs
                entry['tgtC'] += ids
                entry['tgtC_sent_lens'].append(len(entry['tgtC']))

            # tgt - mesh
            mesh = [f'εmesh_{t}' for t in mesh[0].lower().split()]
            ids = self.tokC.convert_tokens_to_ids(mesh)
            ids = list(filter(lambda x: x > 1, ids))  # Remove UNKs
            entry['tgtC'] += ids
            entry['tgtC_sent_lens'].append(len(entry['tgtC']))

            # tgt - keywords
            seq = ' '.join(keywords)
            ids = self.tokC.convert_tokens_to_ids(self.tokC.tokenize(seq))
            ids = list(filter(lambda x: x > 1, ids))  # Remove UNKs
            tgt_tokens.update(ids)
            entry['tgtC'] += ids
            entry['tgtC_sent_lens'].append(len(entry['tgtC']))
            entry['token_labels'] = \
                [1 if t in tgt_tokens else 0 for t in entry['src']]
            sum_ = sum(entry['token_labels'])
            self.ext_lbl_freq[0] += len(entry['token_labels']) - sum_
            self.ext_lbl_freq[1] += sum_
            entry['doc_label'] = 0 if rel == 0 else 1
            rtn.append(entry)
        return rtn

    def build_trec_exs(self, topics, docs):
        """For each topic and doc pair, encode them, and construct example list
        """
        exs = list()
        # Tokenize document using Stanford CoreNLP Tokenizer
        logger.debug('Tokenizing %s documents using Stanford CoreNLP '
                     'Tokenizer...', len(docs))
        self.tokenize(docs)

        # Add positive examples
        for qid in topics:
            for did, rel in topics[qid]['docs']:
                if did not in docs or \
                        len(docs[did]['text']) < self.min_src_nsents:
                    continue
                d = docs[did]
                # Complete keywords: doc_keywords > doc_mesh > q_mesh
                keywords = d['keywords'] if len(d['keywords']) > 0 \
                    else d['mesh_names']
                if len(keywords) == 0 and rel > 0:
                    keywords = [topics[qid]['mesh'][1]]

                exs.append((qid, did, rel, d['text'][:self.max_src_nsents],
                            topics[qid]['fields'], topics[qid]['mesh'],
                            keywords))
                self.doc_lbl_freq[int(rel > 0)] += 1

        # Add negative examples
        neg_docs_ids = [did for did, d in docs.items() if not d['pos']]
        qids = random.choices(list(topics.keys()), k=len(neg_docs_ids))
        for i, did in enumerate(neg_docs_ids):
            exs.append((qids[i], did, 0,
                        docs[did]['text'][:self.max_src_nsents],
                        topics[qid]['fields'], topics[qid]['mesh'], []))
            self.doc_lbl_freq[0] += 1
        random.shuffle(exs)
        rtn = self.encode(exs)

        return rtn

    # todo. Following function will be changed
    def build(self, examples, docs):
        """Bulding examples is done in two modes: one for data preparation and
        the other for prediction.

        In data preparation,
        - `exs` are quries in TREC ref datasets
        - `docs` consists of pos and neg documents prepared by `read_pubmed_docs`

        In prediction,
        - `exs` only contains one query with no labels
        - `docs` the retrieved documents from Solr search results

        """
        # Tokenize documents and build examples with doc_labels
        exs = []
        # Title and Text are multivalued ('text_general' in Solr)
        results = docs
        docs = {}
        for r in results:
            title = ' '.join(r['ArticleTitle']
                             if 'ArticleTitle' in r else [])
            body = ' '.join(r['AbstractText']
                            if 'AbstractText' in r else [])
            docs[r['id']] = (title + ' ' + body).strip()
        logger.debug(f'Tokinizing {len(docs)} retrieved docs...')
        pos_docs = self.tokenize(docs)

        # Build examples (with dummy label -1)
        qid = list(examples.keys())[0]  # There's only one anyways
        logger.info(f'Preparing examples for {qid}...')
        for did, text in pos_docs.items():
            if len(pos_docs[did]) < self.min_src_nsents:
                continue
            exs.append((qid, did, -1, pos_docs[did][:self.max_src_nsents],
                        examples[qid]['topics']))

        data = self.encode(exs)
        return data

def xstr(s):
    return '' if s is None else str(s)

def year2group(txt):
    """Map year to age group"""
    age_group_map = [
        (0, 'infant newborn', 'D007231'),
        (0.5, 'infant', 'D007223'),
        (2, 'child preschool', 'D002675'),
        (5, 'child', 'D002648'),
        (12, 'adolescent', 'D000293'),
        (18, 'adult', 'D000328'),
        (44, 'middle aged', 'D008875'),
        (64, 'aged', 'D000368'),
        (79, 'aged old', 'D000369')
    ]
    pattern = r"(\d+)-year-old\s(male|female)"
    m = re.match(pattern, txt.lower())
    if m:
        age = int(m.group(1))
        group_id = 0
        for i, rec in enumerate(age_group_map):
            if age < rec[0]:
                group_id = i
            else:
                break
        group = age_group_map[group_id][1] + ' ' + m.group(2)
        group_mesh = [age_group_map[group_id][2]]
        if m.group(2) == 'male':
            group_mesh.append('D008297')
        elif m.group(2) == 'female':
            group_mesh.append('D005260')
    return group, group_mesh


def lookup_mesh(phrase, client, mapping):
    """Search MeSH terms in a phrase and build a MeSH dict via UTS API"""
    if phrase in mapping:
        return mapping[phrase]
    rst = client.get_mesh_by_term_search(phrase)
    print('{} {} => {} {}\r'.format(len(mapping), phrase, rst['ui'], ' '*50),
          end='')
    if rst['ui'] != 'NONE':
        mapping[phrase] = (rst['ui'], rst['name'])
    else:
        mapping[phrase] = None
    time.sleep(1)
    return mapping[phrase]


def read_trec_ref(dir_trec, years=None):
    """Read topics and qrels, look up MeSH terms from the fields, and return 
    a list of topics along with the their document relevance judgments"""
    logger.info('=== Reading TREC reference files '.ljust(80, '='))
    # Return cached, if exists
    if os.path.exists(pjoin(dir_trec, 'trec_ref.pkl')):
        logger.warning('Reading topics from a cached file. If you want to '
                       're-construct MeSH mapping, delete the cached file '
                       'and run this script again.')
        topics = pickle.load(open(pjoin(dir_trec, 'trec_ref.pkl'), 'rb'))
        return {k: v for k, v in topics.items() if k[1:5] in years}

    topics = dict()
    fields = ['disease', 'gene', 'demographic']
    for y in [2017, 2018, 2019]:  # First, read all topics (2017--2019) and save
        t_cnt = 0
        # Read topics
        with open(pjoin(dir_trec, f'topics{y}.xml')) as f:
            data = et.parse(f)
            for t in data.iterfind('topic'):
                qid = 't{}-{:02}'.format(y, int(t.get('number')))
                topics[qid] = {'fields': [None] * len(fields), 'docs': []}
                for i, fld in enumerate(fields):
                    txt = t.find(fld).text.lower()
                    if fld == 'demographic':
                        age_group, _ = year2group(txt)
                        txt = age_group
                    else:
                        txt = re.sub(r'[\-,]', ' ', txt)
                        txt = re.sub(r'[()]', '', txt)
                    topics[qid]['fields'][i] = txt
                t_cnt += 1
        logger.info('%s topics found from topics%s.xml', t_cnt, y)

        # Associate doc relevances
        with open(pjoin(dir_trec, f'qrels-abstracts-{y}.txt')) as f:
            for line in f:
                num, _, docid, relevancy = line.split()
                qid = 't{}-{:02}'.format(y, int(num))
                topics[qid]['docs'].append((docid, int(relevancy)))

    # Look up MeSHes
    logger.info('MeSH mapping using a UTS API client...')
    uts = UtsClient()
    mesh_mapping = dict()
    for qid in topics:
        mesh_codes = list()
        mesh_names = list()
        for i, fld in enumerate(fields):
            phrase = topics[qid]['fields'][i]
            if fld == 'disease':
                mesh = lookup_mesh(phrase, uts, mesh_mapping)
                if mesh is not None:
                    mesh_codes.append(mesh[0])
                    mesh_names.append(mesh[1])
            else:
                for token in phrase.split():
                    mesh = lookup_mesh(token, uts, mesh_mapping)
                    if mesh is not None:
                        mesh_codes.append(mesh[0])
                        mesh_names.append(mesh[1])
        topics[qid]['mesh'] = (' '.join(mesh_codes), ' '.join(mesh_names))

    logger.info('Saving topics to trec_ref.pkl')
    pickle.dump(topics, open(pjoin(dir_trec, 'trec_ref.pkl'), 'wb'))

    return {k: v for k, v in topics.items() if k[1:5] in years}


def read_pubmed_docs(topics):
    """Preprocess documents; retrieve all documents which occur at least once
    in the relavance judgements and random sample the same number of
    documents from PubMed to be used as negative examples"""

    logger.info('=== Retrieving PubMed docs '.ljust(80, '='))
    pos_doc_ids = set()
    docs = dict()

    # Identify all distinct documents in topics
    for qid in topics:
        pos_doc_ids.update([did for did, _ in topics[qid]['docs']])

    solr_client = pysolr.Solr(SOLR_URI)
    job_size = 1000

    # Positive docs
    lst_pos_docs = list(pos_doc_ids)
    for i in range(0, len(lst_pos_docs), job_size):
        res = solr_client.search(
            q="id:({})".format(' '.join(lst_pos_docs[i:i+job_size])),
            qf=['ArticleTitle', 'AbstractText'],
            fq=['AbstractText:*'],
            rows=job_size
        )
        for r in res:
            docs[r['id']] = {
                'title': ' '.join(r['ArticleTitle']) if 'ArticleTitle' in r else '',
                'body': ' '.join(r['AbstractText']),
                'keywords': r['Keyword'] if 'Keyword' in r else [],
                'mesh_names': r['MeshDescriptorName'] if 'MeshDescriptorName' in r else [],
                'pos': True
            }
    pos_len = len(docs.keys())
    logger.info('- %s Positive docs retrieved', pos_len)

    # Negative docs
    res = solr_client.search(
        q='*:*',
        sort='random_1234 desc',
        fq=['AbstractText:*'],
        rows=int(pos_len)
    )
    for r in res:
        if r['id'] not in docs.keys():
            docs[r['id']] = {
                'title': ' '.join(r['ArticleTitle']) if 'ArticleTitle' in r else '',
                'body': ' '.join(r['AbstractText']),
                'keywords': r['Keyword'] if 'Keyword' in r else [],
                'mesh_names': r['MeshDescriptorName'] if 'MeshDescriptorName' in r else [],
                'pos': False
            }
    logger.info('- %s Negative docs retrieved', len(docs)-pos_len)

    return docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--dir_trec', type=str, default='data/trec_ref',
                        help='Path to directory of TREC reference data files')
    parser.add_argument('--dir_pubmed', type=str, default='data/pubmed',
                        help='Path to directory of PubMed data files')
    parser.add_argument('--dir_out', type=str, default='data/tasumm',
                        help='Path to directory for TASumm inputs/outputs')

    # Runtime environment
    parser.add_argument('--years', type=str, default='2017,2018,2019',
                        help='Years of TREC reference data to read in')
    parser.add_argument('--num_exs', type=int, default=20000,
                        help='Number of examples to store in each data file')

    # Lengths
    parser.add_argument('--min_src_nsents', type=int, default=2,
                        help='Min number of sentences required for an example')
    parser.add_argument('--max_src_nsents', type=int, default=100,
                        help='Max number of sentences required for an example')
    parser.add_argument('--min_src_ntokens_per_sent', type=int, default=3,
                        help='Min number of tokens in a sentence')
    parser.add_argument('--max_src_ntokens_per_sent', type=int, default=100,
                        help='Max number of tokens in a sentence')

    # Use other embeddings
    parser.add_argument('--file_emb', type=str, default='',
                        help='Path to a word embedding file')
    parser.add_argument('--vocab_size', type=int, default=150000,
                        help='Vocabulary size of word embeddings')

    args = parser.parse_args()

    # Logger
    log_lvl = logging.INFO
    logging.basicConfig(
        level=log_lvl,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Set defaults
    args.years = args.years.split(',')

    # Preprocess --------------------------------------------------------------
    # Step 1) Read reference data;
    #         load from cache (/data/trec_ref/trec_ref.pkl) if exists
    TOPICS = read_trec_ref(args.dir_trec, args.years)

    # Step 2) Retrieve all the associating documents from PubMed baseline data
    DOCS = read_pubmed_docs(TOPICS)

    # Step 3) Build list of examples
    builder = ExsBuilder(file_emb=args.file_emb)
    data = builder.build_trec_exs(TOPICS, DOCS)

    logger.info('REL class frequency [irrel, rel] is %s. ',
                builder.doc_lbl_freq)
    logger.info('EXT class frequency [irrel, rel] is %s. ',
                builder.ext_lbl_freq)

    # Save --------------------------------------------------------------------
    # Save data into PyTorch files
    # ds_keys = {'train': (0, 1.0)}
    ds_keys = {'train': (0, 0.8), 'valid': (0.8, 1.0)}
    if not os.path.exists(args.dir_out):
        logger.info('mkdirs %s', args.dir_out)
        os.makedirs(args.dir_out, exist_ok=True)
    for ds, (lb, ub) in ds_keys.items():
        batch = []
        batch_cnt = 0
        slice_ = slice(int(lb * len(data)), int(ub * len(data)))
        for ex in data[slice_]:
            batch.append(ex)
            if len(batch) >= args.num_exs:
                dest_file = pjoin(args.dir_out, f'{ds}_{batch_cnt}.pt')
                torch.save(batch, dest_file)
                logger.info(f'{dest_file} saved')
                batch_cnt += 1
                batch = []
        if len(batch) > 0:
            dest_file = pjoin(args.dir_out, f'{ds}_{batch_cnt}.pt')
            torch.save(batch, dest_file)
            logger.info(f'{dest_file} saved')
