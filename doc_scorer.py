""" doc_scorer.py
Compute document scores given a patient case. In test mode, run on a list of
topics to compare the system against TREC-PM results.

Scores can be composed of four different measures (original DR score, REL,
ROUGE, and query_sim)

We use Reciprocal Rank Fusion (RRF) to combine scores and produce ranked list
of documents.
"""
import logging
import argparse
from collections import defaultdict
import re
import sys

from rouge import Rouge

import pysolr
import torch

from model import DocRelClassifier
from preprocessing import ExsBuilder, read_trec_ref
from data import DataLoader
from doc2query import Summarizer
import utils

logger = logging.getLogger('doc_scorer')
logging.getLogger('pysolr').setLevel(logging.WARNING)

SOLR_URI = 'http://localhost:8983/solr/pubmed20'

# Solr Query Parameters (https://tinyurl.com/y82kybe8)
SOLR_Q_PARAMS = {
    'defType': 'edismax',
    'qf': ['AbstractText', 'ArticleTitle'],    # query fields
    'fq': "AbstractText:*",                    # filter query
    'fl': '*,score',                           # fields to return
}
SUMM_PARAMS = {
    'n_best': 2,
    'min_length': 1,
    'max_length': 50,
    'beam_size': 4
}
EX = {
    'qid000': {
        'topics': [
            'gastric cancer',
            'ERBB2 amplification',
            'aged male',
            'D013274 D018734 D005784',
            'adenocarcinoma diagnosis erbb receptors esophagogastric junction'
        ]
    }
}

MDL = {'name': '', 'model': None, 'args': None}
EXS_BUILDER = None


class DocSummary:
    """DocSummary contains pseudo-query sentences and ext highlighted words
    that are used in building a document dependant query"""

    def __init__(self, doc_id, src):
        self.doc_id = doc_id  # Document id
        self.src = src
        self.pred_sents = {}
        self.ext_keywords = set()

    def get_doc_q(self):
        terms = set()
        for tp, sent in self.pred_sents.items():
            if isinstance(sent, list):
                sent = ' '.join(sent)
            terms.update([t for t in sent.split() if not t.startswith('[')])
        logger.debug(f'{terms} | {self.ext_keywords}')
        terms.update(self.ext_keywords)
        doc_q = ' '.join(terms)
        return doc_q

    def update(self, tp_token, pred_sent):
        self.pred_sents[tp_token] = pred_sent


def load_rel_mdl(mdl_f):
    if MDL['name'] == 'REL':
        return
    logger.debug(f'Loading REL model from {mdl_f}...')
    MDL['name'] = 'REL'
    data = torch.load(mdl_f, map_location=lambda storage, loc: storage)
    MDL['model'] = DocRelClassifier()
    MDL['model'].load_state_dict(data['model'], strict=True)
    MDL['model'].to(args.device).eval()
    MDL['args'] = data['args']


def load_abs_mdl(mdl_f):
    if MDL['name'] == 'ABS':
        return
    logger.debug('Loading ABS model from {}...'.format(mdl_f))
    MDL['name'] = 'ABS'
    MDL['model'] = Summarizer(mdl_f, **SUMM_PARAMS)
    MDL['model'].eval()
    MDL['args'] = MDL['model'].abs_model.args   # protocol


def compute_rel_scores(q, docs, scoreboard):
    """
    Compute document relevance score by a trained REL model and
    update the given scoreboard
    """
    if MDL['name'] != 'rel':
        load_rel_mdl(args.mdl_rel)
    data = EXS_BUILDER.build(q, docs)
    it = DataLoader(data, 'rel', 32, MDL['args'].max_ntokens_src,
                    *utils.get_special_tokens())
    with torch.no_grad():
        for batch in it:
            _, logits = MDL['model'](batch)
            for s, did in zip(logits, batch.did):
                # scoreboard[qid][did]['scores'].append(s)
                scoreboard[qid][did]['scores'].append(
                    torch.sigmoid(s[1]).item())


def compute_abs_scores(q, docs, scoreboard, ret_queries=False,
                       ext_scores_threshold=1.2):
    """Run a pretrained ABS on retrieved docs, update scoreboard, and return
    pseudo-queries if indicated"""
    if MDL['name'] != 'abs':
        load_abs_mdl(args.mdl_abs)
    data = EXS_BUILDER.build(q, docs)
    qid = list(q.keys())[0]
    topics = q[qid]['topics']  # Renaming for convenience
    ma = MDL['args']
    m = MDL['model']
    tokB = EXS_BUILDER.tokB
    bert_cont_tokens = {i for k, i in tokB.vocab.items() if k.startswith('##')}
    it = DataLoader(data, 'abs', ma.batch_size, ma.max_ntokens_src,
                    m.spt_ids_B, m.spt_ids_C, m.eos_mapping)
    q_ = dict()
    with torch.no_grad():
        for i, batch in enumerate(it):
            print(f'Translating batch #{i}\r', end='')
            results = MDL['model'].translate(batch)
            translations = MDL['model'].results_to_translations(results)
            for t in translations:
                if t.did not in q_:
                    q_[t.did] = DocSummary(t.did, t.src)
                    q_[t.did].ext_keywords = \
                        get_ext_keywords(t.ext_scores, t.src_input,
                                         bert_cont_tokens, tokB,
                                         ext_scores_threshold)
                q_[t.did].update(t.topic, t.pred_sents)

    # For each document, build a doc-query and compute sim score to user-query
    user_query = topics[0].lower() + ' ' + topics[1].lower() + ' '
    user_query += ' '.join(
        [f'εmesh_{t.lower()}' for t in topics[3].split()])
    user_query += ' ' + 'treatment therapy human drugs prognastic clinical'
    logger.debug(f'\n\n{qid} user_query: {user_query}')

    for did, doc_query in q_.items():
        doc_q = doc_query.get_doc_q()
        logger.debug(f'{did} doc_query: {doc_q}')
        score1, score2 = q_sim(user_query, doc_q)
        logger.debug(f'score1: {score1}, score2: {score2}')
        scoreboard[qid][did]['scores'].append(score1)
        scoreboard[qid][did]['scores'].append(score2)
        # self.scoreboard[qid][did]['scores'].append(score2)

    if ret_queries:
        return q_


def q_sim(q, h):
    assert MDL['name'] == 'ABS' and MDL['model'] is not None
    tokenizer = MDL['model'].tokenizerC
    dec_embeddings = MDL['model'].abs_model.decoder.dec_embeddings

    # Unigram r score
    rouge = Rouge()
    score1 = rouge.get_scores(h, q)[0]['rouge-1']['r']

    # cossim between embeddings
    q = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(q))
    h = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(h))
    # remove UNKs
    q = [i for i in q if i != 1]
    h = [i for i in h if i != 1]
    q_emb = dec_embeddings(torch.LongTensor(q).cuda())
    h_emb = dec_embeddings(torch.LongTensor(h).cuda())
    score2 = 0
    for i, h_ in enumerate(h_emb):
        sims = []
        for j, q_ in enumerate(q_emb):
            sims.append(torch.cosine_similarity(h_.view(1, -1),
                                                q_.view(1, -1)).item())
        score2 += max(sims)
    score2 /= len(h_emb)

    return score1, score2


def get_ext_keywords(ext_scores, doc, bert_cont_tokens, tokenizerB,
                     ext_scores_threshold=1.2):
    """Given the extractive model scores and input document, detokenize
     BERT indicies to get a list of keywords sets"""
    mask = ext_scores[:, -1].gt(torch.full(ext_scores.size()[:-1],
                                           ext_scores_threshold).cuda())
    highlighted_words = set()
    # forward
    for i in range(len(mask) - 1):
        if mask[i] and ~mask[i + 1] and \
                doc[i + 1].item() in bert_cont_tokens:
            mask[i + 1] = True
    # backward
    for i in range(len(mask) - 1, 0, -1):
        if mask[i] and ~mask[i - 1] and \
                doc[i].item() in bert_cont_tokens:
            mask[i - 1] = True

    mask_filled = doc.masked_fill(~mask, 103)  # 103 for [MASK] in BERT
    words = tokenizerB.decode(mask_filled, skip_special_tokens=True)
    words = re.sub(r'[\-,\.]', ' ', words)
    highlighted_words.update(words.split())
    return highlighted_words


def compute_RRF(scoreboard, n_scores, k=60):
    for qid, docs in scoreboard.items():
        for i in range(n_scores):
            ranked_docs = sorted(docs.items(), key=lambda t: t[1]['scores'][i],
                                 reverse=True)
            for r, (did, rec) in enumerate(ranked_docs):
                if 'rank_score' not in rec:
                    rec['rank_score'] = []
                rec['rank_score'].append(1 / (k + r + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Run this script in debug mode')
    parser.add_argument('--dir_trec', type=str, default='data/trec_ref',
                        help='Path to directory of TREC reference data files')
    parser.add_argument('--solr_rows', type=int, default=10,
                        help='Number of documents to retrieve by Solr')
    parser.add_argument('--mdl_rel', type=str, default='',
                        help='Path to a trained REL model')
    parser.add_argument('--mdl_abs', type=str, default='',
                        help='Path to a pretrained ABS model')
    parser.add_argument('--f_emb', type=str, default='',
                        help='Path to a pretrained embeddings file')
    parser.add_argument('--ref_year', type=str, default='',
                        help='Year of the TREC PM reference for test')
    parser.add_argument('--ext_scores_threshold', type=float, default=1.2,
                        help='Minimum logit value of EXT model for limiting '
                             'keyword selection from the source document')
    args = parser.parse_args()

    # Logger
    log_lvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_lvl,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Initialization ----------------------------------------------------------
    # Set defaults
    SOLR_Q_PARAMS['rows'] = args.solr_rows
    if not torch.cuda.is_available():
        raise RuntimeError('Running in CPU mode is not available')
    args.device = torch.device('cuda')
    test_mode = (args.ref_year != '')
    scoreboard = defaultdict(dict)

    # Initialize Solr client
    logger.debug('Initializing a Solr client')
    solr_client = pysolr.Solr(SOLR_URI)
    solr_mlt = pysolr.Solr(SOLR_URI, search_handler='/mlt')

    # Initialize example builder
    EXS_BUILDER = ExsBuilder(file_emb=args.f_emb)

    # Run on TREC ref ---------------------------------------------------------
    if test_mode:
        n_scores = 1
        # Read topics from ref file of the year
        exs = read_trec_ref(args.dir_trec, args.ref_year.split(','))

        # Retrieve documents
        ret_docs = {}

        # for qid, data in exs.items():
        #     q = ' '.join(data['fields'][:3])
        #     res = solr_client.search(q, **SOLR_Q_PARAMS)
        #     ret_docs[qid] = res
        #     print(qid, len(res))
        #     for r in res:
        #         scoreboard[qid][r['id']] = {'scores': [r['score']]}

        # debug. more like this
        for qid, data in exs.items():
            q = ' '.join(data['fields'][:3])
            SOLR_Q_PARAMS['rows'] = 10
            res = solr_client.search(q, **SOLR_Q_PARAMS)
            q = ' '.join(['id:{}'.format(r['id']) for r in res])
            params = {
                'mlt.mindf': 100,   
                'mlt.boost': 'true'
            }
            print('mlt:', qid)
            similar = solr_mlt.more_like_this(q,
                                              rows=args.solr_rows,
                                              fl="*,score",
                                              mltfl='AbstractText',
                                              **params)

            ret_docs[qid] = similar
            for r in similar:
                scoreboard[qid][r['id']] = {'scores': [r['score']]}

        # Run REL
        if args.mdl_rel != '':
            n_scores += 1
            load_rel_mdl(args.mdl_rel)
            logger.debug('Computing REL scores...')
            for qid, data in exs.items():
                compute_rel_scores({qid: data}, ret_docs[qid], scoreboard)

        # Run ABS
        if args.mdl_abs != '':
            n_scores += 2
            load_abs_mdl(args.mdl_abs)
            logger.debug('Computing ABS scores...')
            for qid, data in exs.items():
                compute_abs_scores(
                    {qid: data}, ret_docs[qid], scoreboard,
                    ext_scores_threshold=args.ext_scores_threshold
                )

        compute_RRF(scoreboard, n_scores)

        # Sort the results by qid
        print_outs = []
        for qid in sorted(scoreboard, key=lambda k: int(k[-2:])):
            for did, scores in scoreboard[qid].items():
                print_outs.append('{} dummy {} {} {:.8f} run-name\n'
                                  ''.format(int(qid[-2:]), did, len(print_outs),
                                            sum(scores['rank_score'])))
        with open('test.rel', 'w') as f:
            for l in print_outs:
                f.write(l)

    # debug mode ---------------------------------------------------------------
    else:  # debugging with an example query
        # Retrieve documents
        qid = list(EX.keys())[0]
        q = ' '.join(EX[qid]['topics'][:3])
        res = solr_client.search(q, **SOLR_Q_PARAMS)
        logger.debug('%s documents retrieved by Solr query', len(res))
        for r in res:
            scoreboard[qid][r['id']] = {'scores': [r['score']]}

        # Run REL
        if args.mdl_rel != '':
            load_rel_mdl(args.mdl_rel)
            compute_rel_scores(EX, res, scoreboard)

        # Run ABS
        if args.mdl_abs != '':
            load_abs_mdl(args.mdl_abs)
            compute_abs_scores(EX, res, scoreboard,
                               ext_scores_threshold=args.ext_scores_threshold)

