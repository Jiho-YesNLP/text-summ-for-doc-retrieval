""" utils.py

Common tools used in multiple files
"""

import os
import time
import logging
from pathlib import Path
from collections import defaultdict

import torch

from transformers import BertTokenizer

logger = logging.getLogger('utils')

# Globally used constants ------------------------------------------------------
# Functional tokens for BERT and other embeddings ([PAD] should be indexed at 0)
# [unused--] and [unused1--] for each topic
SP_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
for i in range(200):  # Add [unused0] to [unused199]
    SP_TOKENS.append(f'[unused{i}]')


class Statistics:
    """Keep track of loss statistics in training"""

    def __init__(self):
        self.epoch = 0
        self.steps = 0
        self.avg_train_loss = 0
        self.avg_valid_loss = 0
        self.best_valid_loss = 9999
        self.train_loss = []
        self.valid_loss = []
        self.train_ret_scores = [0, 0, 0, 0]  # [n_lbls, pos, true_pos, true]
        self.valid_ret_scores = [0, 0, 0, 0]
        self.lr = []
        self.timer = Timer()

    def update(self, loss, mode, model_type, logits=None, labels=None):
        if mode == 'train':
            self.steps += 1
            scores = self.train_ret_scores
            self.train_loss.append(loss)
        else:
            scores = self.valid_ret_scores
            self.valid_loss.append(loss)
        if model_type in ['rel', 'ext']:
            # Update retrieval stats
            scores[0] += labels.numel()
            scores[1] += labels.sum().item()
            dim = logits.dim() - 1
            scores[2] += (logits.max(dim)[1] & labels).sum().item()
            scores[3] += (1 - logits.max(dim)[1] ^ labels).sum().item()

    def compute_retrieval_scores(self, mode):
        if mode == 'train':
            scores = self.train_ret_scores
        else:
            scores = self.valid_ret_scores
        recall = scores[2] / scores[1] if scores[1] > 0 else -1
        prec = scores[3] / scores[0] if scores[0] > 0 else -1
        return recall, prec

    def report(self, mode='train', model_type='rel'):
        """Report current statistics
        e.g.
            steps: 123 loss: 0.1234 recall: 0.8 precision: 0.9 time-elapsed: 12.34s
        """
        if mode == 'train':
            avg_recall, avg_prec = self.compute_retrieval_scores(mode)
            avg_loss = sum(self.train_loss) / len(self.train_loss)
            self.avg_train_loss = avg_loss
            if model_type in ['rel', 'ext']:
                msg = (
                    'steps: {} loss: {:.4f} recall: {:.4f} prec.: {:.4f} '
                    'lr {:.6f} time: {:.2f}s'
                    ''.format(self.steps, avg_loss, avg_recall, avg_prec,
                              self.lr, self.timer.time())
                )
            else:  # 'abs'
                msg = (
                    'steps: {} loss: {:.4f} lr {} time: {:.2f}s'
                    ''.format(self.steps, avg_loss,
                              ', '.join('p{}/{:.6f}'.format(i, lr)
                                        for i, lr in enumerate(self.lr)),
                              self.timer.time())
                )
            self.train_loss = []
            self.train_ret_scores = [0] * 4
        else:  # 'valid'
            avg_recall, avg_prec = self.compute_retrieval_scores(mode)
            avg_loss = sum(self.valid_loss) / len(self.valid_loss)
            avg_valid_loss = avg_loss
            if model_type in ['rel', 'ext']:
                msg = (
                    'VAL - loss: {:.4f} recall: {:.4f} prec.: {:.4f}'
                    ' time: {:.2f}s'
                    ''.format(avg_loss, avg_recall,
                              avg_prec, self.timer.time())
                )
            else:  # 'abs'
                msg = (
                    'VAL - loss: {:.4f} lr {} time: {:.2f}s'
                    ''.format(avg_loss,
                              ', '.join('p{}/{:.6f}'.format(i, lr)
                                        for i, lr in enumerate(self.lr)),
                              self.timer.time())
                )
            self.valid_loss = []
            self.valid_ret_scores = [0] * 4
        logger.info(msg)

    def is_best(self):
        if self.avg_valid_loss <= self.best_valid_loss:
            self.best_valid_loss = self.avg_valid_loss
            return True
        return False


class Timer:
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class Tokenizer:
    """Tokenizer contains the functionalities for tokenizing sentences and also
    the vocabulary index. The vocabulary is composed of
    (SP_TOKENS, MESHes, regular words) in that order. """

    def __init__(self, vocab_size):
        from nltk.tokenize import word_tokenize
        self.tokenizer = word_tokenize
        self.vocab_size = vocab_size
        self.sym2idx = defaultdict(lambda: len(self.sym2idx))
        self.idx2sym = None
        # Add special tokens
        self.sp_tokens = SP_TOKENS
        _ = [self.sym2idx[k] for k in self.sp_tokens]

    def from_pretrained(self, emb_file):
        """Read pre-trained embeddings separate MeSHes (starts with
        'εmesh_' code) and regular words up to the vocab_size, then construct
        sym2idx and idx2sim"""
        logger.debug('Building a vocabulary from pretrained embeddings...')
        mesh_codes = []
        reg_words = []
        mesh_indicator = 'εmesh_'
        vocab_size_ = self.vocab_size - len(self.sp_tokens)
        with open(emb_file) as f:
            next(f)  # skip the first row which contains vocab_size and dim
            for line in f:
                token = line.split()[0]
                if token.startswith(mesh_indicator):
                    mesh_codes.append(token)
                else:
                    if vocab_size_ > 0:
                        reg_words.append(token)
                        vocab_size_ -= 1
        # Build the Vocab
        if len(mesh_codes) + len(self.sp_tokens) > self.vocab_size:
            raise RuntimeError('Found MeSH codes more than the vocab size')
        logger.debug(
            f"Adding {len(mesh_codes)} MeSH codes into the vocabulary")
        for t in mesh_codes:
            _ = self.sym2idx[t]
        logger.debug("Adding {} regular words into the vocabulary"
                     "".format(self.vocab_size-len(mesh_codes)-len(self.sp_tokens)))
        for t in reg_words:
            _ = self.sym2idx[t]
            if len(self.sym2idx) >= self.vocab_size:
                break
        # Redefine sym2idx to return 'unk' for unknown symbols
        self.sym2idx = defaultdict(lambda: self.sym2idx['[UNK]'], self.sym2idx)
        self.idx2sym = {v: k for k, v in self.sym2idx.items()}

    def convert_tokens_to_ids(self, tokens):
        # Convert a list of tokens to corresponding ids
        return [self.sym2idx[t] for t in tokens]

    def convert_id_to_token(self, id):
        return self.idx2sym[id]

    def tokenize(self, seq):
        return self.tokenizer(seq)

    def decode(self, seq):
        if len(seq) == 0:
            return ''
        if torch.is_tensor(seq[0]):
            symbols = [self.idx2sym[i.item()] for i in seq]
        else:
            symbols = [self.idx2sym[i] for i in seq]
        return ' '.join(symbols)


def tile(x, count, dim=0):
    """ Tiles x on dimension dim count times. """
    perm = list(range(x.dim()))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def save_model(mdl, args, optim, stat):
    checkpoint = {
        'model': mdl.state_dict(),
        'args': args,
        # 'optim': optim,    # todo. not done yet
        # 'stat': stat,
    }
    fname = '{}_{}_{}.pt'.format(args.model_type, stat.steps, args.exp_id)
    fname_pttn = f'{args.model_type}_*_{args.exp_id}.pt'
    for fpath in Path(args.dir_model).glob(fname_pttn):
        fpath.unlink()
    if not os.path.exists(args.dir_model):
        logger.info(f'mkdir {args.dir_model}')
        os.makedirs(args.dir_model, exist_ok=True)
    fpath = os.path.join(args.dir_model, fname)
    if not os.path.exists(fpath):
        torch.save(checkpoint, fpath)
        logger.info(f'    - Saving checkpoint {fname}')


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def get_special_tokens(bert_tokenizer=None):
    """
    return indices of special tokens and eos/bos pair mapping
    """
    spt_ids_B = {}
    spt_ids_C = {}
    eos_mapping = {}
    if bert_tokenizer is None:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for t in SP_TOKENS:
        spt_ids_B[t] = bert_tokenizer.vocab[t]
        spt_ids_C[t] = SP_TOKENS.index(t)
    # BOS/EOS mapping
    for i in range(100):
        t = f'[unused{i}]'
        eos_mapping[SP_TOKENS.index(t)] = SP_TOKENS.index(f'[unused{i+100}]')
    return spt_ids_B, spt_ids_C, eos_mapping
