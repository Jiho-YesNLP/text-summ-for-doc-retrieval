""" Generate a DataLoader """
import logging
import random
from pathlib import Path
import os
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence

from utils import sequence_mask

logger = logging.getLogger('data')


def load_dataset(path, data_type):
    """
    Dataset generator.

    :param path: path to the directory of BERTified data
    :param data_type: 'train', 'valid', or 'test'
    :return: A list of datasets (lazily loaded)
    """
    # Read list of files and sort
    files = sorted(Path(path).glob(f'{data_type}_[0-9]*.pt'))
    # random.shuffle(files)
    for f in files:
        ds = torch.load(f)
        logger.debug(f'Loading {data_type} dataset... (file: '
                     f'{os.path.basename(f)}, num_exs: {len(ds)})')
        yield ds


class DataLoader:
    """Dynamically loads dataset from chunked datafiles and returns a
    DataIterator

    datasets: iterator for datasets (chunked bertified files)
    cur_data_iter: given a dataset (paired docs), this yields batchified examples
    """

    def __init__(self, datasets, model_type, batch_size, max_ntokens_src,
                 spt_ids_B, spt_ids_C, eos_mapping):
        # Book-keeping
        self.datasets = datasets
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_ntokens_src = max_ntokens_src
        self.spt_ids_B = spt_ids_B
        self.spt_ids_C = spt_ids_C
        self.eos_mapping = eos_mapping

        self.cur_data_iter = self._next_ds_iter(datasets)

    def __iter__(self):
        while self.cur_data_iter is not None:
            for batch in self.cur_data_iter:
                yield batch
            self.cur_data_iter = self._next_ds_iter(self.datasets)

    def _next_ds_iter(self, ds_iter):
        try:
            self.cur_dataset = next(ds_iter)
        except TypeError:
            if self.datasets is None:
                return None
            if isinstance(ds_iter, list):  # data from doc_scorer
                self.cur_dataset = ds_iter
                self.datasets = None  # consume
        except StopIteration:
            return None

        return DataIterator(self.cur_dataset,
                            model_type=self.model_type,
                            batch_size=self.batch_size,
                            max_ntokens_src=self.max_ntokens_src,
                            spt_ids_B=self.spt_ids_B,
                            spt_ids_C=self.spt_ids_C,
                            eos_mapping=self.eos_mapping)


class DataIterator:
    """Process and batchify examples"""

    def __init__(self, dataset, model_type, batch_size, max_ntokens_src,
                 spt_ids_B, spt_ids_C, eos_mapping=None):
        self.dataset = dataset
        self.model_type = model_type
        self.batch_size = batch_size
        self.max_ntokens_src = max_ntokens_src
        # Special tokens and bos/eos mappings
        self.spt_ids_B = spt_ids_B  # Indicies by BERT Tokenizer
        self.spt_ids_C = spt_ids_C  # Indicies by Custom embeddings dictionary
        self.eos_mapping = eos_mapping

    def batchify(self):
        """Given examples from a dataset,
         (1) create encoded inputs by interpolating necessary BERT special
             tokens with source and target sequences
         (2) create segment indicating sequence
         and yield minibatch of the processed examples
        """
        # random.shuffle(self.dataset)
        proc_data = []
        for ex in self.dataset:
            if self.model_type == 'rel':  # Src: doc+query, Tgt: doc labels
                # Doc
                inp = [self.spt_ids_B['[CLS]']]
                segs = [0]
                sent_lens = [0] + ex['src_sent_lens']
                for i in range(len(sent_lens) - 1):
                    s, e = sent_lens[i], sent_lens[i+1]
                    if len(inp) + e - s >= self.max_ntokens_src - 1:
                        break
                    inp += ex['src'][s:e] + [self.spt_ids_B['[SEP]']]
                    segs += [0 if segs[-1] == 1 else 1] * (e - s + 1)

                # Query
                sent_lens = [0] + ex['tgtB_sent_lens']
                for i in range(3):    # Topics: 0. disease 1. gene, 2. demo
                    s, e = sent_lens[i], sent_lens[i+1]
                    inp += ex['tgtB'][s:e]
                    segs += [0 if segs[-1] == 1 else 1] * (e - s)
                inp += [self.spt_ids_B['[SEP]']]
                segs += [segs[-1]]
                # With document-level labels
                proc_data.append(
                    (inp, segs, ex['doc_label'], ex['qid'], ex['did']))
            elif self.model_type == 'ext':  # Src: doc, Tgt: token labels
                # Doc
                inp = [self.spt_ids_B['[CLS]']]
                segs = [0]
                sent_lens = [0] + ex['src_sent_lens']
                for i in range(len(sent_lens) - 1):
                    s, e = sent_lens[i], sent_lens[i+1]
                    if len(inp) + e - s >= self.max_ntokens_src - 1:
                        break
                    inp += ex['src'][s:e]
                    segs += [0 if segs[-1] == 1 else 1] * (e - s)
                inp += [self.spt_ids_B['[SEP]']]
                segs += [segs[-1]]
                # With token-level labels
                proc_data.append((inp, segs,
                                  [0] + ex['token_labels'][:len(inp)-2] + [0],
                                  ex['qid'], ex['did']))
            elif self.model_type == 'abs':
                # src: Doc, target: Topic sentences
                if ex['doc_label'] == 0:  # Feed only relevant pairs
                    continue
                # Doc
                inp = [self.spt_ids_B['[CLS]']]
                segs = [0]
                sent_lens = [0] + ex['src_sent_lens']
                for i in range(len(sent_lens) - 1):
                    s, e = sent_lens[i], sent_lens[i+1]
                    if len(inp) + e - s >= self.max_ntokens_src - 1:
                        break
                    inp += ex['src'][s:e]
                    segs += [0 if segs[-1] == 1 else 1] * (e - s)
                inp += [self.spt_ids_B['[SEP]']]
                segs += [segs[-1]]
                sent_lens = [0] + ex['tgtC_sent_lens']
                topics = ('disease', 'gene', 'demo', 'mesh', 'keywords')
                for i, tp in enumerate(topics):
                    if tp == 'demo':  # Ignored intentionally
                        continue
                    s, e = sent_lens[i], sent_lens[i+1]
                    bos = self.spt_ids_C[f'[unused{i}]']
                    eos = self.eos_mapping[bos]
                    tgt = [bos] + ex['tgtC'][s:e] + [eos]
                    proc_data.append((inp, segs, tgt, ex['qid'], ex['did']))

        # This used to be here, and it worked. why not this time.
        # proc_data.sort(key=lambda x: len(x[0]), reverse=True)
        minibatch = []
        for ex in proc_data:
            minibatch.append(ex)
            if len(minibatch) == self.batch_size:
                yield minibatch
                minibatch = []
        if len(minibatch) > 0:
            yield minibatch

    def __iter__(self):
        for batch in self.batchify():
            t_batch = TensorBatch(batch, model_type=self.model_type)
            yield t_batch


class TensorBatch:
    """minibatch of tensorfied examples"""

    def __init__(self, batch, model_type, device='cuda'):
        self.batch_size = len(batch)

        pad_ = partial(pad_sequence, batch_first=True)
        self.inp = pad_([torch.tensor(x[0]) for x in batch]).to(device)
        lens = [next((i for i, v in enumerate(s) if v == 0), len(s))
                for s in self.inp]
        self.src_lens = torch.LongTensor(lens).to(device)
        self.mask_inp = sequence_mask(self.src_lens, self.inp.size(1))
        self.segs = pad_([torch.tensor(x[1]) for x in batch]).to(device)
        if model_type == 'rel':
            self.tgt = torch.tensor([x[2] for x in batch]).to(device)
        elif model_type in ['ext', 'abs']:
            self.tgt = pad_([torch.tensor(x[2]) for x in batch]).to(device)

        self.qid = [x[3] for x in batch]
        self.did = [x[4] for x in batch]

    def __len__(self):
        return self.batch_size
