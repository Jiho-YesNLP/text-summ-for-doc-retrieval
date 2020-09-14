"""doc2query

This file contains Summarizer which loads pretrained ABS model and generate
pseudo query given documents. Generated sentences are wrapped in Translation
class.
"""
import logging
import argparse
import random

from prettytable import PrettyTable
import torch
from transformers import BertTokenizer

from data import DataLoader, load_dataset
from model import AbstractiveSummarizer
from beam_search import BeamSearch
from utils import Tokenizer, get_special_tokens

logger = logging.getLogger('doc2query')
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


class Summarizer:
    """Use a test model to generate fielded query sentences from documents"""

    def __init__(self, f_abs, n_best=1, min_length=1, max_length=50,
                 beam_size=4, bert_model='bert-base-uncased'):
        self.n_best = n_best
        self.min_length = min_length
        self.max_length = max_length
        self.beam_size = beam_size
        self.abs_model = self.load_abs_model(f_abs)
        self.eval()
        logger.info(f'Loading BERT Tokenizer [{bert_model}]...')
        self.tokenizerB = BertTokenizer.from_pretrained('bert-base-uncased')
        self.spt_ids_B, self.spt_ids_C, self.eos_mapping = get_special_tokens()
        logger.info('Loading custom Tokenizer for using WBMET embeddings')
        self.tokenizerC = Tokenizer(self.abs_model.args.vocab_size)
        self.tokenizerC.from_pretrained(self.abs_model.args.file_dec_emb)

    @staticmethod
    def load_abs_model(f_abs):
        """Load a pre-trained abs model"""
        logger.info(f'Loading an abstractive test model from {f_abs}...')
        data = torch.load(f_abs, map_location=lambda storage, loc: storage)
        mdl = AbstractiveSummarizer(data['args'])
        mdl.load_state_dict(data['model']).cuda()
        return mdl

    def translate(self, docs):
        """Translate a batch of documents."""
        batch_size = docs.inp.size(0)
        spt_ids = self.spt_ids_C
        decode_strategy = BeamSearch(self.beam_size, batch_size, self.n_best,
                                     self.min_length, self.max_length,
                                     spt_ids, self.eos_mapping)
        return self._translate_batch_with_strategy(docs, decode_strategy)

    def _translate_batch_with_strategy(self, batch, decode_strategy):
        """Translate a batch of documents step by step using cache

        :param batch (dict): A batch of documentsj
        :param decode_strategy (DecodeStrategy): A decode strategy for
            generating translations step by step. I.e., BeamSearch
        """

        # (1) Run the encoder on the src
        ext_scores, hidden_states = \
            self.abs_model.encoder(batch.inp,
                                   attention_mask=batch.mask_inp,
                                   token_type_ids=batch.segs)

        # (2) Prepare decoder and decode_strategy
        self.abs_model.decoder.init_state(batch.inp)
        field_signals = batch.tgt[:, 0]
        fn_map_state, memory_bank, memory_pad_mask = \
            decode_strategy.initialize(hidden_states[-1], batch.src_lens,
                                       field_signals)
        if fn_map_state is not None:
            self.abs_model.decoder.map_state(fn_map_state)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.unsqueeze(-1)
            dec_out, attns = self.abs_model.decoder(
                decoder_input, memory_bank, memory_pad_mask, step=step
            )
            log_probs = self.abs_model.generator(dec_out[:, -1, :].squeeze(1))
            # Beam advance
            decode_strategy.advance(log_probs, attns)

            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if any_finished:
                # Reorder states.
                memory_bank = memory_bank.index_select(0, select_indices)
                memory_pad_mask = memory_pad_mask.index_select(
                    0, select_indices)

            if self.beam_size > 1 or any_finished:
                self.abs_model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))
        res = {
            'batch': batch,
            'gold_scores':
                self._gold_score(batch, hidden_states[-1], batch.mask_inp),
            'scores': decode_strategy.scores,
            'predictions': decode_strategy.predictions,
            'ext_scores': ext_scores,
            'attentions': decode_strategy.attention
        }
        return res

    def results_to_translations(self, data):
        """Convert results into Translation object"""
        batch = data['batch']
        translations = []
        for i, did in enumerate(batch.did):
            src_input_ = batch.inp[i]
            src_ = self.tokenizerB.decode(src_input_)
            topic_ = \
                self.tokenizerC.convert_id_to_token(batch.tgt[i][0].item())
            pred_sents_ = [
                self.tokenizerC.decode(data['predictions'][i][n])
                for n in range(self.n_best)
            ]
            gold_sent_ = self.tokenizerC.decode(batch.tgt[i])
            x = Translation(did=did, src_input=src_input_, src=src_,
                            topic=topic_, ext_scores=data['ext_scores'][i],
                            pred_sents=pred_sents_,
                            pred_scores=data['scores'][i],
                            gold_sent=gold_sent_,
                            gold_score=data['gold_scores'][i],
                            attentions=data['attentions'][i])
            translations.append(x)
        return translations

    def _gold_score(self, batch, memory_bank, memory_pad_mask):
        if hasattr(batch, 'tgt'):
            gs = self._score_target(batch, memory_bank, memory_pad_mask)
            self.abs_model.decoder.init_state(batch.inp)
        else:
            gs = [0] * batch.batch_size
        return gs

    def _score_target(self, batch, memory_bank, memory_pad_mask):
        tgt_in = batch.tgt[:, :-1]
        dec_out, _ = self.abs_model.decoder(
            tgt_in, memory_bank, memory_pad_mask)
        log_probs = self.abs_model.generator(dec_out)
        gold = batch.tgt[:, 1:]
        tgt_pad_mask = gold.eq(self.spt_ids_C['[PAD]'])
        log_probs[tgt_pad_mask] = 0
        gold_scores = log_probs.gather(2, gold.unsqueeze(-1))
        gold_scores = gold_scores.sum(dim=1).view(-1)
        return gold_scores.tolist()

    def eval(self):
        self.abs_model.eval()


class Translation:
    """Container for a translated sentence.

    Attributes:
        did (str): Source document ID
        field_signal (str): Field signal (e.g., [unused0])
        src (str): Raw source words
        pred_sents (List[str]): Words from the n-best translations
        pred_scores (List[float]): Log-probs of n-best translations
        attns (List[FloatTensor]): Attention distribution for each translation
        highlighted_words (List[str]): Tokens from the source document predicted
            as query candidate
        gold_sent (str): Words from gold translation
        gold_score (float): Log-prob of gold translation
    """

    def __init__(self, did, src_input, src, topic, ext_scores,
                 pred_sents, pred_scores, gold_sent=None, gold_score=None,
                 attentions=None):
        self.did = did
        self.src_input = src_input
        self.src = src
        self.topic = topic
        self.ext_scores = ext_scores
        self.pred_sents = pred_sents
        self.pred_scores = pred_scores
        self.gold_sent = gold_sent
        self.gold_score = gold_score
        self.attentions = attentions

        # Print-outs
        self.table = PrettyTable()
        self.table.field_names = ['Key', 'Values']
        self.table.align['Key'] = 'r'
        self.table.align['Values'] = 'l'
        self.table.max_width['Values'] = 80

    def log(self):
        """Pretty Print"""
        self.table.clear_rows()
        keys = ['did', 'src', 'pred_sents', 'pred_scores']
        if self.gold_sent is not None:
            keys.extend(['gold_sent', 'gold_score'])
        for k in keys:
            val = getattr(self, k)
            if k == 'pred_sents':
                val = '\n'.join(val)
            elif k == 'pred_scores':
                val = f'{val[0]:.4f}'  # best score
                # val = ', '.join([str(s.item()) for s in val])
            elif k == 'gold_score':
                val = f'{val:.4f}'
            self.table.add_row([k, val])
        print(self.table.get_string())


if __name__ == '__main__':
    # Logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    parser = argparse.ArgumentParser(
        'Topic-attended Summarization for Document Retrieval - Query generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Doc2Query
    d2q = parser.add_argument_group('Doc2Query')
    d2q.add_argument('--dir_data', type=str, default='data/tasumm',
                     help='Path to directory where training datasets are')
    d2q.add_argument('--beam_size', type=int, default=4,
                     help='Number of steps in Beam Search algorithm')
    d2q.add_argument('--n_best', type=int, default=2,
                     help='Number of predictions to produce')
    d2q.add_argument('--min_length', type=int, default=1,
                     help='Minimum number of tokens to predict')
    d2q.add_argument('--max_length', type=int, default=50,
                     help='Maximum number of tokens to predict')
    d2q.add_argument('--file_abs_model', type=str, default=None,
                     help='File path to a pre-trained abs model')
    args = parser.parse_args()

    # Set defaults
    args.model_type = 'abs'
    args.cuda = torch.cuda.is_available()
    if not args.cuda:
        logger.error('It seems like running the pretrained BERT on CPU is not'
                     ' available. You should run this on GPU.')
        raise RuntimeError
    args.gpu = -1
    torch.cuda.set_device(args.gpu)
    args.device = torch.device('cuda')
    random.seed(1234)
    torch.manual_seed(1234)

    # Test
    summarizer = Summarizer(args.file_abs_model,
                            n_best=args.n_best,
                            min_length=args.min_length,
                            max_length=args.max_length,
                            beam_size=args.beam_size)
    mdl_args = summarizer.abs_model.args
    test_iter = DataLoader(load_dataset(args.dir_data, 'test'),
                           mdl_args.model_type, mdl_args.batch_size,
                           mdl_args.max_ntokens_src,
                           summarizer.spt_ids_B, summarizer.spt_ids_C,
                           summarizer.eos_mapping)
    with torch.no_grad():
        for batch in test_iter:
            results = summarizer.translate(batch)
            translations = summarizer.results_to_translations(results)
            for x in translations:
                x.log()
