""" train.py

train models (REL, EXT, ABS) and save the best model parameters
"""
import sys
import code
import logging
import argparse
import random
from datetime import datetime

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import BertTokenizer, AdamW

from data import DataLoader, load_dataset
from model import DocRelClassifier, ExtractiveClassifier, AbstractiveSummarizer
import utils
from loss import TASummEncLoss, TASummDecLoss

logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def set_defaults():
    """Set default configurations"""
    assert torch.cuda.is_available(), \
        ('Some of the pretrained BERT library have an issue with CPU mode. '
         'We decide not to consider CPU mode')
    # Random
    random.seed(args.rseed)
    torch.manual_seed(args.rseed)

    args.exp_id = 'exp{}'.format(datetime.now().strftime('%m%d%H%M'))
    logger.info(f'=== Experiment {args.exp_id} '.ljust(90, '='))

    # Runtime
    # Classification weights for imbalanced data
    if args.model_type in ['rel', 'ext'] and args.crit_pos_weight is None:
        args.crit_pos_weight = 0.13 if args.model_type == 'rel' else 0.056


def train_encoder(mdl, crit, optim, sch, stat):
    """Train REL or EXT model"""
    logger.info(f'*** Epoch {stat.epoch} ***')
    mdl.train()
    it = DataLoader(load_dataset(args.dir_data, 'train'),
                    args.model_type, args.batch_size, args.max_ntokens_src,
                    spt_ids_B, spt_ids_C, eos_mapping)
    for batch in it:
        _, logits = mdl(batch)
        mask_inp = utils.sequence_mask(batch.src_lens, batch.inp.size(1))
        loss = crit(logits, batch.tgt, mask_inp)
        loss.backward()
        stat.update(loss, 'train', args.model_type, logits=logits,
                    labels=batch.tgt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optim.step()
        if stat.steps == 0:
            continue
        if stat.steps % args.log_interval == 0:
            stat.lr = optim.param_groups[0]['lr']
            stat.report()
            sch.step(stat.avg_train_loss)
        if stat.steps % args.valid_interval == 0:
            valid_ret(mdl, crit, optim, stat)


def valid_ret(mdl, crit, optim, stat):
    """Run validation steps for 'rel' and 'ext' models"""
    mdl.eval()
    logger.info('Validating...')
    it = DataLoader(load_dataset(args.dir_data, 'valid'),
                    args.model_type, args.batch_size, args.max_ntokens_src,
                    spt_ids_B, spt_ids_C, eos_mapping)
    with torch.no_grad():
        for i, batch in enumerate(it):
            _, logits = mdl(batch)
            lossW = crit(logits, batch.tgt, batch.mask_inp)
            stat.update(lossW, logits=logits, labels=batch.tgt,
                        mode='valid', model_type=args.model_type)
            if i >= args.max_valid_steps:
                logger.info('Max step reached')
                break
    stat.report(mode='valid', model_type=args.model_type)
    if stat.is_best():
        utils.save_model(mdl, args, optim, stat)
    mdl.train()


def train_abs(mdl, crit, optim, sch, stat):
    """Train ABS model"""
    mdl.train()
    it = DataLoader(load_dataset(args.dir_data, 'train'),
                    args.model_type, args.batch_size, args.max_ntokens_src,
                    spt_ids_B, spt_ids_C, eos_mapping)
    logger.info(f'*** Epoch {stat.epoch} ***')
    for batch in it:
        optim.zero_grad()
        outputs = mdl(batch)
        loss, scores = crit.compute_loss(batch, outputs)
        loss.backward()
        stat.update(loss, labels=batch.tgt, mode='train',
                    model_type=args.model_type)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optim.step()
        if stat.steps == 0:
            continue
        if stat.steps % args.log_interval == 0:
            stat.lr = [g['lr'] for g in optim.param_groups]
            stat.report(model_type=args.model_type)
            sch.step(stat.avg_train_loss)

            # Print out sample predictions
            if args.debug:
                this = random.randint(0, len(scores) - 1)
                this_bos = batch.tgt[this][0].item()
                max_indices = scores[this].max(1)[1]
                idx_eos = (max_indices == eos_mapping[this_bos]).nonzero()
                idx_eos = idx_eos[0].item() if len(idx_eos) > 0 \
                    else batch.src_lens[this].item()
                logger.debug('Truth: {}\nPrediction: {}'.format(
                    TokC.decode(batch.tgt[this]),
                    TokC.decode([this_bos] + max_indices[:idx_eos+1].tolist())
                ))
        if stat.steps % args.valid_interval == 0:
            valid_abs(mdl, crit, optim, stat)
            mdl.train()


def valid_abs(mdl, crit, optim, stat):
    """Run validation steps for 'abs'"""
    mdl.eval()
    logger.info('Validating...')
    it = DataLoader(load_dataset(args.dir_data, 'valid'),
                    args.model_type, args.batch_size, args.max_ntokens_src,
                    spt_ids_B, spt_ids_C, eos_mapping)
    with torch.no_grad():
        for i, batch in enumerate(it):
            outputs = mdl(batch)
            loss, _ = crit.compute_loss(batch, outputs)
            stat.update(loss, labels=batch.tgt, mode='valid',
                        model_type=args.model_type)
            if i >= args.max_valid_steps:
                logger.info(
                    f'Max valid steps ({args.max_valid_steps}) reached')
                break
    stat.report(mode='valid', model_type=args.model_type)
    if stat.is_best():
        utils.save_model(mdl, args, optim, stat)


if __name__ == '__main__':
    # Configuration ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        'Topic-attended Summarization for Document Retrieval',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Path
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--dir_data', type=str, default='data/tasumm',
                       help='Path to directory where training datasets are')
    files.add_argument('--dir_model', type=str, default='data/models',
                       help='Path to directory for saving trained models')
    files.add_argument('--file_trained_ext', type=str, default=None,
                       help='Path to a file of trained extractive model')
    # Runtime environmnet
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--debug', action='store_true',
                         help='Run in debug mode (= verbose)')
    runtime.add_argument('--rseed', type=int, default=1234,
                         help='Random seed')
    runtime.add_argument('--model_type', type=str, default='rel',
                         choices=['rel', 'ext', 'abs'],
                         help="Model type: 'rel' for document relevancy "
                         "'ext' for extractive classification "
                         "and 'abs' for doc2query text summarizer")
    runtime.add_argument('--batch_size', type=int, default=12,
                         help='Size of minibatch')
    runtime.add_argument('--epochs', type=int, default=5,
                         help='number of epochs to train')
    runtime.add_argument('--log_interval', type=int, default=100,
                         help='Logging interval in training steps')
    runtime.add_argument('--valid_interval', type=int, default=2000,
                         help='Validation interval in training steps')
    runtime.add_argument('--max_valid_steps', type=int, default=400,
                         help='Maximum number of validation steps')
    # Model (general)
    mdl_cfg = parser.add_argument_group('Model (General)')
    mdl_cfg.add_argument('--bert_model', type=str, default='bert-base-uncased',
                         help='Model name of pre-trained BERT')
    mdl_cfg.add_argument('--max_ntokens_src', default=384, type=int,
                         help='Maximum token length of src text to read')
    mdl_cfg.add_argument('--lr_enc', type=float, default=1e-5,
                         help='Learning rate for an optimizer')
    mdl_cfg.add_argument('--lr_dec', type=float, default=1e-3,
                         help='Learning rate for an optimizer')
    mdl_cfg.add_argument('--crit_pos_weight', type=float, default=None,
                         help='Class weights used in criterion method')
    mdl_cfg.add_argument('--file_dec_emb', type=str, default=None,
                         help='Path to a file that contains word embeddings.')
    mdl_cfg.add_argument('--vocab_size', type=int, default=120000,
                         help='Vocabulary size used in decoder')
    mdl_cfg.add_argument("--dec_dropout", type=float, default=0.1,
                         help='Dropout rate in decoder')
    mdl_cfg.add_argument('--dec_pos_emb_dim', type=int, default=256,
                         help='Position embedding dimension')
    mdl_cfg.add_argument('--dec_max_pos_embeddings', type=int, default=256,
                         help='Maximum length of sequence for decoder position '
                         'embeddings')
    mdl_cfg.add_argument('--dec_layers', type=int, default=6,
                         help='Number of decode layer')
    mdl_cfg.add_argument('--dec_hidden_size', type=int, default=768,
                         help='Weight dimension for decode layer')
    mdl_cfg.add_argument('--dec_heads', type=int, default=8,
                         help='Number of heads in decode layer')
    mdl_cfg.add_argument('--dec_ff_size', type=int, default=2048,
                         help='Feed-forward layer size in decode layer')
    args = parser.parse_args()

    # Logger
    log_lvl = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_lvl,
        format='%(asctime)s %(name)s %(levelname)s: [ %(message)s ]',
        datefmt='%b%d %H:%M'
    )

    # Default settings ---------------------------------------------------------
    set_defaults()

    # Print the configurations
    args_str = 'Experiment Configuration\n'
    for k in vars(args):
        args_str += f'   - {k[:30]}'.ljust(35) + f'{getattr(args, k)}\n'
    logger.debug(args_str)

    # Initialize tokenizer and set the special tokens
    TokB = BertTokenizer.from_pretrained('bert-base-uncased')
    TokC = None
    spt_ids_B, spt_ids_C, eos_mapping = \
        utils.get_special_tokens(bert_tokenizer=TokB)
    if args.model_type == 'abs':
        TokC = utils.Tokenizer(vocab_size=args.vocab_size)
        TokC.from_pretrained(args.file_dec_emb)

    # Model and Criterion ------------------------------------------------------
    if args.model_type == 'rel':
        model = DocRelClassifier(bert_model=args.bert_model).cuda()
        criterion = TASummEncLoss(pos_weight=args.crit_pos_weight,
                                  reduction='mean')
    elif args.model_type == 'ext':
        model = ExtractiveClassifier(args).cuda()
        criterion = TASummEncLoss(pos_weight=args.crit_pos_weight)
    elif args.model_type == 'abs':
        model = AbstractiveSummarizer(args).cuda()
        if args.file_trained_ext is not None:
            model.load_ext_model(args.file_trained_ext)
        criterion = TASummDecLoss(model.generator, 0, model.decoder.vocab_size)

    if args.model_type == 'abs':
        dec_params = [p for n, p in model.decoder.named_parameters()
                      if not n.startswith('encoder')]
        optimizer = AdamW([
            {'params': model.encoder.parameters(), 'lr': args.lr_enc},
            {'params': dec_params, 'lr': args.lr_dec}
        ], lr=1e-3)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr_enc)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.9)

    training_stats = utils.Statistics()

    # Train --------------------------------------------------------------------
    logger.info(f'Start training {args.model_type} model ')
    for epoch in range(1, args.epochs + 1):
        training_stats.epoch = epoch
        if args.model_type in ['rel', 'ext']:
            train_encoder(model, criterion, optimizer,
                          scheduler, training_stats)
        elif args.model_type == 'abs':
            train_abs(model, criterion, optimizer, scheduler, training_stats)
