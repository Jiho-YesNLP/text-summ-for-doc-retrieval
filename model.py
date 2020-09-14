""" model

Three models (REL, EXT, ABS) and their sub modules
"""
import logging

import numpy as np
import torch
from torch import nn

from transformers import BertForSequenceClassification as B4SC
from transformers import BertForTokenClassification as B4TC
from transformers import BertConfig

from onmt import MultiHeadedAttention, PositionwiseFeedForward
from utils import SP_TOKENS

logger = logging.getLogger('model')


class DocRelClassifier(nn.Module):
    """Relevant document classifier"""

    def __init__(self, bert_model='bert-base-uncased'):
        super(DocRelClassifier, self).__init__()
        self.bert = B4SC.from_pretrained(bert_model)

    def forward(self, data):
        outputs = self.bert(data.inp, attention_mask=data.mask_inp,
                            token_type_ids=data.segs, labels=data.tgt)
        return outputs[:2]  # loss, logits


class ExtractiveClassifier(nn.Module):
    """Extractive token classifier that predicts keywords in documents"""

    def __init__(self, args):
        super(ExtractiveClassifier, self).__init__()
        self.args = args
        self.bert = B4TC.from_pretrained(args.bert_model)

    def forward(self, data):
        outputs = self.bert(data.inp, attention_mask=data.mask_inp,
                            token_type_ids=data.segs, labels=data.tgt)
        return outputs[:2]  # loss, logits


class AbstractiveSummarizer(nn.Module):
    """Abstractive model that predicts topic-attended sequence of tokens given
    the outputs from the pre-trained Extractive model"""

    def __init__(self, args, pad_idx=0):
        super(AbstractiveSummarizer, self).__init__()
        self.args = args
        self.pad_idx = pad_idx
        self.encoder = B4TC(BertConfig(hidden_size=args.dec_hidden_size,
                                       output_hidden_states=True))
        self.decoder = TransformerDecoder(args.dec_layers,
                                          args.dec_hidden_size,
                                          args.dec_heads,
                                          args.dec_ff_size,
                                          args.dec_max_pos_embeddings,
                                          args.dec_pos_emb_dim,
                                          args.dec_dropout,
                                          self.pad_idx,
                                          embeddings=self.load_embeddings())
        self.generator = nn.Sequential(
            nn.Linear(args.dec_hidden_size, self.decoder.vocab_size),
            nn.LogSoftmax(dim=-1)
        )
        self.generator[0].weight = self.decoder.dec_embeddings.weight

    def forward(self, data):
        _, hidden_states = self.encoder(data.inp,
                                        attention_mask=data.mask_inp,
                                        token_type_ids=data.segs)
        self.decoder.init_state(data.inp)
        dec_outputs, _ = self.decoder(data.tgt[:, :-1],
                                      hidden_states[-1], data.mask_inp)
        return dec_outputs

    def load_ext_model(self, file):
        """Load pretrained EXT model for the use in ABS"""
        logger.info(f'Loading a pre-trained extractive model from {file}...')
        data = torch.load(file, map_location=lambda storage, loc: storage)
        self.encoder.load_state_dict(
            {n[5:]: p for n, p in data['model'].items()
             if n.startswith('bert.')}
        )

    def load_embeddings(self):
        """In the decoder, we use custom word embeddings (BMET Embeddings)
        In BMET embeddings, we have a subset of words for MeSH codes indicated
        by a special prefix 'εmesh_'
        """
        mesh_indicator = 'εmesh_'
        mesh_codes = []  # Use list to add items sequencially
        reg_words = []
        # Read embeddings from file_dec_emb
        vocab_size = self.args.vocab_size - len(SP_TOKENS)
        with open(self.args.file_dec_emb) as f:
            emb_vocab_size, dim = map(int, next(f).split()[:2])
            assert emb_vocab_size >= vocab_size, \
                ("Not enough vocabulary in BMET embeddings. Check the "
                 "vocab_size in configuration")
            for line in f:
                vals = line.split()
                token, vec = vals[0], list(map(float, vals[1:]))
                if token.startswith(mesh_indicator):
                    mesh_codes.append((token, vec))
                else:
                    reg_words.append((token, vec))
        # Build embeddings
        embs = []
        # Add special tokens
        for _ in SP_TOKENS:
            embs.append(np.random.normal(size=dim).tolist())
        # Add mesh codes
        for _, v in mesh_codes:
            # copy only vector values, assume that indexing is done right
            embs.append(v)
        # Add regular words
        for t, v in reg_words:
            if len(embs) >= self.args.vocab_size:
                break
            embs.append(v)
        weight = torch.FloatTensor(embs).cuda()
        tgt_embeddings = nn.Embedding.from_pretrained(weight)
        return tgt_embeddings


class TransformerDecoderLayer(nn.Module):
    """Transformer layer in the decoder"""

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.context_attn = \
            MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask()
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask,
                previous_input=None, layer_cache=None):
        """
        # T: could be 1 in the case of stepwise decoding or tgt_len
        Args:
            inputs (`FloatTensor`): `[batch_size x T x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x T]`
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size,  T, model_dim]`
            * attn `[batch_size, T, src_len]`
            * all_input `[batch_size, current_step, model_dim]`

        """
        T_ = tgt_pad_mask.size(1)  # tgt_len
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :T_, :T_], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query, _ = self.self_attn(all_input, all_input, input_norm,
                                  mask=dec_mask,
                                  layer_cache=layer_cache,
                                  attn_type="self")
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(memory_bank, memory_bank, query_norm,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        # This attentions are for computing converage penalty
        cov_attn = \
            torch.max(attns, dim=1)[0] / attns.size(0)  # max values over heads
        output = self.feed_forward(self.drop(mid) + query)
        return output, cov_attn, all_input

    @staticmethod
    def _get_attn_subsequent_mask(size=5000):
        """
        Get an attention mask to avoid using the subsequent info.
        Args:
            size: int
        Returns:
            (`LongTensor`):
            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    def __init__(self, dec_nlayers, dec_hidden_size, dec_att_heads, dec_ff_size,
                 dec_max_pos_embeddings, dec_pos_emb_dim, dec_dropout,
                 pad_sym, embeddings=None):
        super(TransformerDecoder, self).__init__()

        # Book-keeping
        self.dec_nlayers = dec_nlayers
        self.dec_hidden_size = dec_hidden_size
        self.dec_heads = dec_att_heads
        self.dec_ff_size = dec_ff_size
        self.dec_max_pos_embeddings = dec_max_pos_embeddings
        self.dec_pos_emb_dim = dec_pos_emb_dim
        self.dec_dropout = dec_dropout
        self.pad_sym = pad_sym

        # Decoder state
        self.state = {}

        # Embeddings
        self.dec_embeddings = embeddings
        self.vocab_size, self.dec_emb_dim = embeddings.weight.size()
        self.dec_pos_embeddings = nn.Embedding(self.dec_max_pos_embeddings,
                                               self.dec_pos_emb_dim)
        self.dec_emb_proj = nn.Linear(self.dec_emb_dim + self.dec_pos_emb_dim,
                                      self.dec_hidden_size)
        self.emb_layer_norm = torch.nn.LayerNorm(self.dec_hidden_size)
        self.emb_dropout = nn.Dropout(self.dec_dropout)

        # Transformer decoder layers
        layers = []
        for i in range(self.dec_nlayers):
            l = TransformerDecoderLayer(self.dec_hidden_size, self.dec_heads,
                                        self.dec_ff_size, self.dec_dropout)
            layers.append(l)
        self.transformer_layers = nn.ModuleList(layers)
        self.dec_out_proj = nn.Linear(self.dec_hidden_size, self.dec_emb_dim)
        self.layer_norm = nn.LayerNorm(self.dec_emb_dim)

    def forward(self, tgt, memory_bank, memory_pad_mask, step=None):
        """
        Decode, possibly stepwise

        Returns:
            * outputs
            * top_attn `[batch_size, head, T, src_len]`
        """
        if step == 0:
            self._init_cache(memory_bank)

        # To embeddings
        if step is None:
            position_ids = torch.arange(tgt.size(1)).expand_as(tgt)
        else:
            position_ids = torch.tensor(step).expand(tgt.size(0), 1)
        position_ids = position_ids.cuda()
        de = self.dec_embeddings(tgt)
        pe = self.dec_pos_embeddings(position_ids)
        emb_ = self.dec_emb_proj(torch.cat((de, pe), 2))
        emb_ = self.emb_layer_norm(emb_)
        # emb_ = self.emb_layer_norm(de + pe)
        output = self.emb_dropout(emb_)

        tgt_pad_mask = tgt.data.eq(self.pad_sym).unsqueeze(1)  # [N, 1, L]

        top_attn = None
        for i, layer in enumerate(self.transformer_layers):
            # print(i, 'memroy_pad_mask', memory_pad_mask)
            layer_cache = self.state['cache'][f'layer_{i}'] \
                if step is not None else None
            output, top_attn, _ = layer(
                output,
                memory_bank,
                memory_pad_mask.unsqueeze(1),
                tgt_pad_mask,
                layer_cache=layer_cache
            )
        output = self.dec_out_proj(output)
        output = self.layer_norm(output)

        return output, top_attn

    def init_state(self, src):
        self.state['src'] = src
        self.state['cache'] = None

    def map_state(self, fn):
        """Apply fn to the state recursively"""
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        for i, layer in enumerate(self.transformer_layers):
            self.state["cache"]["layer_{}".format(i)] = {
                "memory_keys": None, "memory_values": None,
                "self_keys": None, "self_values": None
            }

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()
