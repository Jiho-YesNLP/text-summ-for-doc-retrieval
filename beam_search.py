"""beam_search"""

import logging
import torch

from utils import tile, sequence_mask

logger = logging.getLogger()


class BeamSearch:
    def __init__(self, beam_size, batch_size, n_best, min_length, max_length,
                 spt_ids, eos_mapping, pen_alpha=0.4, pen_beta=0.4):
        """
        Best-first search on branches of predictions with a limited size.

        :param beam_size: (int) Number of beams to use
        :param batch_size:  (int) Number of examples in a batch
        :param n_best: (int) Don't stop until at least this many beams have
            reached EOS.
        :param min_length: Shortest acceptable sequence, not counting BOS or EOS
        :param max_length: Longest acceptable sequence, not counting BOS

        Attributes:  (B is the current batch_size as beams advance)
            - topk_log_probs                      [B x beam_size,]
              The scores used for the topk operation
            - topk_scores                         [B, beam_size]
              The scores that a sequence will receive when it finishes
            - topk_ids                            [B, beam_size]
              The word indicies of the topk predictions
            - _beam_offset                        [batch_size x beam_size,]
              e.g. [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, ...] with beam_size 4
            - _batch_index                        [B, beam_size]
              Beam indicies of topk
            - select_indicies                     [B x beam_size,]
              Flat view of _batch_index
            - _prev_penalty                       [B, beam_size]
              Previous coverage scores
            - _coverage                           [B x beam_size, 1, src_len]
              Current context attention
            - alive_seq                           [B x beam_size, step]
              This sequence grows in the step axis on each call to advance()
            - alive_attn                          [B x beam_size, step, src_len]
              Accumulated attentions
        """
        # Beam parameters
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.unk_id = spt_ids['[UNK]']
        self.sptB_tokens = spt_ids
        self.eos_mapping = eos_mapping
        self.n_best = n_best
        self.predictions = [[] for _ in range(batch_size)]
        self.scores = [[] for _ in range(batch_size)]
        self.attention = [[] for _ in range(batch_size)]
        self.min_length = min_length
        self.max_length = max_length
        self.pen_alpha = pen_alpha
        self.pen_beta = pen_beta

        # Result caching
        self.hypotheses = [[] for _ in range(batch_size)]

        # Beam state
        self.top_beam_finished = torch.zeros([batch_size], dtype=torch.bool)
        self._batch_offset = torch.arange(batch_size, dtype=torch.uint8)
        self.select_indices = None
        self.done = False

        # "Global state" of the old beam
        self._prev_penalty = None
        self._coverage = None

        # Book-keeping; Step-wise status variables
        self.topk_log_probs = None
        self.topk_scores = None
        self.topk_ids = None
        self._beam_offset = None
        self._batch_index = None
        self.alive_attn = None
        self.is_finished = None
        self.alive_seq = None

    def __len__(self):
        return self.alive_seq.shape[1]

    def initialize(self, memory_bank, src_lengths, field_signals, device=None):
        """Initialize search state for each batch input"""
        # Repeat state in beam_size
        def fn_map_state(state, dim):
            return tile(state, self.beam_size, dim=dim)

        src_max_len = memory_bank.size(1)
        memory_bank = tile(memory_bank, self.beam_size)
        memory_pad_mask = tile(~sequence_mask(src_lengths, src_max_len),
                               self.beam_size)
        self.memory_lengths = tile(src_lengths, self.beam_size)
        mb_device = memory_bank.device
        if device is None:
            self.device = mb_device
        self.field_signals = field_signals
        self.alive_seq = field_signals.repeat_interleave(self.beam_size)\
            .unsqueeze(-1).to(self.device)
        self.is_finished = torch.zeros(
            [self.batch_size, self.beam_size], dtype=torch.uint8, device=self.device
        )
        self.best_scores = torch.full(
            [self.batch_size], -1e10, dtype=torch.float, device=self.device
        )
        self._beam_offset = torch.arange(
            0, self.batch_size * self.beam_size, step=self.beam_size,
            dtype=torch.long, device=self.device
        )
        # Give full probability to the first beam on the first step; with no
        # prior information, choose any (the first beam)
        self.topk_log_probs = torch.tensor(
            [0.0] + [float("-inf")] * (self.beam_size - 1), device=self.device
        ).repeat(self.batch_size)
        # buffers for the topk scores and 'backpointer'
        self.topk_scores = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.float, device=self.device
        )
        self.topk_ids = torch.empty(
            (self.batch_size, self.beam_size), dtype=torch.long, device=self.device
        )
        self._batch_index = torch.empty(
            [self.batch_size, self.beam_size], dtype=torch.long, device=self.device
        )
        return fn_map_state, memory_bank, memory_pad_mask

    @property
    def current_predictions(self):
        return self.alive_seq[:, -1]

    @property
    def current_backptr(self):
        # for testing
        return self.select_indices.view(self.batch_size, self.beam_size)\
            .fmod(self.beam_size)

    @property
    def batch_offset(self):
        return self._batch_offset

    def advance(self, log_probs, attn):
        """
        Step-wise prediction over the examples multipied by beam_size

        :param log_probs: log probabilities in generation over the vocabulary
        :param attn: Context attention of the last prediction
                     [B x beam_size, 1, src_length]
        """
        step = len(self)
        _B = log_probs.size(0) // self.beam_size  # batch_size
        vocab_size = log_probs.size(-1)

        # Apply step-wise current coverage penalty
        if self._prev_penalty is not None:
            # Replace with new coverage penalty
            self.topk_log_probs += self._prev_penalty
            cov_pen = \
                self.coverage_wu(self._coverage +
                                 attn).view(_B, self.beam_size)
            self.topk_log_probs -= cov_pen

        # Silence [UNK]s
        log_probs[:, self.unk_id] = -1e10
        # Ensure min_length
        if step <= self.min_length:
            eoses = list(self.eos_mapping.values())
            log_probs.index_fill_(1, torch.LongTensor(eoses).to(self.device),
                                  -1e10)

        # Multiply probs by the beam probabilities.
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        # Length penalty :cite: `wu2016google`
        # If the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.length_wu(step + 1)
        curr_scores = log_probs / length_penalty

        # Flatten probs into a list of possibilities.
        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)
        self.topk_scores, self.topk_ids = curr_scores.topk(
            self.beam_size, dim=-1)

        # Recover log probs.
        self.topk_log_probs = self.topk_scores * length_penalty

        # Resolve beam origin and map to batch index flat representation.
        self._batch_index = self.topk_ids / vocab_size
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)
        self.topk_ids.fmod_(vocab_size)  # in-place op, resolve true word ids

        # Append last prediction.
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        # Coverage
        current_attn = attn.index_select(0, self.select_indices)
        if step == 1:
            self.alive_attn = current_attn
            # Initialize global state
            self._prev_penalty = torch.zeros_like(self.topk_log_probs)
            self._coverage = current_attn
        else:
            self.alive_attn = \
                self.alive_attn.index_select(0, self.select_indices)
            self.alive_attn = torch.cat([self.alive_attn, current_attn], 1)
            self._coverage = \
                self._coverage.index_select(0, self.select_indices)
            self._coverage += current_attn
            self._prev_penalty =\
                self.coverage_wu(self._coverage).view(_B, self.beam_size)

        self.is_finished = torch.zeros_like(self.topk_ids, dtype=torch.bool)
        for i, bos in enumerate(self.field_signals.tolist()):
            for j, eos in enumerate(self.topk_ids[i]):
                self.is_finished[i, j] = self.eos_mapping[bos] == eos

        # Ensure max_length
        if step == self.max_length - 1:
            self.is_finished.fill_(1)

    def update_finished(self):
        """Penalize beams that finished"""
        _B_old = self.topk_log_probs.shape[0]
        step = self.alive_seq.shape[-1]  # 1 greater than the step in advance
        self.topk_log_probs.masked_fill_(self.is_finished, -1e10)
        # it's faster to not move this back to the original device
        self.is_finished = self.is_finished.to('cpu')
        self.top_beam_finished |= self.is_finished[:, 0].eq(1)
        predictions = self.alive_seq.view(_B_old, self.beam_size, step)
        attention = (
            self.alive_attn.view(_B_old, step - 1, self.beam_size,
                                 self.alive_attn.size(-1))
            if self.alive_attn is not None else None)
        non_finished_batch = []
        for i in range(self.is_finished.size(0)):  # for each example
            b = self._batch_offset[i]
            finished_hyp = self.is_finished[i].nonzero().view(-1)
            # Store finished hypotheses for this batch.
            for j in finished_hyp:  # for each finished beam
                # Append (score, predictions, attentions) to hypotheses
                self.hypotheses[b].append((
                    self.topk_scores[i, j],
                    predictions[i, j, :],
                    attention[i, :, j, :self.memory_lengths[i]]
                    if attention is not None else None))
            # If the batch reached the end, save the n_best hypotheses.
            finish_flag = self.top_beam_finished[i] != 0
            if finish_flag and len(self.hypotheses[b]) >= self.n_best:
                ranked_hyp = sorted(
                    self.hypotheses[b], key=lambda x: x[0], reverse=True)
                for n, (score, pred, attn) in enumerate(ranked_hyp):
                    if n >= self.n_best:
                        break
                    self.scores[b].append(score)
                    self.predictions[b].append(pred)
                    self.attention[b].append(
                        attn if attn is not None else [])
            else:
                non_finished_batch.append(i)
        non_finished = torch.tensor(non_finished_batch)
        # If all sentences are translated, no need to go further.
        if len(non_finished) == 0:
            self.done = True
            return

        _B_new = non_finished.shape[0]
        # Remove finished batches for the next step.
        self.top_beam_finished = self.top_beam_finished.index_select(
            0, non_finished)
        self._batch_offset = self._batch_offset.index_select(0, non_finished)
        non_finished = non_finished.to(self.topk_ids.device)
        self.topk_log_probs = self.topk_log_probs.index_select(0,
                                                               non_finished)
        self._batch_index = self._batch_index.index_select(0, non_finished)
        self.select_indices = self._batch_index.view(_B_new * self.beam_size)
        self.alive_seq = predictions.index_select(0, non_finished) \
            .view(-1, self.alive_seq.size(-1))
        self.topk_scores = self.topk_scores.index_select(0, non_finished)
        self.topk_ids = self.topk_ids.index_select(0, non_finished)
        self.field_signals = self.field_signals.index_select(0, non_finished)

        # Coverage
        if self.alive_attn is not None:
            inp_seq_len = self.alive_attn.size(-1)
            self.alive_attn = attention.index_select(0, non_finished) \
                .view(_B_new * self.beam_size, step - 1, inp_seq_len)
            self._coverage = self._coverage \
                .view(_B_old, 1, self.beam_size, inp_seq_len) \
                .index_select(0, non_finished) \
                .view(_B_new * self.beam_size, 1, inp_seq_len)
            self._prev_penalty = self._prev_penalty.index_select(
                0, non_finished)

    def coverage_wu(self, cov):
        x = -torch.min(cov, cov.clone().fill_(1.0)).log()
        x[x == float('Inf')] = 0.  # There's a case with [PAD]; log(0)
        penalty = x.sum(-1)
        return self.pen_beta * penalty

    def length_wu(self, cur_len):
        return ((5 + cur_len) / 6.0) ** self.pen_alpha

