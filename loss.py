"""A class with loss functions for TASumm abstractive model

OpenNMT loss function basically uses NLLLoss or LabelSmoothingLoss
(https://arxiv.org/abs/1512.00567).

"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class TASummEncLoss(nn.Module):
    def __init__(self, pos_weight, reduction='none'):
        super(TASummEncLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(
            reduction=reduction,
            pos_weight=torch.FloatTensor([pos_weight, 1]).cuda()
        )

    def forward(self, logits, target, mask):
        tgt_ = F.one_hot(target, num_classes=2).float()
        loss = self.criterion(logits, tgt_)
        if tgt_.dim() > 2:
            loss = (loss * mask.unsqueeze(-1).expand_as(loss)).sum()
            loss = loss / mask.sum()
        return loss


class TASummDecLoss(nn.Module):
    def __init__(self, generator, pad_symbol, vocab_size, label_smoothing=0.0):
        super(TASummDecLoss, self).__init__()
        self.generator = generator
        self.padding_idx = pad_symbol
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(ignore_index=self.padding_idx,
                                        reduction='none')

    def compute_loss(self, batch, outputs):
        target = batch.tgt[:, 1:]
        scores = self.generator(outputs)  # [N x L x V]
        loss = self.criterion(scores.transpose(1, 2), target)
        mask = (target != self.padding_idx)
        loss = (loss * mask.long()).sum() / mask.sum()
        return loss, scores


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')
