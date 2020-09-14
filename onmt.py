"""Modules imported from OpenNMT"""
import math

import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    """ Multi-Head Attention module originally from OpenNMT"""

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0, \
            f"{model_dim} % {head_count} != 0"
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, attn_type=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        # CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # aeq(self.model_dim % 8, 0)
        # if mask is not None:
        #    batch_, q_len_, k_len_ = mask.size()
        #    aeq(batch_, batch)
        #    aeq(k_len_, k_len)
        #    aeq(q_len_ == q_len)
        # END CHECKS

        def shape(x):
            """Projection"""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """Compute context"""
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"], key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"], value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"], \
                                 layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)
        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3)).float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask.bool(), -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)
        context = unshape(context_original)
        output = self.final_linear(context)
        attns = attn.view(batch_size, head_count, query_len, key_len)

        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        return output, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class AverageAttention(nn.Module):
    """
    Average Attention module from
    "Accelerating Neural Transformer via an Average Attention Network"
    :cite:`DBLP:journals/corr/abs-1805-00631`.

    Args:
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, model_dim, dropout=0.1, aan_useffn=False):
        self.model_dim = model_dim
        self.aan_useffn = aan_useffn
        super(AverageAttention, self).__init__()
        if aan_useffn:
            self.average_layer = PositionwiseFeedForward(model_dim, model_dim,
                                                         dropout)
        self.gating_layer = nn.Linear(model_dim * 2, model_dim * 2)

    def cumulative_average_mask(self, batch_size, inputs_len, device):
        """
        Builds the mask to compute the cumulative average as described in
        :cite:`DBLP:journals/corr/abs-1805-00631` -- Figure 3

        Args:
            batch_size (int): batch size
            inputs_len (int): length of the inputs

        Returns:
            (FloatTensor):

            * A Tensor of shape ``(batch_size, input_len, input_len)``
        """

        triangle = torch.tril(torch.ones(inputs_len, inputs_len,
                              dtype=torch.float, device=device))
        weights = torch.ones(1, inputs_len, dtype=torch.float, device=device) \
            / torch.arange(1, inputs_len + 1, dtype=torch.float, device=device)
        mask = triangle * weights.transpose(0, 1)

        return mask.unsqueeze(0).expand(batch_size, inputs_len, inputs_len)


    def cumulative_average(self, inputs, mask_or_step,
                           layer_cache=None, step=None):
        """
        Computes the cumulative average as described in
        :cite:`DBLP:journals/corr/abs-1805-00631` -- Equations (1) (5) (6)

        Args:
            inputs (FloatTensor): sequence to average
                ``(batch_size, input_len, dimension)``
            mask_or_step: if cache is set, this is assumed
                to be the current step of the
                dynamic decoding. Otherwise, it is the mask matrix
                used to compute the cumulative average.
            layer_cache: a dictionary containing the cumulative average
                of the previous step.

        Returns:
            a tensor of the same shape and type as ``inputs``.
        """

        if layer_cache is not None:
            step = mask_or_step
            average_attention = (inputs + step *
                                 layer_cache["prev_g"]) / (step + 1)
            layer_cache["prev_g"] = average_attention
            return average_attention
        else:
            mask = mask_or_step
            return torch.matmul(mask.to(inputs.dtype), inputs)

    def forward(self, inputs, mask=None, layer_cache=None, step=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor, FloatTensor):

            * gating_outputs ``(batch_size, input_len, model_dim)``
            * average_outputs average attention
                ``(batch_size, input_len, model_dim)``
        """

        batch_size = inputs.size(0)
        inputs_len = inputs.size(1)
        average_outputs = self.cumulative_average(
          inputs, self.cumulative_average_mask(batch_size,
                                               inputs_len, inputs.device)
          if layer_cache is None else step, layer_cache=layer_cache)
        if self.aan_useffn:
            average_outputs = self.average_layer(average_outputs)
        gating_outputs = self.gating_layer(torch.cat((inputs,
                                                      average_outputs), -1))
        input_gate, forget_gate = torch.chunk(gating_outputs, 2, dim=2)
        gating_outputs = torch.sigmoid(input_gate) * inputs + \
            torch.sigmoid(forget_gate) * average_outputs

        return gating_outputs, average_outputs



