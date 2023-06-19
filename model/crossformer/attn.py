import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, channel_grouping, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.channel_grouping = channel_grouping

        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.router = nn.Parameter(torch.randn(seg_num - 1, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x):
        # collect group tensors in lis and init tensor of same size to keep channel order
        tensor_list = []
        final_out = torch.zeros(x.shape).to(x.device)
        for group_idx, channels in self.channel_grouping.items():
            # get sub-tensor with corresponding channels
            x_sub = torch.index_select(x, dim=1, index=torch.LongTensor(channels).to(x.device))

            # Cross Time Stage: Directly apply MSA to each dimension
            # loop over batches to avoid mixing up the batches during attention
            batch_size = x_sub.shape[0]
            batch_list = []
            for batch in range(batch_size):
                time_in = x_sub[batch, :, :, :]
                # time_in = rearrange(x_sub, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
                time_enc = self.time_attention(
                    time_in, time_in, time_in
                )
                dim_in = time_in + self.dropout(time_enc)
                dim_in = self.norm1(dim_in)
                dim_in = dim_in + self.dropout(self.MLP1(dim_in))
                dim_in = self.norm2(dim_in)

                # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages
                # to build the D-to-D connection

                # separate the CLS tokens form the signal segments
                cls_tokens = dim_in[:, 0, :]
                dim_send = dim_in[:, 1:, :]

                # cls_tokens = rearrange(cls_tokens, '(b ts_d) d_model -> b ts_d d_model', b=batch)
                dim_send = rearrange(dim_send, 'ts_d seg_num d_model -> seg_num ts_d d_model')
                # batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model',
                #                       repeat=batch)
                dim_buffer = self.dim_sender(self.router, dim_send, dim_send)
                dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
                dim_enc = dim_send + self.dropout(dim_receive)
                dim_enc = self.norm3(dim_enc)
                dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
                dim_enc = self.norm4(dim_enc)

                dim_enc = rearrange(dim_enc, 'seg_num ts_d d_model -> ts_d seg_num d_model')

                # concat CLS tokens with dim attentions on segment dim and append to group tensor list
                group_channels = dim_enc.shape[0]
                dim_enc = torch.stack(
                    [torch.vstack((cls_tokens[g_channel], dim_enc[g_channel]))
                     for g_channel in range(group_channels)])

                batch_list.append(dim_enc)

            batch_enc = torch.stack(batch_list)
            tensor_list.append(batch_enc)

        # stack group tensors along channel dim
        stacked_tensor = torch.cat(tensor_list, dim=1).to(x.device)

        # rearrange tensor to original channel order
        orig_channels = []
        for channels in self.channel_grouping.values():
            orig_channels += channels

        for current_pos, orig_channel in enumerate(orig_channels):
            final_out[:, orig_channel, :, :] = stacked_tensor[:, current_pos, :, :]

        return final_out
