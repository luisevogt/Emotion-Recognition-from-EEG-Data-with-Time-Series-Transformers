import torch
import torch.nn as nn
from model.crossformer.attn import TwoStageAttentionLayer
from math import ceil


class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''

    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """

        batch_size, ts_d, seg_num, d_model = x.shape

        pad_num = (seg_num - 1) % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        # extract cls tokens and concatenate with itself to match dimension of merged tensor
        # but I really don't know if this is the right thing to do here
        cls_tokens = x[:, :, 0, :]
        cls_merge = [cls_tokens for _ in range(self.win_size)]
        cls_tokens = torch.cat(cls_merge, dim=-1)

        seg_to_merge = []
        for i in range(1, self.win_size + 1):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        # stack cls token on top of x
        x = torch.stack([torch.stack([torch.vstack((cls_tokens[batch_idx, channel], x[batch_idx, channel]))
                                      for channel in range(ts_d)])
                         for batch_idx in range(batch_size)])  # [B, ts_d, seg_num/win_size +1, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''

    def __init__(self, win_size, d_model, n_heads, channel_grouping, d_ff, depth, dropout,
                 seg_num=10, factor=10):
        super(scale_block, self).__init__()

        self.channel_grouping = channel_grouping

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, self.channel_grouping,
                                                             d_ff, dropout))

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''

    def __init__(self, e_blocks, win_size, d_model, n_heads, channel_grouping, d_ff, block_depth, dropout,
                 in_seg_num=10, factor=8):
        super(Encoder, self).__init__()

        self.channel_grouping = channel_grouping

        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, self.channel_grouping, d_ff, block_depth, dropout,
                                              in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, d_model, n_heads, self.channel_grouping, d_ff, block_depth,
                                                  dropout, ceil(in_seg_num / win_size ** i), factor))

    def forward(self, x):

        for block in self.encode_blocks:
            x = block(x)

        return x
