import torch.nn as nn
from einops import rearrange


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, hidden_dim):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, hidden_dim)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape
        # reshape tensor from (batch, ts_len, ts_dim) to (batch * ts_dim * ts_len // seg_len, seg_len) to divide inputs
        # to segments
        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        # embed every segment. dims: (batch * ts_dim * ts_len // seg_len, hidden_dim)
        x_embed = self.linear(x_segment)
        # reshape to input dimensions.
        # From (batch * ts_dim // seg_len, hidden_dim) to (batch, ts_dim, seg_num, hidden_dim)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)

        return x_embed
