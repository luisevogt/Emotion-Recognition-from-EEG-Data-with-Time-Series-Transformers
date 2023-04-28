import torch
import torch.nn as nn
from einops import repeat

from model.crossformer.cross_encoder import Encoder
from model.crossformer.cross_decoder import Decoder
from model.crossformer.cross_embed import DSW_embedding

from math import ceil


# maybe change to nn.Module, depending on what the final model will be
class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size=4,
                 factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3,
                 dropout=0.0, baseline=False):
        # init base model with classification tag
        super(Crossformer, self).__init__()

        self.data_dim = data_dim  # number of dimensions in the data (num of channels)
        self.in_len = in_len  # length of input / history sequence (10 secs * sample frequency)
        self.out_len = out_len  # length of output. Not needed for use case  TODO: delete this
        self.seg_len = seg_len  # length of each time segment
        self.merge_win = win_size  # number of overlapping values in segments

        self.baseline = baseline  # ??? I think not needed anymore, to be deleted later

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth=1,
                               dropout=dropout, in_seg_num=(self.pad_in_len // seg_len), factor=factor)

        # TODO delete this for use case and replace with classification layer
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout,
                               out_seg_num=(self.pad_out_len // seg_len), factor=factor)

    def forward(self, x_seq):
        # don't need
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0
        batch_size = x_seq.shape[0]

        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        # TODO: put classification here
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, :self.out_len, :]
