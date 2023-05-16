from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model.base_model import BaseModel

from math import ceil

from model.crossformer.cross_embed import DSW_embedding
from model.crossformer.cross_encoder import Encoder
from model.classifier.BinaryClassifier import BinaryClassifier


def get_cls_pos_encoding(seq_length, d):
    result = torch.ones(seq_length, d)
    for i in range(seq_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))

    return result


class SelectedCrossTransformer(BaseModel):
    def __init__(self, data_dim, in_length, classification_tag, channel_grouping=None, seg_length=10, win_size=2,
                 factor=10,
                 hidden_dim=512, ff_dim=1024, num_heads=4, e_layers=3, lr=1e-3, lr_decay=0.5, momentum=0.9,
                 weight_decay=1e-2, dropout=0.1, device='cpu', tag='SelectedCrossTransformer', log=True):

        if not channel_grouping:
            self.channel_grouping = {0: [channel_idx for channel_idx in range(data_dim)]}
        else:
            self.channel_grouping = channel_grouping

        # if logging enabled, then create a tensorboard writer, otherwise prevent the
        # parent class to create a summary writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%H%M%S")
            self._tb_path = f"runs/{tag}/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
            print('Summary writer activated.')
        else:
            self._writer = False

        super(SelectedCrossTransformer, self).__init__(classification_tag=classification_tag)

        self._data_dim = data_dim
        self._in_len = in_length
        self._seg_length = seg_length
        self._merge_win = win_size

        self._tag = tag

        self._device = device

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * in_length / seg_length) * seg_length
        self.in_len_add = self.pad_in_len - self._in_len

        # Class tokens
        self.class_token = nn.Parameter(torch.rand(1, hidden_dim))
        self.class_token.requires_grad = True
        self.cls_pos_embedding = nn.Parameter(torch.tensor(get_cls_pos_encoding(1, hidden_dim)))
        self.cls_pos_embedding.requires_grad = False  # do not learn position encoding of cls token

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_length, hidden_dim)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_length), hidden_dim))
        self.pre_norm = nn.LayerNorm(hidden_dim)

        # Encoder
        self.encoder = Encoder(e_blocks=e_layers, win_size=win_size, d_model=hidden_dim, n_heads=num_heads,
                               channel_grouping=self.channel_grouping, d_ff=ff_dim, block_depth=1,
                               dropout=dropout, in_seg_num=(self.pad_in_len // seg_length) + 1, factor=factor)

        # Classification Layer
        self.classification = BinaryClassifier(hidden_dim, dropout)

        # set loss function, optimizer and scheduler for learning rate decay
        self._loss_fn = nn.BCEWithLogitsLoss()
        self._optim = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self._scheduler = None
        if lr_decay:
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optim, gamma=lr_decay)

    def forward(self, x_seq):

        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        # prepare cls tokens
        cls_token = self.class_token + self.cls_pos_embedding

        # embed segments and add cls token
        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        batch_size, channels = x_seq.shape[0], x_seq.shape[1]
        x_seq = torch.stack([torch.stack([torch.vstack((cls_token, x_seq[batch_idx, channel]))
                                          for channel in range(channels)])
                             for batch_idx in range(batch_size)])
        x_seq = self.pre_norm(x_seq)

        # get encoder output
        enc_out = self.encoder(x_seq)

        # classification
        x = self.classification(enc_out)

        return x
