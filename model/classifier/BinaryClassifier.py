from datetime import datetime

import torch
import torch.nn as nn
from einops import rearrange


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_dim, channel_grouping):
        super(BinaryClassifier, self).__init__()

        self.num_cls_tokens = len(channel_grouping)
        self.channel_grouping = channel_grouping
        self.layer_1 = nn.Linear(hidden_dim * self.num_cls_tokens, 3)
        # self.layer_2 = nn.Linear(self._reduced_dim, self._reduced_dim)
        # self.layer_out = nn.Linear(self._reduced_dim, 1)

        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=dropout)

        # self.batchnorm1 = nn.BatchNorm1d(64)
        # self.batchnorm2 = nn.BatchNorm1d(self._reduced_dim)

    def forward(self, inputs):
        # extract cls token per group and take the average
        tokens = []
        for group_idx, channels in self.channel_grouping.items():
            sub_inputs = torch.index_select(inputs, dim=1, index=torch.LongTensor(channels).to(inputs.device))
            cls_token = sub_inputs[:, 0, 0, :]
            tokens.append(cls_token)

        # concat averages
        cls_input = torch.cat(tokens, dim=-1).to(inputs.device)

        x = self.layer_1(cls_input)

        # x = self.relu(self.layer_1(avg_token))
        # x = self.batchnorm1(x)
        # x = self.relu(self.layer_2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        # x = self.layer_out(x)

        return x
