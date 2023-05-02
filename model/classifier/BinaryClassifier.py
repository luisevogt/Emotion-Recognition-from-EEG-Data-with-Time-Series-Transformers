from datetime import datetime

import torch
import torch.nn as nn
from einops import rearrange


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_dim, dropout, reduced_dim=64):
        super(BinaryClassifier, self).__init__()

        self._reduced_dim = reduced_dim

        self.layer_1 = nn.Linear(hidden_dim, self._reduced_dim)
        self.layer_2 = nn.Linear(self._reduced_dim, self._reduced_dim)
        self.layer_out = nn.Linear(self._reduced_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(self._reduced_dim)

    def forward(self, inputs):
        # extract cls token and take the average
        cls_tokens = inputs[:, :, 0, :]
        avg_token = torch.mean(cls_tokens, dim=1)

        x = self.relu(self.layer_1(avg_token))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
