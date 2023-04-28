from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model.base_model import BaseModel


class BinaryClassification(BaseModel):
    def __init__(self, log=True):
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%H%M%S")
            self._tb_path = f"runs/{self.__tb_sub}"  # put tag here in final model
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        super(BinaryClassification, self).__init__(classification_tag='a')

        self.layer_1 = nn.Linear(12, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

        self._loss_fn = nn.BCEWithLogitsLoss()
        self._optim = optim.Adam(self.parameters(), lr=0.001)  # change and put lr in init plus weight decay
        # scheduler here I guess

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"training on {self.device}")

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x
