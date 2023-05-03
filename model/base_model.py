import string
import pandas as pd
from datetime import datetime
from typing import Any, List

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn

from utils.tools import EarlyStopping


def eval_step(engine, batch):
    return batch


class BaseModel(nn.Module):
    def __init__(self, classification_tag: string) -> None:
        super(BaseModel, self).__init__()

        assert classification_tag.lower() in ['a', 'v', 'd'], "Please provide a valid classification tag. " \
                                                              "Valid tags are: a, v, d for arousal, valence and " \
                                                              "dominance. "
        if classification_tag.lower() == 'a':
            self.__class_names = {0: "low arousal",
                                  1: "high arousal"}
        elif classification_tag.lower() == 'v':
            self.__class_names = {0: "low valence",
                                  1: "high valence"}
        elif classification_tag.lower() == 'd':
            self.__class_names = {0: "low dominance",
                                  1: "high dominance"}
        else:
            raise ValueError("Please provide a valid classification tag. Valid tags are: a, v, d for arousal, "
                             "valence and dominance.")

        # enable tensorboard
        if self._writer is None:
            self.__tb_sub = datetime.now().strftime("%H%M%S")
            self._tb_path = f"runs/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        self.__sample_position = 0

        # check for gpu
        self._device = "cpu"
        if torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self._device_name}")

        # to be defined by children
        self._scheduler: _LRScheduler = None
        self._optim: Optimizer = None
        self._loss_fn: Module = None

    @property
    def log_path(self) -> str:
        return self._tb_path

    def use_device(self, device: str) -> None:
        self._device = device
        self.to(self._device)

    @staticmethod
    def save_to_default(model) -> None:
        """This method saves the current model state to the tensorboard 
        directory.
        """
        model_tag = datetime.now().strftime("%H%M%S")
        torch.save(model.state_dict(), f"{model._tb_path}/model_{model_tag}.torch")

    def forward(self, inputs):
        """
        This method performs the forward call.

        Args:
            inputs (Any): The input passed to the defined neural network.

        Raises:
            NotImplementedError: The Base model has no implementation
                                 for this.
        """
        raise NotImplementedError

    @staticmethod
    def __binary_acc(y_pred, y_test):
        y_pred_labels = torch.round(torch.sigmoid(y_pred))

        sum_correct = (y_pred_labels == y_test).sum().float()
        acc = sum_correct / y_test.shape[0]

        return acc

    def learn(self, train, validate=None, test=None, epochs: int = 1, save_every: int = -1):
        print("Training model on: ", self._device if self._device == 'cpu' else self._device_name)

        # set the model into training mode
        self.train()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # run for n epochs specified
        for e in tqdm(range(epochs)):

            # track loss and accuracy
            losses = []
            accuracies = []

            # run for each batch in training set
            for X, y in train:
                X = X.to(self._device)
                y = y.to(self._device)

                # reset the gradient
                self._optim.zero_grad()

                # perform the prediction and measure the loss between the prediction
                # and the expected output
                _y = self(X)

                loss = self._loss_fn(_y, y.unsqueeze(1).type(torch.float32))
                accuracy = self.__binary_acc(_y, y.unsqueeze(1).type(torch.float32))

                # run backpropagation
                loss.backward()
                self._optim.step()

                losses.append(loss.detach().cpu().item())
                accuracies.append(accuracy.detach().cpu().item())

                # log in tensorboard
                log_loss = np.mean(losses, axis=0)
                log_acc = np.mean(accuracies, axis=0)

                self._writer.add_scalar("Train/loss", log_loss, self.__sample_position)
                self._writer.add_scalar("Train/accuracy", log_acc, self.__sample_position)

                self.__sample_position += X.size(0)

            # if there is an adaptive learning rate (scheduler) available
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Train/learning_rate", lr, e)

            # run a validation of the current model state
            if validate:
                # set the model to eval mode, run validation and set to train mode again
                self.eval()
                vali_loss = self.validate(validate, e)
                early_stopping(vali_loss)
                self.train()

            if test:
                self.eval()
                self.test(test, e)
                self.train()

            if save_every > 0 and e % save_every == 0:
                BaseModel.save_to_default(self)

            epoch_loss = np.mean(losses, axis=0)
            epoch_acc = np.mean(accuracies, axis=0)

            print(f'Epoch {(e + 1) + 0:03}: | Loss: {epoch_loss / len(train):.5f} | Acc: {epoch_acc / len(train):.3f}')

            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.eval()
        self._writer.flush()

    def validate(self, dataloader, log_step: int = -1):
        """Method validates model's accuracy based on the given data. In validation, the model
        only looks one step ahead.

        Args:
            dataloader (_type_): The dataloader which contains value, not used for training.
            log_step (int, optional): The step of the logger, can be disabled by setting to -1.

        Returns:
            float: The model's accuracy.
        """
        accuracies = []
        losses = []

        # predict all y's of the validation set and append the model's accuracy 
        # to the list
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self._device)
                y = y.to(self._device)

                _y = self(X)

                loss = self._loss_fn(_y, y)
                losses.append(loss.detach().cpu().item())
                accuracies.append(self.__binary_acc(_y, y))

        # calculate mean accuracy and loss
        accuracy = np.mean(np.array(accuracies))
        loss = np.mean(np.array(losses))

        # log to the tensorboard if wanted
        if log_step != -1:
            self._writer.add_scalar("Validation/accuracy", accuracy, log_step)
            self._writer.add_scalar("Validation/loss", loss, log_step)

        return loss

    def test(self, dataloader, log_step: int = -1) -> None:
        """Method validates model's accuracy based on test data. During testing, the model
        only looks one step ahead.

        Args:
            dataloader (_type_): The dataloader which contains value, not used for training.
            log_step (int, optional): The step of the logger, can be disabled by setting to -1.

        """

        # input to confusion matrix and classification report
        y_pred_list = []
        y_labels = []

        # predict all y's of the test set and log metrics
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(self._device)
                y = y.to(self._device)

                _y = self(X)

                # get 0 or 1 label for confusion matrix and classification report
                _y = torch.sigmoid(_y)
                _y_label = torch.round(_y)
                y_pred_list.append(_y_label.detach().cpu().numpy())

                y_labels.append(y)

        # log metrics to tensorboard if wanted

        y_pred_list = [pred.squeeze().tolist() for pred in y_pred_list]
        y_labels = [label.squeeze().tolist() for label in y_labels]
        print(len(y_labels), len(y_pred_list))
        report = classification_report(y_labels, y_pred_list,
                                       target_names=list(self.__class_names.values()), output_dict=True)

        precision_0 = report[self.__class_names[0]]['precision']
        precision_1 = report[self.__class_names[1]]['precision']

        recall_0 = report[self.__class_names[0]]['recall']
        recall_1 = report[self.__class_names[1]]['recall']

        f1_score_0 = report[self.__class_names[0]]['f1-score']
        f1_score_1 = report[self.__class_names[1]]['f1-score']

        test_accuracy = report['accuracy']

        # get confusion matrix and log to tensorboard if wanted
        cm = confusion_matrix(y, y_pred_list)
        df_cm = pd.DataFrame(cm / np.sum(cm, axis=1)[:, None],
                             index=[value for value in self.__class_names.values()],
                             columns=[value for value in self.__class_names.values()])

        plt.figure(figsize=(12, 7))
        figure = sn.heatmap(df_cm, annot=True).get_figure()

        if log_step != -1:
            self._writer.add_scalar("Test/accuracy", test_accuracy, log_step)
            self._writer.add_scalar(f"Test/precision_{self.__class_names[0]}", precision_0, log_step)
            self._writer.add_scalar(f"Test/precision_{self.__class_names[1]}", precision_1, log_step)
            self._writer.add_scalar(f"Test/recall_{self.__class_names[0]}", recall_0, log_step)
            self._writer.add_scalar(f"Test/recall_{self.__class_names[1]}", recall_1, log_step)
            self._writer.add_scalar(f"Test/f1-score_{self.__class_names[0]}", f1_score_0, log_step)
            self._writer.add_scalar(f"Test/f1-score_{self.__class_names[1]}", f1_score_1, log_step)

            self._writer.add_figure("Confusion matrix", figure, log_step)
