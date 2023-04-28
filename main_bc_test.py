import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# tutorial https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
from torch.utils.data import Dataset, DataLoader
from model.BinaryClassifier import BinaryClassification

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001


## train data
class TrainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


## test data
class TestData(Dataset):

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


def main():
    df = pd.read_csv('datasets/Dataset_spine.csv')

    # remap class labels
    df['Class_att'] = df['Class_att'].astype('category')
    encode_map = {
        'Abnormal': 1,
        'Normal': 0
    }

    df['Class_att'].replace(encode_map, inplace=True)

    # input and output
    X = df.iloc[:, 0:-2]  # first 64 cols
    y = df.iloc[:, -2]  # last col

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)

    # datanormalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = TrainData(torch.FloatTensor(X_train),
                           torch.FloatTensor(y_train))

    test_data = TestData(torch.FloatTensor(X_test))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    model = BinaryClassification()
    model.use_device(model.device)

    print(model)

    model.learn(train=train_loader, test=(test_loader, y_test), epochs=50)


if __name__ == '__main__':
    main()
