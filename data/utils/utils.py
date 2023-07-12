import os.path
import pickle

import torch
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from collections import Counter

from data.dataset import DEAPDataset, DreamerDataset


def get_class_distribution(dataset):
    """
    Calculates the distribution of different classes in a dataset object.
    :param dataset: The dataset where distribution is wanted
    :return: a dictionary that contains the count of all classes in the dataset.
    """

    if isinstance(dataset, DEAPDataset):
        # get class names
        idx_to_class_names = dataset.get_class_names()

        # keep track of counts in a dict
        count_dict = {v: 0 for v in idx_to_class_names.values()}

        # loop over dataset
        for element in dataset:
            label = element[1]
            label = idx_to_class_names[label]
            count_dict[label] += 1

        return count_dict


def get_class_distribution_loaders(dataloader, dataset):
    # get class names
    idx_to_class_names = dataset.get_class_names()

    # keep track of counts in a dict
    count_dict = {v: 0 for v in idx_to_class_names.values()}

    for _, j in dataloader:
        y_idx = j.item()
        label = idx_to_class_names[y_idx]
        count_dict[label] += 1

    return count_dict


def read_targets(filepath):
    return pickle.load(open(filepath, 'rb'))


def stratify_data(split: list, data_dir, data_tag, classification_tag, sample_size=10):
    """
    Performs a stratified split on the indexes of the dataset and returns SubsetRandomSamplers for train, vali
    test that can be given to dataloaders.
    :param split: fractions of the dataset split into train, validation and test
    :param data_dir: directory of data
    :param data_tag: a tag that indicates whether DEAP or our data should be used
    :param classification_tag: A character that indicates which label should be used from data. Valid tags are
        'a' for arousal, 'v' for valence and 'd' for dominance.
    :param sample_size: The size of the sample in seconds. Default is 10.
    :return: SubsetRandomSamplers for train, vali and test set.
    """

    train_size, vali_size, test_size = split

    # get dataset
    if data_tag.lower() == 'deap':
        dataset = DEAPDataset(data_dir, classification_tag, sample_size)

    elif data_tag.lower() == 'dreamer':
        dataset = DreamerDataset(data_dir, classification_tag, sample_size)

    else:
        raise ValueError("Please provide valid dataset. Valid datasets are deap and dreamer.")

    # stratified split
    targets = read_targets(dataset.targets)

    # calculate weights for weighted loss
    # change it to be on train split for inverse number of samples 1/classcount
    class_dist = Counter(targets)
    print(class_dist)
    weight_1 = torch.tensor(class_dist[0] / class_dist[1])

    train_idx, test_idx = train_test_split(np.arange(len(targets)),
                                           test_size=test_size,
                                           shuffle=True,
                                           stratify=targets)

    targets = [targets[i] for i in train_idx]

    train_idx, vali_idx = train_test_split(train_idx,
                                           test_size=vali_size / train_size,
                                           shuffle=True,
                                           stratify=targets)

    train_sampler = SubsetRandomSampler(train_idx)
    vali_sampler = SubsetRandomSampler(vali_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    return dataset, train_sampler, vali_sampler, test_sampler, weight_1

# TODO k-fold ?
