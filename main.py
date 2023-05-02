import torch

from pathlib import Path
from data.dataset import DEAPDataset
from torch.utils.data import random_split, DataLoader

from model.SelectedCrossTransformer import SelectedCrossTransformer


def get_data(split: list, data_dir, data_tag, classification_tag, sample_size=10):
    """
    Splits the dataset into non-overlapping parts of given length.
    :param split: lengths or fractions of splits (as list)
    :param data_dir: directory of data
    :param data_tag: a tag that indicates whether DEAP or DREAMER should be used
    :param classification_tag: A character that indicates which label should be used from data. Valid tags are
        'a' for arousal, 'v' for valence and 'd' for dominance.
    :param sample_size: The size of the sample in seconds. Default is 10.
    :return:
    """
    # TODO k-fold ?

    assert classification_tag.lower() in ['a', 'v', 'd'], "Please provide a valid classification tag. " \
                                                          "Valid tags are: a, v, d for arousal, valence and " \
                                                          "dominance. "

    # set generator for random permutation
    generator = torch.Generator().manual_seed(42)

    # get dataset
    if data_tag.lower() == 'deap':
        dataset = DEAPDataset(data_dir, classification_tag, sample_size)
    else:
        raise ValueError("Please provide valid dataset. Valid datasets are deap and dreamer.")

    return random_split(dataset, split, generator)


if __name__ == '__main__':
    data_dir = Path('datasets/DEAP/data_preprocessed_python')
    data_tag = 'deap'
    classification_tag = 'a'

    batch_size = 32
    epochs = 1

    train_data, test_data = get_data([0.7, 0.3], data_dir=data_dir, data_tag=data_tag,
                                     classification_tag=classification_tag)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    model = SelectedCrossTransformer(data_dim=32, in_length=1280, classification_tag=classification_tag)

    model.learn(train=train_loader, test=test_loader, epochs=epochs)

