import torch
import argparse

from pathlib import Path
from data.dataset import DEAPDataset
from torch.utils.data import random_split, DataLoader

from model.SelectedCrossTransformer import SelectedCrossTransformer
from model.base_model import BaseModel
from config.config import Config

config = Config()
config_dict = None


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
    available_models = {'SelectedCrossTransformer': SelectedCrossTransformer}

    # parse arguments
    parser = argparse.ArgumentParser(description="Thesis")
    parser.add_argument("--config", dest="config", help="A .yaml file (path) that contains "
                                                        "hyperparameters and configurations for a training / testing")

    args = parser.parse_args()

    # get config dict
    config_path = args.config
    if config_path:
        config_dict = config.get_args(config_path)

    # TODO distinguish between load and train

    # prepare data
    print("Load data...")
    classification_tag = config_dict['classification_tag']

    dataset_args = config_dict['dataset_args']
    dataset_args['classification_tag'] = classification_tag

    dataloader_args = config_dict['dataloader_args']

    train_data, vali_data, test_data = get_data(**dataset_args)

    train_loader = DataLoader(dataset=train_data, **dataloader_args, pin_memory=True)
    vali_loader = DataLoader(dataset=vali_data, pin_memory=True)
    test_loader = DataLoader(dataset=test_data, pin_memory=True)

    # get model
    device = config_dict['device']

    model_name = config_dict['model_name']
    model_args = config_dict['model_args']

    if model_args['channel_grouping'] == 'None':
        if dataset_args['data_tag'] == 'deap':
            _, channel_grouping = DEAPDataset.get_channel_grouping()
            model_args['channel_grouping'] = channel_grouping

    sample_size = config_dict['dataset_args']['sample_size']
    sample_freq = DEAPDataset.sample_freq
    model_args['in_length'] = sample_size * sample_freq
    model_args['classification_tag'] = classification_tag

    model = available_models[model_name](**model_args)
    model.use_device(device)

    seed = config_dict['seed']

    if seed != 'None':
        if device == 'cpu':
            torch.manual_seed(seed)
        elif device == 'cuda':
            torch.cuda.manual_seed_all(seed)

    print(f'Start training of model {model_name}.')

    # model.learn(train=train_loader, validate=vali_loader, test=test_loader, epochs=config_dict['train_epochs'],
    #            save_every=config_dict['save_every'])
    model.eval()
    model.test(test_loader, 0)

    # config_dict['evaluation'] = model.log_path
    # config_dict['model_args']['log'] = False
    # config.store_args(f"{model.log_path}/config.yml", config_dict)

    # BaseModel.save_to_default(model)
