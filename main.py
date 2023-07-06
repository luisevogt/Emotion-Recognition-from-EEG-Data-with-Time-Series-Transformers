import argparse

import torch
from torch.utils.data import DataLoader

from config.config import Config
from data.dataset import DEAPDataset, NexusDataset
from data.utils.utils import stratify_data, get_class_distribution
from model.SelectedCrossTransformer import SelectedCrossTransformer
from torchsummary import summary
from model.base_model import BaseModel
from plot.plot import plot_bar_chart

config = Config()
config_dict = None

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
    config_copy = config_dict.copy()
    classification_tag = config_copy['classification_tag']

    dataset_args = config_copy['dataset_args']
    dataset_args['classification_tag'] = classification_tag

    dataloader_args = config_copy['dataloader_args']

    dataset, train_sampler, vali_sampler, test_sampler = stratify_data(**dataset_args)

    train_loader = DataLoader(dataset=dataset, **dataloader_args, sampler=train_sampler, pin_memory=True)
    vali_loader = DataLoader(dataset=dataset, sampler=vali_sampler, batch_size=1, pin_memory=True)
    test_loader = DataLoader(dataset=dataset, sampler=test_sampler, batch_size=1, pin_memory=True)

    # get model
    device = config_copy['device']

    model_name = config_copy['model_name']
    model_args = config_copy['model_args']

    if model_args['channel_grouping'] == 'None':
        model_args['channel_grouping'] = None
    elif model_args['channel_grouping'] == 'deap' and dataset_args['data_tag'] == 'deap':
        _, channel_grouping = DEAPDataset.get_channel_grouping()
        model_args['channel_grouping'] = channel_grouping
        sample_freq = DEAPDataset.sample_freq
    elif model_args['channel_grouping'] == 'nexus' and dataset_args['data_tag'] == 'nexus':
        _, channel_grouping = NexusDataset.get_channel_grouping()
        model_args['channel_grouping'] = channel_grouping
        sample_freq = NexusDataset.sample_freq

    if model_args['lr_decay'] == 'None':
        model_args['lr_decay'] = None

    sample_size = config_copy['dataset_args']['sample_size']
    model_args['in_length'] = sample_size * sample_freq
    model_args['classification_tag'] = classification_tag

    model = available_models[model_name](**model_args)
    model.use_device(device)

    seed = config_copy['seed']

    if seed != 'None':
        if device == 'cpu':
            torch.manual_seed(seed)
        elif device == 'cuda':
            torch.cuda.manual_seed_all(seed)

    print(f'Start training of model {model_name}.')

    # print(summary(model, input_size=(768, 32)))

    model.learn(train=train_loader, validate=vali_loader, test=test_loader, epochs=config_dict['train_epochs'],
                save_every=config_dict['save_every'])

    # model.eval()
    # model.test(dataloader=test_loader)

    config_dict['evaluation'] = model.log_path
    config_dict['model_args']['log'] = False
    config.store_args(f"{model.log_path}/config.yml", config_dict)

    BaseModel.save_to_default(model)
