import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial

from config.config import Config
from data.dataset import DEAPDataset, DreamerDataset
from data.utils.utils import stratify_data, get_class_distribution_loaders
from model.SelectedCrossTransformer import SelectedCrossTransformer
from torchsummary import summary
from model.base_model import BaseModel
from ray.air import Checkpoint, session
from ray import tune
from ray.tune.schedulers import ASHAScheduler

available_models = {'SelectedCrossTransformer': SelectedCrossTransformer}
config = Config()
config_dict = None

max_num_epochs = 15
num_samples = 100


def get_seg_lengths(s_size):
    if s_size == 1:
        return np.random.choice(np.array([2, 4, 8, 16]))
    elif s_size == 5:
        return np.random.choice(np.array([8, 16, 32, 40, 64, 80]))
    elif s_size == 10:
        return np.random.choice(np.array([16, 32, 40, 64, 80, 128, 256]))


hyperparam_config = {
    "sample_size": tune.choice([1, 5, 10]),
    "seg_length": tune.sample_from(lambda spec: get_seg_lengths(spec.config.sample_size)),
    "factor": tune.choice(range(1, 5)),
    "lr": tune.loguniform(1e-5, 1e-2),
}


def main(hyper_param_config, config=None):
    # prepare data
    print("Load data...")
    config_copy = config.copy()
    classification_tag = config_copy['classification_tag']

    dataset_args = config_copy['dataset_args']
    dataset_args['classification_tag'] = classification_tag

    dataloader_args = config_copy['dataloader_args']

    dataset, train_sampler, vali_sampler, test_sampler, weights = stratify_data(
        sample_size=hyper_param_config["sample_size"], **dataset_args)

    train_loader = DataLoader(dataset=dataset, **dataloader_args, sampler=train_sampler, pin_memory=True)
    vali_loader = DataLoader(dataset=dataset, sampler=vali_sampler, batch_size=1, pin_memory=True)
    test_loader = DataLoader(dataset=dataset, sampler=test_sampler, batch_size=1, pin_memory=True)

    # get model
    device = config_copy['device']

    model_name = config_copy['model_name']
    model_args = config_copy['model_args']
    _, channel_grouping = DreamerDataset.get_channel_grouping()
    if model_args['channel_grouping'] == 'None':
        channel_grouping = None
    elif model_args['channel_grouping'] == 'deap':
        _, channel_grouping = DEAPDataset.get_channel_grouping()
    elif model_args['channel_grouping'] == 'dreamer':
        _, channel_grouping = DreamerDataset.get_channel_grouping()

    model_args['channel_grouping'] = channel_grouping

    if dataset_args['data_tag'] == 'deap':
        sample_freq = DEAPDataset.sample_freq
    elif dataset_args['data_tag'] == 'dreamer':
        sample_freq = DreamerDataset.sample_freq

    if model_args['lr_decay'] == 'None':
        model_args['lr_decay'] = None

    sample_size = hyper_param_config["sample_size"]
    model_args['in_length'] = sample_size * sample_freq
    model_args['classification_tag'] = classification_tag

    model = available_models[model_name](seg_length=hyper_param_config["seg_length"],
                                         factor=hyper_param_config["factor"],
                                         lr=hyper_param_config["lr"],
                                         **model_args, weights=None)
    model.use_device(device)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        model._optim.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    seed = config_copy['seed']

    if seed != 'None':
        if device == 'cpu':
            torch.manual_seed(seed)
        elif device == 'cuda':
            torch.cuda.manual_seed_all(seed)

    print(f'Start training of model {model_name}.')

    # print(summary(model, input_size=(768, 32)))
    model.train()
    model.learn(train=train_loader, validate=vali_loader, epochs=max_num_epochs)


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description="Thesis")
    parser.add_argument("--config", dest="config", help="A .yaml file (path) that contains "
                                                        "hyperparameters and configurations for a training / testing")

    args = parser.parse_args()

    # get config dict
    config_path = args.config
    if config_path:
        config_dict = config.get_args(config_path)


    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    result = tune.run(
        partial(main, config=config_dict),
        resources_per_trial={"cpu": 2},
        config=hyperparam_config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("accuracy", "max", "all")
    print(f"Best trial local path: {best_trial.local_path()}")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    print("Load test data...")
    config_copy = config_dict.copy()
    classification_tag = config_dict.copy()['classification_tag']

    dataset_args = config_copy['dataset_args']
    dataset_args['classification_tag'] = classification_tag

    dataloader_args = config_copy['dataloader_args']

    dataset, train_sampler, vali_sampler, test_sampler, weights = stratify_data(
        sample_size=best_trial.config["sample_size"], **dataset_args)

    test_loader = DataLoader(dataset=dataset, sampler=test_sampler, batch_size=1, pin_memory=True)

    model_args = config_dict.copy()['model_args']

    _, channel_grouping = DreamerDataset.get_channel_grouping()
    if model_args['channel_grouping'] == 'None':
        channel_grouping = None
    elif model_args['channel_grouping'] == 'deap':
        _, channel_grouping = DEAPDataset.get_channel_grouping()
    elif model_args['channel_grouping'] == 'dreamer':
        _, channel_grouping = DreamerDataset.get_channel_grouping()

    model_args['channel_grouping'] = channel_grouping

    if config_dict['data_tag'] == 'deap':
        sample_freq = DEAPDataset.sample_freq
    elif config_dict['data_tag'] == 'dreamer':
        sample_freq = DreamerDataset.sample_freq

    if model_args['lr_decay'] == 'None':
        model_args['lr_decay'] = None

    sample_size = best_trial.config["sample_size"]
    model_args['in_length'] = sample_size * sample_freq
    model_args['classification_tag'] = classification_tag

    best_trained_model = SelectedCrossTransformer(seg_length=best_trial.config["seg_length"],
                                                  factor=best_trial.config["factor"],
                                                  lr=best_trial.config["lr"],
                                                  **model_args, weights=None)
    best_trained_model.use_device(config_dict["device"])

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    best_trained_model.eval()
    test_acc = best_trained_model.test(test_loader)
    print("Best trial test set accuracy: {}".format(test_acc))
