import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import math
from pathlib import Path
import pickle
import random

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from
from ray.tune.schedulers import AsyncHyperBandScheduler

import model_utils
import dataset_utils

import models.resnet_imagenet as imagenet_models # only use imagenet models
import models

from models.butterfly_conv import ButterflyConv2d, ButterflyConv2dBBT
from models.low_rank_conv import LowRankConv2d

class TrainableModel(Trainable):
    """Trainable object for a Pytorch model, to be used with Ray's Hyperband tuning.
    """

    def _setup(self, config):
        model_args = config['model']['args']
        device = config['device']
        self.device = device
        torch.manual_seed(config['seed'])
        if self.device == 'cuda':
            torch.cuda.manual_seed(config['seed'])
        self.layer = model_args['layer']
        # make butterfly
        if config['dataset'] == 'cifar10':
            teacher_model = models.__dict__[config['teacher_model']]()
        elif config['dataset'] == 'imagenet':
            teacher_model = imagenet_models.__dict__[config['teacher_model']]()

        modules = set([name for name, _ in teacher_model.named_modules()])
        assert model_args['layer'] in modules, f"{model_args['layer']} not in network"

        # get parameters from layer to replace to use in butterfly
        for name, module in teacher_model.named_modules():
            if name == model_args['layer']:
                try:
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                except:
                    raise ValueError("Only convolutional layers currently supported.")

        # create butterfly for specific layer and train
        if model_args['structure_type'] == 'B':
            structured_layer = ButterflyConv2d(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, tied_weight=False, ortho_init=True,
                param=model_args['param'])
        elif model_args['structure_type'] == 'BBT' or model_args['nblocks'] > 1:
            structured_layer = ButterflyConv2dBBT(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, nblocks=model_args['nblocks'], tied_weight=False,
                ortho_init=True, param=model_args['param'])
        elif model_args['structure_type'] == 'LR':
            assert out_channels >= in_channels, "Out channels < in channels"
            if model_args['nblocks'] == 0:
                rank = int(math.log2(out_channels))
            else:
                rank = int(math.log2(out_channels)) * model_args['nblocks'] * 2
            structured_layer =  LowRankConv2d(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, rank=rank)

        self.model = structured_layer.to(device)

        def load_teacher(traindir):
            teacher_input, teacher_output = dataset_utils.get_mmap_files(traindir, self.layer)
            train = torch.utils.data.TensorDataset(teacher_input, teacher_output)
            train_loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True,
                num_workers=config['workers'], pin_memory=True)
            return train_loader

        # load activations
        self.train_loader = load_teacher(config['train_dir'])
        if config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(structured_layer.parameters(), lr=config['lr'])
        else:
            self.optimizer = optim.SGD(structured_layer.parameters(), lr=config['lr'])

    def _train_iteration(self):
        self.model.train()
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()

    def _test(self):
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.mse_loss(output, target).item()
        training_loss = loss / len(self.train_loader)
        return {"mean_loss": training_loss, "inverse_loss": 1/training_loss}

    def _train(self):
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()
                 }
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        if hasattr(self, 'device'):
            checkpoint = torch.load(checkpoint_path, self.device)
        else:
            checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


ex = Experiment('Distillation_experiment')
ex.observers.append(FileStorageObserver.create('logs'))
slack_config_path = Path('../config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def default_config():
    model = 'resnet18'  # Name of model, see model_utils.py
    model_args = {'structure_type': 'B',
                  'nblocks': 1,
                  'param': 'regular'}  # Arguments to be passed to the model, as a dictionary
    optimizer = 'SGD'  # Which optimizer to use, either Adam or SGD
    ntrials = 8  # Number of trials for hyperparameter tuning
    nmaxepochs = 10  # Maximum number of epochs
    result_dir = project_root + '/cnn/ray_results'  # Directory to store results
    train_dir = '/distillation/imagenet/activations'
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing
    workers = 4
    dataset = 'imagenet'
    teacher_model = 'resnet18'
    iters = 1

@ex.capture
def distillation_experiment(model, model_args, optimizer,
    ntrials, result_dir, train_dir, workers, cuda, smoke_test, teacher_model, dataset, iters):
    assert optimizer in ['Adam', 'SGD'], 'Only Adam and SGD are supported'
    config={
        'optimizer': optimizer,
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(2e-5), math.log(1e-1)) if optimizer == 'Adam'
                                           else random.uniform(math.log(2e-3), math.log(1e-0)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'device': 'cuda' if cuda else 'cpu',
        'model': {'name': model, 'args': model_args},
        'teacher_model': teacher_model,
        'train_dir': train_dir,
        'workers': workers,
        'dataset': dataset
     }
    model_args_print = '_'.join([f'{key}_{value}' for key,value in model_args.items()])
    experiment = RayExperiment(
        name=f'{model}_{model_args_print}_{optimizer}',
        run=TrainableModel,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        checkpoint_freq=1000,  # Just to enable recovery with @max_failures
        max_failures=-1,
        resources_per_trial={'cpu': 4, 'gpu': 1 if cuda else 0},
        stop={"training_iteration": iters},
        config=config,
    )
    return experiment


@ex.automain
def run(model, result_dir, nmaxepochs):
    experiment = distillation_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            ray.init(redis_address=address)
    except:
        ray.init()
    trials = ray.tune.run(experiment, raise_on_failed_trial=True, queue_trials=True)
    trials = [trial for trial in trials if trial.last_result is not None]
    loss = [trial.last_result.get('mean_loss', float('inf')) for trial in trials]

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return min(loss)
