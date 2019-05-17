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
from ray.tune import Trainable, Experiment as RayExperiment, sample_from, grid_search
from ray.tune.schedulers import AsyncHyperBandScheduler


import model_utils
import dataset_utils


class TrainableModel(Trainable):
    """Trainable object for a Pytorch model, to be used with Ray's Hyperband tuning.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        torch.manual_seed(config['seed'])
        if self.device == 'cuda':
            torch.cuda.manual_seed(config['seed'])
        self.model = model_utils.get_model(config['model']).to(device)
        self.train_loader, self.valid_loader, self.test_loader = dataset_utils.get_dataset(config['dataset'])
        structured_params = filter(lambda p: hasattr(p, '_is_structured') and p._is_structured, self.model.parameters())
        unstructured_params = filter(lambda p: not (hasattr(p, '_is_structured') and p._is_structured), self.model.parameters())
        if config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam([{'params': structured_params, 'weight_decay': 0.0},
                                         {'params': unstructured_params}],
                                        lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            self.optimizer = optim.SGD([{'params': structured_params, 'weight_decay': 0.0},
                                        {'params': unstructured_params}],
                                       lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay_period'], gamma=config['lr_decay_factor'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=config['lr_decay_factor'])

    def _train_iteration(self):
        self.model.train()
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

    def _test(self):
        self.model.eval()
        valid_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                valid_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += (pred == target.data.view_as(pred)).long().cpu().sum()
        valid_loss = valid_loss / len(self.valid_loader.dataset)
        valid_accuracy = correct.item() / len(self.valid_loader.dataset)
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += (pred == target.data.view_as(pred)).long().cpu().sum()
        test_loss = test_loss / len(self.test_loader.dataset)
        test_accuracy = correct.item() / len(self.test_loader.dataset)
        return {"mean_loss": valid_loss, "mean_accuracy": valid_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy}

    def _train(self):
        self.scheduler.step()
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        if hasattr(self, 'device'):
            checkpoint = torch.load(checkpoint_path, self.device)
        else:
            checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


ex = Experiment('Cifar10_experiment')
ex.observers.append(FileStorageObserver.create('logs'))
slack_config_path = Path('../config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def default_config():
    model = 'LeNet'  # Name of model, see model_utils.py
    model_args = {}  # Arguments to be passed to the model, as a dictionary
    optimizer = 'Adam'  # Which optimizer to use, either Adam or SGD
    lr_decay = False  # Whether to use learning rate decay
    lr_decay_period = 25  # Period of learning rate decay
    weight_decay = False  # Whether to use weight decay
    ntrials = 20  # Number of trials for hyperparameter tuning
    nmaxepochs = 100  # Maximum number of epochs
    result_dir = project_root + '/cnn/results'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing


@ex.named_config
def sgd():
    optimizer = 'SGD'  # Which optimizer to use, either Adam or SGD
    lr_decay = True  # Whether to use learning rate decay
    lr_decay_period = 25  # Period of learning rate decay
    weight_decay = True  # Whether to use weight decay


@ex.capture
def cifar10_experiment(model, model_args, optimizer, lr_decay, lr_decay_period, weight_decay, ntrials, nmaxepochs, result_dir, cuda, smoke_test):
    assert optimizer in ['Adam', 'SGD'], 'Only Adam and SGD are supported'
    config={
        'optimizer': optimizer,
        # 'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(2e-5), math.log(1e-2)) if optimizer == 'Adam'
        #                                    else random.uniform(math.log(2e-3), math.log(1e-0)))),
        'lr': grid_search([0.025, 0.05, 0.1, 0.2]),
        # 'lr_decay_factor': sample_from(lambda spec: random.choice([0.1, 0.2])) if lr_decay else 1.0,
        'lr_decay_factor': 0.2,
        # 'lr_decay_period': lr_decay_period,
        # 'weight_decay': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-6), math.log(5e-4)))) if weight_decay else 0.0,
        'weight_decay': 5e-4,
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'device': 'cuda' if cuda else 'cpu',
        'model': {'name': model, 'args': model_args},
        'dataset': {'name': 'CIFAR10'}
     }
    experiment = RayExperiment(
        name=f'cifar10_{model}_{model_args}_{optimizer}',
        run=TrainableModel,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        checkpoint_freq=1000,  # Just to enable recovery with @max_failures
        max_failures=-1,
        resources_per_trial={'cpu': 4, 'gpu': 1 if cuda else 0},
        stop={"training_iteration": 1 if smoke_test else nmaxepochs},
        config=config,
    )
    return experiment


@ex.automain
def run(model, model_args, result_dir, nmaxepochs):
    experiment = cifar10_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            ray.init(redis_address=address)
    except:
        ray.init()
    # ahb = AsyncHyperBandScheduler(reward_attr='mean_accuracy', max_t=nmaxepochs)
    # trials = ray.tune.run(experiment, scheduler=ahb, raise_on_failed_trial=False, queue_trials=True)
    trials = ray.tune.run(experiment, raise_on_failed_trial=False, queue_trials=True)
    trials = [trial for trial in trials if trial.last_result is not None]
    accuracy = [trial.last_result.get('mean_accuracy', float('-inf')) for trial in trials]

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return model, model_args, max(accuracy)
