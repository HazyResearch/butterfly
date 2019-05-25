import os, sys, subprocess
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import math
from pathlib import Path
import pickle
import random
import datetime

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
        self.config = config

        device = config['device']
        self.device = device
        torch.manual_seed(config['seed'])
        if self.device == 'cuda':
            torch.cuda.manual_seed(config['seed'])

        # model
        self.model = model_utils.get_model(config['model']).to(device)
        self.model_args = config['model']
        # count parameters
        self.nparameters = sum(param.nelement() for param in self.model.parameters())
        print("Parameter count: ", self.nparameters)

        # dataset
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
        # scheduler
        if config['lr_decay']['milestones'] is not None:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['lr_decay']['milestones'], gamma=config['lr_decay']['factor'])
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay']['period'], gamma=config['lr_decay']['factor'])
        self.switch_ams = config['switch_ams']
        # if self.switch_ams is not None:
        #     self.ams_optimizer = optim.Adam([{'params': structured_params, 'weight_decay': 0.0},
        #                                  {'params': unstructured_params}],
        #                                 lr=config['lr'], weight_decay=config['weight_decay'])

    def _train_iteration(self): #TODO report train loss and acc
        self.model.train()
        # with torch.autograd.set_detect_anomaly(True):
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
        return {"nparameters": self.nparameters, "mean_loss": valid_loss, "mean_accuracy": valid_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy}

    def _train(self):
        if self.switch_ams is not None and self._iteration == self.switch_ams:
            print("Switching to AMSGrad")
            structured_params = filter(lambda p: hasattr(p, '_is_structured') and p._is_structured, self.model.parameters())
            unstructured_params = filter(lambda p: not (hasattr(p, '_is_structured') and p._is_structured), self.model.parameters())
            self.optimizer = optim.Adam([{'params': structured_params, 'weight_decay': 0.0},
                                         {'params': unstructured_params}],
                                        lr=self.config['lr'], weight_decay=self.config['weight_decay'], amsgrad=True)
            # self.optimizer = self.ams_optimizer
            # for group in self.optimizer.param_groups:
            #     group['amsgrad'] = True
        self._train_iteration()
        metrics = self._test()
        self.scheduler.step()
        return metrics

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        full_model = {
            'state': self.model.state_dict(),
            'args': self.model_args,
        }
        state = {'model': full_model,
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        if hasattr(self, 'device'):
            checkpoint = torch.load(checkpoint_path, self.device)
        else:
            checkpoint = torch.load(checkpoint_path)
        self.model = model_utils.get_model(checkpoint['model']['args'])
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model']['state'])

        # TODO: refactor this into an optimizer constructing helper
        structured_params = filter(lambda p: hasattr(p, '_is_structured') and p._is_structured, self.model.parameters())
        unstructured_params = filter(lambda p: not (hasattr(p, '_is_structured') and p._is_structured), self.model.parameters())
        self.optimizer = optim.Adam([{'params': structured_params},
                                     {'params': unstructured_params}],)
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scheduler.optimizer = self.optimizer


ex = Experiment('Cifar10_experiment')
ex.observers.append(FileStorageObserver.create('logs'))
slack_config_path = Path('../config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def default_config():
    dataset = 'CIFAR10'
    model = 'LeNet'  # Name of model, see model_utils.py
    args = {}  # Arguments to be passed to the model, as a dictionary
    optimizer = 'SGD'  # Which optimizer to use, either Adam or SGD
    nmaxepochs = 200  # Maximum number of epochs
    use_hyperband = False
    lr = {'grid': [0.025, 0.05, 0.1, 0.2]}
    lr_decay = {'factor': 0.2, 'period': None, 'milestones': [int(30 * nmaxepochs / 100), int(60 * nmaxepochs / 100), int(80 * nmaxepochs / 100)]}
    # lr_decay = True  # Whether to use learning rate decay
    lr_decay_period = 25  # Period of learning rate decay
    weight_decay = False  # Whether to use weight decay
    ntrials = 20  # Number of trials for hyperparameter tuning
    batch = 128
    grace_period = 25
    # decay_milestones = [int(30 * nmaxepochs / 100), int(60 * nmaxepochs / 100), int(80 * nmaxepochs / 100)]
    resume_pth = None
    result_dir = project_root + '/cnn/results'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing

@ex.named_config
def adam():
    optimizer = 'Adam'  # Which optimizer to use, either Adam or SGD
    use_hyperband = True
    lr = {'min': 1e-4, 'max': 1e-2, 'grid': None}
    # lr_decay = False  # Whether to use learning rate decay
    lr_decay = None # {'factor': 0.2, 'period': 25, 'milestones': [int(30 * nmaxepochs / 100), int(60 * nmaxepochs / 100), int(80 * nmaxepochs / 100)]}}
    lr_decay_period = 25  # Period of learning rate decay
    weight_decay = False  # Whether to use weight decay
    grace_period = 100

@ex.named_config
def sgd():
    # abbreviated sgd schedule for resnet
    optimizer = 'SGD'  # Which optimizer to use, either Adam or SGD
    use_hyperband = True
    lr = {'min': 2e-2, 'max': 2e-1, 'grid': None}
    lr_decay = {'factor': 0.2, 'period': 25, 'milestones': None}
    # lr_decay = True  # Whether to use learning rate decay
    # lr_decay_period = 25  # Period of learning rate decay
    weight_decay = True  # Whether to use weight decay
    nmaxepochs = 100


@ex.capture
def cifar10_experiment(dataset, model, args, optimizer, use_hyperband, lr, lr_decay, weight_decay, ntrials, nmaxepochs, batch, resume_pth, result_dir, cuda, smoke_test):
    assert optimizer in ['Adam', 'SGD'], 'Only Adam and SGD are supported'
    if lr_decay is None:
        lr_decay = {'factor': 1.0, 'period': 1000, 'milestones': None}
    config={
        'optimizer': optimizer,
        'switch_ams': int(0.5 * nmaxepochs) if optimizer == 'Adam' else None,
        'lr': grid_search(lr['grid']) if lr['grid'] is not None else sample_from(lambda spec: math.exp(random.uniform(math.log(lr['min']), math.log(lr['max'])))),
        # 'lr_decay_factor': 0.2 if lr_decay else 1.0,
        # 'lr_decay_period': lr_decay_period if lr_decay else 10000,
        # 'decay_milestones': decay_milestones,
        'lr_decay' : lr_decay,
        'weight_decay': 5e-4 if weight_decay else 0.0,
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'device': 'cuda' if cuda else 'cpu',
        'model': {'name': model, 'args': args},
        'dataset': {'name': dataset, 'batch': batch},
     }
    smoke_str = 'smoke_' if smoke_test else '' # for easy finding and deleting unimportant logs
    args_str = '_'.join([k+':'+str(v) for k,v in args.items()])
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    experiment = RayExperiment(
        name=f'{smoke_str}{dataset.lower()}_{model}_{args_str}_{optimizer}_epochs_{nmaxepochs}_{timestamp}_{commit_id}',
        run=TrainableModel,
        local_dir=result_dir,
        num_samples=ntrials if not smoke_test else 1,
        checkpoint_at_end=True,
        checkpoint_freq=1000,  # Just to enable recovery with @max_failures
        max_failures=0,
        resources_per_trial={'cpu': 4, 'gpu': 1 if cuda else 0},
        stop={"training_iteration": 1 if smoke_test else nmaxepochs},
        restore=resume_pth,
        config=config,
    )
    return experiment


@ex.automain
def run(model, args, result_dir, nmaxepochs, use_hyperband, grace_period):
    experiment = cifar10_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            ray.init(redis_address=address)
    except:
        ray.init()
    if use_hyperband:
        if grace_period == -1: grace_period = nmaxepochs
        ahb = AsyncHyperBandScheduler(reward_attr='mean_accuracy', max_t=nmaxepochs, grace_period=grace_period)
        trials = ray.tune.run(experiment, scheduler=ahb, raise_on_failed_trial=False, queue_trials=True)
    else:
        trials = ray.tune.run(experiment, raise_on_failed_trial=False, queue_trials=True)
    trials = [trial for trial in trials if trial.last_result is not None]
    accuracy = [trial.last_result.get('mean_accuracy', float('-inf')) for trial in trials]
    nparameters = trials[0].last_result['nparameters']

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return max(accuracy), model, nparameters, args
