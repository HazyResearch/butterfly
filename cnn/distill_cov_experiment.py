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

import models

from mobilenet_imagenet import MobileNet, Butterfly1x1Conv

# from models.butterfly_conv import ButterflyConv2d, ButterflyConv2dBBT

N_LBFGS_STEPS_VALIDATION = 50

class TrainableDistillCovModel(Trainable):
    """Trainable object for a Pytorch model, to be used with Ray's Hyperband tuning.
    """

    def _setup(self, config):
        model_args = config['model_args']
        device = config['device']
        self.device = device
        torch.manual_seed(config['seed'])
        if self.device == 'cuda':
            torch.cuda.manual_seed(config['seed'])
        self.layer = model_args['layer']
        # Load teacher model and weight
        if config['dataset'] == 'cifar10':
            teacher_model = models.__dict__[config['teacher_model']]()
        elif config['dataset'] == 'imagenet':
            assert config['teacher_model'].startswith('mobilenetv1')
            width_mult = 1.0
            if len(config['teacher_model'].split('_')) >= 2:
                width_mult = float(config['teacher_model'].split('_')[1])
            teacher_model = MobileNet(width_mult=width_mult)
        teacher_model = teacher_model.to(self.device)
        loaded_state_dict = torch.load(config['teacher_model_path'], map_location=self.device)['state_dict']
        # Strip names to be compatible with Pytorch 1.2, i.e. 'module.conv1.weight' -> 'conv1.weight'
        loaded_state_dict = {name.replace('module.', ''): weight for name, weight in loaded_state_dict.items()}
        teacher_model.load_state_dict(loaded_state_dict)

        module_dict = dict(teacher_model.named_modules())
        assert model_args['layer'] in module_dict, f"{model_args['layer']} not in network"
        # get parameters from layer to replace to use in butterfly
        teacher_module = module_dict[self.layer]
        try:
            in_channels = teacher_module.in_channels
            out_channels = teacher_module.out_channels
            # kernel_size = teacher_module.kernel_size
            # stride = teacher_module.stride
            # padding = teacher_module.padding
        except:
            raise ValueError("Only convolutional layers currently supported.")

        # create butterfly for specific layer and train
        if model_args['structure_type'] == 'B':
            self.student_module = Butterfly1x1Conv(in_channels, out_channels,
                bias=False, tied_weight=model_args['tied_weight'], ortho_init=True,
                param=model_args['param'], nblocks=model_args['nblocks'])
        self.student_module = self.student_module.to(device)

        input_cov = torch.load(config['input_cov_path'], map_location=self.device)[self.layer]
        # input_cov /= torch.norm(input_cov)
        # Normalized so that each entry of cholesky factor has magnitude about 1.0
        # iid standard Gaussian has spectral norm about sqrt(in_channels) + sqrt(out_channels) (Bai-Yin's law)
        # So we normalize the eigenvalues of input_cov to have size (sqrt(in_channels) + sqrt(out_channels))^2.
        input_cov *= (math.sqrt(in_channels) + math.sqrt(out_channels)) ** 2 / torch.symeig(input_cov)[0].max()
        self.input = torch.cholesky(input_cov, upper=True)
        self.input = self.input.reshape(in_channels, in_channels, 1, 1)  # To be compatible with conv2d
        with torch.no_grad():
            self.target = teacher_module(self.input)
        if config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(self.student_module.parameters(), lr=config['lr'])
        else:
            self.optimizer = optim.SGD(self.student_module.parameters(), lr=config['lr'], momentum=config['momentum'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.n_epochs_per_validation = config['n_epochs_per_validation']

    def loss(self):
        output = self.student_module(self.input)
        return F.mse_loss(output, self.target)

    def _train(self):
        self.student_module.train()
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            self.optimizer.step()
        loss = loss.item()
        if (self._iteration + 1) % self.n_epochs_per_validation == 0:
            loss = min(loss, self.polish(N_LBFGS_STEPS_VALIDATION, save_to_self_model=True))
        return {'mean_loss': loss}

    def polish(self, nmaxsteps=50, patience=5, threshold=1e-8, save_to_self_model=False):
        if not save_to_self_model:
            student_module_bak = self.student_module
            self.student_module = copy.deepcopy(self.student_module)
        optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, self.student_module.parameters()))
        def closure():
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            return loss
        n_bad_steps = 0
        best_loss = float('inf')
        for i in range(nmaxsteps):
            loss = optimizer.step(closure)
            if loss.item() < best_loss - threshold:
                best_loss = loss.item()
                n_bad_steps = 0
            else:
                n_bad_steps += 1
            if n_bad_steps > patience:
                break
        if not save_to_self_model:
            self.student_module = student_module_bak
        return loss.item()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        state = {'student_module': self.student_module.state_dict(),
                 'optimizer': self.optimizer.state_dict()
                 }
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        if hasattr(self, 'device'):
            checkpoint = torch.load(checkpoint_path, self.device)
        else:
            checkpoint = torch.load(checkpoint_path)
        self.student_module.load_state_dict(checkpoint['student_module'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


ex = Experiment('ImageNet covariance distillation_experiment')
ex.observers.append(FileStorageObserver.create('distill_cov_logs'))
slack_config_path = Path('../config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def default_config():
    model_args = {'structure_type': 'B',
                  'nblocks': 1,
                  'param': 'odo',
                  'tied_weight': False,
                  'layer': 'layers.6.conv2'}  # Arguments to be passed to the model, as a dictionary
    optimizer = 'SGD'  # Which optimizer to use, either Adam or SGD
    ntrials = 20  # Number of trials for hyperparameter tuning
    nmaxepochs = 100  # Maximum number of epochs
    result_dir = project_root + '/cnn/distill_cov_results'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing
    dataset = 'imagenet'
    teacher_model = 'mobilenetv1_0.5'
    teacher_model_path = project_root + '/cnn/mobilenetv1_0.5/checkpoint.pth.tar'
    input_cov_path = project_root + '/cnn/mobilenetv1_0.5/input_cov.pt'
    min_lr = 1e-4
    max_lr = 1e-2
    grace_period = 10
    momentum = 0.9
    nsteps = 2000  # Number of steps per epoch
    nepochsvalid = nmaxepochs  # Frequency of validation (polishing), in terms of epochs

@ex.capture
def distillation_experiment(model_args, optimizer, ntrials, result_dir,
                            cuda, smoke_test, teacher_model, teacher_model_path,
                            input_cov_path, dataset, min_lr, max_lr, momentum,
                            nsteps, nepochsvalid):
    assert optimizer in ['Adam', 'SGD'], 'Only Adam and SGD are supported'
    config={
        'optimizer': optimizer,
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(min_lr), math.log(max_lr)) if optimizer == 'Adam'
                                           else random.uniform(math.log(min_lr), math.log(max_lr)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'device': 'cuda' if cuda else 'cpu',
        'model_args': dict(model_args),  # Need to copy @encoder as sacred created a read-only dict
        'teacher_model': teacher_model,
        'teacher_model_path': teacher_model_path,
        'input_cov_path': input_cov_path,
        'dataset': dataset,
        'momentum': momentum,
        'n_steps_per_epoch': nsteps,
        'n_epochs_per_validation': nepochsvalid,
        }
    model_args_print = '_'.join([f'{key}_{value}' for key,value in model_args.items()])
    experiment = RayExperiment(
        name=f'{model_args_print}_{optimizer}',
        run=TrainableDistillCovModel,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        checkpoint_freq=1000,  # Just to enable recovery with @max_failures
        max_failures=-1,
        resources_per_trial={'cpu': 2, 'gpu': 0.5 if cuda else 0},
        stop={"training_iteration": 1 if smoke_test else 9999},
        config=config,
    )
    return experiment


@ex.automain
def run(model_args, result_dir, nmaxepochs, grace_period):
    experiment = distillation_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            ray.init(redis_address=address)
    except:
        ray.init()
    ahb = AsyncHyperBandScheduler(metric='mean_loss', mode='min', grace_period=grace_period,
                                  max_t=nmaxepochs)
                                  #  reduction_factor=2, brackets=3, max_t=nmaxepochs)
    trials = ray.tune.run(experiment, scheduler=ahb, raise_on_failed_trial=False,
                          queue_trials=True, reuse_actors=True).trials
    trials = [trial for trial in trials if trial.last_result is not None]
    loss = [trial.last_result.get('mean_loss', float('inf')) for trial in trials]

    return model_args, min(loss)
