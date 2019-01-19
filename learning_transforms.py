import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np

import torch
from torch import nn
from torch import optim

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from
from ray.tune.schedulers import AsyncHyperBandScheduler

from tune import run_experiments

from butterfly import Block2x2DiagProduct, BlockPerm, BlockPermProduct, FixedPermutation
from semantic_loss import semantic_loss_exactly_one
from training import PytorchTrainable, TrainableMatrixFactorization
from utils import bitreversal_permutation
from complex_utils import real_to_complex, complex_matmul
from target_matrix import named_target_matrix


N_LBFGS_STEPS = 300
N_LBFGS_STEPS_VALIDATION = 15
N_TRIALS_TO_POLISH = 16


class TrainableButterfly(TrainableMatrixFactorization):
    """Product of butterfly matrices, with fixed bit-reversal permutation matrix.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float).to(device)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        complex = self.target_matrix.dim() == 3 or config['complex']
        torch.manual_seed(config['seed'])
        self.model = Block2x2DiagProduct(size=size, complex=complex).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.n_epochs_per_validation = config['n_epochs_per_validation']
        self.input = torch.eye(size)[:, torch.tensor(bitreversal_permutation(size))].to(device)
        if complex:
            self.input = real_to_complex(self.input)


class TrainableBP(TrainableMatrixFactorization):
    """Product of butterfly matrices and product of block permutation matrices.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float).to(device)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        complex = self.target_matrix.dim() == 3 or config['complex']
        torch.manual_seed(config['seed'])
        self.model = nn.Sequential(
            BlockPermProduct(size=size, complex=complex, share_logit=config['share_logit_0']),
            Block2x2DiagProduct(size=size, complex=complex)
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.n_epochs_per_validation = config['n_epochs_per_validation']
        self.input = torch.eye(size).to(device)
        if complex:
            self.input = real_to_complex(self.input)

    def freeze(self):
        if not isinstance(self.model[0], FixedPermutation):
            self.model[0] = FixedPermutation(self.model[0].argmax(), complex=self.model[0].complex)


class TrainablePBT(TrainableMatrixFactorization):
    """Product of block permutation matrices and product butterfly matrices, but
    with increasing size (i.e. transpose of the normal butterfly block).
    """
    # Transposing the permutation product won't capture the FFT, since we'll
    # need permutations that interleave the first half and second half (inverse
    # of the permutation that separates the even and the odd). However, using
    # the permutation product with increasing size will work since it can
    # represent bit reversal, which is its own inverse.

    def _setup(self, config):
        device = config['device']
        self.device = device
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float).to(device)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        complex = self.target_matrix.dim() == 3 or config['complex']
        torch.manual_seed(config['seed'])
        self.model = nn.Sequential(
            Block2x2DiagProduct(size=size, complex=complex, decreasing_size=False),
            BlockPermProduct(size=size, complex=complex, share_logit=config['share_logit_0'], increasing_size=True),
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.n_epochs_per_validation = config['n_epochs_per_validation']
        self.input = torch.eye(size).to(device)
        if complex:
            self.input = real_to_complex(self.input)

    def freeze(self):
        if not isinstance(self.model[1], FixedPermutation):
            self.model[1] = FixedPermutation(self.model[1].argmax(), complex=self.model[1].complex)


class TrainableBPP(TrainableMatrixFactorization):
    """Product of butterfly matrices and product of block permutation matrices,
    plus an extra permutation for more flexibility.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float).to(device)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        complex = self.target_matrix.dim() == 3 or config['complex']
        torch.manual_seed(config['seed'])
        self.model = nn.Sequential(
            BlockPerm(size=size, complex=complex),
            BlockPermProduct(size=size, complex=complex, share_logit=config['share_logit_0']),
            Block2x2DiagProduct(size=size, complex=complex)
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.n_epochs_per_validation = config['n_epochs_per_validation']
        self.input = torch.eye(size).to(device)
        if complex:
            self.input = real_to_complex(self.input)

    def freeze(self):
        if not isinstance(self.model[0], FixedPermutation):
            self.model[0] = FixedPermutation(self.model[0].argmax(), complex=self.model[0].complex)
        if not isinstance(self.model[1], FixedPermutation):
            self.model[1] = FixedPermutation(self.model[1].argmax(), complex=self.model[1].complex)


class TrainableBPBP(TrainableMatrixFactorization):
    """Combine two (BP) blocks, where B is product of butterflies and P is
    product of block permutations.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float).to(device)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        complex = self.target_matrix.dim() == 3 or config['complex']
        torch.manual_seed(config['seed'])
        self.model = nn.Sequential(
            BlockPermProduct(size=size, complex=complex, share_logit=config['share_logit_0']),
            Block2x2DiagProduct(size=size, complex=complex),
            BlockPermProduct(size=size, complex=complex, share_logit=config['share_logit_0']),
            Block2x2DiagProduct(size=size, complex=complex),
        ).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.n_epochs_per_validation = config['n_epochs_per_validation']
        self.input = torch.eye(size).to(device)
        if complex:
            self.input = real_to_complex(self.input)

    def freeze(self):
        if not isinstance(self.model[0], FixedPermutation):
            self.model[0] = FixedPermutation(self.model[0].argmax(), complex=self.model[0].complex)
        if not isinstance(self.model[2], FixedPermutation):
            self.model[2] = FixedPermutation(self.model[2].argmax(), complex=self.model[2].complex)


def polish(trial):
    """Load model from checkpoint, then fix the order of the factor
    matrices (using the largest logits), and re-optimize using L-BFGS to find
    the nearest local optima.
    """
    # Polish on CPUs, not GPUs
    device = trial.config['device']
    trial.config['device'] = 'cpu'
    trainable = eval(trial.trainable_name)(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    loss = trainable.polish(N_LBFGS_STEPS, save_to_self_model=True)
    torch.save(trainable.model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))
    trial.config['device'] = device
    return loss


model_name_to_trainable = {
    'B': TrainableButterfly,
    'BP': TrainableBP,
    'PBT': TrainablePBT,
    'BPP': TrainableBPP,
    'BPBP': TrainableBPBP,
}


ex = Experiment('Transform_factorization')
ex.observers.append(FileStorageObserver.create('logs_new'))
slack_config_path = Path('config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def default_config():
    model = 'BP'
    target = 'dft'  # The target matrix to factor ('dft', 'idft', 'dct', 'hadamard')
    size = 8  # Size of matrix to factor, must be power of 2
    complex = True  # Whether to use complex factorization or real factorization
    fixed_order = True  # Whether the order of the factors are fixed
    softmax_fn = 'softmax'  # Whether to use softmax (+ semantic loss) or sparsemax
    ntrials = 20  # Number of trials for hyperparameter tuning
    nsteps = 400  # Number of steps per epoch
    nepochsvalid = 5  # Frequency of validation (polishing), in terms of epochs
    nmaxepochs = 200  # Maximum number of epochs
    result_dir = 'results_new'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    nthreads = 1  # Number of CPU threads per job
    smoke_test = False  # Finish quickly for testing


@ex.capture
def transform_experiment(trainable, target, size, complex, ntrials, nsteps, nepochsvalid, result_dir, cuda, nthreads, smoke_test):
    config={
        'target_matrix': named_target_matrix(target, size),
        'complex': complex,
        'share_logit_0': sample_from(lambda spec: random.choice([True, False])),
        'share_logit_1': sample_from(lambda spec: random.choice([True, False])),
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_steps_per_epoch': nsteps,
        'n_epochs_per_validation': nepochsvalid,
        'device': 'cuda' if cuda else 'cpu',
     }
    experiment = RayExperiment(
        name=f'{target}_factorization_{trainable.__name__}_{complex}_{size}',
        run=trainable,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': nthreads, 'gpu': 0.25 if cuda else 0},
        stop={
            'training_iteration': 1 if smoke_test else 99999,
            'negative_loss': -1e-8
        },
        config=config,
    )
    return experiment


@ex.automain
def run(model, result_dir, nmaxepochs, nthreads):
    trainable = model_name_to_trainable[model]
    experiment = transform_experiment(trainable)
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb, raise_on_failed_trial=False, early_stop_all_trials=True)
    trials = [trial for trial in trials if trial.last_result is not None]
    losses = [-trial.last_result.get('negative_loss', float('inf')) for trial in trials]

    # Polish solutions with L-BFGS
    pool = mp.Pool()
    sorted_trials = sorted(trials, key=lambda trial: -trial.last_result['negative_loss'])
    polished_losses = pool.map(polish, sorted_trials[:N_TRIALS_TO_POLISH])
    # polished_losses = [-trial.last_result['polished_negative_loss'] for trial in sorted_trials[:N_TRIALS_TO_POLISH]]
    pool.close()
    pool.join()
    for i in range(min(N_TRIALS_TO_POLISH, len(trials))):
        sorted_trials[i].last_result['polished_negative_loss'] = -polished_losses[i]
    print(np.array(losses))
    print(np.sort(losses))
    # print(np.sort(losses)[:N_TRIALS_TO_POLISH])
    print(np.sort(polished_losses))

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return min(losses + polished_losses)
