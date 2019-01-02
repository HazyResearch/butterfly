import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from scipy.fftpack import dct

import torch
from torch import nn
from torch import optim

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler

from butterfly import Butterfly, ButterflyProduct
from semantic_loss import semantic_loss_exactly_one
from utils import PytorchTrainable, bitreversal_permutation
from complex_utils import complex_mul, complex_matmul


N_LBFGS_STEPS = 300
N_TRIALS_TO_POLISH = 20


def dct_test():
    # DCT matrix for n = 4
    size = 4
    # Need to transpose as dct acts on rows of matrix np.eye, not columns
    DCT = torch.tensor(dct(np.eye(size)).T, dtype=torch.float)
    M0diag=torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    M0subdiag=torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    M0superdiag=torch.tensor([[1.0, 0.0], [0.0, -1.0]], requires_grad=True)
    M0 = Butterfly(size, diagonal=2, complex=True, diag=M0diag, subdiag=M0subdiag, superdiag=M0superdiag)
    M1 = Butterfly(size,
                   diagonal=1,
                   complex=True,
                   diag=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True),
                   subdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True),
                   superdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True))
    arange_ = np.arange(size)
    dct_perm = np.concatenate((arange_[::2], arange_[::-2]))
    br_perm = bitreversal_permutation(size)
    perm = torch.arange(size)[dct_perm][br_perm]
    arange_ = torch.arange(size, dtype=torch.float)
    diag_real = 2 * torch.cos(-math.pi * arange_ / (2 * size))
    diag_imag = 2 * torch.sin(-math.pi * arange_ / (2 * size))
    diag = torch.stack((torch.diag(diag_real), torch.diag(diag_imag)), dim=-1)
    assert torch.allclose(complex_matmul(diag, complex_matmul(M0.matrix(), M1.matrix()))[:, perm, 0], DCT)
    D = torch.stack((diag_real, diag_imag), dim=-1)
    DM0 = Butterfly(size,
                    diagonal=2,
                    complex=True,
                    diag=complex_mul(D, M0diag),
                    subdiag=complex_mul(D[2:], M0subdiag),
                    superdiag=complex_mul(D[:2], M0superdiag))
    assert torch.allclose(complex_matmul(DM0.matrix(), M1.matrix())[:, perm, 0], DCT)


class TrainableDct(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'],
                                      complex=True,
                                      fixed_order=config['fixed_order'],
                                      softmax_fn=config['softmax_fn'])
        if (not config['fixed_order']) and config['softmax_fn'] == 'softmax':
            self.semantic_loss_weight = config['semantic_loss_weight']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        size = config['size']
        # Need to transpose as dct acts on rows of matrix np.eye, not columns
        self.target_matrix = torch.tensor(dct(np.eye(size)).T, dtype=torch.float)
        arange_ = np.arange(size)
        dct_perm = np.concatenate((arange_[::2], arange_[::-2]))
        br_perm = bitreversal_permutation(size)
        self.perm = torch.arange(size)[dct_perm][br_perm]

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()[:, self.perm, 0]
            loss = nn.functional.mse_loss(y, self.target_matrix)
            if (not self.model.fixed_order) and hasattr(self, 'semantic_loss_weight'):
                semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
                loss += self.semantic_loss_weight * semantic_loss.mean()
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableDctTempAnnealing(TrainableDct):

    def _train(self):
        temperature = 1.0 / (0.1 * self._iteration + 1)
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix(temperature)[:, self.perm, 0]
            loss = nn.functional.mse_loss(y, self.target_matrix)
            if (not self.model.fixed_order) and hasattr(self, 'semantic_loss_weight'):
                semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
                loss += self.semantic_loss_weight * semantic_loss.mean()
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


def polish_dct(trial):
    """Load model from checkpoint, then fix the order of the butterflies
    matrices (using the largest logits), and re-optimize using L-BFGS to find
    the nearest local optima.
    """
    trainable = eval(trial.trainable_name)(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    model = trainable.model
    config = trial.config
    polished_model = ButterflyProduct(size=config['size'], complex=model.complex, fixed_order=True)
    if not model.fixed_order:
        prob = model.softmax_fn(model.logit)
        maxes, argmaxes = torch.max(prob, dim=-1)
        polished_model.butterflies = nn.ModuleList([model.butterflies[argmax] for argmax in argmaxes])
    else:
        polished_model.butterflies = model.butterflies
    optimizer = optim.LBFGS(polished_model.parameters())
    def closure():
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.perm, 0], trainable.target_matrix)
        loss.backward()
        return loss
    for i in range(N_LBFGS_STEPS):
        optimizer.step(closure)
    torch.save(polished_model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))
    loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.perm, 0], trainable.target_matrix)
    return loss.item()


ex = Experiment('Dct_factorization')
ex.observers.append(FileStorageObserver.create('logs'))
slack_config_path = Path('config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.named_config
def softmax_config():
    fixed_order = False  # Whether the order of the factors are fixed
    softmax_fn = 'softmax'  # Whether to use softmax (+ semantic loss) or sparsemax


@ex.named_config
def sparsemax_config():
    fixed_order = False  # Whether the order of the factors are fixed
    softmax_fn = 'sparsemax'  # Whether to use softmax (+ semantic loss) or sparsemax


@ex.config
def fixed_order_config():
    fixed_order = True  # Whether the order of the factors are fixed
    softmax_fn = 'softmax'  # Whether to use softmax (+ semantic loss) or sparsemax
    size = 8  # Size of matrix to factor, must be power of 2
    ntrials = 20  # Number of trials for hyperparameter tuning
    nsteps = 400  # Number of steps per epoch
    nmaxepochs = 200  # Maximum number of epochs
    result_dir = 'results'  # Directory to store results
    nthreads = 1  # Number of CPU threads per job
    smoke_test = False  # Finish quickly for testing


@ex.capture
def dct_experiment(fixed_order, softmax_fn, size, ntrials, nsteps, result_dir, nthreads, smoke_test):
    assert softmax_fn in ['softmax', 'sparsemax']
    config={
        'fixed_order': fixed_order,
        'softmax_fn': softmax_fn,
        'size': size,
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_steps_per_epoch': nsteps,
     }
    if (not fixed_order) and softmax_fn == 'softmax':
        config['semantic_loss_weight'] = sample_from(lambda spec: math.exp(random.uniform(math.log(5e-3), math.log(5e-1))))
    experiment = RayExperiment(
        name=f'Dct_factorization_{fixed_order}_{softmax_fn}_{size}',
        run=TrainableDct,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': nthreads, 'gpu': 0},
        stop={
            'training_iteration': 1 if smoke_test else 99999,
            'negative_loss': -1e-8
        },
        config=config,
    )
    return experiment


@ex.automain
def run(result_dir, nmaxepochs, nthreads):
    experiment = dct_experiment()
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb, raise_on_failed_trial=False)
    losses = [-trial.last_result['negative_loss'] for trial in trials]

    # Polish solutions with L-BFGS
    pool = mp.Pool()
    sorted_trials = sorted(trials, key=lambda trial: -trial.last_result['negative_loss'])
    polished_losses = pool.map(polish_dct, sorted_trials[:N_TRIALS_TO_POLISH])
    pool.close()
    pool.join()
    for i in range(N_TRIALS_TO_POLISH):
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
