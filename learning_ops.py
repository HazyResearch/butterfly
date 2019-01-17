import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from numpy.polynomial import legendre

import torch
from torch import nn
from torch import optim

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler

from hstack_diag import HstackDiagProduct
from utils import PytorchTrainable, bitreversal_permutation


N_LBFGS_STEPS = 300
N_TRIALS_TO_POLISH = 20


class TrainableOps(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = HstackDiagProduct(size=config['size'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        size = config['size']
        # Target: Legendre polynomials
        P = np.zeros((size, size), dtype=np.float64)
        for i, coef in enumerate(np.eye(size)):
            P[i, :i + 1] = legendre.leg2poly(coef)
        self.target_matrix = torch.tensor(P)
        self.br_perm = bitreversal_permutation(size)
        self.input = (torch.eye(size)[:, :, None, None] * torch.eye(2)).unsqueeze(-1)
        self.input_permuted = self.input[:, self.br_perm]

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model(self.input_permuted)
            loss = nn.functional.mse_loss(y.double(), self.target_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


def polish_ops(trial):
    """Load model from checkpoint, and re-optimize using L-BFGS to find
    the nearest local optimum.
    """
    trainable = eval(trial.trainable_name)(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    model = trainable.model
    config = trial.config
    polished_model = HstackDiagProduct(size=config['size'])
    polished_model.factors = model.factors
    polished_model.P_init = model.P_init
    optimizer = optim.LBFGS(polished_model.parameters())
    def closure():
        optimizer.zero_grad()
        eye = torch.eye(polished_model.size)
        x = (eye[:, :, None, None] * torch.eye(2)).unsqueeze(-1)
        y = polished_model(x[:, trainable.br_perm])
        loss = nn.functional.mse_loss(y, trainable.target_matrix)
        loss.backward()
        return loss
    for i in range(N_LBFGS_STEPS):
        optimizer.step(closure)
    torch.save(polished_model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))
    eye = torch.eye(polished_model.size)
    x = (eye[:, :, None, None] * torch.eye(2)).unsqueeze(-1)
    y = polished_model(x[:, trainable.br_perm])
    loss = nn.functional.mse_loss(y, trainable.target_matrix)
    return loss.item()


ex = Experiment('Ops_factorization')
ex.observers.append(FileStorageObserver.create('logs'))
slack_config_path = Path('config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def fixed_order_config():
    size = 8  # Size of matrix to factor, must be power of 2
    ntrials = 20  # Number of trials for hyperparameter tuning
    nsteps = 400  # Number of steps per epoch
    nmaxepochs = 200  # Maximum number of epochs
    result_dir = 'results'  # Directory to store results
    nthreads = 1  # Number of CPU threads per job
    smoke_test = False  # Finish quickly for testing


@ex.capture
def ops_experiment(size, ntrials, nsteps, result_dir, nthreads, smoke_test):
    config={
        'size': size,
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_steps_per_epoch': nsteps,
     }
    experiment = RayExperiment(
        name=f'Ops_factorization_{size}',
        run=TrainableOps,
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
    experiment = ops_experiment()
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    os.environ['OMP_NUM_THREADS'] = str(nthreads)  # For some reason we need this for OPs otherwise it'll thrash
    ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb, raise_on_failed_trial=False)
    losses = [-trial.last_result['negative_loss'] for trial in trials]

    # Polish solutions with L-BFGS
    pool = mp.Pool()
    sorted_trials = sorted(trials, key=lambda trial: -trial.last_result['negative_loss'])
    polished_losses = pool.map(polish_ops, sorted_trials[:N_TRIALS_TO_POLISH])
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

# TODO: there might be a memory leak, trying to find it here
# import gc
# import operator as op
# from functools import reduce

# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             print(reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
#             # print(type(obj), obj.size(), obj.type())
#     except:
#         pass
