import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from scipy.linalg import hadamard

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
from utils import PytorchTrainable

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


N_LBFGS_STEPS = 300
N_TRIALS_TO_POLISH = 20


def hadamard_test():
    # Hadamard matrix for n = 4
    size = 4
    M0 = Butterfly(size,
                   diagonal=2,
                   diag=torch.tensor([1.0, 1.0, -1.0, -1.0], requires_grad=True),
                   subdiag=torch.ones(2, requires_grad=True),
                   superdiag=torch.ones(2, requires_grad=True))
    M1 = Butterfly(size,
                   diagonal=1,
                   diag=torch.tensor([1.0, -1.0, 1.0, -1.0], requires_grad=True),
                   subdiag=torch.tensor([1.0, 0.0, 1.0], requires_grad=True),
                   superdiag=torch.tensor([1.0, 0.0, 1.0], requires_grad=True))
    H = M0.matrix() @ M1.matrix()
    assert torch.allclose(H, torch.tensor(hadamard(4), dtype=torch.float))
    M = ButterflyProduct(size, fixed_order=True)
    M.butterflies[0] = M0
    M.butterflies[1] = M1
    assert torch.allclose(M.matrix(), H)


class TrainableHadamardFactorFixedOrder(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'], fixed_order=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        # self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['momentum'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.target_matrix = torch.tensor(hadamard(config['size']), dtype=torch.float)

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()
            loss = nn.functional.mse_loss(y, self.target_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableHadamardFactorSoftmax(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'], fixed_order=False)
        self.semantic_loss_weight = config['semantic_loss_weight']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        # detach to set H.requires_grad = False
        self.target_matrix = torch.tensor(hadamard(config['size']), dtype=torch.float).detach()

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()
            loss = nn.functional.mse_loss(y, self.target_matrix)
            semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
            total_loss = loss + self.semantic_loss_weight * semantic_loss.mean()
            total_loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableHadamardFactorSparsemax(TrainableHadamardFactorFixedOrder):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'], softmax_fn='sparsemax')
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        # detach to set H.requires_grad = False
        self.target_matrix = torch.tensor(hadamard(config['size']), dtype=torch.float).detach()


def hadamard_factorization_fixed_order(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Hadamard matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Hadamard_factorization_fixed_order_{args.size}',
        run=TrainableHadamardFactorFixedOrder,
        local_dir=args.result_dir,
        num_samples=args.ntrials,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': args.nthreads, 'gpu': 0},
        stop={
            'training_iteration': 1 if args.smoke_test else 99999,
            'negative_loss': -1e-8
        },
        config={
            'size': args.size,
            'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
            # 'momentum': sample_from(lambda spec: random.uniform(0.0, 0.99)),
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def hadamard_factorization_softmax(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Hadamard matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--fixed-order', action='store_true', help='Whether the order of the butterfly matrices are fixed or learned')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Hadamard_factorization_softmax_{args.size}',
        run=TrainableHadamardFactorSoftmax,
        local_dir=args.result_dir,
        num_samples=args.ntrials,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': args.nthreads, 'gpu': 0},
        stop={
            'training_iteration': 1 if args.smoke_test else 99999,
            'negative_loss': -1e-8
        },
        config={
            'size': args.size,
            'fixed_order': args.fixed_order,
            'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
            # 'momentum': sample_from(lambda spec: random.uniform(0.0, 0.99)),
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'semantic_loss_weight': sample_from(lambda spec: math.exp(random.uniform(math.log(5e-4), math.log(5e-1)))),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def hadamard_factorization_sparsemax(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Hadamard matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Hadamard_factorization_sparsemax_{args.size}',
        run=TrainableHadamardFactorSparsemax,
        local_dir=args.result_dir,
        num_samples=args.ntrials,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': args.nthreads, 'gpu': 0},
        stop={
            'training_iteration': 1 if args.smoke_test else 99999,
            'negative_loss': -1e-8
        },
        config={
            'size': args.size,
            'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
            # 'momentum': sample_from(lambda spec: random.uniform(0.0, 0.99)),
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


# argv = ['--size', '8']  # for dirty testing with ipython

# if __name__ == '__main__':
#     # experiment, args = hadamard_factorization_fixed_order(sys.argv[1:])
#     experiment, args = hadamard_factorization_softmax(sys.argv[1:])
#     # experiment, args = hadamard_factorization_sparsemax(sys.argv[1:])
#     # We'll use multiple processes so disable MKL multithreading
#     os.environ['MKL_NUM_THREADS'] = str(args.nthreads)
#     ray.init()
#     ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=args.nmaxepochs)
#     trials = run_experiments(experiment, scheduler=ahb, raise_on_failed_trial=False)
#     losses = [-trial.last_result['negative_loss'] for trial in trials]
#     print(np.array(losses))
#     print(np.sort(losses))

#     checkpoint_path = Path(args.result_dir) / experiment.name
#     checkpoint_path.mkdir(parents=True, exist_ok=True)
#     checkpoint_path /= 'trial.pkl'
#     with checkpoint_path.open('wb') as f:
#         pickle.dump(trials, f)

class TrainableHadamard(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'],
                                      fixed_order=config['fixed_order'],
                                      softmax_fn=config['softmax_fn'])
        if (not config['fixed_order']) and config['softmax_fn'] == 'softmax':
            self.semantic_loss_weight = config['semantic_loss_weight']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.target_matrix = torch.tensor(hadamard(config['size']), dtype=torch.float)

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()
            loss = nn.functional.mse_loss(y, self.target_matrix)
            if (not self.model.fixed_order) and hasattr(self, 'semantic_loss_weight'):
                semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
                loss += self.semantic_loss_weight * semantic_loss.mean()
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


def polish_hadamard(trial):
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
        loss = nn.functional.mse_loss(polished_model.matrix(), trainable.target_matrix)
        loss.backward()
        return loss
    for i in range(N_LBFGS_STEPS):
        optimizer.step(closure)
    torch.save(polished_model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))
    loss = nn.functional.mse_loss(polished_model.matrix(), trainable.target_matrix)
    return loss.item()


ex = Experiment('Hadamard_factorization')
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
def hadamard_experiment(fixed_order, softmax_fn, size, ntrials, nsteps, result_dir, nthreads, smoke_test):
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
        name=f'Hadamard_factorization_{fixed_order}_{softmax_fn}_{size}',
        run=TrainableHadamard,
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
    experiment = hadamard_experiment()
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb, raise_on_failed_trial=False)
    losses = [-trial.last_result['negative_loss'] for trial in trials]

    # Polish solutions with L-BFGS
    pool = mp.Pool()
    sorted_trials = sorted(trials, key=lambda trial: -trial.last_result['negative_loss'])
    polished_losses = pool.map(polish_hadamard, sorted_trials[:N_TRIALS_TO_POLISH])
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
