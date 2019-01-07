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
from ray.tune import Trainable, Experiment as RayExperiment, sample_from, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler

from butterfly import Butterfly, ButterflyProduct
from semantic_loss import semantic_loss_exactly_one
from utils import PytorchTrainable, bitreversal_permutation
from complex_utils import complex_matmul


N_LBFGS_STEPS = 300
N_TRIALS_TO_POLISH = 20


def fft_test():
    # DFT matrix for n = 4
    size = 4
    DFT = torch.fft(torch.stack((torch.eye(size), torch.zeros((size, size))), dim=-1), 1)
    P = torch.stack((torch.tensor([[1., 0., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 1., 0., 0.],
                                   [0., 0., 0., 1.]]),
                     torch.zeros((size, size))), dim=-1)
    M0 = Butterfly(size,
                   diagonal=2,
                   complex=True,
                   diag=torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                   subdiag=torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True),
                   superdiag=torch.tensor([[1.0, 0.0], [0.0, -1.0]], requires_grad=True))
    M1 = Butterfly(size,
                   diagonal=1,
                   complex=True,
                   diag=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True),
                   subdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True),
                   superdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True))
    assert torch.allclose(complex_matmul(M0.matrix(), complex_matmul(M1.matrix(), P)), DFT)
    br_perm = torch.tensor(bitreversal_permutation(size))
    assert torch.allclose(complex_matmul(M0.matrix(), M1.matrix())[:, br_perm], DFT)
    D = complex_matmul(DFT, P.transpose(0, 1))
    assert torch.allclose(complex_matmul(M0.matrix(), M1.matrix()), D)


class TrainableFftFactorFixedOrder(PytorchTrainable):

    def _setup(self, config):
        size = config['size']
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=size, complex=True, fixed_order=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.target_matrix = torch.fft(torch.stack((torch.eye(size), torch.zeros((size, size))), dim=-1), 1)
        self.br_perm = torch.tensor(bitreversal_permutation(size))

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()[:, self.br_perm]
            loss = nn.functional.mse_loss(y, self.target_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableFftFactorSoftmax(PytorchTrainable):

    def _setup(self, config):
        size = config['size']
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=size, complex=True, fixed_order=False)
        self.semantic_loss_weight = config['semantic_loss_weight']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.target_matrix = torch.fft(torch.stack((torch.eye(size), torch.zeros((size, size))), dim=-1), 1)
        self.br_perm = torch.tensor(bitreversal_permutation(size))

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()[:, self.br_perm]
            loss = nn.functional.mse_loss(y, self.target_matrix)
            semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
            total_loss = loss + self.semantic_loss_weight * semantic_loss.mean()
            total_loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableFftFactorSparsemax(TrainableFftFactorFixedOrder):

    def _setup(self, config):
        size = config['size']
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=size, complex=True, fixed_order=False, softmax_fn='sparsemax')
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.target_matrix = torch.fft(torch.stack((torch.eye(size), torch.zeros((size, size))), dim=-1), 1)
        self.br_perm = torch.tensor(bitreversal_permutation(size))


class TrainableFftFactorSparsemaxNoPerm(TrainableFftFactorSparsemax):

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()
            loss = nn.functional.mse_loss(y, self.target_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableFftFactorSoftmaxNoPerm(TrainableFftFactorSoftmax):

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()
            loss = nn.functional.mse_loss(y, self.target_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableRandnFactorSoftmaxNoPerm(PytorchTrainable):

    def _setup(self, config):
        size = config['size']
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=size, complex=False, fixed_order=False, softmax_fn='softmax')
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.target_matrix = torch.rand(size, size, requires_grad=False)

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()
            loss = nn.functional.mse_loss(y, self.target_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableFftFactorSparsemaxPermFront(TrainableFftFactorSparsemax):

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()[self.br_perm, :]
            loss = nn.functional.mse_loss(y, self.target_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}



def fft_factorization_fixed_order(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Fft matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Fft_factorization_fixed_order_{args.size}',
        run=TrainableFftFactorFixedOrder,
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
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def fft_factorization_softmax(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Fft matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Fft_factorization_softmax_{args.size}',
        run=TrainableFftFactorSoftmax,
        local_dir=args.result_dir,
        num_samples=args.ntrials,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': args.nthreads, 'gpu': 0},
        stop={
            'training_iteration': 1 if args.smoke_test else 99999,
            'is_nan': True,
            'negative_loss': -1e-8
        },
        config={
            'size': args.size,
            'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'semantic_loss_weight': sample_from(lambda spec: math.exp(random.uniform(math.log(5e-4), math.log(5e-1)))),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def fft_factorization_sparsemax(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Fft matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Fft_factorization_sparsemax_{args.size}',
        run=TrainableFftFactorSparsemax,
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
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def fft_factorization_sparsemax_no_perm(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Fft matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Fft_factorization_sparsemax_no_perm_{args.size}',
        run=TrainableFftFactorSparsemaxNoPerm,
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
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def fft_factorization_softmax_no_perm(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Fft matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Fft_factorization_softmax_no_perm_{args.size}',
        run=TrainableFftFactorSoftmaxNoPerm,
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
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def randn_factorization_softmax_no_perm(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Fft matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Randn_factorization_softmax_no_perm_{args.size}',
        run=TrainableRandnFactorSoftmaxNoPerm,
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
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


def fft_factorization_sparsemax_perm_front(argv):
    parser = argparse.ArgumentParser(description='Learn to factor Fft matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of CPU threads per job')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args(argv)
    experiment = RayExperiment(
        name=f'Fft_factorization_sparsemax_perm_front_{args.size}',
        run=TrainableFftFactorSparsemaxPermFront,
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
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
            'n_steps_per_epoch': args.nsteps,
        },
    )
    return experiment, args


# if __name__ == '__main__':
#     # experiment, args = fft_factorization_fixed_order(sys.argv[1:])
#     experiment, args = fft_factorization_softmax(sys.argv[1:])
#     # experiment, args = fft_factorization_sparsemax(sys.argv[1:])
#     # experiment, args = fft_factorization_sparsemax_no_perm(sys.argv[1:])
#     # experiment, args = fft_factorization_softmax_no_perm(sys.argv[1:])
#     # experiment, args = randn_factorization_softmax_no_perm(sys.argv[1:])
#     # experiment, args = fft_factorization_sparsemax_perm_front(sys.argv[1:])
#     # We'll use multiple processes so disable MKL multithreading
#     os.environ['MKL_NUM_THREADS'] = str(args.nthreads)
#     ray.init()
#     ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=args.nmaxepochs)
#     trials = run_experiments(experiment, scheduler=ahb)
#     losses = [-trial.last_result['negative_loss'] for trial in trials]
#     print(np.array(losses))
#     print(np.sort(losses))

#     checkpoint_path = Path(args.result_dir) / experiment.name
#     checkpoint_path.mkdir(parents=True, exist_ok=True)
#     checkpoint_path /= 'trial.pkl'
#     with checkpoint_path.open('wb') as f:
#         pickle.dump(trials, f)


class TrainableFft(PytorchTrainable):

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
        self.target_matrix = torch.fft(torch.stack((torch.eye(size), torch.zeros((size, size))), dim=-1), 1)
        self.br_perm = torch.tensor(bitreversal_permutation(size))

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix()[:, self.br_perm]
            loss = nn.functional.mse_loss(y, self.target_matrix)
            if (not self.model.fixed_order) and hasattr(self, 'semantic_loss_weight'):
                semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
                loss += self.semantic_loss_weight * semantic_loss.mean()
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableFftTempAnnealing(TrainableFft):

    def _train(self):
        temperature = 1.0 / (0.1 * self._iteration + 1)
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix(temperature)[:, self.br_perm]
            loss = nn.functional.mse_loss(y, self.target_matrix)
            if (not self.model.fixed_order) and hasattr(self, 'semantic_loss_weight'):
                semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
                loss += self.semantic_loss_weight * semantic_loss.mean()
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


def polish_fft(trial):
    """Load model from checkpoint, then fix the order of the factor
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
        polished_model.factors = nn.ModuleList([model.factors[argmax] for argmax in argmaxes])
    else:
        polished_model.factors = model.factors
    optimizer = optim.LBFGS(polished_model.parameters())
    def closure():
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.br_perm], trainable.target_matrix)
        loss.backward()
        return loss
    for i in range(N_LBFGS_STEPS):
        optimizer.step(closure)
    torch.save(polished_model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))
    loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.br_perm], trainable.target_matrix)
    return loss.item()


ex = Experiment('Fft_factorization')
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
def fft_experiment(fixed_order, softmax_fn, size, ntrials, nsteps, result_dir, nthreads, smoke_test):
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
        name=f'Fft_factorization_{fixed_order}_{softmax_fn}_{size}',
        run=TrainableFft,
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


@ex.capture
def fft_experiment_temp_annealing(fixed_order, softmax_fn, size, ntrials, nsteps, result_dir, nthreads, smoke_test):
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
        name=f'Fft_factorization_Temp_{fixed_order}_{softmax_fn}_{size}',
        run=TrainableFftTempAnnealing,
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
    experiment = fft_experiment()
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb, raise_on_failed_trial=False)
    losses = [-trial.last_result['negative_loss'] for trial in trials]

    # Polish solutions with L-BFGS
    pool = mp.Pool()
    sorted_trials = sorted(trials, key=lambda trial: -trial.last_result['negative_loss'])
    polished_losses = pool.map(polish_fft, sorted_trials[:N_TRIALS_TO_POLISH])
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
