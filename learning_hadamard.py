import argparse
import math
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

import ray
from ray.tune import Trainable, Experiment, sample_from, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler

from butterfly import Butterfly, ButterflyProduct
from semantic_loss import semantic_loss_exactly_one
from utils import PytorchTrainable

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# semantic_loss_weight = 0.05
# model = ButterflyProduct(size, fixed_order=True)
# optimizer = optim.Adam(model.parameters(), lr=0.03)
# for i in range(15000):
#     optimizer.zero_grad()
#     # x = torch.randn(64, size)
#     # y = model(x)
#     # loss = nn.functional.mse_loss(y, x @ H.t())
#     y = model.matrix()
#     # semantic_loss = semantic_loss_exactly_one(nn.functional.softmax(model.logit, dim=-1), dim=-1)
#     # loss = nn.functional.mse_loss(y, H) + semantic_loss_weight * semantic_loss.mean()
#     loss = nn.functional.mse_loss(y, H)
#     loss.backward()
#     optimizer.step()
#     if i % 500 == 0:
#         # y = model.matrix()
#         # loss = nn.functional.mse_loss(y, H)
#         print(f'Loss: {loss.item()}')


class TrainableHadamardFactorFixedOrder(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'], fixed_order=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        # self.optimizer = optim.SGD(self.model.parameters(), lr=config['lr'], momentum=config['momentum'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        # detach to set H.requires_grad = False
        self.target_matrix = torch.tensor(hadamard(config['size']), dtype=torch.float).detach()

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
            semantic_loss = semantic_loss_exactly_one(nn.functional.softmax(self.model.logit, dim=-1), dim=-1)
            total_loss = loss + self.semantic_loss_weight * semantic_loss.mean()
            total_loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableHadamardFactorSparsemax(TrainableHadamardFactorFixedOrder):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'], softmax='sparsemax')
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
    experiment = Experiment(
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
    experiment = Experiment(
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
    experiment = Experiment(
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

if __name__ == '__main__':
    # experiment, args = hadamard_factorization_fixed_order(sys.argv[1:])
    # experiment, args = hadamard_factorization_softmax(sys.argv[1:])
    experiment, args = hadamard_factorization_sparsemax(sys.argv[1:])
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(args.nthreads)
    ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=args.nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb)
    losses = [-trial.last_result['negative_loss'] for trial in trials]
    print(np.array(losses))
    print(np.sort(losses))

    checkpoint_path = Path(args.result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    # with checkpoint_path.open('rb') as f:
    #     trials = pickle.load(f)

    # best_trial = max(trials, key=lambda trial: trial.last_result['negative_loss'])
    # train_model = best_trial._get_trainable_cls()(best_trial.config)
    # train_model.restore(str(Path(best_trial.logdir) / best_trial._checkpoint.value))
    # model = train_model.model

    # train_model.optimizer.lr
    # for i in range(200):
    #     train_model.train()
