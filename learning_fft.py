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

from butterfly import Butterfly, ButterflyProduct, sinkhorn, Block2x2DiagProduct, BlockPermProduct, FixedPermutation
from semantic_loss import semantic_loss_exactly_one
from utils import PytorchTrainable, bitreversal_permutation, TrainableMatrixFactorization
from complex_utils import real_to_complex, complex_matmul
from target_matrix import named_target_matrix


N_LBFGS_STEPS = 300
N_LBFGS_STEPS_VALIDATION = 15
N_TRIALS_TO_POLISH = 60


class TrainableFftFactorFixedOrder(PytorchTrainable):

    def _setup(self, config):
        size = config['size']
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=size, complex=True, fixed_order=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.target_matrix = torch.fft(real_to_complex(torch.eye(size)), 1)
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
        self.target_matrix = torch.fft(real_to_complex(torch.eye(size)), 1)
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
        self.target_matrix = torch.fft(real_to_complex(torch.eye(size)), 1)
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
        self.target_matrix = torch.fft(real_to_complex(torch.eye(size)), 1)
        self.br_perm = torch.tensor(bitreversal_permutation(size))
        # br_perm = bitreversal_permutation(size)
        # br_reverse = torch.tensor(list(br_perm[::-1]))
        # br_reverse = torch.cat((torch.tensor(list(br_perm[:size//2][::-1])), torch.tensor(list(br_perm[size//2:][::-1]))))
        # Same as [6, 2, 4, 0, 7, 3, 5, 1], which is [0, 1]^4 * [0, 2, 1, 3]^2 * [6, 4, 2, 0, 7, 5, 3, 1]
        # br_reverse = torch.cat((torch.tensor(list(br_perm[:size//4][::-1])), torch.tensor(list(br_perm[size//4:size//2][::-1])), torch.tensor(list(br_perm[size//2:3*size//4][::-1])), torch.tensor(list(br_perm[3*size//4:][::-1]))))
        # self.br_perm = br_reverse
        # self.br_perm = torch.tensor([0, 7, 4, 3, 2, 5, 6, 1])  # Doesn't work
        # self.br_perm = torch.tensor([7, 3, 0, 4, 2, 6, 5, 1])  # Doesn't work
        # self.br_perm = torch.tensor([4, 0, 6, 2, 5, 1, 7, 3])  # This works, [0, 1]^4 * [2, 0, 3, 1]^2 * [0, 2, 4, 6, 1, 3, 5, 7] or [1, 0]^4 * [0, 2, 1, 3]^2 * [0, 2, 4, 6, 1, 3, 5, 7]
        # self.br_perm = torch.tensor([4, 0, 2, 6, 5, 1, 3, 7])  # Doesn't work, [0, 1]^4 * [2, 0, 1, 3]^2 * [0, 2, 4, 6, 1, 3, 5, 7]
        # self.br_perm = torch.tensor([1, 5, 3, 7, 0, 4, 2, 6])  # This works, [0, 1]^4 * [4, 6, 5, 7, 0, 4, 2, 6]
        # self.br_perm = torch.tensor([4, 0, 6, 2, 5, 1, 3, 7])  # Doesn't work
        # self.br_perm = torch.tensor([4, 0, 6, 2, 1, 5, 3, 7])  # Doesn't work
        # self.br_perm = torch.tensor([0, 4, 6, 2, 1, 5, 7, 3])  # Doesn't work
        # self.br_perm = torch.tensor([4, 1, 6, 2, 5, 0, 7, 3])  # This works, since it's just swapping 0 and 1
        # self.br_perm = torch.tensor([5, 1, 6, 2, 4, 0, 7, 3])  # This works, since it's swapping 4 and 5

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


class TrainableFftBlock2x2(TrainableMatrixFactorization):

    def _setup(self, config):
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        torch.manual_seed(config['seed'])
        self.model = Block2x2DiagProduct(size=size, complex=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.input = real_to_complex(torch.eye(size)[:, torch.tensor(bitreversal_permutation(size))])


class TrainableFftBlockPerm(TrainableMatrixFactorization):

    def _setup(self, config):
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        torch.manual_seed(config['seed'])
        self.model = nn.Sequential(
            BlockPermProduct(size=size, complex=True, share_logit=False),
            Block2x2DiagProduct(size=size, complex=True)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.input = real_to_complex(torch.eye(size))

    def freeze(self):
        self.model[0] = FixedPermutation(self.model[0].argmax(), complex=self.model[0].complex)


class TrainableFftBlockPermTranspose(TrainableMatrixFactorization):

    def _setup(self, config):
        self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        size = self.target_matrix.shape[0]
        torch.manual_seed(config['seed'])
        # Transposing the permutation product won't capture the FFT, since we'll
        # permutations that interleave the first half and second half (inverse
        # of the permutation that separates the even and the odd).
        # However, using the permutation product with increasing size will work
        # since it can represent bit reversal, which is its own inverse.
        self.model = nn.Sequential(
            Block2x2DiagProduct(size=size, complex=True, decreasing_size=False),
            BlockPermProduct(size=size, complex=True, share_logit=False, increasing_size=True),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.input = real_to_complex(torch.eye(size))

    def freeze(self):
        self.model[1] = FixedPermutation(self.model[1].argmax(), complex=self.model[1].complex)


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


class TrainableFftLearnPerm(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'],
                                      complex=True,
                                      fixed_order=config['fixed_order'],
                                      softmax_fn=config['softmax_fn'],
                                      learn_perm=True)
        if (not config['fixed_order']) and config['softmax_fn'] == 'softmax':
            self.semantic_loss_weight = config['semantic_loss_weight']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        size = config['size']
        self.target_matrix = torch.fft(real_to_complex(torch.eye(size)), 1)

    def _train(self):
        temperature = 1.0 / (0.3 * self._iteration + 1)
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            y = self.model.matrix(temperature)
            loss = nn.functional.mse_loss(y, self.target_matrix)
            if (not self.model.fixed_order) and hasattr(self, 'semantic_loss_weight'):
                semantic_loss = semantic_loss_exactly_one(nn.functional.log_softmax(self.model.logit, dim=-1))
                loss += self.semantic_loss_weight * semantic_loss.mean()
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -polished_loss_fft_learn_perm(self)}


def polish(trial):
    """Load model from checkpoint, then fix the order of the factor
    matrices (using the largest logits), and re-optimize using L-BFGS to find
    the nearest local optima.
    """
    trainable = eval(trial.trainable_name)(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    loss = trainable.polish(N_LBFGS_STEPS)
    torch.save(trainable.model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))
    return loss


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


def polish_fft_learn_perm(trial):
    """Load model from checkpoint, then fix the order of the factor
    matrices (using the largest logits), and re-optimize using L-BFGS to find
    the nearest local optima.
    """
    trainable = eval(trial.trainable_name)(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    model = trainable.model
    config = trial.config
    polished_model = ButterflyProduct(size=config['size'], complex=model.complex, fixed_order=True)
    temperature = 1.0 / (0.3 * trainable._iteration + 1)
    trainable.perm = torch.argmax(sinkhorn(model.perm_logit / temperature), dim=1)
    if not model.fixed_order:
        prob = model.softmax_fn(model.logit)
        maxes, argmaxes = torch.max(prob, dim=-1)
        polished_model.factors = nn.ModuleList([model.factors[argmax] for argmax in argmaxes])
    else:
        polished_model.factors = model.factors
    optimizer = optim.LBFGS(polished_model.parameters())
    def closure():
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.perm], trainable.target_matrix)
        loss.backward()
        return loss
    for i in range(N_LBFGS_STEPS):
        optimizer.step(closure)
    torch.save(polished_model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))
    loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.perm], trainable.target_matrix)
    return loss.item()


def polished_loss_fft_learn_perm(trainable):
    model = trainable.model
    polished_model = ButterflyProduct(size=model.size, complex=model.complex, fixed_order=True)
    temperature = 1.0 / (0.3 * trainable._iteration + 1)
    trainable.perm = torch.argmax(sinkhorn(model.perm_logit / temperature), dim=1)
    if not model.fixed_order:
        prob = model.softmax_fn(model.logit)
        maxes, argmaxes = torch.max(prob, dim=-1)
        polished_model.factors = nn.ModuleList([model.factors[argmax] for argmax in argmaxes])
    else:
        polished_model.factors = model.factors
    preopt_loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.perm], trainable.target_matrix)
    optimizer = optim.LBFGS(polished_model.parameters())
    def closure():
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.perm], trainable.target_matrix)
        loss.backward()
        return loss
    for i in range(N_LBFGS_STEPS_VALIDATION):
        optimizer.step(closure)
    loss = nn.functional.mse_loss(polished_model.matrix()[:, trainable.perm], trainable.target_matrix)
    # return loss.item() if not torch.isnan(loss) else preopt_loss.item() if not torch.isnan(preopt_loss) else float('inf')
    return loss.item() if not torch.isnan(loss) else preopt_loss.item() if not torch.isnan(preopt_loss) else 9999.0


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


@ex.capture
def fft_experiment_learn_perm(fixed_order, softmax_fn, size, ntrials, nsteps, result_dir, nthreads, smoke_test):
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
        name=f'Fft_factorization_Learnperm_{fixed_order}_{softmax_fn}_{size}',
        run=TrainableFftLearnPerm,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': nthreads, 'gpu': 0},
        stop={
            'training_iteration': 1 if smoke_test else 99999,
            # 'negative_loss': -1e-8
        },
        config=config,
    )
    return experiment


@ex.capture
def fft_experiment_block2x2(size, ntrials, nsteps, result_dir, nthreads, smoke_test):
    config={
        'target_matrix': named_target_matrix('dft', size),
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_steps_per_epoch': nsteps,
     }
    experiment = RayExperiment(
        name=f'Fft_factorization_block_{size}',
        run=TrainableFftBlock2x2,
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
def fft_experiment_blockperm(size, ntrials, nsteps, result_dir, nthreads, smoke_test):
    config={
        'target_matrix': named_target_matrix('dft', size),
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_steps_per_epoch': nsteps,
     }
    experiment = RayExperiment(
        name=f'Fft_factorization_block_perm_{size}',
        run=TrainableFftBlockPerm,
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
def fft_experiment_blockperm_transpose(size, ntrials, nsteps, result_dir, nthreads, smoke_test):
    config={
        'target_matrix': named_target_matrix('dft', size),
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_steps_per_epoch': nsteps,
     }
    experiment = RayExperiment(
        name=f'Fft_factorization_block_perm_transpose_{size}',
        run=TrainableFftBlockPermTranspose,
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
    # experiment = fft_experiment()
    # experiment = fft_experiment_temp_annealing()
    # experiment = fft_experiment_learn_perm()
    # experiment = fft_experiment_block2x2()
    experiment = fft_experiment_blockperm()
    # experiment = fft_experiment_blockperm_transpose()
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb, raise_on_failed_trial=False)
    losses = [-trial.last_result['negative_loss'] for trial in trials]

    # Polish solutions with L-BFGS
    pool = mp.Pool()
    sorted_trials = sorted(trials, key=lambda trial: -trial.last_result['negative_loss'])
    # polished_losses = pool.map(polish_fft, sorted_trials[:N_TRIALS_TO_POLISH])
    # polished_losses = pool.map(polish_fft_learn_perm, sorted_trials[:N_TRIALS_TO_POLISH])
    # polished_losses = pool.map(polish_fft_block2x2, sorted_trials[:N_TRIALS_TO_POLISH])
    # polished_losses = pool.map(polish_fft_blockperm, sorted_trials[:N_TRIALS_TO_POLISH])
    # polished_losses = pool.map(polish_fft_blockperm_transpose, sorted_trials[:N_TRIALS_TO_POLISH])
    polished_losses = pool.map(polish, sorted_trials[:N_TRIALS_TO_POLISH])
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

    polished_losses = [-trial.last_result['polished_negative_loss'] for trial in sorted_trials[:N_TRIALS_TO_POLISH]]
