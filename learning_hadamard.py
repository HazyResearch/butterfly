import argparse
import math
import os
import pickle
import random

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

# size = 64
# H = torch.tensor(hadamard(size), dtype=torch.float)
# H = H.detach()  # to set H.requires_grad = False


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


class TrainHadamardFactor(PytorchTrainable):

    def _setup(self, config):
        torch.manual_seed(config['seed'])
        self.model = ButterflyProduct(size=config['size'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        # detach to set H.requires_grad = False
        self.hadamard_matrix = torch.tensor(hadamard(config['size']), dtype=torch.float).detach()

    def _train(self):
        for _ in range(N_STEPS_PER_EPOCH):
            self.optimizer.zero_grad()
            y = self.model.matrix()
            loss = nn.functional.mse_loss(y, self.hadamard_matrix)
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


# args_strings = ['--size', '8']  # for dirty testing with ipython

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn to factor Hadamard matrix')
    parser.add_argument('--size', type=int, default=8, help='Size of matrix to factor, must be power of 2')
    parser.add_argument('--ntrials', type=int, default=20, help='Number of trials for hyperparameter tuning')
    parser.add_argument('--nsteps', type=int, default=200, help='Number of steps per epoch')
    parser.add_argument('--nmaxepochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--result-dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--smoke-test', action='store_true', help='Finish quickly for testing')
    args = parser.parse_args()
    N_STEPS_PER_EPOCH = args.nsteps

    ray.init()

    experiment = Experiment(
        name=f'Hadamard_factorization_{args.size}',
        run=TrainHadamardFactor,
        local_dir='./results',
        num_samples=args.ntrials,
        checkpoint_at_end=True,
        stop={
            'training_iteration': 1 if args.smoke_test else 99999,
            'negative_loss': -1e-8
        },
        config={
            'size': args.size,
            'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
            'seed': sample_from(lambda spec: random.randint(0, 1 << 16))
        },
    )
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=args.nmaxepochs)
    trials = run_experiments(experiment, scheduler=ahb)
    losses = [-trial.last_result['negative_loss'] for trial in trials]
    print(np.array(losses))

    with open('trials.pkl', 'wb') as f:
        pickle.dump(trials, f)

    # with open('trials.pkl', 'rb') as f:
    #     trials = pickle.load(f)

    # best_trial = max(trials, key=lambda trial: trial.last_result['negative_loss'])
    # train_model = TrainHadamardFactor(best_trial.config)
    # train_model.restore(best_trial.logdir + '/' + best_trial._checkpoint.value)
    # model = train_model.model

    # train_model.optimizer.lr
    # for i in range(200):
    #     train_model.train()
