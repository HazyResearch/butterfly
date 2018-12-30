import os
import pickle
from pathlib import Path
import numpy as np

import multiprocessing as mp

import torch
from torch import nn
from torch import optim

import ray

from butterfly import ButterflyProduct
from learning_hadamard import TrainableHadamardFactorFixedOrder, TrainableHadamardFactorSoftmax, TrainableHadamardFactorSparsemax
from learning_fft import TrainableFftFactorFixedOrder, TrainableFftFactorSoftmax, TrainableFftFactorSparsemax


N_LBFGS_STEPS = 300
N_TRIALS_TO_POLISH = 20

# We'll use multiple processes so disable MKL multithreading
os.environ['MKL_NUM_THREADS'] = '1'

# @ray.remote
def polish_hadamard(trial):
    trainable = eval(trial.trainable_name)(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    model = trainable.model
    config = trial.config
    polished_model = ButterflyProduct(size=config['size'], complex=model.complex, fixed_order=True)
    if not model.fixed_order:
        prob = model.softmax_fn(model.logit)
        maxes, argmaxes = torch.max(prob, dim=-1)
        # print(maxes)
        # if torch.all(maxes >= 0.99):
        polished_model.butterflies = nn.ModuleList([model.butterflies[argmax] for argmax in argmaxes])
        # else:
        #     return -trial.last_result['negative_loss']
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


def polish_fft(trial):
    trainable = eval(trial.trainable_name)(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    model = trainable.model
    config = trial.config
    polished_model = ButterflyProduct(size=config['size'], complex=model.complex, fixed_order=True)
    if not model.fixed_order:
        prob = model.softmax_fn(model.logit)
        maxes, argmaxes = torch.max(prob, dim=-1)
        # print(maxes)
        # if torch.all(maxes >= 0.99):
        polished_model.butterflies = nn.ModuleList([model.butterflies[argmax] for argmax in argmaxes])
        # else:
        #     return -trial.last_result['negative_loss']
    else:
        polished_model.butterflies = model.butterflies
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



# ray.init()

result_dir = 'results'
experiment_names = [[f'Hadamard_factorization_fixed_order_{size}' for size in [8, 16, 32, 64, 128, 256]]]
experiment_names += [[f'Hadamard_factorization_softmax_{size}' for size in [8, 16, 32, 64, 128, 256]]]
experiment_names += [[f'Hadamard_factorization_sparsemax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_fixed_order_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_softmax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_sparsemax_{size}' for size in [8, 16, 32, 64, 128]]]

pool = mp.Pool()
for experiment_names_ in experiment_names:
    # print(experiment_names_[0])
    for experiment_name in experiment_names_:
        print(experiment_name)
        checkpoint_path = Path(result_dir) / experiment_name / 'trial.pkl'
        with checkpoint_path.open('rb') as f:
            trials = pickle.load(f)
        sorted_trials = sorted(trials, key=lambda trial: -trial.last_result['negative_loss'])
        losses = [-trial.last_result['negative_loss'] for trial in sorted_trials]
        # polished_losses = ray.get([polish.remote(trial) for trial in sorted_trials[:N_TRIALS_TO_POLISH]])
        if experiment_name.startswith('Hadamard'):
            polished_losses = pool.map(polish_hadamard, sorted_trials[:20])
        elif experiment_name.startswith('Fft'):
            polished_losses = pool.map(polish_fft, sorted_trials[:20])
        else:
            assert False, 'Unknown experiment'
        print(np.sort(losses)[:N_TRIALS_TO_POLISH])
        for i in range(N_TRIALS_TO_POLISH):
            sorted_trials[i].last_result['polished_negative_loss'] = -polished_losses[i]
        print(np.array([trial.last_result['polished_negative_loss'] for trial in sorted_trials[:N_TRIALS_TO_POLISH]]))
        with checkpoint_path.open('wb') as f:
            pickle.dump(trials, f)
