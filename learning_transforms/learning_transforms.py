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

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from
from ray.tune.schedulers import AsyncHyperBandScheduler

from tune import run

from butterfly import Butterfly
from butterfly.permutation import Permutation, FixedPermutation, PermutationFactor
from butterfly.utils import bitreversal_permutation
from butterfly.complex_utils import real_to_complex
from training import PytorchTrainable, TrainableMatrixFactorization
from target_matrix import named_target_matrix


N_LBFGS_STEPS = 50
N_TRIALS_TO_POLISH = 16


class TrainableBP(TrainableMatrixFactorization):
    """Product of butterfly matrices and permutation matrices.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        size = config['size']
        if isinstance(config['target_matrix'], str):
            self.target_matrix = torch.tensor(named_target_matrix(config['target_matrix'], size), dtype=torch.float).to(device)
        else:
            self.target_matrix = torch.tensor(config['target_matrix'], dtype=torch.float).to(device)
        assert self.target_matrix.shape[0] == self.target_matrix.shape[1], 'Only square matrices are supported'
        assert self.target_matrix.dim() in [2, 3], 'target matrix must be 2D if real of 3D if complex'
        complex = self.target_matrix.dim() == 3 or config['complex']
        torch.manual_seed(config['seed'])
        if config['model'] == 'B':
            self.model = nn.Sequential(
                FixedPermutation(torch.tensor(bitreversal_permutation(size))),
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, ortho_init=True)
            ).to(device)
        elif config['model'] == 'BP':
            self.model = nn.Sequential(
                Permutation(size=size, share_logit=config['share_logit'][0]),
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, ortho_init=True)
            ).to(device)
        elif config['model'] == 'PBT':
            self.model = nn.Sequential(
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, increasing_stride=False, ortho_init=True),
                Permutation(size=size, share_logit=config['share_logit'][0])
            ).to(device)
        elif config['model'] == 'BPP':
            self.model = nn.Sequential(
                PermutationFactor(size=size),
                Permutation(size=size, share_logit=config['share_logit'][0]),
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, ortho_init=True)
            ).to(device)
        elif config['model'] == 'BPBP':
            self.model = nn.Sequential(
                Permutation(size=size, share_logit=config['share_logit'][0]),
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, ortho_init=True),
                Permutation(size=size, share_logit=config['share_logit'][1]),
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, ortho_init=True)
            ).to(device)
        elif config['model'] == 'BBT':
            # param_type = 'regular' if complex else 'perm'
            param_type = config['param']
            self.model = nn.Sequential(
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, param=param_type, increasing_stride=False),
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, param=param_type, increasing_stride=True)
            )
        elif config['model'][0] == 'B' and (config['model'][1:]).isdigit():
            depth = int(config['model'][1:])
            param_type = config['param']
            self.model = nn.Sequential(
                *[
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, param=param_type, increasing_stride=True)
                    for _ in range(depth)
                ]
            )
        elif config['model'][0] == 'T' and (config['model'][1:]).isdigit():
            depth = int(config['model'][1:])
            param_type = config['param']
            self.model = nn.Sequential(
                *[
                Butterfly(in_size=size, out_size=size, bias=False, complex=complex, param=param_type, increasing_stride=False)
                    for _ in range(depth)
                ]
            )
        else:
            assert False, f"Model {config['model']} not implemented"
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.n_steps_per_epoch = config['n_steps_per_epoch']
        self.n_epochs_per_validation = config['n_epochs_per_validation']
        self.input = torch.eye(size).to(device)
        if complex:
            self.input = real_to_complex(self.input)

    def freeze(self):
        for i, m in enumerate(self.model):
            if isinstance(m, Permutation) or isinstance(m, PermutationFactor):
                self.model[i] = FixedPermutation(m.argmax())


def polish(trial):
    """Load model from checkpoint, then fix the order of the factor
    matrices (using the largest logits), and re-optimize using L-BFGS to find
    the nearest local optima.
    """
    # Hack: create new instance without call __init__, since trainable.__init__
    # creates result_dir and log_dir in the wrong place (~/ray_results)
    trainable_cls = TrainableBP
    trainable = trainable_cls.__new__(trainable_cls)
    trainable._setup(trial.config)
    trainable.restore(str(Path(trial.logdir) / trial._checkpoint.value))
    loss = trainable.polish(N_LBFGS_STEPS, save_to_self_model=True)
    torch.save(trainable.model.state_dict(), str((Path(trial.logdir) / trial._checkpoint.value).parent / 'polished_model.pth'))

    # round for permutation experiments
    def proj(m):
        if isinstance(m, Butterfly):
            m.round_to_perm()

    trainable.model.apply(proj)
    loss = trainable.loss().item()
    return loss


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
    param = 'regular' # How to constrain the parameters
    ntrials = 20  # Number of trials for hyperparameter tuning
    nsteps = 400  # Number of steps per epoch
    nepochsvalid = 5  # Frequency of validation (polishing), in terms of epochs
    nmaxepochs = 200  # Maximum number of epochs
    result_dir = project_root + '/learning_transforms/results_new'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    nthreads = 1  # Number of CPU threads per job
    smoke_test = False  # Finish quickly for testing


@ex.capture
def transform_experiment(model, target, size, complex, param, ntrials, nsteps, nepochsvalid, result_dir, cuda, nthreads, smoke_test):
    # assert model in ['B', 'BP', 'PBT', 'BPP', 'BPBP', 'BBT', 'BBB'], f'Model {model} not implemented'
    config={
        'model': model,
        'target_matrix': target,
        'size': size,
        'complex': complex,
        # 'share_logit': sample_from(lambda spec: np.random.choice((True, False), size=2)),
        'share_logit': True,
        'param': param,
        'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(5e-1)))),
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_steps_per_epoch': nsteps,
        'n_epochs_per_validation': nepochsvalid,
        'device': 'cuda' if cuda else 'cpu',
     }
    experiment = RayExperiment(
        name=f'{target}_factorization_{model}_{complex}_{size}_{param}',
        run=TrainableBP,
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
def run(result_dir, nmaxepochs, nthreads, cuda):
    experiment = transform_experiment()
    # We'll use multiple processes so disable MKL multithreading
    os.environ['MKL_NUM_THREADS'] = str(nthreads)
    torch.set_num_threads(nthreads)
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            ray.init(redis_address=address)
    except:
        ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='negative_loss', max_t=nmaxepochs)
    trials = run(experiment, scheduler=ahb, raise_on_failed_trial=False, queue_trials=True, early_stop_all_trials=True)
    trials = [trial for trial in trials if trial.last_result is not None]
    losses = [-trial.last_result.get('negative_loss', float('-inf')) for trial in trials]
    print(np.array(losses))

    # Polish solutions with L-BFGS
    polish_fn = ray.remote(num_gpus=0.25 if cuda else 0)(polish)
    sorted_trials = sorted(trials, key=lambda trial: -trial.last_result.get('negative_loss', float('-inf')))
    polished_losses = ray.get([polish_fn.remote(trial) for trial in sorted_trials[:N_TRIALS_TO_POLISH]])
    for i in range(min(N_TRIALS_TO_POLISH, len(trials))):
        sorted_trials[i].last_result['polished_negative_loss'] = -polished_losses[i]
    print(np.sort(losses)[:N_TRIALS_TO_POLISH])
    print(np.sort(polished_losses))

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return min(losses + polished_losses)
