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
import torch.nn.functional as F

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from
from ray.tune.schedulers import AsyncHyperBandScheduler


import model_utils
import dataset_utils
import permutation_utils as perm


class TrainableModel(Trainable):
    """Trainable object for a Pytorch model, to be used with Ray's Hyperband tuning.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        torch.manual_seed(config['seed'])
        if self.device == 'cuda':
            torch.cuda.manual_seed(config['seed'])
        self.model = model_utils.get_model(config['model']).to(device)
        self.train_loader, self.test_loader = dataset_utils.get_dataset(config['dataset'])
        permutation_params = filter(lambda p: hasattr(p, '_is_perm_param') and p._is_perm_param, self.model.parameters())
        unstructured_params = filter(lambda p: not (hasattr(p, '_is_perm_param') and p._is_perm_param), self.model.parameters())
        if config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam([{'params': permutation_params, 'weight_decay': 0.0, 'lr': config['plr']},
                                         {'params': unstructured_params}],
                                        lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            self.optimizer = optim.SGD([{'params': structured_params, 'weight_decay': 0.0},
                                        {'params': unstructured_params}],
                                       lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay_period'], gamma=config['lr_decay_factor'])

        #
        self.unsupervised = config['unsupervised']
        self.tv = config['tv']
        self.anneal_entropy_factor = config['anneal_entropy']

    def _train_iteration(self):
        self.model.train()
        inv_temp = self.anneal_entropy_factor * self._iteration
        print(f"ITERATION {self._iteration} INV TEMP {inv_temp}")
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.unsupervised:
                p = self.model.get_permutations()
                # print("train_iteration REQUIRES GRAD: ", p.requires_grad)
                loss = perm.tv(output, norm=self.tv['norm'], p=self.tv['p'], symmetric=self.tv['sym']) \
                    + inv_temp * perm.entropy(p, reduction='mean')
                # print("LOSS ", loss.item())
            else:
                # target = target.expand(output.size()[:-1]).flatten()
                # outupt = output.flatten(0, -2)
                # print(output.shape, target.shape)
                # assert self.model.samples == output.size(0) // target.size(0)
                target = target.repeat(output.size(0) // target.size(0))
                # print(output.shape, target.shape)
                loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

    def _test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_samples += output.size(0)
                if self.unsupervised:
                    test_loss += perm.tv(output).item()
                    # p = self.model.get_permutations()
                else:
                    target = target.repeat(output.size(0)//target.size(0))
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += (pred == target.data.view_as(pred)).long().cpu().sum().item()

            if self.unsupervised:
                true = self.test_loader.true_permutation[0]
                # gp = self.model.get_permutations()
                # import pdb; pdb.set_trace()
                p = self.model.get_permutations() # (rank, sample, n, n)
                # print(f"PERMUTATIONS SHAPE {p.shape}")
                p0 = p[0]
                elements = p0[..., torch.arange(len(true)), true]
                print("max in true perm elements", elements.max(dim=-1)[0])
                # if len(p0.size()) > 2: p0 = p0[0]
                # print(p0)
                sample_ent = perm.entropy(p, reduction='mean')
                sample_nll = perm.dist(p, self.test_loader.true_permutation, fn='nll')
                sample_was = perm.dist(p, self.test_loader.true_permutation, fn='was')
                # print("sample ENTROPY ", sample_ent)
                # print("sample NLL ", sample_nll)
                # print("sample TRANSPORT ", sample_was)

                # correct = perm.dist(p, self.test_loader.true_permutation).item()
                # correct = correct * total_samples # hack to normalize properly

                mean = self.model.get_mean_perms() # (rank, sample, n, n)
                mean_ent = perm.entropy(mean, reduction='mean')
                mean_nll = perm.dist(mean, self.test_loader.true_permutation, fn='nll')
                mean_was = perm.dist(mean, self.test_loader.true_permutation, fn='was')
                print("mean ENTROPY ", mean_ent)
                print("mean NLL ", mean_nll)
                print("mean TRANSPORT ", mean_was)

                mle = self.model.get_mle_perms() # (rank, sample, n, n)
                mle_ent = perm.entropy(mle, reduction='mean')
                mle_nll = perm.dist(mle, self.test_loader.true_permutation, fn='nll')
                mle_was = perm.dist(mle, self.test_loader.true_permutation, fn='was')
                print("mle ENTROPY ", mle_ent)
                print("mle NLL ", mle_nll)
                print("mle TRANSPORT ", mle_was)

                return {"mean_loss": test_loss / total_samples,
                        "sample_ent": sample_ent.item(), "sample_nll": sample_nll.item(), "sample_was": sample_was.item,
                        "mean_ent": mean_ent.item(), "mean_nll": mean_nll.item(), "mean_was": mean_was.item(),
                        "mle_ent": mle_ent.item(), "mle_nll": mle_nll.item(), "mle_was": mle_was.item(),
                        "mean_accuracy": 1364.0-mean_was.item()}

        # test_loss = test_loss / len(self.test_loader.dataset)
        # accuracy = correct / len(self.test_loader.dataset)
        test_loss = test_loss / total_samples
        accuracy = correct / total_samples
        # accuracy = 0.0
        # print(f"test_loss {test_loss}, accuracy {accuracy}")
        return {"mean_loss": test_loss, "mean_accuracy": accuracy}

    def _train(self):
        self.scheduler.step()
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        if hasattr(self, 'device'):
            checkpoint = torch.load(checkpoint_path, self.device)
        else:
            checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


ex = Experiment('Cifar10_experiment')
ex.observers.append(FileStorageObserver.create('logs'))
slack_config_path = Path('../config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def default_config():
    dataset = 'PPCIFAR10'
    model = 'PResNet18'  # Name of model, see model_utils.py
    args = {}  # Arguments to be passed to the model, as a dictionary
    optimizer = 'Adam'  # Which optimizer to use, either Adam or SGD
    lr_decay = True  # Whether to use learning rate decay
    lr_decay_period = 18  # Period of learning rate decay
    plr_min = 1e-4
    plr_max = 1e-2
    weight_decay = True  # Whether to use weight decay
    ntrials = 20  # Number of trials for hyperparameter tuning
    nmaxepochs = 72  # Maximum number of epochs
    result_dir = project_root + '/cnn/results'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing
    unsupervised = False
    batch = 128
    tv_norm = 2
    tv_p = 1
    tv_sym = False
    anneal_entropy = 0.0


@ex.named_config
def sgd():
    optimizer = 'SGD'  # Which optimizer to use, either Adam or SGD
    lr_decay = True  # Whether to use learning rate decay
    lr_decay_period = 25  # Period of learning rate decay
    weight_decay = True  # Whether to use weight decay


@ex.capture
def cifar10_experiment(dataset, model, args, optimizer, nmaxepochs, lr_decay, lr_decay_period, plr_min, plr_max, weight_decay, ntrials, result_dir, cuda, smoke_test, unsupervised, batch, tv_norm, tv_p, tv_sym, anneal_entropy):
    assert optimizer in ['Adam', 'SGD'], 'Only Adam and SGD are supported'
    config={
        'optimizer': optimizer,
        # 'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(2e-5), math.log(1e-2)) if optimizer == 'Adam'
        'lr': 2e-4 if optimizer == 'Adam' else random.uniform(math.log(2e-3), math.log(1e-0)),
        'plr': sample_from(lambda spec: math.exp(random.uniform(math.log(plr_min), math.log(plr_max)))),
        # 'lr_decay_factor': sample_from(lambda spec: random.choice([0.1, 0.2])) if lr_decay else 1.0,
        'lr_decay_factor': 0.12 if lr_decay else 1.0,
        'lr_decay_period': lr_decay_period,
        # 'weight_decay': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-6), math.log(5e-4)))) if weight_decay else 0.0,
        'weight_decay': 2e-4 if weight_decay else 0.0,
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'device': 'cuda' if cuda else 'cpu',
        'model': {'name': model, 'args': args},
        # 'dataset': {'name': 'CIFAR10'}
        # 'dataset': {'name': 'PCIFAR10'}
        'dataset': {'name': dataset, 'batch': batch},
        'unsupervised': unsupervised,
        'tv': {'norm': tv_norm, 'p': tv_p, 'sym': tv_sym},
        'anneal_entropy': anneal_entropy,
     }
    experiment = RayExperiment(
        # name=f'pcifar10_{model}_{args}_{optimizer}_lr_decay_{lr_decay}_weight_decay_{weight_decay}',
        name=f'{dataset.lower()}_{model}_{args}_{optimizer}_epochs_{nmaxepochs}_lr_decay_{lr_decay}_plr_{plr_min}-{plr_max}_tvsym_{tv_sym}',
        run=TrainableModel,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        checkpoint_freq=1000,  # Just to enable recovery with @max_failures
        max_failures=-1,
        resources_per_trial={'cpu': 4, 'gpu': 1 if cuda else 0},
        stop={"training_iteration": 1 if smoke_test else 9999},
        config=config,
    )
    return experiment


@ex.automain
def run(model, result_dir, nmaxepochs):
    experiment = cifar10_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            # ray.init(redis_address=address, temp_dir='/tmp/ray2/')
            ray.init(redis_address=address)
    except:
        ray.init()
    ahb = AsyncHyperBandScheduler(reward_attr='mean_accuracy', max_t=nmaxepochs)
    trials = ray.tune.run(experiment, scheduler=ahb, raise_on_failed_trial=False, queue_trials=True)
    trials = [trial for trial in trials if trial.last_result is not None]
    accuracy = [trial.last_result.get('mean_accuracy', float('-inf')) for trial in trials]

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return max(accuracy)
