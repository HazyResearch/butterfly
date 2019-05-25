import os, sys, subprocess
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import math
from pathlib import Path
import pickle
import random
import datetime

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

        self.model = model_utils.get_model(config['model'])
        # restore permutation
        if config['restore_perm'] is not None:
            # checkpoint = torch.load(config['restore_perm'], self.device)
            checkpoint = torch.load(config['restore_perm'])
            # don't restore args, so that you can change temp etc when plugging into end model
            # TODO: implement an update_args() method for the models
            # self.model.permute = model_utils.get_model(checkpoint['model']['args'])
            self.model.permute.load_state_dict(checkpoint['model']['state'])
        self.model.to(self.device)
        self.nparameters = sum(param.nelement() for param in self.model.parameters())

        self.train_loader, self.test_loader = dataset_utils.get_dataset(config['dataset'])
        permutation_params = filter(lambda p: hasattr(p, '_is_perm_param') and p._is_perm_param, self.model.parameters())
        unstructured_params = filter(lambda p: not (hasattr(p, '_is_perm_param') and p._is_perm_param), self.model.parameters())
        if config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam([{'params': permutation_params, 'weight_decay': config['pwd'], 'lr': config['plr']},
                                         {'params': unstructured_params}],
                                        lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            self.optimizer = optim.SGD([{'params': permutation_params, 'weight_decay': config['pwd'], 'lr': config['plr']},
                                        {'params': unstructured_params}],
                                       lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config['lr_decay_period'], gamma=config['lr_decay_factor'])

        #
        self.unsupervised = config['unsupervised']
        self.tv = config['tv']
        self.anneal_entropy_factor = config['anneal_entropy'] # NOTE: restoring a model does not call the sample_from function
        self.anneal_sqrt = config['anneal_sqrt']
        self.entropy_p = config['entropy_p']
        self.model_args = config['model']

    def _train_iteration(self):
        self.model.train()
        inv_temp = math.sqrt(self._iteration) if self.anneal_sqrt else self._iteration
        # print(f"ITERATION {self._iteration} INV TEMP {inv_temp} ANNEAL ENTROPY {self.anneal_entropy_factor}")
        inv_temp *= self.anneal_entropy_factor
        print(f"ITERATION {self._iteration} INV TEMP {inv_temp}")
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            if self.unsupervised:
                # p = self.model.get_permutations()
                # assert p.requires_grad
                # print("train_iteration REQUIRES GRAD: ", p.requires_grad)
                # H = perm.entropy(p, reduction='mean')
                H = self.model.entropy(p=self.entropy_p)
                assert H.requires_grad
                loss = perm.tv(output, norm=self.tv['norm'], p=self.tv['p'], symmetric=self.tv['sym']) + inv_temp * H
                # loss = perm.tv(output, norm=self.tv['norm'], p=self.tv['p'], symmetric=self.tv['sym']) # + inv_temp * H
                # print("LOSS ", loss.item())
            else:
                # target = target.expand(output.size()[:-1]).flatten()
                # outupt = output.flatten(0, -2)
                # print(output.shape, target.shape)
                # assert self.model.samples == output.size(0) // target.size(0)
                target = target.repeat(output.size(0) // target.size(0))
                # print(output.shape, target.shape)
                loss = F.cross_entropy(output, target)
            # tw0 = list(self.model.permute)[0].twiddle
            # tw1 = list(self.model.permute)[1].twiddle
            # assert torch.all(tw0 == tw0)
            # assert torch.all(tw1 == tw1)
            loss.backward()
            # breakpoint()
            self.optimizer.step()
            # tw0 = list(self.model.permute)[0].twiddle
            # tw1 = list(self.model.permute)[1].twiddle
            # assert torch.all(tw0 == tw0)
            # assert torch.all(tw1 == tw1)

    def _test(self):
        self.model.eval()
        test_loss     = 0.0
        correct       = 0.0
        total_samples = 0
        if self.unsupervised:
            mean_loss = 0.0
            mle_loss  = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_samples += output.size(0)
                if self.unsupervised:
                    test_loss  += perm.tv(output, reduction='sum').item()

                    mean_output = self.model(data, perm ='mean')
                    mean_loss  += perm.tv(mean_output, reduction='sum').item()
                    mle_output  = self.model(data, perm ='mle')
                    mle_loss   += perm.tv(mle_output, reduction='sum').item()
                else:
                    target = target.repeat(output.size(0)//target.size(0))
                    test_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += (pred == target.data.view_as(pred)).long().cpu().sum().item()

            if self.unsupervised:
                true = self.test_loader.true_permutation[0]
                p = self.model.get_permutations() # (rank, sample, n, n)
                # p0 = p[0]
                # elements = p0[..., torch.arange(len(true)), true]
                # print("max in true perm elements", elements.max(dim=-1)[0])
                # print(p0)
                sample_ent = perm.entropy(p, reduction='mean')
                sample_nll = perm.dist(p, self.test_loader.true_permutation, fn='nll')
                # sample_was = perm.dist(p, self.test_loader.true_permutation, fn='was')
                sample_was1, sample_was2 = perm.dist(p, self.test_loader.true_permutation, fn='was')

                mean = self.model.get_permutations(perm='mean') # (rank, n, n)
                # print("MEAN PERMUTATION", mean)
                mean_ent = perm.entropy(mean, reduction='mean')
                mean_nll = perm.dist(mean, self.test_loader.true_permutation, fn='nll')
                # mean_was = perm.dist(mean, self.test_loader.true_permutation, fn='was')
                mean_was1, mean_was2 = perm.dist(mean, self.test_loader.true_permutation, fn='was')
                mean_was1_abs, mean_was2_abs = torch.abs(682.-mean_was1), torch.abs(682.-mean_was2)
                unif = torch.ones_like(mean) / mean.size(-1)
                # mean_unif_dist = nn.functional.mse_loss(mean, unif, reduction='sum')
                mean_unif_dist = torch.sum((mean-unif)**2) / mean.size(0)

                mle = self.model.get_permutations(perm='mle') # (rank, n, n)
                # mle_ent = perm.entropy(mle, reduction='mean')
                # mle_nll = perm.dist(mle, self.test_loader.true_permutation, fn='nll')
                # mle_was = perm.dist(mle, self.test_loader.true_permutation, fn='was')
                mle_was1, mle_was2 = perm.dist(mle, self.test_loader.true_permutation, fn='was')

                H = self.model.entropy(p=self.entropy_p)

                # TODO calculate average case wasserstein automatically in terms of rank/dims and power p
                return {
                    "sample_loss": test_loss / total_samples,
                    "sample_ent": sample_ent.item(),
                    "sample_nll": sample_nll.item(),
                    # "sample_was": sample_was.item(),
                    "sample_was1": sample_was1.item(),
                    "sample_was2": sample_was2.item(),
                    "mean_loss": mean_loss / total_samples,
                    "mean_ent": mean_ent.item(),
                    "mean_nll": mean_nll.item(),
                    "mean_was1": mean_was1.item(),
                    "mean_was2": mean_was2.item(),
                    "neg_was2": 682.0-mean_was2.item(),
                    "mle_loss": mle_loss / total_samples,
                    # "mle_ent": mle_ent.item(),
                    # "mle_nll": mle_nll.item(),
                    "mle_was1": mle_was1.item(),
                    "mle_was2": mle_was2.item(),
                    # "mean_accuracy": 682.0-mean_was2.item(),
                    # "neg_sample_loss": -test_loss / total_samples,
                    "mean_unif_dist": mean_unif_dist.item(),
                    # "mean_was1_abs": mean_was1_abs.item(),
                    # "mean_was2_abs": mean_was2_abs.item(),
                    "model_ent": H.item(),
                    # "neg_ent_floor": -int(mean_ent.item()),
                    "neg_ent": -H.item(),
                }

        # test_loss = test_loss / len(self.test_loader.dataset)
        # accuracy = correct / len(self.test_loader.dataset)
        test_loss = test_loss / total_samples
        accuracy = correct / total_samples
        return {"mean_loss": test_loss, "mean_accuracy": accuracy}

    def _train(self):
        # self.scheduler.step()
        # self._train_iteration()
        # return self._test()
        self._train_iteration()
        metrics = self._test()
        self.scheduler.step()
        return metrics

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        full_model = {
            'state': self.model.state_dict(),
            'args': self.model_args,
        }
        state = {'model': full_model,
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(state, checkpoint_path)

        # model_path = os.path.join(checkpoint_dir, "saved_model.pth")
        # torch.save(full_model, model_path)
        # model_args = os.path.join(checkpoint_dir, "saved_model.args")
        # torch.save(self.model_args, model_args)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        if hasattr(self, 'device'):
            checkpoint = torch.load(checkpoint_path, self.device)
        else:
            checkpoint = torch.load(checkpoint_path)
        # saved_model = torch.load(checkpoint_path + '.args')
        # self.model = model_utils.get_model(saved_model['args'])
        self.model = model_utils.get_model(checkpoint['model']['args'])
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model']['state'])

        permutation_params = filter(lambda p: hasattr(p, '_is_perm_param') and p._is_perm_param, self.model.parameters())
        unstructured_params = filter(lambda p: not (hasattr(p, '_is_perm_param') and p._is_perm_param), self.model.parameters())
        self.optimizer = optim.Adam([{'params': permutation_params},
                                     {'params': unstructured_params}],)
        # self.optimizer = optim.Adam([{'params': permutation_params, 'weight_decay': 0.0, 'lr': config['plr']},
        #                              {'params': unstructured_params}],
        #                             lr=config['lr'], weight_decay=config['weight_decay'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.optimizer.param_groups[1].update({'weight_decay': 0.0})
        # self.optimizer.param_groups[0].update({'params': permutation_params})
        # self.optimizer.param_groups[1].update({'params': unstructured_params})

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer)
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scheduler.optimizer = self.optimizer


ex = Experiment('Permutation_experiment')
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
    lr_decay_period = 50  # Period of learning rate decay
    plr_min = 1e-4
    plr_max = 1e-3
    weight_decay = True  # Whether to use weight decay
    pwd = True
    pwd_min = 1e-4
    pwd_max = 5e-4
    ntrials = 20  # Number of trials for hyperparameter tuning
    nmaxepochs = 200  # Maximum number of epochs
    unsupervised = False
    batch = 128
    tv_norm = 2
    tv_p = 1
    tv_sym = None
    anneal_ent_min = 0.0
    anneal_ent_max = 0.0
    anneal_sqrt = True
    entropy_p = None
    temp_min = None
    temp_max = None
    restore_perm = None
    resume_pth = None
    result_dir = project_root + '/cnn/results'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing


@ex.named_config
def sgd():
    optimizer = 'SGD'  # Which optimizer to use, either Adam or SGD
    lr_decay = True  # Whether to use learning rate decay
    lr_decay_period = 25  # Period of learning rate decay
    weight_decay = True  # Whether to use weight decay


@ex.capture
def cifar10_experiment(dataset, model, args, optimizer, nmaxepochs, lr_decay, lr_decay_period, plr_min, plr_max, weight_decay, pwd, pwd_min, pwd_max, ntrials, result_dir, cuda, smoke_test, unsupervised, batch, tv_norm, tv_p, tv_sym, temp_min, temp_max, anneal_ent_min, anneal_ent_max, anneal_sqrt, entropy_p, restore_perm, resume_pth): # TODO clean up and set min,max to pairs/dicts
    assert optimizer in ['Adam', 'SGD'], 'Only Adam and SGD are supported'
    assert restore_perm is None or resume_pth is None # If we're fully resuming training from the checkpoint, no point in restoring any part of the model
    if restore_perm is not None:
        restore_perm = '/dfs/scratch1/albertgu/learning-circuits/cnn/saved_perms/' + restore_perm
        print("Restoring permutation from", restore_perm)

    args_rand = args.copy()
    if temp_min is not None and temp_max is not None:
        args_rand['temp'] = sample_from(lambda spec: math.exp(random.uniform(math.log(temp_min), math.log(temp_max))))
    # args_rand['samples'] = sample_from(lambda _: np.random.choice((8,16)))
    # args_rand['sig'] = sample_from(lambda _: np.random.choice(('BT1', 'BT4')))

    tv = {'norm': tv_norm, 'p': tv_p}
    if tv_sym is 'true':
        tv['sym'] = sample_from(lambda _: np.random.choice((True,)))
    elif tv_sym is 'false':
        tv['sym'] = sample_from(lambda _: np.random.choice((False,)))
    elif tv_sym is 'random':
        tv['sym'] = sample_from(lambda _: np.random.choice((True,False)))
    else:
        assert tv_sym is None, 'tv_sym must be true, false, or random'
        tv['sym'] = False

    if anneal_ent_max == 0.0:
        anneal_entropy = 0.0
    else:
        anneal_entropy = sample_from(lambda _: math.exp(random.uniform(math.log(anneal_ent_min), math.log(anneal_ent_max)))),

    name_smoke_test = 'smoke_' if smoke_test else '' # for easy finding and deleting unimportant logs
    name_args = '_'.join([k+':'+str(v) for k,v in args.items()])
    config={
        'optimizer': optimizer,
        # 'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(2e-5), math.log(1e-2)) if optimizer == 'Adam'
        'lr': 2e-4 if optimizer == 'Adam' else math.exp(random.uniform(math.log(0.025), math.log(0.2))),
        'plr': sample_from(lambda spec: math.exp(random.uniform(math.log(plr_min), math.log(plr_max)))),
        # 'lr_decay_factor': sample_from(lambda spec: random.choice([0.1, 0.2])) if lr_decay else 1.0,
        'lr_decay_factor': 0.2 if lr_decay else 1.0,
        'lr_decay_period': lr_decay_period,
        # 'weight_decay':  sample_from(lambda spec: math.exp(random.uniform(math.log(1e-6), math.log(5e-4)))) if weight_decay else 0.0,
        'weight_decay':    5e-4 if weight_decay else 0.0,
        'pwd': sample_from(lambda spec: math.exp(random.uniform(math.log(pwd_min), math.log(pwd_max)))) if pwd else 0.0,
        'seed':            sample_from(lambda spec: random.randint(0, 1 << 16)),
        'device':          'cuda' if cuda else 'cpu',
        'model':           {'name': model, 'args': args_rand},
        # 'model':           {'name': model, 'args': args.update({'temp': sample_from(lambda spec: math.exp(random.uniform(math.log(temp_min), math.log(temp_max))))})},

        'dataset':         {'name': dataset, 'batch': batch},
        'unsupervised':    unsupervised,
        # 'tv':              {'norm': tv_norm, 'p': tv_p, 'sym': tv_sym},
        # 'tv':              {'norm': tv_norm, 'p': tv_p, 'sym': sample_from(lambda _: np.random.choice((True,False)))},
        'tv': tv if unsupervised else None,
        # 'anneal_entropy':  anneal_entropy,
        # 'anneal_entropy':  sample_from(lambda _: random.uniform(anneal_ent_min, anneal_ent_max)),
        'anneal_entropy':  0.0 if anneal_ent_max==0.0 else sample_from(lambda _: math.exp(random.uniform(math.log(anneal_ent_min), math.log(anneal_ent_max)))),
        'anneal_sqrt':  anneal_sqrt,
        'entropy_p': entropy_p,
        'restore_perm': restore_perm,
     }
    timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
    stopping_criteria = {"training_iteration": 1 if smoke_test else nmaxepochs}
    if unsupervised: # TODO group all the unsupervised casework together
        stopping_criteria.update({'model_ent': 200, 'neg_ent': -5.0})

    experiment = RayExperiment(
        # name=f'pcifar10_{model}_{args}_{optimizer}_lr_decay_{lr_decay}_weight_decay_{weight_decay}',
        name=f'{name_smoke_test}{dataset.lower()}_{model}_{name_args}_{optimizer}_epochs_{nmaxepochs}_plr_{plr_min}-{plr_max}_{timestamp}_{commit_id}',
        # name=f'{dataset.lower()}_{model}_{args_orig}_{optimizer}_epochs_{nmaxepochs}_lr_decay_{lr_decay}_plr_{plr_min}-{plr_max}_tvsym_{tv_sym}_{timestamp}_{commit_id}',
        run=TrainableModel,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=True,
        checkpoint_freq=500,  # Just to enable recovery with @max_failures
        max_failures=0,
        # resources_per_trial={'cpu': 4, 'gpu': 0.5 if cuda else 0},
        resources_per_trial={'cpu': 4, 'gpu': 1 if cuda else 0},
        # stop={"training_iteration": 1 if smoke_test else nmaxepochs, 'model_ent': 200, 'neg_ent': -5.0},
        stop=stopping_criteria,
        # stop={"training_iteration": 1 if smoke_test else nmaxepochs},
        restore=resume_pth,
        config=config,
    )
    return experiment


@ex.automain
def run(model, result_dir, nmaxepochs, unsupervised):
    experiment = cifar10_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            # ray.init(redis_address=address, temp_dir='/tmp/ray2/')
            ray.init(redis_address=address)
    except:
        ray.init()
    if unsupervised:
        ahb = AsyncHyperBandScheduler(reward_attr='neg_was2', max_t=nmaxepochs, grace_period=nmaxepochs, reduction_factor=2, brackets=1)
    else:
        ahb = AsyncHyperBandScheduler(reward_attr='mean_accuracy', max_t=nmaxepochs)
    trials = ray.tune.run(
        experiment, scheduler=ahb,
        raise_on_failed_trial=False, queue_trials=True,
        # with_server=True, server_port=4321,
    )
    trials = [trial for trial in trials if trial.last_result is not None]
    accuracy = [trial.last_result.get('mean_accuracy', float('-inf')) for trial in trials]

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return max(accuracy)
