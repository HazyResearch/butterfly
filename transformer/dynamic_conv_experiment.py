import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(1, project_root + '/fairseq')
sys.path.insert(2, project_root + '/fairseq/scripts')
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + project_root + '/fairseq:' + project_root + '/fairseq/scripts' + os.environ.get('PYTHONPATH', '')

import math
from pathlib import Path
import pickle
import random
import glob
import subprocess  # To run fairseq-train
import re  # To extract BLEU score from output of fairseq-generate

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from, grid_search
from ray.tune.schedulers import AsyncHyperBandScheduler

# # Fairseq scripts
from train import train_translation
from generate import generate_translation
from average_checkpoints import main as avg_checkpoints


class TrainableModel(Trainable):
    """Trainable object for a Pytorch model, to be used with Ray's Hyperband tuning.
    """

    def _setup(self, config):
        device = config['device']
        self.device = device
        torch.manual_seed(config['seed'])
        if self.device == 'cuda':
            torch.cuda.manual_seed(config['seed'])
        model = config['model']
        train_args = [project_root + '/fairseq/data-bin/iwslt14.tokenized.de-en']
        train_args += ['--clip-norm', '0'] if model['name'] == 'DynamicConv' else []
        train_args += ['--optimizer', 'adam']
        train_args += ['--lr', str(config['lr'])]
        train_args += ['--source-lang', 'de']
        train_args += ['--target-lang', 'en']
        train_args += ['--max-tokens', '4000']
        train_args += ['--no-progress-bar']
        train_args += ['--log-interval', '100']
        train_args += ['--min-lr', "1e-09"]
        train_args += ['--weight-decay', str(config['weight_decay'])]
        train_args += ['--criterion', 'label_smoothed_cross_entropy']
        train_args += ['--label-smoothing', '0.1']
        train_args += ['--lr-scheduler', 'inverse_sqrt']
        train_args += ['--ddp-backend=no_c10d']
        train_args += ['--max-update', str(config['nmaxupdates'])]
        train_args += ['--warmup-updates', '4000']
        train_args += ['--warmup-init-lr', "1e-07"]
        train_args += ['--adam-betas=(0.9, 0.98)']
        train_args += ['--keep-last-epochs', '10']
        train_args += ['-a', 'lightconv_butterfly_iwslt_de_en'] if model['name'] == 'DynamicConv' else ['-a', 'transformer_butterfly_iwslt_de_en']
        train_args += ['--dropout', str(config['dropout'])]
        train_args += ['--attention-dropout', '0.1'] if model['name'] == 'DynamicConv' else []
        train_args += ['--weight-dropout', '0.1'] if model['name'] == 'DynamicConv' else []
        train_args += ['--encoder-glu', '0'] if model['name'] == 'DynamicConv' else []
        train_args += ['--decoder-glu', '0 '] if model['name'] == 'DynamicConv' else []
        train_args += ['--seed', str(config['seed'])]
        self._save_dir = config['result_dir'] + f"/seed={config['seed']}"
        train_args += ['--save-dir', self._save_dir]
        n_encoder_layer = 7 if model['name'] == 'DynamicConv' else 6
        structure_type = config['structure_type']
        structure_nblocks = config['structure_nblocks']
        n_encoder_structure_layer = config['n_encoder_structure_layer']
        encoder_structure_type = ['Linear'] * (n_encoder_layer - n_encoder_structure_layer) + [structure_type] * n_encoder_structure_layer
        encoder_structure_nblocks = [0] * (n_encoder_layer - n_encoder_structure_layer) + [structure_nblocks] * n_encoder_structure_layer
        n_decoder_structure_layer = config['n_decoder_structure_layer']
        decoder_structure_type = ['Linear'] * (6 - n_decoder_structure_layer) + [structure_type] * n_decoder_structure_layer
        decoder_structure_nblocks = [0] * (6 - n_decoder_structure_layer) + [structure_nblocks] * n_decoder_structure_layer
        train_args += ['--encoder-structure-type-list', str(encoder_structure_type)]
        train_args += ['--decoder-structure-type-list', str(decoder_structure_type)]
        train_args += ['--encoder-structure-nblocks-list', str(encoder_structure_nblocks)] if model['name'] != 'DynamicConv' else []
        train_args += ['--decoder-structure-nblocks-list', str(decoder_structure_nblocks)] if model['name'] != 'DynamicConv' else []
        if config['structured_attention']:
            encoder_structured_attention = [False] * (6 - n_encoder_structure_layer) + [True] * n_encoder_structure_layer
            train_args += ['--encoder-structured-attention-list', str(encoder_structured_attention)] if model['name'] != 'DynamicConv' else []
            decoder_structured_attention = [False] * (6 - n_decoder_structure_layer) + [True] * n_decoder_structure_layer
            train_args += ['--decoder-structured-attention-list', str(decoder_structured_attention)]

        avg_args = [
            '--inputs=' + self._save_dir, '--num-epoch-checkpoints=10',
            '--output=' + self._save_dir + '/model.pt'
        ]
        gen_args = [project_root + '/fairseq/data-bin/iwslt14.tokenized.de-en', '--batch-size=128', '--remove-bpe',
                    '--beam=4', '--quiet',
                   ]
        self._train_args = train_args
        self._avg_args = avg_args
        self._gen_args = gen_args

    def _train(self):
        os.makedirs(self._save_dir, exist_ok=True)
        stdout = sys.stdout
        with open(self._save_dir + '/logs.txt', 'w+') as log:
            sys.stdout = log
            train_translation(self._train_args)
            avg_checkpoints(cmdline_args=self._avg_args)
            # Delete checkpoints to save disk space
            last_model = os.path.join(self._save_dir, 'checkpoint_last.pt')
            best_model = os.path.join(self._save_dir, 'checkpoint_best.pt')
            ensemble_model = os.path.join(self._save_dir, 'model.pt')
            for ckpt_file in glob.glob(os.path.join(self._save_dir, '*.pt')):
                if ckpt_file != last_model and ckpt_file != ensemble_model \
                                        and ckpt_file != best_model:
                    os.remove(ckpt_file)
            _, BLEU_last_valid = generate_translation(
                self._gen_args + ['--gen-subset=valid', '--path=' + last_model])
            _, BLEU_ensm_valid = generate_translation(
                self._gen_args + ['--gen-subset=valid', '--path=' + ensemble_model])
            _, BLEU_last_test = generate_translation(
                self._gen_args + ['--gen-subset=test', '--path=' + last_model])
            _, BLEU_ensm_test = generate_translation(
                self._gen_args + ['--gen-subset=test', '--path=' + ensemble_model])
        sys.stdout = stdout
        return {
            'final_bleu_valid': BLEU_last_valid,
            'ensemble_bleu_valid': BLEU_ensm_valid,
            'final_bleu_test': BLEU_last_test,
            'ensemble_bleu_test': BLEU_ensm_test,
        }

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        return checkpoint_path

    def _restore(self, checkpoint_path):
        pass


ex = Experiment('Transformer_experiment')
ex.observers.append(FileStorageObserver.create('logs'))
slack_config_path = Path('../config/slack.json')  # Add webhook_url there for Slack notification
if slack_config_path.exists():
    ex.observers.append(SlackObserver.from_config(str(slack_config_path)))


@ex.config
def default_config():
    model = 'DynamicConv'  # Name of model, either 'DynamicConv' or 'Transformer'
    model_args = {}  # Arguments to be passed to the model, as a dictionary
    n_encoder_structure_layer = 0  # Number of structured layer in the encoder
    n_decoder_structure_layer = 0  # Number of structured layer in the decoder
    structure_type = 'B'  # 'B' for butterfly or BBT for product of 2 butterflies
    nblocks = 0  # Number of (BB^T) blocks in structure
    structured_attention = False  # Whether attention layers are structured
    ntrials = 20  # Number of trials for hyperparameter tuning
    nmaxupdates = 50000  # Maximum number of updates
    result_dir = project_root + '/transformer/results'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing


@ex.capture
def dynamic_conv_experiment(model, model_args, n_encoder_structure_layer, n_decoder_structure_layer, structure_type, nblocks, structured_attention,
                            nmaxupdates, ntrials, result_dir, cuda, smoke_test):
    name=f'{model}_{model_args}_type_{structure_type}_nblocks_{nblocks}_encstruct_{n_encoder_structure_layer}_decstruct_{n_decoder_structure_layer}_attstruct_{structured_attention}'
    config={
        # 'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(1e-3)))),
        # 'lr': grid_search([5e-4, 7e-4, 9e-4, 11e-4]),
        'lr': grid_search([9e-4, 11e-4]),
        'weight_decay': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-6), math.log(5e-4)))) if model == 'DynamicConv' else 1e-4,
        # Transformer seems to need dropout 0.3
        'dropout': sample_from(lambda spec: random.uniform(0.1, 0.3)) if model == 'DynamicConv' else 0.3,
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'n_encoder_structure_layer': n_encoder_structure_layer,
        'n_decoder_structure_layer': n_decoder_structure_layer,
        'structure_type': structure_type,
        'structure_nblocks': nblocks,
        'structured_attention': structured_attention,
        'device': 'cuda' if cuda else 'cpu',
        'model': {'name': model, 'args': model_args},
        'nmaxupdates': nmaxupdates,
        'result_dir': result_dir + '/' + name
     }
    experiment = RayExperiment(
        name=name,
        run=TrainableModel,
        local_dir=result_dir,
        num_samples=ntrials,
        checkpoint_at_end=False,
        checkpoint_freq=1000,  # Just to enable recovery with @max_failures
        max_failures=-1,
        resources_per_trial={'cpu': 4, 'gpu': 1 if cuda else 0},
        stop={"training_iteration": 1},
        config=config,
    )
    return experiment


@ex.automain
def run(model, structure_type, nblocks, result_dir, nmaxupdates):
    experiment = dynamic_conv_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            ray.init(redis_address=address)
    except:
        ray.init()
    trials = ray.tune.run(experiment, raise_on_failed_trial=False, queue_trials=True)
    trials = [trial for trial in trials if trial.last_result is not None]
    bleu = [trial.last_result.get('ensemble_bleu_test', float('-inf')) for trial in trials]

    checkpoint_path = Path(result_dir) / experiment.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path /= 'trial.pkl'
    with checkpoint_path.open('wb') as f:
        pickle.dump(trials, f)

    ex.add_artifact(str(checkpoint_path))
    return model, structure_type, nblocks, max(bleu)
