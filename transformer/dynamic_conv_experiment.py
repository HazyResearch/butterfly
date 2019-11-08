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
import subprocess  # To run fairseq-generate
import re  # To extract BLEU score from output of fairseq-generate
import socket  # For hostname

import numpy as np

import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import ray
from ray.tune import Trainable, Experiment as RayExperiment, sample_from, grid_search
from ray.tune.schedulers import AsyncHyperBandScheduler

# Fairseq scripts
from train import train_translation
from generate import generate_translation
from average_checkpoints import main as avg_checkpoints


def evaluate_translation(gen_args):
    # gen_process = subprocess.run(['fairseq-generate'] + gen_args, capture_output=True)
    # Need to use sys.executable to call the correct Python interpreter
    gen_process = subprocess.run([sys.executable, str(Path(project_root) / 'fairseq/generate.py')] + gen_args,
                                 capture_output=True)
    err = gen_process.stderr.decode(sys.stdout.encoding)
    out = gen_process.stdout.decode(sys.stdout.encoding)
    sys.stderr.write(err)
    sys.stdout.write(out)
    m = re.search(r'BLEU4 = ([-+]?\d*\.\d+|\d+),', out)
    return None, float(m.group(1))  # Return a pair to be compatible with generate_translation


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
        train_args = [str(Path(project_root) / 'fairseq/data-bin/iwslt14.tokenized.de-en')]
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
        # Always train from scratch, to overwrite previous runs, so point to nonexistent checkpoint file
        # train_args += ['--restore-file', 'nonexistent_checkpoint.pt']
        train_args += ['-a', 'lightconv_butterfly_iwslt_de_en'] if model['name'] == 'DynamicConv' else ['-a', 'transformer_butterfly_iwslt_de_en']
        train_args += ['--dropout', str(config['dropout'])]
        train_args += ['--attention-dropout', '0.1'] if model['name'] == 'DynamicConv' else []
        train_args += ['--weight-dropout', '0.1'] if model['name'] == 'DynamicConv' else []
        train_args += ['--encoder-glu', '0'] if model['name'] == 'DynamicConv' else []
        train_args += ['--decoder-glu', '0 '] if model['name'] == 'DynamicConv' else []
        train_args += ['--seed', str(config['seed'])]
        self._save_dir = Path(config['result_dir']) / f"seed={config['seed']}"
        train_args += ['--save-dir', str(self._save_dir)]
        train_args += ['--encoder-layers', str(len(config['encoder']))]
        train_args += ['--decoder-layers', str(len(config['decoder']))]
        train_args += ['--encoder-structure-type-list', str(config['encoder'])]
        train_args += ['--decoder-structure-type-list', str(config['decoder'])]
        train_args += ['--structure-lr-multiplier', str(config['structure-lr-multiplier'])]
        print(f'Host: {socket.gethostname()}, save_dir: {self._save_dir}')

        avg_args = [
            '--inputs=' + str(self._save_dir), '--num-epoch-checkpoints=10',
            '--output=' + str(self._save_dir / 'model.pt')
        ]
        gen_args = [project_root + '/fairseq/data-bin/iwslt14.tokenized.de-en',
                    '--batch-size=64', '--remove-bpe',
                    '--beam=4', '--quiet', '--no-progress-bar'
                   ]
        self._train_args = train_args
        self._avg_args = avg_args
        self._gen_args = gen_args

    def _train(self):
        self._save_dir.mkdir(parents=True, exist_ok=True)
        stdout = sys.stdout
        with open(self._save_dir / 'logs.txt', 'w+') as log:
            sys.stdout = log
            # [2019-08-02] For some reason ray gets stuck when I call train_translation
            # or generate_translation.
            # Workaround: use subprocess to call fairseq-generate in another process
            # train_translation(self._train_args)
            subprocess.run([sys.executable, str(Path(project_root) / 'fairseq/train.py')] + self._train_args,
                           stdout=log)
            avg_checkpoints(cmdline_args=self._avg_args)
            last_model = self._save_dir / 'checkpoint_last.pt'
            best_model = self._save_dir / 'checkpoint_best.pt'
            ensemble_model = self._save_dir / 'model.pt'
            # Delete checkpoints to save disk space
            # for ckpt_file in Path(self._save_dir).glob('*.pt'):
            #     if ckpt_file != last_model and ckpt_file != ensemble_model \
            #                             and ckpt_file != best_model:
            #         ckpt_file.unlink()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            _, BLEU_last_valid = evaluate_translation(
                self._gen_args + ['--gen-subset=valid', '--path=' + str(last_model)])
            _, BLEU_ensm_valid = evaluate_translation(
                self._gen_args + ['--gen-subset=valid', '--path=' + str(ensemble_model)])
            _, BLEU_last_test = evaluate_translation(
                self._gen_args + ['--gen-subset=test', '--path=' + str(last_model)])
            _, BLEU_ensm_test = evaluate_translation(
                self._gen_args + ['--gen-subset=test', '--path=' + str(ensemble_model)])
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
    encoder = ['D'] * (7 if model == 'DynamicConv' else 6)  # Layers in the encoder
    decoder = ['D'] * 6  # Layers in the decoder
    structure_lr_multiplier = 1.0  # Learning rate multiplier for structured parameters
    ntrials = 3  # Number of trials for hyperparameter tuning
    nmaxupdates = 50000  # Maximum number of updates
    result_dir = project_root + '/transformer/results'  # Directory to store results
    cuda = torch.cuda.is_available()  # Whether to use GPU
    smoke_test = False  # Finish quickly for testing


@ex.capture
def dynamic_conv_experiment(model, model_args, encoder, decoder, structure_lr_multiplier,
                            nmaxupdates, ntrials, result_dir, cuda, smoke_test):
    # name=f"{model}_{model_args}_encoder_[{'-'.join(encoder)}]_decoder_[{'-'.join(decoder)}]_structlr_{structure_lr_multiplier}"
    name=f"{model}_{model_args}_encoder_[{'-'.join(encoder)}]_decoder_[{'-'.join(decoder)}]_structlr_grid"
    config={
        # 'lr': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-4), math.log(1e-3)))),
        # 'lr': grid_search([5e-4, 7e-4, 9e-4, 11e-4]),
        # 'lr': grid_search([1e-4, 2.5e-4, 5e-4, 7.5e-4]),
        'lr': 5e-4,
        # 'weight_decay': sample_from(lambda spec: math.exp(random.uniform(math.log(1e-6), math.log(5e-4)))) if model == 'DynamicConv' else 1e-4,
        'weight_decay': 1e-4,
        # Transformer seems to need dropout 0.3
        # 'dropout': sample_from(lambda spec: random.uniform(0.1, 0.3)) if model == 'DynamicConv' else 0.3,
        'dropout': 0.3,
        'seed': sample_from(lambda spec: random.randint(0, 1 << 16)),
        'encoder': list(encoder),  # Need to copy @encoder as sacred created a read-only list
        'decoder': list(decoder),
        # 'structure-lr-multiplier': structure_lr_multiplier,
        'structure-lr-multiplier': grid_search([0.25, 0.5, 1.0, 2.0, 4.0]),
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
        resources_per_trial={'cpu': 2, 'gpu': 1 if cuda else 0},
        stop={"training_iteration": 1},
        config=config,
    )
    return experiment


@ex.automain
def run(model, encoder, decoder, result_dir):
    experiment = dynamic_conv_experiment()
    try:
        with open('../config/redis_address', 'r') as f:
            address = f.read().strip()
            ray.init(redis_address=address)
    except:
        ray.init()
    trials = ray.tune.run(experiment, raise_on_failed_trial=False, queue_trials=True).trials
    trials = [trial for trial in trials if trial.last_result is not None]
    bleu = [(trial.last_result.get('ensemble_bleu_valid', float('-inf')),
             trial.last_result.get('ensemble_bleu_test', float('-inf'))) for trial in trials]
    max_bleu = max(bleu, key=lambda x: x[0])[1]
    return model, encoder, decoder, max_bleu
