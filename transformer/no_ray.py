import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(1, project_root + '/fairseq')
sys.path.insert(2, project_root + '/fairseq/scripts')
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + project_root + '/fairseq:' + project_root + '/fairseq/scripts' + os.environ.get('PYTHONPATH', '')

from pathlib import Path
import random
import json
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

import re
import subprocess

import ray
from ray.tune import Trainable

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
        train_args += ['--clip-norm', '0'] if 'DynamicConv' in model['name'] else []
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
        if model['name'] == 'DynamicConv':
            train_args += ['-a', 'lightconv_butterfly_iwslt_de_en']
        elif model['name'] == 'DynamicConvBasic':
            train_args += ['-a', 'lightconv_iwslt_de_en']
        elif model['name'] == 'TransformerBasic':
            train_args += ['-a', 'transformer_iwslt_de_en']
        elif model['name'] == 'Transformer':
            train_args += ['-a', 'transformer_butterfly_iwslt_de_en']
        train_args += ['--dropout', str(config['dropout'])]
        train_args += ['--attention-dropout', '0.1'] if 'DynamicConv' in model['name'] else []
        train_args += ['--weight-dropout', '0.1'] if 'DynamicConv' in model['name'] else []
        train_args += ['--encoder-glu', '0'] if 'DynamicConv' in model['name'] else []
        train_args += ['--decoder-glu', '0 '] if 'DynamicConv' in model['name'] else []
        train_args += ['--seed', str(config['seed'])]
        self._save_dir = Path(config['result_dir']) / f"structlr={config['structure-lr-multiplier']}_seed={config['seed']}"
        train_args += ['--save-dir', str(self._save_dir)]
        train_args += ['--encoder-layers', str(len(config['encoder']))]
        train_args += ['--decoder-layers', str(len(config['decoder']))]
        train_args += ['--encoder-structure-type-list', str(config['encoder'])] if model['name'] == 'Transformer' else []
        train_args += ['--decoder-structure-type-list', str(config['decoder'])] if model['name'] == 'Transformer' else []
        train_args += ['--structure-lr-multiplier', str(config['structure-lr-multiplier'])]
        if config['density'] < 1.0:
            train_args += ['--sparse', '--density', str(config['density']), '--redistribution', 'none', '--verbose']
            if 'DynamicConv' not in model['name']: train_args += ['--force-qkv-separate']
        print(f'save_dir: {self._save_dir}')

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
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def _train(self):
        stdout = sys.stdout
        with open(self._save_dir / 'logs.txt', 'w+') as log:
            sys.stdout = log
            train_translation(self._train_args)
            subprocess.run([sys.executable, str(Path(project_root) / 'fairseq/average_checkpoints.py')] + self._avg_args, stdout=log)

            last_model = self._save_dir / 'checkpoint_last.pt'
            best_model = self._save_dir / 'checkpoint_best.pt'
            ensemble_model = self._save_dir / 'model.pt'
            # Delete checkpoints to save disk space
            for ckpt_file in Path(self._save_dir).glob('*.pt'):
                if ckpt_file != last_model and ckpt_file != ensemble_model \
                                        and ckpt_file != best_model:
                    ckpt_file.unlink()
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


class default_config:
  def __init__(self, model_name='TransformerBasic'):
    self.model = model_name
    self.model_args = {}  # Arguments to be passed to the model, as a dictionary
    self.encoder = ['D'] * (7 if 'DynamicConv' in model_name else 6)
    self.decoder = ['D'] * 6  # Layers in the decoder
    self.density = 1.0  # if less than 1.0, use sparse
    self.structure_lr_multiplier = 1.0  # Learning rate multiplier for structured parameters
    self.nmaxupdates = 50000  # Maximum number of updates
    self.result_dir = project_root + '/transformer/results'  # Directory to store results
    self.cuda = torch.cuda.is_available()  # Whether to use GPU
    self.smoke_test = False  # Finish quickly for testing

def main():
    config = default_config(model_name=sys.argv[1].split('=')[1]).__dict__
    config['density'] = float(sys.argv[2].split('=')[1])
    config['structure-lr-multiplier'] = float(sys.argv[3].split('=')[1])
    c2 = {
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'dropout': 0.3,
        'device': 'cuda' if config['cuda'] else 'cpu',
        'model': {'name': config['model'], 'args': config['model_args']},
        'result_dir': config['result_dir'] + '/' + f"{config['model']}_{config['density']}"
    }
    config.update(**c2)

    trainable_cls = TrainableModel
    trainable = trainable_cls.__new__(trainable_cls)
    config.update(seed=random.randint(0, 1<<16))
    trainable._setup(config)
    savedir = trainable._save_dir
    with open(savedir / 'params.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    result = trainable._train()
    with open(savedir / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
