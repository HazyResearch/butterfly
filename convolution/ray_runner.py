from pathlib import Path
project_root = Path(__file__).parent.absolute()

import os
import random
import math
from collections.abc import Sequence
from functools import partial

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from munch import Munch
from omegaconf.listconfig import ListConfig

import ray
from ray import tune
from ray.tune import Trainable, Experiment, SyncConfig, sample_from, grid_search
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.wandb import WandbLogger

from pl_runner import pl_train
from utils import remove_postfix, dictconfig_to_munch, munch_to_dictconfig


HYDRA_TUNE_KEYS = ['_grid', '_sample', '_sample_uniform', '_sample_log_uniform']


def munchconfig_to_tune_munchconfig(cfg):
    """Convert config to one compatible with Ray Tune.
    Entry as list whose first element is "_grid" is converted to ray.tune.grid_search.
    "_sample" is converted to ray.tune.sample_from.
    "_sample_uniform" is converted to ray.tune.sample_from with uniform distribution [min, max).
    "_sample_log_uniform" is converted to ray.tune.sample_from with uniform distribution
        exp(uniform(log(min), log(max)))
    Examples:
        lr=1e-3 for a specific learning rate
        lr=[_grid, 1e-3, 1e-4, 1e-5] means grid search over those values
        lr=[_sample, 1e-3, 1e-4, 1e-5] means randomly sample from those values
        lr=[_sample_uniform, 1e-4, 3e-4]  means randomly sample from those min/max
        lr=[_sample_log_uniform, 1e-4, 1e-3]  means randomly sample from those min/max but
            distribution is log uniform: exp(uniform(log 1e-4, log 1e-3))
    """
    def convert_value(v):
        # The type is omegaconf.listconfig.ListConfig and not list, so we test if it's a Sequence
        # In hydra 0.11, more precisely omegaconf 1.4.1, ListConfig isn't an instance of Sequence.
        # So we have to test it directly.
        if not (isinstance(v, (Sequence, ListConfig)) and len(v) > 0 and v[0] in HYDRA_TUNE_KEYS):
            return v
        else:
            if v[0] == '_grid':
                # grid_search requires list for some reason
                return grid_search(list(v[1:]))
            elif v[0] == '_sample':
                # Python's lambda doesn't capture the object, it only captures the variable name
                # So we need extra argument to capture the object
                # https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
                # https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
                # Switching back to not capturing variable since (i) ray 1.0 doesn't like that
                # (ii) v isn't changing in this scope
                return sample_from(lambda _: random.choice(v[1:]))
            elif v[0] == '_sample_uniform':
                min_, max_ = v[1:]
                if isinstance(min_, int) and isinstance(max_, int):
                    return sample_from(lambda _: random.randint(min_, max_))
                else:
                    return sample_from(lambda _: random.uniform(min_, max_))
            elif v[0] == '_sample_log_uniform':
                min_, max_ = v[1:]
                return sample_from(lambda _: math.exp(random.uniform(math.log(min_), math.log(max_))))
            else:
                assert False

    def convert(cfg):
        return Munch({k: convert(v) if isinstance(v, Munch) else
                      convert_value(v) for k, v in cfg.items()})

    return convert(cfg)


class TuneReportCheckpointCallback(Callback):
    # We group train and val reporting into one, otherwise tune thinks there're 2 different epochs.
    def on_epoch_end(self, trainer, pl_module):
        results = {remove_postfix(k, '_epoch'): v for k, v in trainer.logged_metrics.items()
                   if (k.startswith('train_') or k.startswith('val_')) and not k.endswith('_step')}
        results['mean_loss'] = results.get('val_loss', results['train_loss'])
        if 'val_accuracy' in results:
            results['mean_accuracy'] = results['val_accuracy']
        # Checkpointing should be done *before* reporting
        # https://docs.ray.io/en/master/tune/api_docs/trainable.html
        with tune.checkpoint_dir(step=trainer.current_epoch) as checkpoint_dir:
            trainer.save_checkpoint(os.path.join(checkpoint_dir,
                                                 f"{type(pl_module).__name__}.ckpt"))
        tune.report(**results)

    def on_test_epoch_end(self, trainer, pl_module):
        results = {remove_postfix(k, '_epoch'): v for k, v in trainer.logged_metrics.items()
                   if k.startswith('test_') and not k.endswith('_step')}
        tune.report(**results)


def pl_train_with_tune(cfg, pl_module_cls, checkpoint_dir=None):
    cfg = munch_to_dictconfig(Munch(cfg))
    checkpoint_path = (None if not checkpoint_dir
                       else os.path.join(checkpoint_dir, f"{pl_module_cls.__name__}.ckpt"))
    trainer_extra_args = dict(
        gpus=1 if cfg.gpu else None,
        progress_bar_refresh_rate=0,
        resume_from_checkpoint=checkpoint_path,
        callbacks=[TuneReportCheckpointCallback()]
    )
    pl_train(cfg, pl_module_cls, **trainer_extra_args)


def ray_train(cfg, pl_module_cls):
    # We need Munch to hold tune functions. DictConfig can only hold static config.
    cfg = munchconfig_to_tune_munchconfig(dictconfig_to_munch(cfg))
    ray_config={
        'model': cfg.model,
        'dataset': cfg.dataset,
        'train': cfg.train,
        'seed': cfg.seed,
        'wandb': cfg.wandb,
        'gpu': cfg.runner.gpu_per_trial != 0.0,
    }
    dataset_str = cfg.dataset._target_.split('.')[-1]
    model_str = cfg.model._target_.split('.')[-1]
    args_str = '_'
    # If we're writing to dfs or efs already, no need to sync explicitly
    # This needs to be a noop function, not just False. If False, ray won't restore failed spot instances
    sync_to_driver = None if not cfg.runner.nfs else lambda source, target: None
    experiment = Experiment(
        name=f'{dataset_str}_{model_str}',
        run=partial(pl_train_with_tune, pl_module_cls=pl_module_cls),
        local_dir=cfg.runner.result_dir,
        num_samples=cfg.runner.ntrials if not cfg.smoke_test else 1,
        resources_per_trial={'cpu': 1 + cfg.dataset.num_workers, 'gpu': cfg.runner.gpu_per_trial},
        # epochs + 1 because calling trainer.test(model) counts as one epoch
        stop={"training_iteration": 1 if cfg.smoke_test else cfg.train.epochs + 1},
        config=ray_config,
        loggers=[WandbLogger],
        keep_checkpoints_num=1,  # Save disk space, just need 1 for recovery
        # checkpoint_at_end=True,
        # checkpoint_freq=1000,  # Just to enable recovery with @max_failures
        max_failures=-1,
        sync_to_driver=sync_to_driver,  # As of Ray 1.0.0, still need this here
    )

    if cfg.smoke_test or cfg.runner.local:
        ray.init(num_gpus=torch.cuda.device_count())
    else:
        try:
            ray.init(address='auto')
        except:
            try:
                with open(project_root / 'ray_config/redis_address', 'r') as f:
                    address = f.read().strip()
                with open(project_root / 'ray_config/redis_password', 'r') as f:
                    password = f.read().strip()
                    ray.init(address=address, _redis_password=password)
            except:
                ray.init(num_gpus=torch.cuda.device_count())
                import warnings
                warnings.warn("Running Ray with just one node")

    if cfg.runner.hyperband:
        scheduler = AsyncHyperBandScheduler(metric='mean_accuracy', mode='max',
                                            max_t=cfg.train.epochs + 1,
                                            grace_period=cfg.runner.grace_period)
    else:
        scheduler = None
    trials = ray.tune.run(experiment,
                          scheduler=scheduler,
                          # sync_config=SyncConfig(sync_to_driver=sync_to_driver),
                          raise_on_failed_trial=False,
                          queue_trials=True)
    return trials
