from pathlib import Path

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback


def pl_train(cfg, pl_module_cls, **kwargs):
    trainer_args = dict(
        gpus=1,
        max_epochs=1 if cfg.smoke_test else cfg.train.epochs,
        checkpoint_callback=False,  # Disable checkpointing to save disk space
        progress_bar_refresh_rate=1,
        **cfg.train.pltrainer,
    )
    trainer_args.update(kwargs)
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)
    model = pl_module_cls(cfg.model, cfg.dataset, cfg.train)
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, model.datamodule)

    if 'save_checkpoint_path' in cfg.train:
        path = cfg.train.save_checkpoint_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(path))

    if cfg.train.run_test:
        trainer.test(model, model.datamodule)

    return trainer, model
