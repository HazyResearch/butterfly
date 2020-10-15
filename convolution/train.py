from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.absolute()
import os
# Add to $PYTHONPATH so that ray workers can see
os.environ['PYTHONPATH'] = str(PROJECT_ROOT) + ":" + os.environ.get('PYTHONPATH', '')

import torch
import pytorch_lightning as pl

import hydra
from omegaconf import OmegaConf

import models
import datamodules
import tasks
from pl_runner import pl_train
from utils import to_scalar, dictconfig_to_munch
from tee import StdoutTee, StderrTee


class LightningModel(pl.LightningModule):

    def __init__(self, model_cfg, dataset_cfg, train_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.datamodule = hydra.utils.instantiate(dataset_cfg, batch_size=train_cfg.batch_size)
        self.model = hydra.utils.instantiate(model_cfg, num_classes=self.datamodule.num_classes)
        self.task = hydra.utils.instantiate(self.train_cfg.task)

    def forward(self, input):
        return self.model.forward(input)

    def training_step(self, batch, batch_idx, prefix='train'):
        batch_x, batch_y = batch
        out = self.forward(batch_x)
        loss = self.task.loss(out, batch_y)
        metrics = self.task.metrics(out, batch_y)
        metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
        self.log(f'{prefix}_loss', loss, on_epoch=True, prog_bar=False)
        self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, prefix='test')

    def configure_optimizers(self):
        # Very important that the twiddle factors shouldn't have weight decay
        structured_params = filter(lambda p: getattr(p, '_is_structured', False),
                                   self.model.parameters())
        unstructured_params = filter(lambda p: not getattr(p, '_is_structured', False),
                                     self.model.parameters())
        params_dict = [{'params': structured_params, 'weight_decay': 0.0},
                       {'params': unstructured_params}]
        optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, params_dict)
        if 'lr_scheduler' not in self.train_cfg:
            return optimizer
        else:
            lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler, optimizer)
            return [optimizer], [lr_scheduler]


@hydra.main(config_path="cfg", config_name="config.yaml")
def main(cfg: OmegaConf):
    with StdoutTee('train.stdout'), StderrTee('train.stderr'):
        print(OmegaConf.to_yaml(cfg))
        if cfg.runner.name == 'pl':
            trainer, model = pl_train(cfg, LightningModel)
        else:
            assert cfg.runner.name == 'ray', 'Only pl and ray runners are supported'
            # Shouldn't need to install ray unless doing distributed training
            from ray_runner import ray_train
            ray_train(cfg, LightningModel)


if __name__ == "__main__":
    main()
