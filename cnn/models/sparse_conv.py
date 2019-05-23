import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import torch
import torch.nn as nn
import models.resnet_imagenet as imagenet_models # only use imagenet models
import models
import numpy as np

class SparseConv2d(nn.Conv2d):
    def __init__(self, nparams, layer, pretrained, dataset, model, device, **kwargs):
        # create conv2d layer
        super().__init__(**kwargs)
        self.nparams = nparams

        # load pretrained model and get layer
        if dataset == 'imagenet':
            pretrained_model = imagenet_models.__dict__[model]()
        elif dataset == 'cifar10':
            pretrained_model = models.__dict__[model]()

        # create conv2d layer and copy weights
        pretrained_weight = self._get_pretrained_weight(pretrained_model, layer)
        self.weight.data.copy_(pretrained_weight)

        # create mask
        self.mask = self._sparse_projection().to(device)

    def _get_pretrained_weight(self, pretrained_model, layer):
        weight = None
        for name, param in pretrained_model.named_parameters():
            if layer in name:
                weight = param
        assert weight is not None, "Could not find weight for layer"
        return weight

    def _sparse_projection(self):
        flat_weight = self.weight.flatten().abs()
        mask = torch.zeros_like(self.weight)
        flat_mask = mask.view(-1)
        # sort indices after taking absolute values
        flat_weight_indices = flat_weight.argsort(descending=True)[:self.nparams]
        # assign large magnitudes to 1
        flat_mask[flat_weight_indices] = 1
        return mask

    def forward(self, x):
        # sparsify weights
        self.weight.data = self.weight * self.mask
        output = super().forward(x)
        return output
