import math
import os

import numpy as np

import torch
from ray.tune import Trainable


class PytorchTrainable(Trainable):
    """Abstract Trainable class for Pytorch models, which checkpoints the model
    and the optimizer.
    Subclass must initialize self.model and self.optimizer in _setup.
    """

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model_optimizer.pth")
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def bitreversal_permutation(n):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return perm.squeeze(0)
