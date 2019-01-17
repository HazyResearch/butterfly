import math
import os

import numpy as np

import torch
from torch import nn
from torch import optim

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


class TrainableFixedData(PytorchTrainable):
    """Abstract Trainable class for Pytorch models with fixed data.
    Subclass must initialize self.model, self.optimizer, and
    self.n_steps_per_epoch in _setup, and have to implement self.loss().
    """
    def loss(self):
        raise NotImplementedError

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            self.optimizer.step()
        return {'negative_loss': -loss.item()}


class TrainableMatrixFactorization(TrainableFixedData):
    """Abstract Trainable class for Pytorch models that factor a target matrix.
    Subclass must initialize self.model, self.optimizer,
    self.n_steps_per_epoch, self.target_matrix, and self.input in _setup, and
    may override self.freeze() to freeze model (e.g. taking argmax of logit
    instead of logit).
    """
    def forward(self):
        return self.model(self.input)

    def loss(self):
        output = self.forward()
        if self.target_matrix.dim() == 2 and output.dim() == 3:  # Real target matrix, take real part
            output = output[:, :, 0]
        return nn.functional.mse_loss(output, self.target_matrix)

    def freeze(self):
        pass

    def polish(self, nsteps):
        self.freeze()
        optimizer = optim.LBFGS(self.model.parameters())
        def closure():
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            return loss
        for i in range(nsteps):
            loss = optimizer.step(closure)
        return loss.item()


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
