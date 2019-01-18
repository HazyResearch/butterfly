import copy
import os

import torch
from torch import nn
from torch import optim

from ray.tune import Trainable


N_LBFGS_STEPS_VALIDATION = 15


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
    self.n_steps_per_epoch, self.n_epochs_per_validation, self.target_matrix,
    and self.input in _setup, and may override self.freeze() to freeze model
    (e.g. taking argmax of logit instead of logit).

    """
    def forward(self):
        return self.model(self.input)

    def loss(self):
        # Take transpose since the transform acts on the rows of the input
        output = self.forward().transpose(0, 1)
        if self.target_matrix.dim() == 2 and output.dim() == 3:  # Real target matrix, take real part
            output = output[:, :, 0]
        return nn.functional.mse_loss(output, self.target_matrix)

    def freeze(self):
        pass

    def polish(self, nsteps, save_to_self_model=False):
        if not save_to_self_model:
            model_bak = self.model
            self.model = copy.deepcopy(self.model)
        self.freeze()
        optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()))
        def closure():
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            return loss
        for i in range(nsteps):
            loss = optimizer.step(closure)
        if not save_to_self_model:
            self.model = model_bak
        return loss.item()

    def _train(self):
        for _ in range(self.n_steps_per_epoch):
            self.optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            self.optimizer.step()
        loss = loss.item()
        if (self._iteration + 1) % self.n_epochs_per_validation == 0:
            loss = min(loss, self.polish(N_LBFGS_STEPS_VALIDATION, save_to_self_model=False))
        return {'negative_loss': -loss}
