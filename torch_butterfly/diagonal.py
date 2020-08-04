import math

import numpy as np

import torch
from torch import nn

from torch_butterfly.complex_utils import complex_mul


class Diagonal(nn.Module):

    def __init__(self, size, diagonal_init=None):
        """Multiply by diagonal matrix
        Parameter:
            size: int
            diagonal_init: (n, )
        """
        super().__init__()
        self.size = size
        if diagonal_init is not None:
            self.diagonal = nn.Parameter(diagonal_init.detach().clone())
            self.complex = self.diagonal.is_complex()
            if self.complex:
                self.diagonal = nn.Parameter(torch.view_as_real(self.diagonal))
        else:
            init_stddev = math.sqrt(1. / self.size)
            self.diagonal = nn.Parameter(torch.randn(size) * init_stddev)
            self.complex = False

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, size)
        Return:
            output: (batch, *, size)
        """
        diagonal = self.diagonal if not self.complex else torch.view_as_complex(self.diagonal)
        return complex_mul(input, diagonal)
