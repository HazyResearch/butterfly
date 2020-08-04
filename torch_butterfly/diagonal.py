import math

import numpy as np

import torch
from torch import nn

from torch_butterfly.complex_utils import complex_mul, real_dtype_to_complex


class Diagonal(nn.Module):

    def __init__(self, size=None, complex=False, diagonal_init=None):
        """Multiply by diagonal matrix
        Parameter:
            size: int
            diagonal_init: (n, )
        """
        super().__init__()
        if diagonal_init is not None:
            self.size = diagonal_init.shape
            self.diagonal = nn.Parameter(diagonal_init.detach().clone())
            self.complex = self.diagonal.is_complex()
        else:
            assert size is not None
            self.size = size
            dtype = torch.get_default_dtype() if not complex else real_dtype_to_complex[torch.get_default_dtype()]
            self.diagonal = nn.Parameter(torch.randn(size, dtype=dtype))
            self.complex = complex
        if self.complex:
            self.diagonal = nn.Parameter(torch.view_as_real(self.diagonal))

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, size)
        Return:
            output: (batch, *, size)
        """
        diagonal = self.diagonal if not self.complex else torch.view_as_complex(self.diagonal)
        return complex_mul(input, diagonal)
