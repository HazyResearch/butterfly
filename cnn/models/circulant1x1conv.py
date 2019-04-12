import math
import torch
from torch import nn

from butterfly.complex_utils import complex_mul


class CirculantLinear(nn.Module):

    def __init__(self, size, nstack=1):
        super().__init__()
        self.size = size
        self.nstack = nstack
        init_stddev = math.sqrt(1. / self.size)
        c = torch.randn(nstack, size) * init_stddev
        self.c_f = nn.Parameter(torch.rfft(c, 1))

    def forward(self, input):
        """
        Parameters:
            input: (batch, size)
        Return:
            output: (batch, nstack * size)
        """
        batch = input.shape[0]
        input_f = torch.rfft(input, 1)
        prod = complex_mul(self.c_f, input_f.unsqueeze(1))
        return torch.irfft(prod, 1, signal_sizes=(self.size, )).view(batch, self.nstack * self.size)


class Circulant1x1Conv(CirculantLinear):

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w)
        Return:
            output: (batch, nstack * c, h, w)
        """
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)
