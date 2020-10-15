import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_butterfly
from torch_butterfly import Butterfly
from torch_butterfly.complex_utils import real2complex, complex2real
from torch_butterfly.complex_utils import complex_mul, complex_matmul
from torch_butterfly.combine import TensorProduct


class KOP2d(nn.Module):

    def __init__(self, in_size, in_ch, out_ch, kernel_size, nblocks=1):
        super().__init__()
        self.in_size = in_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.nblocks = nblocks
        if isinstance(self.in_size, int):
            self.in_size = (self.in_size, self.in_size)
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        # Just to use nn.Conv2d's initialization
        self.weight = nn.Parameter(nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size,
                                             padding=padding, bias=False).weight)

        self.Kd = TensorProduct(
            Butterfly(self.in_size[0], self.in_size[0], bias=False,
                increasing_stride=False, complex=True, init='ortho', nblocks=nblocks),
            Butterfly(self.in_size[1], self.in_size[1], bias=False,
                increasing_stride=False, complex=True, init='ortho', nblocks=nblocks)
        )
        self.K1 = TensorProduct(
            Butterfly(self.in_size[0], self.in_size[0], bias=False,
                increasing_stride=False, complex=True, init='ortho', nblocks=nblocks),
            Butterfly(self.in_size[1], self.in_size[1], bias=False,
                increasing_stride=False, complex=True, init='ortho', nblocks=nblocks)
        )
        self.K2 = TensorProduct(
            Butterfly(self.in_size[0], self.in_size[0], bias=False,
                increasing_stride=True, complex=True, init='ortho', nblocks=nblocks),
            Butterfly(self.in_size[1], self.in_size[1], bias=False,
                increasing_stride=True, complex=True, init='ortho', nblocks=nblocks)
        )

    def forward(self, x):
        # (batch, in_ch, h, w)
        x_f = self.K1(real2complex(x))
        # (out_ch, in_ch, h, w)
        w_f = self.Kd(real2complex(self.weight)) * math.sqrt(self.in_size[0] * self.in_size[1])
        # prod = complex_mul(x_f.unsqueeze(1), w_f).sum(dim=2)
        prod = complex_matmul(x_f.permute(2, 3, 0, 1), w_f.permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
        out = complex2real(self.K2(prod))
        return out
