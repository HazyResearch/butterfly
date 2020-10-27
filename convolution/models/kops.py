import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_butterfly
from torch_butterfly import Butterfly
from torch_butterfly.complex_utils import Real2Complex, Complex2Real
from torch_butterfly.complex_utils import complex_mul, complex_matmul
from torch_butterfly.combine import TensorProduct


class KOP2d(nn.Module):

    def __init__(self, in_size, in_ch, out_ch, kernel_size, complex=True, init='ortho', nblocks=1,
                 base=2, zero_pad=True):
        super().__init__()
        self.in_size = in_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.complex = complex
        assert init in ['ortho', 'fft']
        if init == 'fft':
            assert self.complex, 'fft init requires complex=True'
        self.init = init
        self.nblocks = nblocks
        assert base in [2, 4]
        self.base = base
        self.zero_pad = zero_pad
        if isinstance(self.in_size, int):
            self.in_size = (self.in_size, self.in_size)
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        self.padding = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        # Just to use nn.Conv2d's initialization
        self.weight = nn.Parameter(nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size,
                                             padding=self.padding, bias=False).weight)

        increasing_strides = [False, False, True]
        inits = ['ortho'] * 3 if self.init == 'ortho' else ['fft_no_br', 'fft_no_br', 'ifft_no_br']
        self.Kd, self.K1, self.K2 = [
            TensorProduct(
                Butterfly(self.in_size[-1], self.in_size[-1], bias=False, complex=complex,
                    increasing_stride=incstride, init=i, nblocks=nblocks),
                Butterfly(self.in_size[-2], self.in_size[-2], bias=False, complex=complex,
                    increasing_stride=incstride, init=i, nblocks=nblocks)
            )
            for incstride, i in zip(increasing_strides, inits)
        ]
        with torch.no_grad():
            self.Kd.map1 *= math.sqrt(self.in_size[-1])
            self.Kd.map2 *= math.sqrt(self.in_size[-2])

        if base == 4:
            self.Kd.map1, self.Kd.map2 = self.Kd.map1.to_base4(), self.Kd.map2.to_base4()
            self.K1.map1, self.K1.map2 = self.K1.map1.to_base4(), self.K1.map2.to_base4()
            self.K2.map1, self.K2.map2 = self.K2.map1.to_base4(), self.K2.map2.to_base4()

        if complex:
            self.Kd = nn.Sequential(Real2Complex(), self.Kd)
            self.K1 = nn.Sequential(Real2Complex(), self.K1)
            self.K2 = nn.Sequential(self.K2, Complex2Real())

    def forward(self, x):
        if self.zero_pad:
            w = F.pad(self.weight.flip([-1]),
                        (0, self.in_size[-1] - self.kernel_size[-1])).roll(-self.padding[-1],
                                                                           dims=-1)
            w = F.pad(w.flip([-2]),
                    (0, 0, 0, self.in_size[-2] - self.kernel_size[-2])).roll(-self.padding[-2],
                                                                            dims=-2)
        else:
            w = self.weight
        # (batch, in_ch, h, w)
        x_f = self.K1(x)
        # (out_ch, in_ch, h, w)
        # w_f = self.Kd(self.weight) * math.sqrt(self.in_size[0] * self.in_size[1])
        # w_f = self.Kd(self.weight)
        w_f = self.Kd(w)
        # prod = complex_mul(x_f.unsqueeze(1), w_f).sum(dim=2)
        prod = complex_matmul(x_f.permute(2, 3, 0, 1), w_f.permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
        out = self.K2(prod)
        return out
