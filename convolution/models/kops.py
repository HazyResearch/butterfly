import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_butterfly
from torch_butterfly import Butterfly
from torch_butterfly.complex_utils import Real2Complex, Complex2Real
from torch_butterfly.complex_utils import complex_matmul
from torch_butterfly.combine import TensorProduct
from torch_butterfly.permutation import bitreversal_permutation


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
        self.weight = nn.Parameter(
            nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size, padding=self.padding,
                      bias=False).weight.flip([-1, -2])
        )

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
        if self.zero_pad:
            # Instead of zero-padding and calling weight.roll(-self.padding[-1], dims=-1) and
            # weight.roll(-self.padding[-2], dims=-2), we multiply self.Kd by complex exponential
            # instead, using the Shift theorem.
            # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Shift_theorem
            with torch.no_grad():
                n1, n2 = self.Kd.map1.n, self.Kd.map2.n
                device = self.Kd.map1.twiddle.device
                br1 = bitreversal_permutation(n1, pytorch_format=True).to(device)
                br2 = bitreversal_permutation(n2, pytorch_format=True).to(device)
                diagonal1 = torch.exp(1j * 2 * math.pi / n1 * self.padding[-1]
                                      * torch.arange(n1, device=device))[br1]
                diagonal2 = torch.exp(1j * 2 * math.pi / n2 * self.padding[-2]
                                      * torch.arange(n2, device=device))[br2]
                # We multiply the 1st block instead of the last block (only the first block is not
                # the identity if init=fft). This seems to perform a tiny bit better.
                # If init=ortho, this won't correspond exactly to rolling the weight.
                self.Kd.map1.twiddle[:, 0, -1, :, 0, :] *= diagonal1[::2].unsqueeze(-1)
                self.Kd.map1.twiddle[:, 0, -1, :, 1, :] *= diagonal1[1::2].unsqueeze(-1)
                self.Kd.map2.twiddle[:, 0, -1, :, 0, :] *= diagonal2[::2].unsqueeze(-1)
                self.Kd.map2.twiddle[:, 0, -1, :, 1, :] *= diagonal2[1::2].unsqueeze(-1)

        if base == 4:
            self.Kd.map1, self.Kd.map2 = self.Kd.map1.to_base4(), self.Kd.map2.to_base4()
            self.K1.map1, self.K1.map2 = self.K1.map1.to_base4(), self.K1.map2.to_base4()
            self.K2.map1, self.K2.map2 = self.K2.map1.to_base4(), self.K2.map2.to_base4()

        if complex:
            self.Kd = nn.Sequential(Real2Complex(), self.Kd)
            self.K1 = nn.Sequential(Real2Complex(), self.K1)
            self.K2 = nn.Sequential(self.K2, Complex2Real())

    def forward(self, x):
        # (batch, in_ch, h, w)
        x_f = self.K1(x)
        # (out_ch, in_ch, h, w)
        # w_f = self.Kd(self.weight) * math.sqrt(self.in_size[0] * self.in_size[1])
        # w_f = self.Kd(self.weight)
        w_f = self.Kd(self.weight)
        # prod = (x_f.unsqueeze(1) * w_f).sum(dim=2)
        prod = complex_matmul(x_f.permute(2, 3, 0, 1), w_f.permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
        out = self.K2(prod)
        return out
