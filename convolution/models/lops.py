import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

import torch_butterfly
from torch_butterfly import Butterfly
from torch_butterfly.complex_utils import ComplexLinear
from torch_butterfly.complex_utils import Real2Complex, Complex2Real
from torch_butterfly.complex_utils import complex_matmul
from torch_butterfly.combine import TensorProduct


class LOP2d(nn.Module):
    """Similar to KOP2d, but we use nn.Linear instead of Butterfly.
    """

    def __init__(self, in_size, in_ch, out_ch, kernel_size, complex=True, init='random'):
        super().__init__()
        self.in_size = in_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.complex = complex
        assert init in ['random', 'fft']
        if init == 'fft':
            assert self.complex, 'fft init requires complex=True'
        self.init = init
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

        linear_cls = nn.Linear if not complex else ComplexLinear
        self.Kd, self.K1, self.K2 = [
            TensorProduct(
                linear_cls(self.in_size[-1], self.in_size[-1], bias=False),
                linear_cls(self.in_size[-2], self.in_size[-2], bias=False)
            )
            for _ in range(3)
        ]
        if init == 'fft':
            eye1 = torch.eye(self.in_size[-1], dtype=torch.complex64)
            eye2 = torch.eye(self.in_size[-2], dtype=torch.complex64)
            # These are symmetric so we don't have to take transpose
            fft_mat1 = torch.fft.fft(eye1, norm='ortho')
            fft_mat2 = torch.fft.fft(eye2, norm='ortho')
            ifft_mat1 = torch.fft.ifft(eye1, norm='ortho')
            ifft_mat2 = torch.fft.ifft(eye2, norm='ortho')
            with torch.no_grad():
                self.Kd.map1.weight.copy_(fft_mat1)
                self.Kd.map2.weight.copy_(fft_mat2)
                self.K1.map1.weight.copy_(fft_mat1)
                self.K1.map2.weight.copy_(fft_mat2)
                self.K2.map1.weight.copy_(ifft_mat1)
                self.K2.map2.weight.copy_(ifft_mat2)
        with torch.no_grad():
            self.Kd.map1.weight *= math.sqrt(self.in_size[-1])
            self.Kd.map2.weight *= math.sqrt(self.in_size[-2])
        self.Kd.map1.weight._is_structured = True
        self.Kd.map2.weight._is_structured = True
        self.K1.map1.weight._is_structured = True
        self.K1.map2.weight._is_structured = True
        self.K2.map1.weight._is_structured = True
        self.K2.map2.weight._is_structured = True

        if complex:
            self.Kd = nn.Sequential(Real2Complex(), self.Kd)
            self.K1 = nn.Sequential(Real2Complex(), self.K1)
            self.K2 = nn.Sequential(self.K2, Complex2Real())

    def forward(self, x):
        w = F.pad(self.weight,
                  (0, self.in_size[-1] - self.kernel_size[-1])).roll(-self.padding[-1], dims=-1)
        w = F.pad(w,
                  (0, 0, 0, self.in_size[-2] - self.kernel_size[-2])).roll(-self.padding[-2],
                                                                           dims=-2)
        # (batch, in_ch, h, w)
        x_f = self.K1(x)
        # (out_ch, in_ch, h, w)
        w_f = self.Kd(w)
        # prod = (x_f.unsqueeze(1) * w_f).sum(dim=2)
        prod = complex_matmul(x_f.permute(2, 3, 0, 1), w_f.permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
        out = self.K2(prod)
        return out


