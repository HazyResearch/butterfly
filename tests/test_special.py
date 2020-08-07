import math
import unittest

import numpy as np
from scipy import linalg as la

import torch
from torch import nn
from torch.nn import functional as F

import pywt  # To test wavelet

import torch_butterfly
from torch_butterfly.complex_utils import view_as_real, view_as_complex


class ButterflySpecialTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_fft(self):
        batch_size = 10
        n = 16
        input = torch.randn(batch_size, n, dtype=torch.complex64)
        for normalized in [False, True]:
            out_torch = view_as_complex(torch.fft(view_as_real(input),
                                                  signal_ndim=1, normalized=normalized))
            for br_first in [True, False]:
                b = torch_butterfly.special.fft(n, normalized=normalized, br_first=br_first)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_ifft(self):
        batch_size = 10
        n = 16
        input = torch.randn(batch_size, n, dtype=torch.complex64)
        for normalized in [False, True]:
            out_torch = view_as_complex(torch.ifft(view_as_real(input),
                                                   signal_ndim=1, normalized=normalized))
            for br_first in [True, False]:
                b = torch_butterfly.special.ifft(n, normalized=normalized, br_first=br_first)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_circulant(self):
        batch_size = 10
        n = 13
        for complex in [False, True]:
            dtype = torch.float32 if not complex else torch.complex64
            col = torch.randn(n, dtype=dtype)
            C = la.circulant(col.numpy())
            input = torch.randn(batch_size, n, dtype=dtype)
            out_torch = torch.tensor(input.detach().numpy() @ C.T)
            out_np = torch.tensor(np.fft.ifft(np.fft.fft(input.numpy()) * np.fft.fft(col.numpy())),
                                  dtype=dtype)
            self.assertTrue(torch.allclose(out_torch, out_np, self.rtol, self.atol))
            for separate_diagonal in [True, False]:
                b = torch_butterfly.special.circulant(col, transposed=False,
                                                      separate_diagonal=separate_diagonal)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

            row = torch.randn(n, dtype=dtype)
            C = la.circulant(row.numpy()).T
            input = torch.randn(batch_size, n, dtype=dtype)
            out_torch = torch.tensor(input.detach().numpy() @ C.T)
            # row is the reverse of col, except the 0-th element stays put
            # This corresponds to the same reversal in the frequency domain.
            # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Time_and_frequency_reversal
            row_f = np.fft.fft(row.numpy())
            row_f_reversed = np.hstack((row_f[:1], row_f[1:][::-1]))
            out_np = torch.tensor(np.fft.ifft(np.fft.fft(input.numpy())
                                              * row_f_reversed), dtype=dtype)
            self.assertTrue(torch.allclose(out_torch, out_np, self.rtol, self.atol))
            for separate_diagonal in [True, False]:
                b = torch_butterfly.special.circulant(row, transposed=True,
                                                      separate_diagonal=separate_diagonal)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_hadamard(self):
        batch_size = 10
        n = 16
        H = torch.tensor(la.hadamard(n), dtype=torch.float32)
        input = torch.randn(batch_size, n)
        out_torch = F.linear(input, H) / math.sqrt(n)
        for increasing_stride in [True, False]:
            b = torch_butterfly.special.hadamard(n, normalized=True,
                                                 increasing_stride=increasing_stride)
            out = b(input)
            self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_conv1d_circular_singlechannel(self):
        batch_size = 10
        for n in [13, 16]:
            for kernel_size in [1, 3, 5, 7]:
                conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2,
                                 padding_mode='circular', bias=False)
                weight = conv.weight
                input = torch.randn(batch_size, 1, n)
                out_torch = conv(input)
                for separate_diagonal in [True, False]:
                    b = torch_butterfly.special.conv1d_circular_singlechannel(n, weight,
                                                                              separate_diagonal)
                    out = b(input)
                    self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_conv1d_circular_multichannel(self):
        batch_size = 10
        in_channels = 3
        out_channels = 4
        for n in [13, 16]:
            for kernel_size in [1, 3, 5, 7]:
                conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                 padding=(kernel_size - 1) // 2, padding_mode='circular',
                                 bias=False)
                weight = conv.weight
                input = torch.randn(batch_size, in_channels, n)
                out_torch = conv(input)
                b = torch_butterfly.special.conv1d_circular_multichannel(n, weight)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_fft2d(self):
        batch_size = 10
        n1 = 16
        n2 = 32
        input = torch.randn(batch_size, n2, n1, dtype=torch.complex64)
        for normalized in [False, True]:
            out_torch = view_as_complex(torch.fft(view_as_real(input),
                                                  signal_ndim=2, normalized=normalized))
            for br_first in [True, False]:
                b = torch_butterfly.special.fft2d(n1, n2, normalized=normalized, br_first=br_first)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_ifft2d(self):
        batch_size = 10
        n1 = 16
        n2 = 32
        input = torch.randn(batch_size, n2, n1, dtype=torch.complex64)
        for normalized in [False, True]:
            out_torch = view_as_complex(torch.ifft(view_as_real(input),
                                                   signal_ndim=2, normalized=normalized))
            for br_first in [True, False]:
                b = torch_butterfly.special.ifft2d(
                    n1, n2, normalized=normalized, br_first=br_first)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_conv2d_circular_multichannel(self):
        batch_size = 10
        in_channels = 3
        out_channels = 4
        for n1 in [13, 16]:
            for n2 in [27, 32]:
                for kernel_size1 in [1, 3, 5, 7]:
                    for kernel_size2 in [1, 3, 5, 7]:
                        padding1 = (kernel_size1 - 1) // 2
                        padding2 = (kernel_size2 - 1) // 2
                    conv = nn.Conv2d(in_channels, out_channels, (kernel_size2, kernel_size1),
                                     padding=(padding2, padding1), padding_mode='circular',
                                     bias=False)
                    weight = conv.weight
                    input = torch.randn(batch_size, in_channels, n2, n1)
                    out_torch = conv(input)
                    b = torch_butterfly.special.conv2d_circular_multichannel(n1, n2, weight)
                    out = b(input)
                    self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_wavelet_haar(self):
        batch_size = 10
        n = 32
        input = torch.randn(batch_size, n)
        out_pywt = torch.tensor(np.hstack(pywt.wavedec(input.numpy(), 'haar')))
        b = torch_butterfly.special.wavelet_haar(n)
        out = b(input)
        self.assertTrue(torch.allclose(out, out_pywt, self.rtol, self.atol))


if __name__ == "__main__":
    unittest.main()
