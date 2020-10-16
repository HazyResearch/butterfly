import math
import unittest

import numpy as np
from scipy import linalg as la
import scipy.fft

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

    def test_fft_unitary(self):
        batch_size = 10
        n = 16
        input = torch.randn(batch_size, n, dtype=torch.complex64)
        normalized = True
        out_torch = view_as_complex(torch.fft(view_as_real(input),
                                              signal_ndim=1, normalized=normalized))
        for br_first in [True, False]:
            b = torch_butterfly.special.fft_unitary(n, br_first=br_first)
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

    def test_ifft_unitary(self):
        batch_size = 10
        n = 16
        input = torch.randn(batch_size, n, dtype=torch.complex64)
        normalized = True
        out_torch = view_as_complex(torch.ifft(view_as_real(input),
                                               signal_ndim=1, normalized=normalized))
        for br_first in [True, False]:
            b = torch_butterfly.special.ifft_unitary(n, br_first=br_first)
            out = b(input)
            self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_dct(self):
        batch_size = 10
        n = 16
        input = torch.randn(batch_size, n)
        for type in [2, 3, 4]:
            for normalized in [False, True]:
                out_sp = torch.tensor(scipy.fft.dct(input.numpy(), type=type,
                                                    norm=None if not normalized else 'ortho'))
                b = torch_butterfly.special.dct(n, type=type, normalized=normalized)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_sp, self.rtol, self.atol))

    def test_dst(self):
        batch_size = 1
        n = 16
        input = torch.randn(batch_size, n)
        for type in [2, 4]:
            for normalized in [False, True]:
                out_sp = torch.tensor(scipy.fft.dst(input.numpy(), type=type,
                                                    norm=None if not normalized else 'ortho'))
                b = torch_butterfly.special.dst(n, type=type, normalized=normalized)
                out = b(input)
                self.assertTrue(torch.allclose(out, out_sp, self.rtol, self.atol))

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
            # Just to show how to implement circulant multiply with FFT
            if complex:
                input_f = view_as_complex(torch.fft(view_as_real(input), signal_ndim=1))
                col_f = view_as_complex(torch.fft(view_as_real(col), signal_ndim=1))
                prod_f = input_f * col_f
                out_fft = view_as_complex(torch.ifft(view_as_real(prod_f), signal_ndim=1))
                self.assertTrue(torch.allclose(out_torch, out_fft, self.rtol, self.atol))
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

    def test_toeplitz(self):
        batch_size = 10
        for n, m in [(13, 38), (27, 11)]:
            for complex in [False, True]:
                dtype = torch.float32 if not complex else torch.complex64
                col = torch.randn(n, dtype=dtype)
                row = torch.randn(m, dtype=dtype)
                T = la.toeplitz(col.numpy(), row.numpy())
                input = torch.randn(batch_size, m, dtype=dtype)
                out_torch = torch.tensor(input.detach().numpy() @ T.T)
                for separate_diagonal in [True, False]:
                    b = torch_butterfly.special.toeplitz(col, row,
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

    def test_hadamard_diagonal(self):
        batch_size = 10
        n = 16
        H = torch.tensor(la.hadamard(n), dtype=torch.float32) / math.sqrt(n)
        for k in [1, 2, 3]:
            diagonals = torch.randint(0, 2, (k, n)) * 2 - 1.0
            input = torch.randn(batch_size, n)
            out_torch = input
            for diagonal in diagonals.unbind():
                out_torch = F.linear(out_torch * diagonal, H)
            for increasing_stride in [True, False]:
                for separate_diagonal in [True, False]:
                    b = torch_butterfly.special.hadamard_diagonal(
                        diagonals, normalized=True, increasing_stride=increasing_stride,
                        separate_diagonal=separate_diagonal
                    )
                    out = b(input)
                    self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_conv1d_circular_singlechannel(self):
        batch_size = 10
        for n in [13, 16]:
            for kernel_size in [1, 3, 5, 7]:
                padding = (kernel_size - 1) // 2
                conv = nn.Conv1d(1, 1, kernel_size, padding=padding, padding_mode='circular',
                                 bias=False)
                weight = conv.weight
                input = torch.randn(batch_size, 1, n)
                out_torch = conv(input)
                # Just to show how to implement conv1d with FFT
                input_f = view_as_complex(torch.rfft(input, signal_ndim=1))
                col = F.pad(weight.flip(dims=(-1,)), (0, n - kernel_size)).roll(-padding, dims=-1)
                col_f = view_as_complex(torch.rfft(col, signal_ndim=1))
                prod_f = input_f * col_f
                out_fft = torch.irfft(view_as_real(prod_f), signal_ndim=1, signal_sizes=(n,))
                self.assertTrue(torch.allclose(out_torch, out_fft, self.rtol, self.atol))
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
                padding = (kernel_size - 1) // 2
                conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding,
                                 padding_mode='circular', bias=False)
                weight = conv.weight
                input = torch.randn(batch_size, in_channels, n)
                out_torch = conv(input)
                # Just to show how to implement conv1d with FFT
                input_f = view_as_complex(torch.rfft(input, signal_ndim=1))
                col = F.pad(weight.flip(dims=(-1,)), (0, n - kernel_size)).roll(-padding, dims=-1)
                col_f = view_as_complex(torch.rfft(col, signal_ndim=1))
                prod_f = (input_f.unsqueeze(1) * col_f).sum(dim=2)
                out_fft = torch.irfft(view_as_real(prod_f), signal_ndim=1, signal_sizes=(n,))
                self.assertTrue(torch.allclose(out_torch, out_fft, self.rtol, self.atol))
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
            # Just to show how fft2d is exactly 2 ffts on each dimension
            input_f = view_as_complex(torch.fft(view_as_real(input), signal_ndim=1,
                                                normalized=normalized))
            out_fft = view_as_complex(
                torch.fft(view_as_real(input_f.transpose(-1, -2)),
                          signal_ndim=1, normalized=normalized)).transpose(-1, -2)
            self.assertTrue(torch.allclose(out_torch, out_fft, self.rtol, self.atol))
            for br_first in [True, False]:
                for flatten in [False, True]:
                    b = torch_butterfly.special.fft2d(n1, n2, normalized=normalized,
                                                      br_first=br_first, flatten=flatten)
                    out = b(input)
                    self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_fft2d_unitary(self):
        batch_size = 10
        n1 = 16
        n2 = 32
        input = torch.randn(batch_size, n2, n1, dtype=torch.complex64)
        normalized = True
        out_torch = view_as_complex(torch.fft(view_as_real(input),
                                              signal_ndim=2, normalized=normalized))
        for br_first in [True, False]:
            b = torch_butterfly.special.fft2d_unitary(n1, n2, br_first=br_first)
            out = b(input)
            self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_ifft2d(self):
        batch_size = 10
        n1 = 32
        n2 = 16
        input = torch.randn(batch_size, n2, n1, dtype=torch.complex64)
        for normalized in [False, True]:
            out_torch = view_as_complex(torch.ifft(view_as_real(input),
                                                   signal_ndim=2, normalized=normalized))
            # Just to show how ifft2d is exactly 2 iffts on each dimension
            input_f = view_as_complex(torch.ifft(view_as_real(input), signal_ndim=1,
                                                 normalized=normalized))
            out_fft = view_as_complex(
                torch.ifft(view_as_real(input_f.transpose(-1, -2)),
                           signal_ndim=1, normalized=normalized)).transpose(-1, -2)
            self.assertTrue(torch.allclose(out_torch, out_fft, self.rtol, self.atol))
            for br_first in [True, False]:
                for flatten in [False, True]:
                    b = torch_butterfly.special.ifft2d(n1, n2, normalized=normalized,
                                                       br_first=br_first, flatten=flatten)
                    out = b(input)
                    self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_ifft2d_unitary(self):
        batch_size = 10
        n1 = 16
        n2 = 32
        input = torch.randn(batch_size, n2, n1, dtype=torch.complex64)
        normalized = True
        out_torch = view_as_complex(torch.ifft(view_as_real(input),
                                               signal_ndim=2, normalized=normalized))
        for br_first in [True, False]:
            b = torch_butterfly.special.ifft2d_unitary(n1, n2, br_first=br_first)
            out = b(input)
            self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_conv2d_circular_multichannel(self):
        batch_size = 10
        in_channels = 3
        out_channels = 4
        for n1 in [13, 16]:
            for n2 in [27, 32]:
                # flatten is only supported for powers of 2 for now
                if n1 == 1 << int(math.log2(n1)) and n2 == 1 << int(math.log2(n2)):
                    flatten_cases = [False, True]
                else:
                    flatten_cases = [False]
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
                        # Just to show how to implement conv2d with FFT
                        input_f = view_as_complex(torch.rfft(input, signal_ndim=2))
                        col = F.pad(weight.flip(dims=(-1,)), (0, n1 - kernel_size1)).roll(
                            -padding1, dims=-1)
                        col = F.pad(col.flip(dims=(-2,)), (0, 0, 0, n2 - kernel_size2)).roll(
                            -padding2, dims=-2)
                        col_f = view_as_complex(torch.rfft(col, signal_ndim=2))
                        prod_f = (input_f.unsqueeze(1) * col_f).sum(dim=2)
                        out_fft = torch.irfft(view_as_real(prod_f), signal_ndim=2,
                                              signal_sizes=(n2, n1))
                        self.assertTrue(torch.allclose(out_torch, out_fft, self.rtol, self.atol))
                        for flatten in flatten_cases:
                            b = torch_butterfly.special.conv2d_circular_multichannel(
                                n1, n2, weight, flatten=flatten)
                            out = b(input)
                            self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_fastfood(self):
        batch_size = 10
        n = 32
        H = torch.tensor(la.hadamard(n), dtype=torch.float32) / math.sqrt(n)
        diag1 = torch.randint(0, 2, (n,)) * 2 - 1.0
        diag2, diag3 = torch.randn(2, n)
        permutation = torch.randperm(n)
        input = torch.randn(batch_size, n)
        out_torch = F.linear(input * diag1, H)[:, permutation]
        out_torch = F.linear(out_torch * diag2, H) * diag3
        for increasing_stride in [True, False]:
            for separate_diagonal in [True, False]:
                b = torch_butterfly.special.fastfood(
                    diag1, diag2, diag3, permutation, normalized=True,
                    increasing_stride=increasing_stride, separate_diagonal=separate_diagonal
                )
                out = b(input)
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_acdc(self):
        batch_size = 10
        n = 32
        input = torch.randn(batch_size, n)
        diag1, diag2 = torch.randn(2, n)
        for separate_diagonal in [True, False]:
            out_sp = torch.tensor(scipy.fft.dct(input.numpy(), norm='ortho')) * diag1
            out_sp = torch.tensor(scipy.fft.idct(out_sp.numpy(), norm='ortho')) * diag2
            b = torch_butterfly.special.acdc(diag1, diag2, dct_first=True,
                                             separate_diagonal=separate_diagonal)
            out = b(input)
            self.assertTrue(torch.allclose(out, out_sp, self.rtol, self.atol))
            out_sp = torch.tensor(scipy.fft.idct(input.numpy(), norm='ortho')) * diag1
            out_sp = torch.tensor(scipy.fft.dct(out_sp.numpy(), norm='ortho')) * diag2
            b = torch_butterfly.special.acdc(diag1, diag2, dct_first=False,
                                             separate_diagonal=separate_diagonal)
            out = b(input)
            self.assertTrue(torch.allclose(out, out_sp, self.rtol, self.atol))

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
