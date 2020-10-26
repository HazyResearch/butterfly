import math
import unittest

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torch_butterfly
from torch_butterfly import Butterfly
from torch_butterfly.complex_utils import view_as_complex, complex_matmul
from torch_butterfly.combine import TensorProduct
from torch_butterfly.complex_utils import real2complex


class ButterflyTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_butterfly(self):
        batch_size = 10
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for increasing_stride in [True, False]:
                        for init in ['randn', 'ortho', 'identity']:
                            for nblocks in [1, 2, 3]:
                                b = Butterfly(in_size, out_size, True, complex, increasing_stride,
                                              init, nblocks=nblocks).to(device)
                                dtype = torch.float32 if not complex else torch.complex64
                                input = torch.randn(batch_size, in_size, dtype=dtype, device=device)
                                output = b(input)
                                self.assertTrue(output.shape == (batch_size, out_size),
                                                (output.shape, device, (in_size, out_size), complex, init, nblocks))
                                if init == 'ortho':
                                    twiddle = b.twiddle if not b.complex else view_as_complex(b.twiddle)
                                    twiddle_np = twiddle.detach().to('cpu').numpy()
                                    twiddle_np = twiddle_np.reshape(-1, 2, 2)
                                    twiddle_norm = np.linalg.norm(twiddle_np, ord=2, axis=(1, 2))
                                    self.assertTrue(np.allclose(twiddle_norm, 1),
                                                    (twiddle_norm, device, (in_size, out_size), complex, init))

    def test_fft_init(self):
        batch_size = 10
        n = 16
        input = torch.randn(batch_size, n, dtype=torch.complex64)
        br = torch_butterfly.permutation.bitreversal_permutation(n, pytorch_format=True)
        for increasing_stride in [True, False]:
            for nblocks in [1, 2, 3]:
                out_torch = torch.view_as_complex(torch.fft(torch.view_as_real(input),
                                                            signal_ndim=1, normalized=True))
                b = Butterfly(n, n, False, complex=True, increasing_stride=increasing_stride,
                              init='fft_no_br', nblocks=nblocks)
                out = b(input[..., br]) if increasing_stride else b(input)[..., br]
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))
                out_torch = torch.view_as_complex(torch.ifft(torch.view_as_real(input),
                                                             signal_ndim=1, normalized=True))
                b = Butterfly(n, n, False, complex=True, increasing_stride=increasing_stride,
                              init='ifft_no_br', nblocks=nblocks)
                out = b(input[..., br]) if increasing_stride else b(input)[..., br]
                self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_fft2d_init(self):
        batch_size = 10
        in_channels = 3
        out_channels = 4
        n1, n2 = 16, 32
        input = torch.randn(batch_size, in_channels, n2, n1)
        for kernel_size1 in [1, 3, 5, 7]:
            for kernel_size2 in [1, 3, 5, 7]:
                padding1 = (kernel_size1 - 1) // 2
                padding2 = (kernel_size2 - 1) // 2
                conv = nn.Conv2d(in_channels, out_channels, (kernel_size2, kernel_size1),
                                padding=(padding2, padding1), padding_mode='circular',
                                bias=False)
                out_torch = conv(input)
                weight = conv.weight
                w = F.pad(weight.flip(dims=(-1,)), (0, n1 - kernel_size1)).roll(
                    -padding1, dims=-1)
                w = F.pad(w.flip(dims=(-2,)), (0, 0, 0, n2 - kernel_size2)).roll(
                    -padding2, dims=-2)
                increasing_strides = [False, False, True]
                inits = ['fft_no_br', 'fft_no_br', 'ifft_no_br']
                for nblocks in [1, 2, 3]:
                    Kd, K1, K2 = [
                        TensorProduct(
                            Butterfly(n1, n1, bias=False, complex=complex,
                                increasing_stride=incstride, init=i, nblocks=nblocks),
                            Butterfly(n2, n2, bias=False, complex=complex,
                                increasing_stride=incstride, init=i, nblocks=nblocks)
                        )
                        for incstride, i in zip(increasing_strides, inits)
                    ]
                    with torch.no_grad():
                        Kd.map1 *= math.sqrt(n1)
                        Kd.map2 *= math.sqrt(n2)
                    out = K2(
                        complex_matmul(K1(real2complex(input)).permute(2, 3, 0, 1),
                                       Kd(real2complex(w)).permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
                    ).real
                    self.assertTrue(torch.allclose(out, out_torch, self.rtol, self.atol))

    def test_autograd(self):
        """Check if autograd works (especially for complex), by trying to match a 4x4 matrix.
        """
        size = 4
        niters = 10000
        true_model = nn.Linear(size, size, bias=False)
        x = torch.eye(size)
        with torch.no_grad():
            y = true_model(x)
        for complex in [False, True]:
            if complex:
                model = nn.Sequential(
                    torch_butterfly.complex_utils.Real2Complex(),
                    Butterfly(size, size, bias=False, complex=complex),
                    torch_butterfly.complex_utils.Complex2Real(),
                )
            else:
                model = Butterfly(size, size, bias=False, complex=complex)
            with torch.no_grad():
                inital_loss = F.mse_loss(model(x), y)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            for i in range(niters):
                out = model(x)
                loss = F.mse_loss(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # At least loss should decrease
            # print(inital_loss, loss)
            self.assertTrue(loss.item() < inital_loss.item())

    def test_transpose_conjugate_multiply(self):
        n = 16
        for complex in [False, True]:
            for increasing_stride in [True, False]:
                for nblocks in [1, 2, 3]:
                    b = Butterfly(n, n, False, complex, increasing_stride, nblocks=nblocks)
                    dtype = torch.float32 if not complex else torch.complex64
                    input = torch.eye(n, dtype=dtype)
                    matrix = b(input).t()
                    matrix_t = b.forward(input, transpose=True).t()
                    matrix_conj = b.forward(input, conjugate=True).t()
                    matrix_t_conj = b.forward(input, transpose=True, conjugate=True).t()
                    self.assertTrue(torch.allclose(matrix.t(), matrix_t, self.rtol, self.atol),
                                    (complex, increasing_stride, nblocks))
                    self.assertTrue(torch.allclose(matrix.conj(), matrix_conj,
                                                   self.rtol, self.atol),
                                    (complex, increasing_stride, nblocks))
                    self.assertTrue(torch.allclose(matrix.t().conj(), matrix_t_conj,
                                                   self.rtol, self.atol),
                                    (complex, increasing_stride, nblocks))

    def test_subtwiddle(self):
        batch_size = 10
        n = 16
        input_size = 8
        for complex in [False, True]:
            for increasing_stride in [True, False]:
                for nblocks in [1, 2, 3]:
                    b = Butterfly(n, n, True, complex, increasing_stride, nblocks=nblocks)
                    dtype = torch.float32 if not complex else torch.complex64
                    input = torch.randn(batch_size, input_size, dtype=dtype)
                    output = b(input, subtwiddle=True)
                    self.assertTrue(output.shape == (batch_size, input_size),
                                    (output.shape, n, input_size, complex, nblocks))

    def test_butterfly_imul(self):
        batch_size = 10
        device = 'cpu'
        for in_size, out_size in [(7, 15), (15, 7)]:
            for complex in [False, True]:
                for increasing_stride in [True, False]:
                    for init in ['randn', 'ortho', 'identity']:
                        for nblocks in [1, 2, 3]:
                            for scale in [0.13, 2.75]:
                                b = Butterfly(in_size, out_size, False, complex, increasing_stride,
                                              init, nblocks=nblocks).to(device)
                                dtype = torch.float32 if not complex else torch.complex64
                                input = torch.randn(batch_size, in_size, dtype=dtype, device=device)
                                output = b(input)
                                with torch.no_grad():
                                    b *= scale
                                output_scaled = b(input)
                                self.assertTrue(torch.allclose(output * scale, output_scaled,
                                                               self.rtol, self.atol),
                                                (output.shape, device, (in_size, out_size), complex, init, nblocks))

    def test_butterfly_to_base4(self):
        batch_size = 10
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for increasing_stride in [True, False]:
                        for init in ['randn', 'ortho', 'identity']:
                            for nblocks in [1, 2, 3]:
                                b = Butterfly(in_size, out_size, True, complex, increasing_stride,
                                              init, nblocks=nblocks).to(device)
                                dtype = torch.float32 if not complex else torch.complex64
                                input = torch.randn(batch_size, in_size, dtype=dtype, device=device)
                                output = b(input)
                                b4 = b.to_base4()
                                output_base4 = b4(input)
                                self.assertTrue(torch.allclose(output, output_base4,
                                                               self.rtol, self.atol),
                                                (output.shape, device, (in_size, out_size), complex, init, nblocks))

    def test_butterfly_unitary(self):
        # Test shape
        batch_size = 10
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for increasing_stride in [True, False]:
                    for nblocks in [1, 2, 3]:
                        b = torch_butterfly.ButterflyUnitary(in_size, out_size, True,
                                                                increasing_stride, nblocks=nblocks).to(device)
                        dtype = torch.complex64
                        input = torch.randn(batch_size, in_size, dtype=dtype, device=device)
                        output = b(input)
                        self.assertTrue(output.shape == (batch_size, out_size),
                                        (output.shape, device, (in_size, out_size), nblocks))
        # Test that it's actually unitary
        size = 32
        for increasing_stride in [True, False]:
            for nblocks in [1, 2, 3]:
                b = torch_butterfly.ButterflyUnitary(size, size, False,
                                                     increasing_stride, nblocks=nblocks)
                eye = torch.eye(size, dtype=torch.complex64)
                twiddle_matrix_np = b(eye).t().detach().numpy()
                self.assertTrue(np.allclose(twiddle_matrix_np @ twiddle_matrix_np.T.conj(),
                                            np.eye(size), self.rtol, self.atol))

    def test_butterfly_bmm(self):
        batch_size = 10
        matrix_batch = 3
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for in_size, out_size in [(7, 15), (15, 7)]:
                for complex in [False, True]:
                    for increasing_stride in [True, False]:
                        for nblocks in [1, 2, 3]:
                            # Test shape
                            b_bmm = torch_butterfly.ButterflyBmm(in_size, out_size, matrix_batch, True,
                                                             complex, increasing_stride, nblocks=nblocks).to(device)
                            dtype = torch.float32 if not complex else torch.complex64
                            input = torch.randn(batch_size, matrix_batch, in_size, dtype=dtype, device=device)
                            output = b_bmm(input)
                            self.assertTrue(output.shape == (batch_size, matrix_batch, out_size),
                                            (output.shape, device, (in_size, out_size), nblocks))
                            # Check that the result is the same as looping over butterflies
                            output_loop = []
                            for i in range(matrix_batch):
                                b = Butterfly(in_size, out_size, True, complex, increasing_stride,
                                              nblocks=nblocks).to(device)
                                with torch.no_grad():
                                    b.twiddle.copy_(b_bmm.twiddle[i * b_bmm.nstacks:(i + 1) * b_bmm.nstacks])
                                    b.bias.copy_(b_bmm.bias[i])
                                output_loop.append(b(input[:, i]))
                            output_loop = torch.stack(output_loop, dim=1)
                            self.assertTrue(torch.allclose(output, output_loop),
                                            ((output - output_loop).abs().max().item(), output.shape, device, (in_size, out_size), complex))

    def test_butterfly_bmm_tensorproduct(self):
        # Just to show how to do TensorProduct (e.g., Conv2d) with ButterflyBmm
        batch_size = 10
        in_channels = 3
        out_channels = 6
        n1, n2 = 32, 16
        dtype = torch.complex64
        input = torch.randn(batch_size, in_channels, n2, n1, dtype=dtype)
        # Generate out_channels x in_channels butterfly matrices and loop over them
        b1s = [Butterfly(n1, n1, bias=False, complex=True)
               for _ in range(out_channels * in_channels)]
        b2s = [Butterfly(n2, n2, bias=False, complex=True)
               for _ in range(out_channels * in_channels)]
        b_tp = [torch_butterfly.combine.TensorProduct(b1, b2) for b1, b2 in zip(b1s, b2s)]
        outputs = []
        for o in range(out_channels):
            output = []
            for i in range(in_channels):
                index = o * in_channels + i
                output.append(b_tp[index](input[:, i]))
            outputs.append(torch.stack(output, dim=1))
        out = torch.stack(outputs, dim=1)
        assert out.shape == (batch_size, out_channels, in_channels, n2, n1)
        # Use ButterflyBmm instead
        b1_bmm = torch_butterfly.ButterflyBmm(n1, n1, matrix_batch=out_channels * in_channels,
                                              bias=False, complex=True)
        with torch.no_grad():
            b1_bmm.twiddle.copy_(torch.cat([b1.twiddle for b1 in b1s]))
        b2_bmm = torch_butterfly.ButterflyBmm(n2, n2, matrix_batch=out_channels * in_channels,
                                              bias=False, complex=True)
        with torch.no_grad():
            b2_bmm.twiddle.copy_(torch.cat([b2.twiddle for b2 in b2s]))
        input_reshaped = input.transpose(1, 2).reshape(batch_size, n2, 1, in_channels, n1)
        input_expanded = input_reshaped.expand(batch_size, n2, out_channels, in_channels, n1)
        out_bmm = b1_bmm(input_expanded.reshape(batch_size, n2, out_channels * in_channels, n1))
        out_bmm = out_bmm.transpose(1, 3)  # (batch_size, n1, out_channels * in_channels, n2)
        out_bmm = b2_bmm(out_bmm)  # (batch_size, n1, out_channels * in_channels, n2)
        out_bmm = out_bmm.permute(0, 2, 3, 1)  # (batch_size, out_channels * in_channels, n2, n1)
        out_bmm = out_bmm.reshape(batch_size, out_channels, in_channels, n2, n1)
        self.assertTrue(torch.allclose(out_bmm, out))


if __name__ == "__main__":
    unittest.main()
