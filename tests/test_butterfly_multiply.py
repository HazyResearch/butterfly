import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import torch
import torch.nn.functional as F

from butterfly import Butterfly
from cnn.models.butterfly_conv import ButterflyConv2d

from butterfly.butterfly_multiply import butterfly_mult_torch, butterfly_mult, butterfly_mult_inplace, butterfly_mult_factors
from butterfly.butterfly_multiply import butterfly_mult_untied_torch, butterfly_mult_untied
from butterfly.butterfly_multiply import butterfly_mult_conv2d_torch, butterfly_mult_conv2d
from butterfly.butterfly_multiply import butterfly_mult_untied_svd_torch, butterfly_mult_untied_svd
from butterfly.butterfly_multiply import butterfly_mult_conv2d_svd_torch, butterfly_mult_conv2d_svd
from factor_multiply import butterfly_multiply_untied_batch

class ButterflyMultTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_butterfly(self):
        batch_size = 10
        n = 4096
        nstack = 2
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for complex in [False, True]:
                for increasing_stride in [True, False]:
                    scaling = 1 / math.sqrt(2) if not complex else 1 / 2
                    twiddle = torch.randn((nstack, n - 1, 2, 2) + (() if not complex else (2, )), requires_grad=True, device=device) * scaling
                    input = torch.randn((batch_size, nstack, n) + (() if not complex else (2, )), requires_grad=True, device=twiddle.device)
                    output = butterfly_mult(twiddle, input, increasing_stride)
                    output_torch = butterfly_mult_torch(twiddle, input, increasing_stride)
                    self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), device, complex, increasing_stride))
                    grad = torch.randn_like(output_torch)
                    d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                    d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_torch).abs().max().item(), device, complex, increasing_stride))
                    # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                                    (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(), device, complex, increasing_stride))

    def test_butterfly_untied(self):
        for batch_size, n in [(10, 4096), (8192, 256)]:  # Test size smaller than 1024 and large batch size for race conditions
            m = int(math.log2(n))
            nstack = 2
            for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
                for complex in [False, True]:
                    for increasing_stride in [True, False]:
                        if batch_size > 1024 and (device == 'cpu' or complex):
                            continue
                        scaling = 1 / math.sqrt(2) if not complex else 1 / 2
                        twiddle = torch.randn((nstack, m, n // 2, 2, 2) + (() if not complex else (2, )), requires_grad=True, device=device) * scaling
                        input = torch.randn((batch_size, nstack, n) + (() if not complex else (2, )), requires_grad=True, device=twiddle.device)
                        output = butterfly_mult_untied(twiddle, input, increasing_stride)
                        output_torch = butterfly_mult_untied_torch(twiddle, input, increasing_stride)
                        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, complex, increasing_stride))
                        grad = torch.randn_like(output_torch)
                        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                        ((d_input - d_input_torch).abs().max().item(), device, complex, increasing_stride))
                        # if device == 'cuda' and batch_size > 1024 and not complex and increasing_stride:
                        #     print((d_twiddle - d_twiddle_torch).abs().mean(dim=(0, 2, 3, 4)))
                        #     print(((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().mean(dim=(0, 2, 3, 4)))
                        #     i = ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().argmax()
                        #     print(d_twiddle.flatten()[i])
                        #     print(d_twiddle_torch.flatten()[i])
                        #     print(d_twiddle.flatten()[i-5:i+5])
                        #     print(d_twiddle_torch.flatten()[i-5:i+5])
                        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                       atol=self.atol * (10 if batch_size > 1024 else 1)),
                                        (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(),
                                         (batch_size, n), device, complex, increasing_stride))

    def test_butterfly_untied_batch(self):
        for batch_size, n in [(8, 256), (10, 512)]:
            m = int(math.log2(n))
            nstack = 2
            for device in ['cpu']:
                for complex in [ True]:
                    for increasing_stride in [True, False]:
                        scaling = 1 / math.sqrt(2)
                        twiddle = torch.randn((nstack, m, n // 2, 2, 2), requires_grad=True, device=device) * scaling
                        input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                        output = butterfly_mult_untied(twiddle, input, increasing_stride, False) # is_training = False
                        output_torch = butterfly_mult_untied_torch(twiddle, input, increasing_stride)
                        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, complex, increasing_stride))

                        # also call directly in case butterfly_mult_untied gets changed
                        if batch_size % 8 != 0:
                            continue
                        output = butterfly_multiply_untied_batch(twiddle, input, increasing_stride)
                        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, complex, increasing_stride))

    def test_butterfly_untied_svd(self):
        for batch_size, n in [(10, 4096), (99, 128)]:  # Test size smaller than 1024
            m = int(math.log2(n))
            nstack = 2
            for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
                for increasing_stride in [True, False]:
                    scaling = 1 / math.sqrt(2)
                    twiddle = torch.randn((nstack, m, n // 2, 2, 2), requires_grad=True, device=device) * scaling
                    input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                    output = butterfly_mult_untied_svd(twiddle, input, increasing_stride)
                    output_torch = butterfly_mult_untied_svd_torch(twiddle, input, increasing_stride)
                    self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), device, increasing_stride))
                    grad = torch.randn_like(output_torch)
                    d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                    d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_torch).abs().max().item(), device, increasing_stride))
                    # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                                    (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(), device, increasing_stride))

    # @unittest.skip("Not numerically stable if twiddle factors aren't orthogonal")
    def test_butterfly_inplace_cpu(self):
        batch_size = 10
        n = 4096
        # TODO: in-place implementation doesn't support nstack for now
        nstack = 1
        b = Butterfly(n, n, bias=False, ortho_init=True)
        twiddle = b.twiddle
        input = torch.randn(batch_size, nstack, n, requires_grad=True)
        output_inplace = butterfly_mult_inplace(twiddle.squeeze(0), input.squeeze(1))
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output_inplace, output_torch, rtol=self.rtol, atol=self.atol),
                        (output_inplace - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle_inplace, d_input_inplace = torch.autograd.grad(output_inplace, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input_inplace, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input_inplace - d_input_torch).abs().max().item())
        # print((d_twiddle_inplace - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle_inplace, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle_inplace - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    # @unittest.skip("Not numerically stable if twiddle factors aren't orthogonal")
    def test_butterfly_complex_inplace_cpu(self):
        batch_size = 10
        n = 4096
        # TODO: in-place implementation doesn't support nstack for now
        nstack = 1
        b = Butterfly(n, n, bias=False, complex=True, ortho_init=True)
        twiddle = b.twiddle
        input = torch.randn(batch_size, nstack, n, 2, requires_grad=True)
        output_inplace = butterfly_mult_inplace(twiddle.squeeze(0), input.squeeze(1))
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output_inplace, output_torch, rtol=self.rtol, atol=self.atol),
                        (output_inplace - output_torch).abs().max().item())

    # @unittest.skip("Not numerically stable if twiddle factors aren't orthogonal")
    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_inplace_cuda(self):
        batch_size = 10
        n = 4096
        # TODO: in-place implementation doesn't support nstack for now
        nstack = 1
        b = Butterfly(n, n, bias=False, ortho_init=True).to('cuda')
        twiddle = b.twiddle
        input = torch.randn(batch_size, nstack, n, requires_grad=True, device=twiddle.device)
        output_inplace = butterfly_mult_inplace(twiddle.squeeze(0), input.squeeze(1))
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output_inplace, output_torch, rtol=self.rtol, atol=self.atol),
                        (output_inplace - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle_inplace, d_input_inplace = torch.autograd.grad(output_inplace, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input_inplace, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input_inplace - d_input_torch).abs().max().item())
        # print((d_twiddle_inplace - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle_inplace, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle_inplace - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    def test_butterfly_factors(self):
        batch_size = 10
        n = 4096
        nstack = 1  # Does not support nstack
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for complex in [False, True]:
                for increasing_stride in [True, False]:
                    scaling = 1 / math.sqrt(2) if not complex else 1 / 2
                    twiddle = torch.randn((nstack, n - 1, 2, 2) + (() if not complex else (2, )), requires_grad=True, device=device) * scaling
                    input = torch.randn((batch_size, nstack, n) + (() if not complex else (2, )), requires_grad=True, device=twiddle.device)
                    output = butterfly_mult_factors(twiddle.squeeze(0), input.squeeze(1), increasing_stride=increasing_stride)
                    output_torch = butterfly_mult_torch(twiddle, input, increasing_stride=increasing_stride).squeeze(1)
                    self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), device, complex, increasing_stride))
                    grad = torch.randn_like(output_torch)
                    d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                    d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_torch).abs().max().item(), device, complex, increasing_stride))
                    # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                                    (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(), device, complex, increasing_stride))

    def test_butterfly_conv2d(self):
        device = 'cuda'
        c_in = 256
        kernel_size = 3
        batch_size = 128
        f_dim = 8
        padding = 1
        for c_out in [c_in, 2*c_in]:
            nstack = c_out // c_in * kernel_size * kernel_size
            m = int(math.log2(c_in))
            for increasing_stride in [True, False]:
                scaling = 1 / math.sqrt(2)
                twiddle = torch.randn((nstack, m, c_in // 2, 2, 2), requires_grad=True, device=device) * scaling
                input_ = torch.randn(batch_size, c_in, f_dim, f_dim,
                                    requires_grad=True).to(device)
                # test forward pass
                output_torch = butterfly_mult_conv2d_torch(twiddle, input_, kernel_size,
                                        padding, increasing_stride)
                output = butterfly_mult_conv2d(twiddle, input_, kernel_size,
                                        padding, increasing_stride)
                self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, c_out, increasing_stride))
                # test backward pass
                grad = torch.randn_like(output_torch)
                d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input_),
                                                        grad, retain_graph=True)
                d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch,
                    (twiddle, input_), grad, retain_graph=True)
                self.assertTrue(torch.allclose(d_input, d_input_torch,
                                               rtol=self.rtol, atol=self.atol),
                                ((d_input - d_input_torch).abs().max().item(), device, c_out, increasing_stride))
                self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch,
                                               rtol=self.rtol * 10, atol=self.atol * 10),
                                (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(), device, c_out, increasing_stride))

    def test_butterfly_conv2d_svd(self):
        device = 'cuda'
        c_in = 256
        kernel_size = 3
        batch_size = 128
        f_dim = 8
        padding = 1
        for c_out in [c_in, 2*c_in]:
            nstack = c_out // c_in * kernel_size * kernel_size
            m = int(math.log2(c_in))
            for increasing_stride in [True, False]:
                scaling = 1 / math.sqrt(2)
                twiddle = torch.randn((nstack, m, c_in // 2, 2, 2), requires_grad=True, device=device) * scaling
                input_ = torch.randn(batch_size, c_in, f_dim, f_dim,
                                    requires_grad=True).to(device)
                # test forward pass
                output_torch = butterfly_mult_conv2d_svd_torch(twiddle, input_, kernel_size,
                                        padding, increasing_stride)
                output = butterfly_mult_conv2d_svd(twiddle, input_, kernel_size,
                                        padding, increasing_stride)
                self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, c_out, increasing_stride))
                # test backward pass
                grad = torch.randn_like(output_torch)
                d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input_),
                                                        grad, retain_graph=True)
                d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch,
                    (twiddle, input_), grad, retain_graph=True)
                self.assertTrue(torch.allclose(d_input, d_input_torch,
                                               rtol=self.rtol, atol=self.atol),
                                ((d_input - d_input_torch).abs().max().item(), device, c_out, increasing_stride))
                self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch,
                                               rtol=self.rtol * 10, atol=self.atol * 10),
                                (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(), device, c_out, increasing_stride))

if __name__ == "__main__":
    unittest.main()
