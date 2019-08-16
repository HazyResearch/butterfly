import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import torch

from butterfly import Butterfly
from butterfly.utils import twiddle_normal_to_fast_format
from cnn.models.butterfly_conv import ButterflyConv2d

from butterfly.butterfly_multiply import butterfly_mult_torch, butterfly_mult, butterfly_mult_inplace, butterfly_mult_factors
from butterfly.butterfly_multiply import butterfly_mult_untied_torch, butterfly_mult_untied
from butterfly.butterfly_multiply import butterfly_ortho_mult_tied_torch, butterfly_ortho_mult_tied
from butterfly.butterfly_multiply import butterfly_ortho_mult_untied_torch, butterfly_ortho_mult_untied
from butterfly.butterfly_multiply import bbt_mult_untied_torch, bbt_mult_untied
from butterfly.butterfly_multiply import bbt_ortho_mult_untied_torch, bbt_ortho_mult_untied
from butterfly.butterfly_multiply import butterfly_mult_conv2d_torch, butterfly_mult_conv2d
from butterfly.butterfly_multiply import bbt_mult_conv2d_torch, bbt_mult_conv2d
from butterfly.butterfly_multiply import butterfly_mult_untied_svd_torch, butterfly_mult_untied_svd
from butterfly.butterfly_multiply import butterfly_mult_conv2d_svd_torch, butterfly_mult_conv2d_svd
# from factor_multiply import butterfly_multiply_untied_eval

from factor_multiply_fast import butterfly_multiply_untied_forward_fast
from factor_multiply_fast import butterfly_multiply_untied_forward_backward_fast
from factor_multiply_fast import butterfly_bbs_multiply_untied_forward_fast
from factor_multiply_fast import butterfly_bbs_multiply_untied_forward_backward_fast
from factor_multiply_fast import butterfly_ortho_multiply_untied_forward_fast
from factor_multiply_fast import butterfly_ortho_multiply_untied_backward_fast
from factor_multiply_fast import butterfly_odo_multiply_untied_forward_fast
from factor_multiply_fast import butterfly_odo_multiply_untied_backward_fast
from factor_multiply_fast import butterfly_odo_multiply_untied_forward_backward_fast


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

    def test_butterfly_untied_eval(self):
        for batch_size, n in [(1, 256), (2, 512), (8, 512), (10, 512)]:
            m = int(math.log2(n))
            nstack = 2
            for device in ['cpu']:
                for complex in [ True]:
                    for increasing_stride in [True, False]:
                        scaling = 1 / math.sqrt(2)
                        twiddle = torch.randn((nstack, m, n // 2, 2, 2), requires_grad=True, device=device) * scaling
                        input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                        output = butterfly_multiply_untied_eval(twiddle, input, increasing_stride)
                        output_torch = butterfly_mult_untied_torch(twiddle, input, increasing_stride)
                        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, complex, increasing_stride))

    def test_butterfly_ortho_tied(self):
        for batch_size, n in [(10, 4096), (8192, 256)]:  # Test size smaller than 1024 and large batch size for race conditions
            m = int(math.log2(n))
            nstack = 2
            for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
                for increasing_stride in [True, False]:
                    if batch_size > 1024 and (device == 'cpu'):
                        continue
                    twiddle = torch.rand((nstack, n - 1), requires_grad=True, device=device) * 2 * math.pi
                    input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                    output = butterfly_ortho_mult_tied(twiddle, input, increasing_stride)
                    output_torch = butterfly_ortho_mult_tied_torch(twiddle, input, increasing_stride)
                    self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), device, increasing_stride))
                    grad = torch.randn_like(output_torch)
                    d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                    d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_torch).abs().max().item(), device, increasing_stride))
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                    atol=self.atol * (10 if batch_size > 1024 else 1)),
                                    (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(),
                                        (batch_size, n), device, increasing_stride))

    def test_butterfly_ortho_untied(self):
        for batch_size, n in [(10, 4096), (8192, 256)]:  # Test size smaller than 1024 and large batch size for race conditions
            m = int(math.log2(n))
            nstack = 2
            for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
                for increasing_stride in [True, False]:
                    if batch_size > 1024 and (device == 'cpu'):
                        continue
                    twiddle = torch.rand((nstack, m, n // 2), requires_grad=True, device=device) * 2 * math.pi
                    input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                    output = butterfly_ortho_mult_untied(twiddle, input, increasing_stride)
                    output_torch = butterfly_ortho_mult_untied_torch(twiddle, input, increasing_stride)
                    self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), device, increasing_stride))
                    grad = torch.randn_like(output_torch)
                    d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                    d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_torch).abs().max().item(), device, increasing_stride))
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                    atol=self.atol * (10 if batch_size > 1024 else 1)),
                                    (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(),
                                        (batch_size, n), device, increasing_stride))

    def test_bbt_untied(self):
        for batch_size, n in [(2048, 512), (10, 4096)]:
            for nblocks in list(range(1, 4)) + [10, 14]:  # Test nblocks >= 7
                m = int(math.log2(n))
                nstack = 2
                for device in ([] if not torch.cuda.is_available() else ['cuda']) + ['cpu']:
                    if batch_size > 1024 and device == 'cpu':
                        continue
                    scaling = 1 / 2
                    twiddle = torch.randn((nstack, nblocks * 2 * m, n // 2, 2, 2), requires_grad=True, device=device) * scaling
                    input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                    output = bbt_mult_untied(twiddle, input)
                    output_torch = bbt_mult_untied_torch(twiddle, input)
                    self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), nblocks, device))
                    grad = torch.randn_like(output_torch)
                    d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                    d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_torch).abs().max().item(), nblocks, device))
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
                                     (batch_size, n), nblocks, device))

    def test_bbt_ortho_untied(self):
        for batch_size, n in [(2048, 512), (10, 4096)]:
            for nblocks in list(range(1, 4)) + [10, 14]:  # Test nblocks >= 7
                m = int(math.log2(n))
                nstack = 2
                for device in ([] if not torch.cuda.is_available() else ['cuda']) + ['cpu']:
                    if batch_size > 1024 and device == 'cpu':
                        continue
                    twiddle = torch.rand((nstack, nblocks * 2 * m, n // 2), requires_grad=True, device=device) * 2 * math.pi
                    input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                    output = bbt_ortho_mult_untied(twiddle, input)
                    output_torch = bbt_ortho_mult_untied_torch(twiddle, input)
                    self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), (batch_size, n), nblocks, device))
                    grad = torch.randn_like(output_torch)
                    d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
                    d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_torch).abs().max().item(), (batch_size, n), nblocks, device))
                    # if device == 'cuda' and batch_size > 1024 and nblocks == 14:
                    #     print((d_twiddle - d_twiddle_torch).abs().mean(dim=(0, 2)))
                    #     print(((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().mean(dim=(0, 2)))
                    #     i = ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().argmax()
                    #     print(d_twiddle.flatten()[i])
                    #     print(d_twiddle_torch.flatten()[i])
                    #     print(d_twiddle.flatten()[i-5:i+5])
                    #     print(d_twiddle_torch.flatten()[i-5:i+5])
                    # Seems to fail for large nblocks because there's likely to be a d_twiddle that's really small.
                    # I guess it's fine.
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                atol=self.atol * (10 if batch_size > 1024 else 1)),
                                    (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(),
                                     (batch_size, n), nblocks, device))

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

    def test_bbt_conv2d(self):
        device = 'cuda'
        c_in = 256
        kernel_size = 3
        batch_size = 128
        f_dim = 8
        padding = 1
        for c_out in [c_in, 2*c_in]:
            nstack = c_out // c_in * kernel_size * kernel_size
            m = int(math.log2(c_in))
            # for nblocks in list(range(1, 4)) + [10, 14]:  # Test nblocks >= 7
            for nblocks in list(range(1, 3)):  # Test nblocks >= 7
                scaling = 1 / math.sqrt(2)
                twiddle = torch.randn((nstack, nblocks * 2 * m, c_in // 2, 2, 2), requires_grad=True, device=device) * scaling
                input_ = torch.randn(batch_size, c_in, f_dim, f_dim,
                                    requires_grad=True).to(device)
                # test forward pass
                output_torch = bbt_mult_conv2d_torch(twiddle, input_, kernel_size, padding)
                output = bbt_mult_conv2d(twiddle, input_, kernel_size, padding)
                self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                        ((output - output_torch).abs().max().item(), device, nblocks, c_out))
                # test backward pass
                grad = torch.randn_like(output_torch)
                d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input_),
                                                        grad, retain_graph=True)
                d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch,
                    (twiddle, input_), grad, retain_graph=True)
                self.assertTrue(torch.allclose(d_input, d_input_torch,
                                               rtol=self.rtol, atol=self.atol),
                                ((d_input - d_input_torch).abs().max().item(), device, nblocks, c_out))
                self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch,
                                               rtol=self.rtol * 10, atol=self.atol * 10),
                                (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(), device, nblocks, c_out))

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

    def test_butterfly_untied_fast(self):
        for batch_size, n in [(2048, 512)]:
            m = int(math.log2(n))
            nstack = 1
            # for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for device in ['cuda']:
                # for complex in [False, True]:
                for complex in [False]:
                    for increasing_stride in [True, False]:
                    # for increasing_stride in [False]:
                        if batch_size > 1024 and (device == 'cpu' or complex):
                            continue
                        scaling = 1 / math.sqrt(2) if not complex else 1 / 2
                        twiddle = torch.randn((nstack, m, n // 2, 2, 2) + (() if not complex else (2, )), requires_grad=True, device=device) * scaling
                        # twiddle = torch.arange(2 * n, dtype=torch.float, device=device, requires_grad=True).reshape(n // 2, 2, 2).unsqueeze(0).repeat(m, 1, 1, 1).unsqueeze(0)
                        twiddle_fast = twiddle_normal_to_fast_format(twiddle)
                        if not increasing_stride:
                            twiddle_fast = twiddle_fast.flip(1)
                        input = torch.randn((batch_size, nstack, n) + (() if not complex else (2, )), requires_grad=True, device=twiddle.device)
                        # input = torch.arange(n, dtype=torch.float, device=device, requires_grad=True).unsqueeze(0).unsqueeze(1).expand(batch_size, -1, -1)
                        output = butterfly_multiply_untied_forward_fast(twiddle_fast, input, increasing_stride)
                        # output_old = butterfly_mult_untied_torch(twiddle, input, increasing_stride)
                        output_old = butterfly_mult_untied(twiddle, input, increasing_stride)
                        self.assertTrue(torch.allclose(output, output_old, rtol=self.rtol, atol=self.atol),
                                        ((output - output_old).abs().max().item(), device, complex, increasing_stride))
                        if n > 4096:
                            continue
                        grad = torch.randn_like(output)
                        d_twiddle, d_input = butterfly_multiply_untied_forward_backward_fast(twiddle_fast, input,
                                                                                             grad, increasing_stride)
                        # d_twiddle, d_input = torch.autograd.grad(output, (twiddle_fast, input), grad, retain_graph=True)
                        d_twiddle_old, d_input_old = torch.autograd.grad(output_old, (twiddle, input), grad, retain_graph=True)
                        self.assertTrue(torch.allclose(d_input, d_input_old, rtol=self.rtol, atol=self.atol),
                                        ((d_input - d_input_old).abs().max().item(), device, complex, increasing_stride))
                        # # if device == 'cuda' and batch_size > 1024 and not complex and increasing_stride:
                        # #     print((d_twiddle - d_twiddle_torch).abs().mean(dim=(0, 2, 3, 4)))
                        # #     print(((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().mean(dim=(0, 2, 3, 4)))
                        # #     i = ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().argmax()
                        # #     print(d_twiddle.flatten()[i])
                        # #     print(d_twiddle_torch.flatten()[i])
                        # #     print(d_twiddle.flatten()[i-5:i+5])
                        # #     print(d_twiddle_torch.flatten()[i-5:i+5])
                        d_twiddle_old = twiddle_normal_to_fast_format(d_twiddle_old)
                        if not increasing_stride:
                            d_twiddle_old = d_twiddle_old.flip(1)
                        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_old, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                       atol=self.atol * (10 if batch_size > 1024 else 1)),
                                        (((d_twiddle - d_twiddle_old) / d_twiddle_old).abs().max().item(),
                                         (batch_size, n), device, complex, increasing_stride))

    def test_butterfly_bbs_untied_fast(self):
        for batch_size, n in [(2048, 512)]:
            m = int(math.log2(n))
            nstack = 1
            nblocks = 3
            # for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for device in ['cuda']:
                if batch_size > 1024 and (device == 'cpu'):
                    continue
                scaling = 1 / math.sqrt(2)
                twiddle = torch.randn((nstack, nblocks * 2 * m, n // 2, 2, 2), requires_grad=True, device=device) * scaling
                # twiddle = torch.arange(16.0, requires_grad=True, device=device).view(nstack, nblocks * 2 * m, n // 2, 2, 2)
                input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                # input = torch.arange(2.0, requires_grad=True, device=twiddle.device).view(batch_size, nstack, n)
                twiddle_fast = []
                for i, chunk in enumerate(twiddle.chunk(nblocks * 2, dim=1)):
                    chunk_fast = twiddle_normal_to_fast_format(chunk)
                    if i % 2 == 0:
                        chunk_fast = chunk_fast.flip(1)
                    twiddle_fast.append(chunk_fast)
                twiddle_fast = torch.cat(twiddle_fast, dim=1)
                output = butterfly_bbs_multiply_untied_forward_fast(twiddle_fast, input)
                output_old = input
                for block in range(nblocks):
                    output_old = butterfly_mult_untied(twiddle[:, block * 2 * m:(block * 2 + 1) * m], output_old, False)
                    output_old = butterfly_mult_untied(twiddle[:, (block * 2 + 1) * m:(block + 1) * 2 * m], output_old, True)
                self.assertTrue(torch.allclose(output, output_old, rtol=self.rtol, atol=self.atol),
                                ((output - output_old).abs().max().item(), device))
                grad = torch.randn_like(output)
                # grad = input.clone()
                d_twiddle, d_input = butterfly_bbs_multiply_untied_forward_backward_fast(twiddle_fast, input, grad)
                # d_twiddle, d_input = torch.autograd.grad(output, (twiddle_fast, input), grad, retain_graph=True)
                d_twiddle_old, d_input_old = torch.autograd.grad(output_old, (twiddle, input), grad, retain_graph=True)
                self.assertTrue(torch.allclose(d_input, d_input_old, rtol=self.rtol, atol=self.atol),
                                ((d_input - d_input_old).abs().max().item(), device))
                d_twiddle_temp = []
                for i, chunk in enumerate(d_twiddle_old.chunk(nblocks * 2, dim=1)):
                    chunk_fast = twiddle_normal_to_fast_format(chunk)
                    if i % 2 == 0:
                        chunk_fast = chunk_fast.flip(1)
                    d_twiddle_temp.append(chunk_fast)
                d_twiddle_old = torch.cat(d_twiddle_temp, dim=1)
                self.assertTrue(torch.allclose(d_twiddle, d_twiddle_old, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                atol=self.atol * (10 if batch_size > 1024 else 1)),
                                (((d_twiddle - d_twiddle_old) / d_twiddle_old).abs().max().item(),
                                    (batch_size, n), device))


    def test_butterfly_ortho_untied_fast(self):
        for batch_size, n in [(2048, 4096)]:
            m = int(math.log2(n))
            nstack = 1
            # for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for device in ['cuda']:
                for increasing_stride in [True, False]:
                    if batch_size > 1024 and (device == 'cpu'):
                        continue
                    twiddle = torch.rand((nstack, m, n // 2), requires_grad=True, device=device) * 2 * math.pi
                    # twiddle = torch.ones((nstack, m, n // 2), requires_grad=True, device=device) * 2 * math.pi * 0.3
                    twiddle_fast = twiddle if increasing_stride else twiddle.flip(1)
                    input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                    twiddle_fast_cos, twiddle_fast_sin = twiddle_fast.cos(), twiddle_fast.sin()
                    output = butterfly_ortho_multiply_untied_forward_fast(twiddle_fast_cos, twiddle_fast_sin, input, increasing_stride)
                    # output_old = butterfly_ortho_mult_untied_torch(twiddle, input)
                    output_old = butterfly_ortho_mult_untied(twiddle, input, increasing_stride)
                    self.assertTrue(torch.allclose(output, output_old, rtol=self.rtol, atol=self.atol),
                                    ((output - output_old).abs().max().item(), device, increasing_stride))
                    grad = torch.randn_like(output)
                    d_twiddle, d_input = butterfly_ortho_multiply_untied_backward_fast(twiddle_fast_cos, twiddle_fast_sin,
                                                                                       output, grad, increasing_stride)
                    # d_twiddle, d_input = torch.autograd.grad(output, (twiddle_fast, input), grad, retain_graph=True)
                    d_twiddle_old, d_input_old = torch.autograd.grad(output_old, (twiddle, input), grad, retain_graph=True)
                    self.assertTrue(torch.allclose(d_input, d_input_old, rtol=self.rtol, atol=self.atol),
                                    ((d_input - d_input_old).abs().max().item(), device, increasing_stride))
                    if not increasing_stride:
                        d_twiddle_old = d_twiddle_old.flip(1)
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_old, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                   atol=self.atol * (10 if batch_size > 1024 else 1)),
                                    (((d_twiddle - d_twiddle_old) / d_twiddle_old).abs().max().item(),
                                     (batch_size, n), device, increasing_stride))

    def test_butterfly_odo_untied_fast(self):
        for batch_size, n in [(2048, 512)]:
            m = int(math.log2(n))
            nstack = 1
            nblocks = 4
            # for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for device in ['cuda']:
                if batch_size > 1024 and (device == 'cpu'):
                    continue
                twiddle = torch.rand((nstack, nblocks * 2 * m, n // 2), requires_grad=True, device=device) * 2 * math.pi
                # diagonal = torch.randn((nstack, nblocks, n), requires_grad=True, device=device)
                # Not numerically stable so we need diagonals to be away from zero
                diagonal = torch.rand((nstack, nblocks, n), requires_grad=True, device=device) + 0.1
                # diagonal = torch.ones((nstack, nblocks, n), requires_grad=True, device=device) * 0.1
                input = torch.randn((batch_size, nstack, n), requires_grad=True, device=twiddle.device)
                twiddle_fast_cos, twiddle_fast_sin = twiddle.cos(), twiddle.sin()
                output = butterfly_odo_multiply_untied_forward_fast(twiddle_fast_cos, twiddle_fast_sin,
                                                                    diagonal, input)
                # output_old = butterfly_odo_mult_untied_torch(twiddle, input)
                output_old = input
                for block in range(nblocks):
                    output_old = butterfly_ortho_mult_untied(twiddle[:, block * 2 * m:(block * 2 + 1) * m].flip(1), output_old, False)
                    output_old = output_old * diagonal[:, block]
                    output_old = butterfly_ortho_mult_untied(twiddle[:, (block * 2 + 1) * m:(block + 1) * 2 * m], output_old, True)
                self.assertTrue(torch.allclose(output, output_old, rtol=self.rtol, atol=self.atol),
                                ((output - output_old).abs().max().item(), device))
                grad = torch.randn_like(output)
                # d_twiddle, d_diagonal, d_input = butterfly_odo_multiply_untied_backward_fast(twiddle_fast_cos, twiddle_fast_sin,
                #                                                                  diagonal, output, grad)
                d_twiddle, d_diagonal, d_input = butterfly_odo_multiply_untied_forward_backward_fast(twiddle_fast_cos, twiddle_fast_sin,
                                                                                 diagonal, input, grad)
                # d_twiddle, d_input = torch.autograd.grad(output, (twiddle_fast, input), grad, retain_graph=True)
                d_twiddle_old, d_diagonal_old, d_input_old = torch.autograd.grad(output_old, (twiddle, diagonal, input), grad, retain_graph=True)
                self.assertTrue(torch.allclose(d_input, d_input_old, rtol=self.rtol, atol=self.atol),
                                ((d_input - d_input_old).abs().max().item(), device))
                self.assertTrue(torch.allclose(d_diagonal, d_diagonal_old, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                               atol=self.atol * (10 if batch_size > 1024 else 1)),
                                ((d_diagonal - d_diagonal_old).abs().max().item(), device))
                self.assertTrue(torch.allclose(d_twiddle, d_twiddle_old, rtol=self.rtol * (10 if batch_size > 1024 else 1),
                                                atol=self.atol * (10 if batch_size > 1024 else 1)),
                                (((d_twiddle - d_twiddle_old) / d_twiddle_old).abs().max().item(),
                                    (batch_size, n), device))


if __name__ == "__main__":
    unittest.main()
