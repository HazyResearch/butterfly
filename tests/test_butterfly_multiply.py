import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import torch
import torch.nn.functional as F

from butterfly import Butterfly
from butterfly.butterfly_multiply import butterfly_mult_torch, butterfly_mult, butterfly_mult_inplace, butterfly_mult_factors
from butterfly.butterfly_multiply import butterfly_mult_untied_torch, butterfly_mult_untied
from butterfly.butterfly_multiply import butterfly_conv2d
from cnn.models.butterfly_conv import ButterflyConv2d

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
        batch_size = 10
        n = 4096
        m = int(math.log2(n))
        nstack = 2
        for device in ['cpu'] + ([] if not torch.cuda.is_available() else ['cuda']):
            for complex in [False, True]:
                for increasing_stride in [True, False]:
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
                    # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
                    self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                                    (((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item(), device, complex, increasing_stride))

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
        complex = False 
        in_planes = 4
        out_planes = 4
        kernel_size = 1
        batch_size = 1
        f_dim = 2
        padding = 1
        return_intermediates = False
        dilation = 1
        bfly = ButterflyConv2d(in_planes, out_planes, kernel_size=kernel_size, 
            padding=padding, bias=False, tied_weight=False).to(device)
        input_ = torch.randn(batch_size, in_planes, f_dim, f_dim, requires_grad=True).to(device)
        # print(input_)
        h_out = (f_dim + 2 * padding - dilation * (kernel_size - 1) - 1)  + 1
        w_out = (f_dim + 2 * padding - dilation * (kernel_size - 1) - 1)  + 1
        input_patches = F.unfold(input_, kernel_size, dilation, padding).view(batch_size, in_planes, kernel_size * kernel_size, h_out * w_out)
        input_reshape = input_patches.permute(0, 3, 2, 1).reshape(batch_size * h_out * w_out, kernel_size * kernel_size, in_planes)
        # print(input_reshape)
        for increasing_stride in [True]:
            output_torch = bfly.forward(input_)
            output = butterfly_conv2d(bfly.twiddle, input_, kernel_size, padding, return_intermediates).mean(dim=1)
            output = output.view(
                batch_size, 
                h_out * w_out, out_planes).transpose(1, 2).view(batch_size, out_planes, 
                h_out, w_out)
            print(output)
            print(output_torch)
            # print(output_torch.shape, output.shape)
            self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                    ((output - output_torch).abs().max().item(), device, complex, increasing_stride))
                                    
if __name__ == "__main__":
    unittest.main()
