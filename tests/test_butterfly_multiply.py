import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import torch

from butterfly.butterfly_multiply import butterfly_mult_torch, butterfly_mult, butterfly_mult_inplace, butterfly_mult_factors

class ButterflyMultTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_butterfly_cpu(self):
        batch_size = 10
        n = 4096
        nstack = 2
        twiddle = torch.randn(nstack, n - 1, 2, 2, requires_grad=True) / math.sqrt(2)
        input = torch.randn(batch_size, n, requires_grad=True)
        output = butterfly_mult(twiddle, input)
        output_torch = butterfly_mult_torch(twiddle, input)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    def test_butterfly_complex_cpu(self):
        batch_size = 10
        n = 4096
        nstack = 2
        twiddle = torch.randn(nstack, n - 1, 2, 2, 2, requires_grad=True) / 2
        input = torch.randn(batch_size, n, 2, requires_grad=True)
        output = butterfly_mult(twiddle, input)
        output_torch = butterfly_mult_torch(twiddle, input)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_cuda(self):
        batch_size = 10
        n = 4096
        nstack = 2
        twiddle = torch.randn(nstack, n - 1, 2, 2, requires_grad=True, device='cuda') / math.sqrt(2)
        input = torch.randn(batch_size, n, requires_grad=True, device=twiddle.device)
        output = butterfly_mult(twiddle, input)
        output_torch = butterfly_mult_torch(twiddle, input)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_complex_cuda(self):
        batch_size = 10
        n = 4096
        nstack = 2
        twiddle = torch.randn(nstack, n - 1, 2, 2, 2, requires_grad=True, device='cuda') / 2
        input = torch.randn(batch_size, n, 2, requires_grad=True, device=twiddle.device)
        output = butterfly_mult(twiddle, input)
        output_torch = butterfly_mult_torch(twiddle, input)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    @unittest.skip("Not numerically stable if twiddle factors aren't orthogonal")
    def test_butterfly_inplace_cpu(self):
        batch_size = 10
        n = 4096
        # TODO: in-place implementation doesn't support nstack for now
        nstack = 1
        twiddle = torch.randn(nstack, n - 1, 2, 2, requires_grad=True) / math.sqrt(2)
        input = torch.randn(batch_size, n, requires_grad=True)
        output_inplace = butterfly_mult_inplace(twiddle.squeeze(0), input)
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

    @unittest.skip("Not numerically stable if twiddle factors aren't orthogonal")
    def test_butterfly_complex_inplace_cpu(self):
        batch_size = 10
        n = 4096
        # TODO: in-place implementation doesn't support nstack for now
        nstack = 1
        twiddle = torch.randn(nstack, n - 1, 2, 2, 2, requires_grad=True) / 2
        input = torch.randn(batch_size, n, 2, requires_grad=True)
        output_inplace = butterfly_mult_inplace(twiddle.squeeze(0), input)
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output_inplace, output_torch, rtol=self.rtol, atol=self.atol),
                        (output_inplace - output_torch).abs().max().item())

    @unittest.skip("Not numerically stable if twiddle factors aren't orthogonal")
    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_inplace_cuda(self):
        batch_size = 10
        n = 4096
        # TODO: in-place implementation doesn't support nstack for now
        nstack = 1
        twiddle = torch.randn(nstack, n - 1, 2, 2, requires_grad=True, device='cuda') / math.sqrt(2)
        input = torch.randn(batch_size, n, requires_grad=True, device=twiddle.device)
        output_inplace = butterfly_mult_inplace(twiddle.squeeze(0), input)
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

    def test_butterfly_factors_cpu(self):
        batch_size = 10
        n = 4096
        nstack = 1  # Does not support nstack
        twiddle = torch.randn(nstack, n - 1, 2, 2, requires_grad=True) / math.sqrt(2)
        input = torch.randn(batch_size, n, requires_grad=True)
        output = butterfly_mult_factors(twiddle.squeeze(0), input)
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    def test_butterfly_factors_complex_cpu(self):
        batch_size = 10
        n = 4096
        nstack = 1  # Does not support nstack
        twiddle = torch.randn(nstack, n - 1, 2, 2, 2, requires_grad=True) / 2
        input = torch.randn(batch_size, n, 2, requires_grad=True)
        output = butterfly_mult_factors(twiddle.squeeze(0), input)
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_factors_cuda(self):
        batch_size = 10
        n = 4096
        nstack = 1  # Does not support nstack
        twiddle = torch.randn(nstack, n - 1, 2, 2, requires_grad=True, device='cuda') / math.sqrt(2)
        input = torch.randn(batch_size, n, requires_grad=True, device=twiddle.device)
        output = butterfly_mult_factors(twiddle.squeeze(0), input)
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())

    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_factors_complex_cuda(self):
        batch_size = 10
        n = 4096
        nstack = 1  # Does not support nstack
        twiddle = torch.randn(nstack, n - 1, 2, 2, 2, requires_grad=True, device='cuda') / 2
        input = torch.randn(batch_size, n, 2, requires_grad=True, device=twiddle.device)
        output = butterfly_mult_factors(twiddle.squeeze(0), input)
        output_torch = butterfly_mult_torch(twiddle, input).squeeze(1)
        self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                        (output - output_torch).abs().max().item())
        grad = torch.randn_like(output_torch)
        d_twiddle, d_input = torch.autograd.grad(output, (twiddle, input), grad, retain_graph=True)
        d_twiddle_torch, d_input_torch = torch.autograd.grad(output_torch, (twiddle, input), grad, retain_graph=True)
        self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                        (d_input - d_input_torch).abs().max().item())
        # print((d_twiddle - d_twiddle_torch) / d_twiddle_torch)
        self.assertTrue(torch.allclose(d_twiddle, d_twiddle_torch, rtol=self.rtol, atol=self.atol),
                        ((d_twiddle - d_twiddle_torch) / d_twiddle_torch).abs().max().item())


if __name__ == "__main__":
    unittest.main()
