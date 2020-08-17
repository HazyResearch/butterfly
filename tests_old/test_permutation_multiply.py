import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import torch

from butterfly.permutation_multiply import permutation_mult_torch, permutation_mult
from butterfly.permutation_multiply import permutation_mult_single_factor_torch, permutation_mult_single


class PermutationMultTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_permutation_cpu(self):
        batch_size = 10
        n = 4096
        m = int(math.log2(n))
        prob = torch.rand(m - 1, 3, requires_grad=True)
        for complex in [False, True]:
            for increasing_stride in [False, True]:
                input = torch.randn((batch_size, n) + (() if not complex else (2, )), requires_grad=True)
                output = permutation_mult(prob, input)
                output_torch = permutation_mult_torch(prob, input)
                self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                (complex, (output - output_torch).abs().max().item()))
                grad = torch.randn_like(output_torch)
                d_prob, d_input = torch.autograd.grad(output, (prob, input), grad, retain_graph=True)
                d_prob_torch, d_input_torch = torch.autograd.grad(output_torch, (prob, input), grad, retain_graph=True)
                self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                (complex, (d_input - d_input_torch).abs().max().item()))
                # print((d_prob - d_prob_torch) / d_prob_torch)
                self.assertTrue(torch.allclose(d_prob, d_prob_torch, rtol=self.rtol, atol=self.atol),
                                (complex, ((d_prob - d_prob_torch) / d_prob_torch).abs().max().item()))

    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_permutation_cuda(self):
        batch_size = 10
        n = 4096
        m = int(math.log2(n))
        prob = torch.rand(m - 1, 3, device='cuda', requires_grad=True)
        for complex in [False, True]:
            for increasing_stride in [False, True]:
                input = torch.randn((batch_size, n) + (() if not complex else (2, )), device=prob.device, requires_grad=True)
                output = permutation_mult(prob, input)
                output_torch = permutation_mult_torch(prob, input)
                self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                                (complex, (output - output_torch).abs().max().item()))
                grad = torch.randn_like(output_torch)
                d_prob, d_input = torch.autograd.grad(output, (prob, input), grad, retain_graph=True)
                d_prob_torch, d_input_torch = torch.autograd.grad(output_torch, (prob, input), grad, retain_graph=True)
                self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                                (complex, (d_input - d_input_torch).abs().max().item()))
                # print((d_prob - d_prob_torch) / d_prob_torch)
                self.assertTrue(torch.allclose(d_prob, d_prob_torch, rtol=self.rtol, atol=self.atol),
                                (complex, ((d_prob - d_prob_torch) / d_prob_torch).abs().max().item()))

    def test_permutation_single_cpu(self):
        batch_size = 10
        n = 4096
        m = int(math.log2(n))
        prob = torch.rand(3, requires_grad=True)
        for complex in [False, True]:
            input = torch.randn((batch_size, n) + (() if not complex else (2, )), requires_grad=True)
            output = permutation_mult_single(prob, input)
            output_torch = permutation_mult_single_factor_torch(prob, input)
            self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                            (complex, (output - output_torch).abs().max().item()))
            grad = torch.randn_like(output_torch)
            d_prob, d_input = torch.autograd.grad(output, (prob, input), grad, retain_graph=True)
            d_prob_torch, d_input_torch = torch.autograd.grad(output_torch, (prob, input), grad, retain_graph=True)
            self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                            (complex, (d_input - d_input_torch).abs().max().item()))
            # print((d_prob - d_prob_torch) / d_prob_torch)
            self.assertTrue(torch.allclose(d_prob, d_prob_torch, rtol=self.rtol, atol=self.atol),
                            (complex, ((d_prob - d_prob_torch) / d_prob_torch).abs().max().item()))

    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_permutation_single_cuda(self):
        batch_size = 10
        n = 4096
        m = int(math.log2(n))
        prob = torch.rand(3, device='cuda', requires_grad=True)
        for complex in [False, True]:
            input = torch.randn((batch_size, n) + (() if not complex else (2, )), device=prob.device, requires_grad=True)
            output = permutation_mult_single(prob, input)
            output_torch = permutation_mult_single_factor_torch(prob, input)
            self.assertTrue(torch.allclose(output, output_torch, rtol=self.rtol, atol=self.atol),
                            (complex, (output - output_torch).abs().max().item()))
            grad = torch.randn_like(output_torch)
            d_prob, d_input = torch.autograd.grad(output, (prob, input), grad, retain_graph=True)
            d_prob_torch, d_input_torch = torch.autograd.grad(output_torch, (prob, input), grad, retain_graph=True)
            self.assertTrue(torch.allclose(d_input, d_input_torch, rtol=self.rtol, atol=self.atol),
                            (complex, (d_input - d_input_torch).abs().max().item()))
            # print((d_prob - d_prob_torch) / d_prob_torch)
            self.assertTrue(torch.allclose(d_prob, d_prob_torch, rtol=self.rtol, atol=self.atol),
                            (complex, ((d_prob - d_prob_torch) / d_prob_torch).abs().max().item()))


if __name__ == "__main__":
    unittest.main()
