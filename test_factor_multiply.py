import unittest

import torch

from butterfly_factor import butterfly_factor_mult
from butterfly import Block2x2DiagProduct
from complex_utils import complex_mul

class ButterflyFactorTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-4
        self.atol = 1e-6

    def test_butterfly_factor_cpu(self):
        batch_size = 10
        n = 1024
        B = Block2x2DiagProduct(n)
        input_ = torch.randn(batch_size, n, requires_grad=True)
        output = input_
        for factor in B.factors[::-1]:
            prev = output
            output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(prev.shape)
            output_slow = ((factor.ABCD * prev.view(-1, 1, 2, factor.size // 2)).sum(dim=-2)).view(prev.shape)
            self.assertTrue(torch.allclose(output, output_slow, rtol=self.rtol, atol=self.atol), (output - output_slow).abs().max().item())
            grad = torch.randn_like(output)
            d_twiddle, d_input = torch.autograd.grad(output, (factor.ABCD, prev), grad, retain_graph=True)
            d_twiddle_slow, d_input_slow = torch.autograd.grad(output_slow, (factor.ABCD, prev), grad, retain_graph=True)
            print((d_input - d_input_slow).abs().max().item())
            self.assertTrue(torch.allclose(d_twiddle, d_twiddle_slow, rtol=self.rtol, atol=self.atol), (d_twiddle - d_twiddle_slow).abs().max().item())
            self.assertTrue(torch.allclose(d_input, d_input_slow, rtol=self.rtol, atol=self.atol), (d_input - d_input_slow).abs().max().item())


    def test_butterfly_factor_complex_cpu(self):
        batch_size = 10
        n = 1024
        B = Block2x2DiagProduct(n, complex=True)
        input_ = torch.randn(batch_size, n, 2, requires_grad=True)
        output = input_
        for factor in B.factors[::-1]:
            prev = output
            output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2, 2)).view(prev.shape)
            output_slow = (complex_mul(factor.ABCD, prev.view(-1, 1, 2, factor.size // 2, 2)).sum(dim=-3)).view(prev.shape)
            self.assertTrue(torch.allclose(output, output_slow, rtol=self.rtol, atol=self.atol), (output - output_slow).abs().max().item())
            grad = torch.randn_like(output)
            d_twiddle, d_input = torch.autograd.grad(output, (factor.ABCD, prev), grad, retain_graph=True)
            d_twiddle_slow, d_input_slow = torch.autograd.grad(output_slow, (factor.ABCD, prev), grad, retain_graph=True)
            print((d_input - d_input_slow).abs().max().item())
            self.assertTrue(torch.allclose(d_twiddle, d_twiddle_slow, rtol=self.rtol, atol=self.atol), (d_twiddle - d_twiddle_slow).abs().max().item())
            self.assertTrue(torch.allclose(d_input, d_input_slow, rtol=self.rtol, atol=self.atol), (d_input - d_input_slow).abs().max().item())


    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_factor_cuda(self):
        pass


if __name__ == "__main__":
    unittest.main()
