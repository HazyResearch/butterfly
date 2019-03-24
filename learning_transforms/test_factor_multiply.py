import unittest

import torch

from butterfly_factor import butterfly_factor_mult, butterfly_factor_mult_inplace
from butterfly import Block2x2DiagProduct
from complex_utils import complex_mul

from factor_multiply import butterfly_factor_multiply_intermediate


def twiddle_list_concat(B: Block2x2DiagProduct):
    # Assume ordering from largest size to smallest size
    if not B.complex:
        return torch.cat([factor.ABCD.permute(2, 0, 1) for factor in B.factors[::-1]])
    else:
        return torch.cat([factor.ABCD.permute(2, 0, 1, 3) for factor in B.factors[::-1]])


class ButterflyFactorTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_butterfly_factor_cpu(self):
        batch_size = 10
        n = 4096
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
            self.assertTrue(torch.allclose(d_twiddle, d_twiddle_slow, rtol=self.rtol, atol=self.atol), (d_twiddle - d_twiddle_slow).abs().max().item())
            self.assertTrue(torch.allclose(d_input, d_input_slow, rtol=self.rtol, atol=self.atol), (d_input - d_input_slow).abs().max().item())


    def test_butterfly_factor_complex_cpu(self):
        batch_size = 10
        n = 4096
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
            self.assertTrue(torch.allclose(d_twiddle, d_twiddle_slow, rtol=self.rtol, atol=self.atol), (d_twiddle - d_twiddle_slow).abs().max().item())
            self.assertTrue(torch.allclose(d_input, d_input_slow, rtol=self.rtol, atol=self.atol), (d_input - d_input_slow).abs().max().item())


    @unittest.skipIf(not torch.cuda.is_available(), "need CUDA")
    def test_butterfly_factor_cuda(self):
        batch_size = 100
        n = 4096  # To test n > MAX_BLOCK_SIZE
        B = Block2x2DiagProduct(n).to('cuda')
        input_ = torch.randn(batch_size, n, device='cuda', requires_grad=True)
        output = input_
        for factor in B.factors[::-1]:
            prev = output
            output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(prev.shape)
            output_slow = ((factor.ABCD * prev.view(-1, 1, 2, factor.size // 2)).sum(dim=-2)).view(prev.shape)
            self.assertTrue(torch.allclose(output, output_slow, rtol=self.rtol, atol=self.atol), (output - output_slow).abs().max().item())
            grad = torch.randn_like(output)
            d_twiddle, d_input = torch.autograd.grad(output, (factor.ABCD, prev), grad, retain_graph=True)
            d_twiddle_slow, d_input_slow = torch.autograd.grad(output_slow, (factor.ABCD, prev), grad, retain_graph=True)
            self.assertTrue(torch.allclose(d_twiddle, d_twiddle_slow, rtol=self.rtol, atol=self.atol), (factor.size, (d_twiddle - d_twiddle_slow).abs().max().item()))
            self.assertTrue(torch.allclose(d_input, d_input_slow, rtol=self.rtol, atol=self.atol), (d_input - d_input_slow).abs().max().item())

    @unittest.skip('Not numerically stable')
    def test_butterfly_factor_inplace_cpu(self):
        batch_size = 10
        n = 4096
        B = Block2x2DiagProduct(n)
        input_ = torch.randn(batch_size, n, requires_grad=True)
        twiddle = twiddle_list_concat(B)
        output_inplace = butterfly_factor_mult_inplace(twiddle, input_)
        output = B(input_)
        self.assertTrue(torch.allclose(output_inplace, output, rtol=self.rtol, atol=self.atol), (output_inplace - output).abs().max().item())
        grad = torch.randn_like(output)
        d_twiddle_inplace, d_input_inplace = torch.autograd.grad(output_inplace, (twiddle, input_), grad, retain_graph=True)
        output.backward(grad, retain_graph=True)
        d_input = input_.grad
        d_twiddle = torch.cat([factor.ABCD.grad.permute(2, 0, 1) for factor in B.factors[::-1]])
        self.assertTrue(torch.allclose(d_input_inplace, d_input, rtol=self.rtol, atol=self.atol), (d_input_inplace - d_input).abs().max().item())
        # self.assertTrue(torch.allclose(d_twiddle_inplace, d_twiddle, rtol=self.rtol, atol=self.atol), (d_twiddle_inplace - d_twiddle).abs().max().item())
        print((d_twiddle_inplace - d_twiddle) / d_twiddle)
        # output = input_
        # for factor in B.factors[::-1]:
        #     prev = output
        #     output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(prev.shape)
        #     d_twiddle, d_input = torch.autograd.grad(output, (factor.ABCD, prev), grad, retain_graph=True)
        # self.assertTrue(torch.allclose(d_twiddle_inplace, d_twiddle, rtol=self.rtol, atol=self.atol), (factor.size, (d_twiddle - d_twiddle_slow).abs().max().item()))

    def test_butterfly_factor_complex_inplace_cpu(self):
        batch_size = 10
        n = 4096
        B = Block2x2DiagProduct(n, complex=True)
        input_ = torch.randn(batch_size, n, 2, requires_grad=True)
        twiddle = twiddle_list_concat(B)
        output_inplace = butterfly_factor_mult_inplace(twiddle, input_)
        output = B(input_)
        self.assertTrue(torch.allclose(output_inplace, output, rtol=self.rtol, atol=self.atol), (output_inplace - output).abs().max().item())

    @unittest.skip('Not numerically stable')
    def test_butterfly_factor_inplace_cuda(self):
        batch_size = 10
        n = 4096
        B = Block2x2DiagProduct(n).to('cuda')
        input_ = torch.randn(batch_size, n, device='cuda', requires_grad=True)
        twiddle = twiddle_list_concat(B)
        output_inplace = butterfly_factor_mult_inplace(twiddle, input_)
        output = B(input_)
        self.assertTrue(torch.allclose(output_inplace, output, rtol=self.rtol, atol=self.atol), (output_inplace - output).abs().max().item())
        grad = torch.randn_like(output)
        d_twiddle_inplace, d_input_inplace = torch.autograd.grad(output_inplace, (twiddle, input_), grad, retain_graph=True)
        output.backward(grad, retain_graph=True)
        d_input = input_.grad
        d_twiddle = torch.cat([factor.ABCD.grad.permute(2, 0, 1) for factor in B.factors[::-1]])
        self.assertTrue(torch.allclose(d_input_inplace, d_input, rtol=self.rtol, atol=self.atol), (d_input_inplace - d_input).abs().max().item())
        # self.assertTrue(torch.allclose(d_twiddle_inplace, d_twiddle, rtol=self.rtol, atol=self.atol), (d_twiddle_inplace - d_twiddle).abs().max().item())
        print((d_twiddle_inplace - d_twiddle) / d_twiddle)
        # print(d_twiddle)
        # print(d_twiddle_inplace)

    def test_butterfly_factor_intermediate_cpu(self):
        batch_size = 10
        n = 4096
        B = Block2x2DiagProduct(n)
        input_ = torch.randn(batch_size, n, requires_grad=True)
        twiddle = twiddle_list_concat(B)
        output_intermediate = butterfly_factor_multiply_intermediate(twiddle, input_)
        output = [input_]
        for factor in B.factors[::-1]:
            output.append(butterfly_factor_mult(factor.ABCD, output[-1].view(-1, 2, factor.size // 2)).view(output[-1].shape))
        output = torch.stack(output)
        self.assertTrue(torch.allclose(output_intermediate, output, rtol=self.rtol, atol=self.atol), (output_intermediate - output).abs().max().item())

    def test_butterfly_factor_intermediate_complex_cpu(self):
        batch_size = 10
        n = 4096
        B = Block2x2DiagProduct(n, complex=True)
        input_ = torch.randn(batch_size, n, 2, requires_grad=True)
        twiddle = twiddle_list_concat(B)
        output_intermediate = butterfly_factor_multiply_intermediate(twiddle, input_)
        output = [input_]
        for factor in B.factors[::-1]:
            output.append(butterfly_factor_mult(factor.ABCD, output[-1].view(-1, 2, factor.size // 2, 2)).view(output[-1].shape))
        output = torch.stack(output)
        self.assertTrue(torch.allclose(output_intermediate, output, rtol=self.rtol, atol=self.atol), (output_intermediate - output).abs().max().item())

    def test_butterfly_factor_intermediate_cuda(self):
        batch_size = 10
        n = 4096
        B = Block2x2DiagProduct(n).to('cuda')
        input_ = torch.randn(batch_size, n, device='cuda', requires_grad=True)
        twiddle = twiddle_list_concat(B)
        output_intermediate = butterfly_factor_multiply_intermediate(twiddle, input_)
        output = [input_]
        for factor in B.factors[::-1]:
            output.append(butterfly_factor_mult(factor.ABCD, output[-1].view(-1, 2, factor.size // 2)).view(output[-1].shape))
        output = torch.stack(output)
        self.assertTrue(torch.allclose(output_intermediate, output, rtol=self.rtol, atol=self.atol), (output_intermediate - output).abs().max().item())



if __name__ == "__main__":
    unittest.main()

    # batch_size = 2
    # n = 4
    # B = Block2x2DiagProduct(n).to('cuda')
    # # input_ = torch.randn(batch_size, n, device='cuda', requires_grad=True)
    # input_ = torch.arange(batch_size * n, dtype=torch.float, device='cuda', requires_grad=True).view(batch_size, n)
    # output = input_
    # factor = B.factors[0]
    # prev = output
    # output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(prev.shape)
    # output_slow = ((factor.ABCD * prev.view(-1, 1, 2, factor.size // 2)).sum(dim=-2)).view(prev.shape)
    # grad = input_
    # d_twiddle, d_input = torch.autograd.grad(output, (factor.ABCD, prev), grad, retain_graph=True)
    # d_twiddle_slow, d_input_slow = torch.autograd.grad(output_slow, (factor.ABCD, prev), grad, retain_graph=True)
    # print(d_twiddle)
    # print(d_twiddle_slow)
    # print((factor.size, (d_twiddle - d_twiddle_slow).abs().max().item()))
