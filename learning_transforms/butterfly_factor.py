import math

import torch
from torch import nn

from complex_utils import real_to_complex, complex_mul, complex_matmul
# from butterfly import Block2x2Diag, Block2x2DiagProduct

from factor_multiply import butterfly_factor_multiply, butterfly_factor_multiply_backward
from factor_multiply import butterfly_factor_multiply_inplace, butterfly_factor_multiply_inplace_backward
from factor_multiply import butterfly_factor_multiply_intermediate, butterfly_factor_multiply_intermediate_backward
from ABCD_mult import ABCD_mult


class ButterflyFactorMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, input):
        ctx.save_for_backward(coefficients, input)
        return butterfly_factor_multiply(coefficients, input)
        # output = torch.empty_like(input)
        # ABCD_mult(coefficients.detach().numpy(), input.detach().numpy(), output.detach().numpy())
        # return output

    @staticmethod
    def backward(ctx, grad):
        coefficients, input = ctx.saved_tensors
        # assert grad.shape == input.shape
        # d_coefficients = torch.einsum('abc, adc -> bdc', (grad, input))
        # d_input = ButterflyFactorMult.apply(coefficients.transpose(0, 1), grad)
        # return d_coefficients, d_input
        d_coefficients, d_input = butterfly_factor_multiply_backward(grad, coefficients, input)
        # d_coefficients = torch.zeros_like(coefficients)
        # d_input = torch.zeros_like(input)
        # d_coefficients = (grad.permute(2, 1, 0) @ input.permute(2, 0, 1)).permute(1, 2, 0)  # Extremely slow on CUDA
        # d_input = butterfly_factor_multiply(coefficients.transpose(0, 1), grad)
        return d_coefficients, d_input

butterfly_factor_mult = ButterflyFactorMult.apply


class ButterflyFactorMultInplace(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        output = butterfly_factor_multiply_inplace(twiddle, input)
        ctx.save_for_backward(twiddle, output)
        return output

    @staticmethod
    def backward(ctx, grad):
        twiddle, output = ctx.saved_tensors
        d_coefficients, d_input = butterfly_factor_multiply_inplace_backward(grad, twiddle, output)
        return d_coefficients, d_input

butterfly_factor_mult_inplace = ButterflyFactorMultInplace.apply


class ButterflyFactorMultIntermediate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        output = butterfly_factor_multiply_intermediate(twiddle, input)
        ctx.save_for_backward(twiddle, output)
        return output[-1]

    @staticmethod
    def backward(ctx, grad):
        twiddle, output = ctx.saved_tensors
        d_coefficients, d_input = butterfly_factor_multiply_intermediate_backward(grad, twiddle, output)
        return d_coefficients, d_input

butterfly_factor_mult_intermediate = ButterflyFactorMultIntermediate.apply


def test_butterfly_factor_multiply():
    import time
    n = 1024
    batch_size = 1000
    ntrials = 100
    m = int(math.log2(n))
    x = torch.randn(n, requires_grad=True)
    sizes = [n >> i for i in range(m)]
    first = time.perf_counter()
    for size in sizes:
        bf = Block2x2Diag(size)
        x = x.view(-1, 2 * bf.ABCD.shape[-1])
        # result_slow = bf(x)
        start = time.perf_counter()
        result = butterfly_factor_mult(bf.ABCD, x.view(-1, 2, bf.ABCD.shape[-1])).view(x.shape)
        [butterfly_factor_mult(bf.ABCD, x.view(-1, 2, bf.ABCD.shape[-1])).view(x.shape) for _ in range(ntrials)]
        # assert torch.allclose(result, result_slow)
        grad = torch.randn_like(x)
        # d_coef_slow, d_x_slow = torch.autograd.grad(result_slow, (bf.ABCD, x), grad, retain_graph=True)
        # d_coef, d_x = torch.autograd.grad(result, (bf.ABCD, x), grad, retain_graph=True)
        [torch.autograd.grad(result, (bf.ABCD, x), grad, retain_graph=True) for _ in range(ntrials)]
        end = time.perf_counter()
        print(end - start)
        # assert torch.allclose(d_coef, d_coef_slow)
        # assert torch.allclose(d_x, d_x_slow)
    last = time.perf_counter()
    print(last - first)


def test_butterfly_factor_multiply_bmm():
    import time
    n = 1024
    batch_size = 1000
    ntrials = 100
    m = int(math.log2(n))
    x = torch.randn(n, requires_grad=True)
    sizes = [n >> i for i in range(m)]
    first = time.perf_counter()
    for size in sizes:
        bf = Block2x2Diag(size)
        ABCD = bf.ABCD.permute(2, 0, 1).clone()
        x = x.view(ABCD.shape[0], 2, -1)
        start = time.perf_counter()
        result = ABCD @ x
        [ABCD @ x for _ in range(ntrials)]
        # assert torch.allclose(result, result_slow)
        grad = torch.randn_like(x)
        # d_coef, d_x = torch.autograd.grad(result, (ABCD, x), grad, retain_graph=True)
        [torch.autograd.grad(result, (ABCD, x), grad, retain_graph=True) for _ in range(ntrials)]
        end = time.perf_counter()
        print(end - start)
        # assert torch.allclose(d_coef, d_coef_slow)
        # assert torch.allclose(d_x, d_x_slow)
    last = time.perf_counter()
    print(last - first)


def test_butterfly_factor_complex_multiply():
    from complex_utils import complex_mul
    n = 1024
    m = int(math.log2(n))
    x = torch.randn((n, 2), requires_grad=True)
    sizes = [n >> i for i in range(m)]
    for size in sizes:
        bf = Block2x2Diag(size, complex=True)
        x = x.view(-1, 2 * bf.ABCD.shape[-2], 2)
        result_slow = (complex_mul(bf.ABCD, x.view(x.shape[:-2] + (1, 2, size // 2, 2))).sum(dim=-3)).view(x.shape)
        result = butterfly_factor_mult(bf.ABCD, x.view(-1, 2, bf.ABCD.shape[-2], 2)).view(x.shape)
        assert torch.allclose(result, result_slow, atol=1e-6)
        grad = torch.randn_like(x)
        d_coef_slow, d_x_slow = torch.autograd.grad(result_slow, (bf.ABCD, x), grad, retain_graph=True)
        d_coef, d_x = torch.autograd.grad(result, (bf.ABCD, x), grad, retain_graph=True)
        assert torch.allclose(d_coef, d_coef_slow, atol=1e-6)
        assert torch.allclose(d_x, d_x_slow, atol=1e-6)
