import math
import operator
import functools

import torch
from torch import nn

from butterfly.complex_utils import real_to_complex, complex_mul, complex_matmul

from factor_multiply import permutation_factor_even_odd_multiply, permutation_factor_even_odd_multiply_backward
from factor_multiply import permutation_factor_reverse_multiply, permutation_factor_reverse_multiply_backward


class PermutationFactorEvenOddMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, p, input):
        ctx.save_for_backward(p, input)
        return permutation_factor_even_odd_multiply(p, input)

    @staticmethod
    def backward(ctx, grad):
        p, input = ctx.saved_tensors
        d_p, d_input = permutation_factor_even_odd_multiply_backward(grad, p, input)
        return d_p, d_input


permutation_factor_even_odd_mult = PermutationFactorEvenOddMult.apply


class PermutationFactorReverseMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, p, input):
        ctx.save_for_backward(p, input)
        return permutation_factor_reverse_multiply(p, input)

    @staticmethod
    def backward(ctx, grad):
        p, input = ctx.saved_tensors
        d_p, d_input = permutation_factor_reverse_multiply_backward(grad, p, input)
        return d_p, d_input


permutation_factor_reverse_mult = PermutationFactorReverseMult.apply


def test_permutation_factor_even_odd_multiply():
    import time
    n = 1024
    m = int(math.log2(n))
    x = torch.randn(n, requires_grad=True)
    sizes = [n >> i for i in range(m)]
    # first = time.perf_counter()
    for size in sizes:
        x = x.reshape(-1, size)
        p = torch.randn(3, requires_grad=True)
        result_slow = ((1 - p[0]) * x.reshape(x.shape[:-1] + (2, x.shape[-1] // 2)) + p[0] * x.reshape(x.shape[:-1] + (x.shape[-1] // 2, 2)).transpose(-1, -2)).reshape(x.shape)
        # start = time.perf_counter()
        result = permutation_factor_even_odd_mult(p[:1], x)
        # [permutation_factor_even_odd_mult(bf.ABCD, x.reshape(-1, 2, bf.ABCD.shape[-1])).reshape(x.shape) for _ in range(10000)]
        # end = time.perf_counter()
        # print(end - start)
        assert torch.allclose(result, result_slow, atol=1e-6)
        grad = torch.randn_like(x)
        d_p_slow, d_x_slow = torch.autograd.grad(result_slow, (p, x), grad, retain_graph=True)
        d_p, d_x = torch.autograd.grad(result, (p, x), grad, retain_graph=True)
        assert torch.allclose(d_p, d_p_slow, atol=1e-6)
        assert torch.allclose(d_x, d_x_slow, atol=1e-6)
    # last = time.perf_counter()
    # print(last - first)


def test_permutation_factor_reverse_multiply():
    import time
    n = 1024
    m = int(math.log2(n))
    x = torch.randn(n, requires_grad=True)
    sizes = [n >> i for i in range(m)]
    # first = time.perf_counter()
    for size in sizes:
        x = x.reshape(-1, size)
        p = torch.randn(3, requires_grad=True)
        result_slow = ((1 - p[1:]).unsqueeze(-1) * x.reshape(-1, 2, x.shape[-1] // 2) + p[1:].unsqueeze(-1) * x.reshape((-1, 2, x.shape[-1] // 2)).flip(-1)).reshape(x.shape)
        # start = time.perf_counter()
        result = permutation_factor_reverse_mult(p[1:], x)
        # [permutation_factor_reverse_mult(bf.ABCD, x.reshape(-1, 2, bf.ABCD.shape[-1])).reshape(x.shape) for _ in range(10000)]
        # end = time.perf_counter()
        # print(end - start)
        assert torch.allclose(result, result_slow, atol=1e-6)
        grad = torch.randn_like(x)
        d_p_slow, d_x_slow = torch.autograd.grad(result_slow, (p, x), grad, retain_graph=True)
        d_p, d_x = torch.autograd.grad(result, (p, x), grad, retain_graph=True)
        assert torch.allclose(d_p, d_p_slow, atol=1e-6)
        assert torch.allclose(d_x, d_x_slow, atol=1e-6)
    # last = time.perf_counter()
    # print(last - first)
