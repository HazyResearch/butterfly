import math
import operator
import functools

import torch
from torch import nn

from complex_utils import real_to_complex, complex_mul, complex_matmul
from butterfly import Block2x2Diag, Block2x2DiagProduct

from butterfly_factor_multiply import butterfly_factor_multiply


class ButterflyFactorMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coefficients, input):
        ctx.save_for_backward(coefficients, input)
        return butterfly_factor_multiply(coefficients, input)

    @staticmethod
    def backward(ctx, grad):
        coefficients, input = ctx.saved_tensors
        assert grad.shape == input.shape
        d_coefficients = torch.einsum('abc, adc -> bdc', (grad, input))
        d_input = ButterflyFactorMult.apply(coefficients.transpose(0, 1), grad)
        return d_coefficients, d_input

butterfly_factor_mult = ButterflyFactorMult.apply


def test_butterfly_factor_multiply():
    n = 512
    m = int(math.log2(n))
    x = torch.randn(n, requires_grad=True)
    sizes = [n >> i for i in range(m)]
    for size in sizes:
        bf = Block2x2Diag(size)
        x = x.reshape(-1, 2 * bf.ABCD.shape[-1])
        result_slow = bf(x)
        result = butterfly_factor_mult(bf.ABCD, x.reshape(-1, 2, bf.ABCD.shape[-1])).reshape(x.shape)
        assert torch.allclose(result, result_slow)
        grad = torch.randn_like(x)
        d_coef_slow, d_x_slow = torch.autograd.grad(result_slow, (bf.ABCD, x), grad, retain_graph=True)
        d_coef, d_x = torch.autograd.grad(result, (bf.ABCD, x), grad, retain_graph=True)
        assert torch.allclose(d_coef, d_coef_slow)
        assert torch.allclose(d_x, d_x_slow)
