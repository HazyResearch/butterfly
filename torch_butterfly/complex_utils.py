import torch
from torch import nn


real_dtype_to_complex = {torch.float32: torch.complex64, torch.float64: torch.complex128}
complex_dtype_to_real = {torch.complex64: torch.float32, torch.complex128: torch.float64}


# Autograd for complex isn't implemented yet so we have to manually write the backward
class ComplexMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return X * Y

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = None, None
        if ctx.needs_input_grad[0]:
            grad_X = (grad * Y.conj()).sum_to_size(*X.shape)
        if ctx.needs_input_grad[1]:
            grad_Y = (grad * X.conj()).sum_to_size(*Y.shape)
        return grad_X, grad_Y


complex_mul = ComplexMul.apply


def real2complex(X):
    return X.to(real_dtype_to_complex[X.dtype])


def complex2real(X):
    return X.to(complex_dtype_to_real[X.dtype])


# nn.Module form just to support convenient use of nn.Sequential
class Real2Complex(nn.Module):
    def forward(self, input):
        return real2complex(input)


class Complex2Real(nn.Module):
    def forward(self, input):
        return complex2real(input)
