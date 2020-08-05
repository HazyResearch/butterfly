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


# In Pytorch 1.6, torch.view_as_real and torch.view_as_complex conjugate their gradients.
# This follow Jax's convention. However, we currently follow Tensorflow's convention, where
# the gradient should be as if everything is done with real numbers.
# See the discussion here: https://github.com/pytorch/pytorch/issues/41857
# As a result, we redefine these functions with the gradient following Tensorflow's convention.
# For now, DO NOT use torch.view_as_real and torch.view_as_complex directly.
# Only use view_as_real and view_as_complex defined in this file.

class ViewAsReal(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return torch.view_as_real(X)

    @staticmethod
    def backward(ctx, grad):
        return torch.view_as_complex(grad)


view_as_real = ViewAsReal.apply


class ViewAsComplex(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        return torch.view_as_complex(X)

    @staticmethod
    def backward(ctx, grad):
        return torch.view_as_real(grad)


view_as_complex = ViewAsComplex.apply


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
