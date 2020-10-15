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


class ComplexMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return view_as_complex(torch.stack([X.real @ Y.real - X.imag @ Y.imag,
                                            X.real @ Y.imag + X.imag @ Y.real], dim=-1))

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = None, None
        if ctx.needs_input_grad[0]:
            Y_t = Y.transpose(-1, -2)
            # grad_X = (grad @ Y_t.conj()).sum_to_size(*X.shape)
            grad_X = view_as_complex(
                torch.stack([grad.real @ Y_t.real + grad.imag @ Y_t.imag,
                             -grad.real @ Y_t.imag + grad.imag @ Y_t.real], dim=-1)
            ).sum_to_size(*X.shape)
        if ctx.needs_input_grad[1]:
            X_t = X.transpose(-1, -2)
            # grad_Y = (X_t.conj() @ grad).sum_to_size(*Y.shape)
            grad_Y = view_as_complex(
                torch.stack([X_t.real @ grad.real + X_t.imag @ grad.imag,
                             X_t.real @ grad.imag - X_t.imag @ grad.real], dim=-1)
            ).sum_to_size(*Y.shape)
        return grad_X, grad_Y


complex_matmul = ComplexMatmul.apply


# In Pytorch 1.6, torch.view_as_real and torch.view_as_complex conjugate their gradients.
# This follows Jax's convention. However, we currently follow Tensorflow's convention, where
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


# Pytorch 1.6.0 doesn't have indexing_backward for complex on GPU so we have to write the backward
# pass explicitly
class IndexLastDim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, permutation):
        ctx.save_for_backward(permutation)
        return X[..., permutation]

    @staticmethod
    def backward(ctx, grad):
        permutation, = ctx.saved_tensors
        output = torch.empty_like(grad)
        output[..., permutation] = grad
        return output, None


index_last_dim = IndexLastDim.apply
