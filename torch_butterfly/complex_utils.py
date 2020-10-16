import numpy as np

import torch
from torch import nn
from torch.utils.dlpack import to_dlpack, from_dlpack

# Check if cupy is available
if torch.cuda.is_available():
    use_cupy = True
    try:
        import cupy as cp
    except:
        use_cupy = False
        import warnings
        warnings.warn("Cupy isn't installed or isn't working properly. Will use Pytorch's complex matmul, which is slower.")
else:
    use_cupy = False


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


complex_torch_dtype_to_np = {torch.complex64: np.complex64, torch.complex128: np.complex128}
complex_np_dtype_to_real = {np.complex64: np.float32, np.complex128: np.float64,
                            cp.dtype('complex64'): np.float32, cp.dtype('complex128'): np.float64}

def torch2np(X):
    """Convert a torch tensor to a numpy array, sharing the same memory.
    """
    return X.detach().numpy()


def np2torch(X):
    return torch.from_numpy(X)


def torch2cp(tensor):
    # Need contiguous, or else it will error
    return cp.fromDlpack(to_dlpack(torch.view_as_real(tensor.cuda().contiguous()))).view(
        complex_torch_dtype_to_np[tensor.dtype]).squeeze(-1)


def cp2torch(tensor):
    return torch.view_as_complex(from_dlpack(cp.ascontiguousarray(tensor)[..., None].view(
        complex_np_dtype_to_real[tensor.dtype]).toDlpack()))


def complex_matmul_torch(X, Y):
    # return X.real @ Y.real - X.imag @ Y.imag + 1j * (X.real @ Y.imag + X.imag @ Y.real)
    return view_as_complex(torch.stack([X.real @ Y.real - X.imag @ Y.imag,
                                        X.real @ Y.imag + X.imag @ Y.real], dim=-1))


class ComplexMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        # return view_as_complex(torch.stack([X.real @ Y.real - X.imag @ Y.imag,
        #                                     X.real @ Y.imag + X.imag @ Y.real], dim=-1))
        # return complex_matmul_torch(X, Y)
        if not X.is_cuda:
            return np2torch(torch2np(X) @ torch2np(Y))
        else:
            return (cp2torch(torch2cp(X) @ torch2cp(Y))
                    if use_cupy else complex_matmul_torch(X, Y))

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = None, None
        if ctx.needs_input_grad[0]:
            Y_t = Y.transpose(-1, -2)
            # grad_X = (grad @ Y_t.conj()).sum_to_size(*X.shape)
            # grad_X = view_as_complex(
            #     torch.stack([grad.real @ Y_t.real + grad.imag @ Y_t.imag,
            #                  -grad.real @ Y_t.imag + grad.imag @ Y_t.real], dim=-1)
            # ).sum_to_size(*X.shape)
            # grad_X = complex_matmul_torch(grad, Y_t.conj()).sum_to_size(*X.shape)
            if not Y.is_cuda:
                grad_X = np2torch(torch2np(grad) @ torch2np(Y_t.conj())).sum_to_size(*X.shape)
            else:
                grad_X = (cp2torch(torch2cp(grad) @ torch2cp(Y_t.conj())) if use_cupy
                          else complex_matmul_torch(grad, Y_t.conj())).sum_to_size(*X.shape)
        if ctx.needs_input_grad[1]:
            X_t = X.transpose(-1, -2)
            # grad_Y = (X_t.conj() @ grad).sum_to_size(*Y.shape)
            # grad_Y = view_as_complex(
            #     torch.stack([X_t.real @ grad.real + X_t.imag @ grad.imag,
            #                  X_t.real @ grad.imag - X_t.imag @ grad.real], dim=-1)
            # ).sum_to_size(*Y.shape)
            # grad_Y = complex_matmul_torch(X_t.conj(), grad).sum_to_size(*Y.shape)
            if not X.is_cuda:
                grad_Y = np2torch(torch2np(X_t.conj()) @ torch2np(grad)).sum_to_size(*Y.shape)
            else:
                grad_Y = (cp2torch(torch2cp(X_t.conj()) @ torch2cp(grad)) if use_cupy
                          else complex_matmul_torch(X_t.conj(), grad)).sum_to_size(*Y.shape)
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
