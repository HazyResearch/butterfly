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
        # import warnings
        # warnings.warn("Cupy isn't installed or isn't working properly. Will use Pytorch's complex matmul, which is slower.")
else:
    use_cupy = False


real_dtype_to_complex = {torch.float32: torch.complex64, torch.float64: torch.complex128}

complex_torch_dtype_to_np = {torch.complex64: np.complex64, torch.complex128: np.complex128}
if use_cupy:
    complex_np_dtype_to_real = {np.complex64: np.float32, np.complex128: np.float64,
                                cp.dtype('complex64'): np.float32,
                                cp.dtype('complex128'): np.float64}


def torch2cp(tensor):
    # Need contiguous, or else it will error
    return cp.fromDlpack(to_dlpack(torch.view_as_real(tensor.cuda().contiguous()))).view(
        complex_torch_dtype_to_np[tensor.dtype]).squeeze(-1)


def cp2torch(tensor):
    return torch.view_as_complex(from_dlpack(cp.ascontiguousarray(tensor)[..., None].view(
        complex_np_dtype_to_real[tensor.dtype]).toDlpack()))


def complex_matmul_torch(X, Y):
    # return X.real @ Y.real - X.imag @ Y.imag + 1j * (X.real @ Y.imag + X.imag @ Y.real)
    return torch.view_as_complex(torch.stack([X.real @ Y.real - X.imag @ Y.imag,
                                              X.real @ Y.imag + X.imag @ Y.real], dim=-1))


class ComplexMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        # return torch.view_as_complex(torch.stack([X.real @ Y.real - X.imag @ Y.imag,
        #                                           X.real @ Y.imag + X.imag @ Y.real], dim=-1))
        # return complex_matmul_torch(X, Y)
        if not X.is_cuda:
            return X @ Y
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
            # grad_X = torch.view_as_complex(
            #     torch.stack([grad.real @ Y_t.real + grad.imag @ Y_t.imag,
            #                  -grad.real @ Y_t.imag + grad.imag @ Y_t.real], dim=-1)
            # ).sum_to_size(*X.shape)
            # grad_X = complex_matmul_torch(grad, Y_t.conj()).sum_to_size(*X.shape)
            if not Y.is_cuda:
                grad_X = (grad @ Y_t.conj()).sum_to_size(*X.shape)
            else:
                grad_X = (cp2torch(torch2cp(grad) @ torch2cp(Y_t.conj())) if use_cupy
                          else complex_matmul_torch(grad, Y_t.conj())).sum_to_size(*X.shape)
        if ctx.needs_input_grad[1]:
            X_t = X.transpose(-1, -2)
            # grad_Y = (X_t.conj() @ grad).sum_to_size(*Y.shape)
            # grad_Y = torch.view_as_complex(
            #     torch.stack([X_t.real @ grad.real + X_t.imag @ grad.imag,
            #                  X_t.real @ grad.imag - X_t.imag @ grad.real], dim=-1)
            # ).sum_to_size(*Y.shape)
            # grad_Y = complex_matmul_torch(X_t.conj(), grad).sum_to_size(*Y.shape)
            if not X.is_cuda:
                grad_Y = (X_t.conj() @ grad).sum_to_size(*Y.shape)
            else:
                grad_Y = (cp2torch(torch2cp(X_t.conj()) @ torch2cp(grad)) if use_cupy
                          else complex_matmul_torch(X_t.conj(), grad)).sum_to_size(*Y.shape)
        return grad_X, grad_Y


def complex_matmul(X, Y):
    return X @ Y if not X.is_complex() else ComplexMatmul.apply(X, Y)


def real2complex(X):
    return X.to(real_dtype_to_complex[X.dtype])


# nn.Module form just to support convenient use of nn.Sequential
class Real2Complex(nn.Module):
    def forward(self, input):
        return real2complex(input)


class Complex2Real(nn.Module):
    def forward(self, input):
        return input.real


# Pytorch 1.7 doesn't have indexing_backward for complex so we have to write the backward
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


# Pytorch 1.7 doesn't support complex reshape backward for non-contiguous tensors (fixed in nightly)
def complex_reshape(x, *shape):
    if not x.is_complex():
        return x.reshape(*shape)
    else:
        return torch.view_as_complex(torch.view_as_real(x).reshape(*shape, 2))
