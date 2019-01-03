''' Utility functions for handling complex tensors: conjugate and complex_mul.
Pytorch (as of 1.0) does not support complex tensors, so we store them as
float tensors where the last dimension is 2 (real and imaginary parts).
'''

import numpy as np
import torch


def conjugate(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


def complex_mul(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)

def complex_matmul_torch(X, Y):
    """Multiply two complex matrices.
    Parameters:
       X: (..., n, m, 2)
       Y: (..., m, p, 2)
    Return:
       Z: (..., n, p, 2)
    """
    return complex_mul(X.unsqueeze(-2), Y.unsqueeze(-4)).sum(dim=-3)


class ComplexMatmulNp(torch.autograd.Function):
    """Multiply two complex matrices, in numpy.
    Parameters:
        X: (n, m, 2)
        Y: (m, p, 2)
    Return:
        Z: (n, p, 2)
    """

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        X_np = X.detach().contiguous().numpy().view('complex64').squeeze(-1)
        Y_np = Y.detach().contiguous().numpy().view('complex64').squeeze(-1)
        prod = torch.from_numpy(np.expand_dims(X_np @ Y_np, -1).view('float32'))
        return prod

    @staticmethod
    def backward(ctx, grad):
        X, Y  = ctx.saved_tensors
        X_np = X.detach().contiguous().numpy().view('complex64').squeeze(-1)
        Y_np = Y.detach().contiguous().numpy().view('complex64').squeeze(-1)
        grad_np = grad.detach().contiguous().numpy().view('complex64').squeeze(-1)
        dX = torch.from_numpy(np.expand_dims(grad_np @ Y_np.conj().T, -1).view('float32'))
        dY = torch.from_numpy(np.expand_dims(X_np.conj().T @ grad_np, -1).view('float32'))
        return dX, dY


complex_matmul = ComplexMatmulNp.apply


def test_complex_mm():
    n = 5
    m = 7
    p = 4
    X = torch.rand(n, m, 2, requires_grad=True)
    Y = torch.rand(m, p, 2, requires_grad=True)
    Z = complex_matmul(X, Y)
    assert Z.shape == (n, p, 2)
    batch_size = 3
    # X = torch.rand(batch_size, n, m, 2)
    # Y = torch.rand(batch_size, m, p, 2)
    # Z = complex_matmul(X, Y)
    # assert Z.shape == (batch_size, n, p, 2)
    X_np = X.detach().contiguous().numpy().view('complex64').squeeze(-1)
    Y_np = Y.detach().contiguous().numpy().view('complex64').squeeze(-1)
    Z_np = np.expand_dims(X_np @ Y_np, axis=-1).view('float32')
    assert np.allclose(Z.numpy(), Z_np)
    Z_torch = complex_matmul_torch(X, Y)
    assert torch.allclose(Z, Z_torch)
    g = torch.rand_like(Z)
    dX, dY = torch.autograd.grad(Z, (X, Y), g)
    dX_torch, dY_torch = torch.autograd.grad(Z_torch, (X, Y), g)
    assert torch.allclose(dX, dX_torch)
    assert torch.allclose(dY, dY_torch)
