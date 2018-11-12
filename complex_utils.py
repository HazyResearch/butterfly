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

def complex_matmul(X, Y):
    """Multiply two complex matrices.
    Parameters:
       X: (..., n, m, 2)
       Y: (..., m, p, 2)
    Return:
       Z: (..., n, p, 2)
    """
    return complex_mul(X.unsqueeze(-2), Y.unsqueeze(-4)).sum(dim=-3)

def test_complex_mm():
    n = 5
    m = 7
    p = 4
    X = torch.rand(n, m, 2)
    Y = torch.rand(m, p, 2)
    Z = complex_matmul(X, Y)
    assert Z.shape == (n, p, 2)
    batch_size = 3
    X = torch.rand(batch_size, n, m, 2)
    Y = torch.rand(batch_size, m, p, 2)
    Z = complex_matmul(X, Y)
    assert Z.shape == (batch_size, n, p, 2)
    X_np = X.numpy().view('complex64').squeeze(-1)
    Y_np = Y.numpy().view('complex64').squeeze(-1)
    Z_np = np.expand_dims(X_np @ Y_np, axis=-1).view('float32')
    assert np.allclose(Z.numpy(), Z_np)
