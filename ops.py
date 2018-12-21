import numpy as np
import torch
from torch.nn import functional as F

from numpy.polynomial import chebyshev, legendre


def polymatmul(A, B):
    """Batch-multiply two matrices of polynomials
    Parameters:
        A: (N, batch_size, n, m, d1)
        B: (batch_size, m, p, d2)
    Returns:
        AB: (N, batch_size, n, p, d1 + d2 - 1)
    """
    unsqueezed = False
    if A.dim() == 4:
        unsqueezed = True
        A = A.unsqueeze(0)
    N, batch_size, n, m, d1 = A.shape
    batch_size_, m_, p, d2 = B.shape
    assert batch_size == batch_size_
    assert m == m_
    # Naive implementation using conv1d and loop, slower but easier to understand
    # Bt_flipped = B.transpose(1, 2).flip(-1)
    # result = torch.stack([
    #     F.conv1d(A[:, i].reshape(-1, m, d1), Bt_flipped[i], padding=d2 - 1).reshape(N, n, p, -1)
    #     for i in range(batch_size)
    # ], dim=1)
    # Batched implementation using grouped convolution, faster
    result = F.conv1d(A.transpose(1, 2).reshape(N * n, batch_size * m, d1),
                      B.transpose(1, 2).reshape(batch_size * p, m, d2).flip(-1),
                      padding=d2 - 1,
                      groups=batch_size).reshape(N, n, batch_size, p, d1 + d2 - 1).transpose(1, 2)
    return result.squeeze(0) if unsqueezed else result


def ops_transpose_mult(a, b, c, p0, p1, v):
    """Fast algorithm to multiply P^T v where P is the matrix of coefficients of
    OPs, specified by the coefficients a, b, c, and the starting polynomials p0,
    p_1.
    In particular, the recurrence is
    P_{n+2}(x) = (a[n] x + b[n]) P_{n+1}(x) + c[n] P_n(x).
    Parameters:
        a: array of length n
        b: array of length n
        c: array of length n
        p0: real number representing P_0(x).
        p1: pair of real numbers representing P_1(x).
        v: (batch_size, n)
    Return:
        result: P^T v.
    """
    n = v.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, "Length n must be a power of 2."

    # Preprocessing: compute T_{i:j}, the transition matrix from p_i to p_j.
    T = [None] * (m + 1)
    # Lowest level, filled with T_{i:i+1}
    # n matrices, each 2 x 2, with coefficients being polynomials of degree <= 1
    T[0] = torch.zeros(n, 2, 2, 2)
    T[0][:, 0, 0, 1] = a
    T[0][:, 0, 0, 0] = b
    T[0][:, 0, 1, 0] = c
    T[0][:, 1, 0, 0] = 1.0
    for i in range(1, m + 1):
        T[i] = polymatmul(T[i - 1][1::2], T[i - 1][::2])

    P_init = torch.tensor([p1, [p0, 0.0]], dtype=torch.float)  # [p_1, p_0]
    P_init = P_init.unsqueeze(0).unsqueeze(-2)
    # Check that T is computed correctly
    # These should be the polynomials P_{n+1} and P_n
    # Pnp1n = polymatmul(T[m], P_init).squeeze()

    # Bottom-up multiplication algorithm to avoid recursion
    S = [None] * m
    Tidentity = torch.eye(2).unsqueeze(0).unsqueeze(3)
    S[0] = v[:, 1::2, None, None, None] * T[0][::2]
    S[0][:, :, :, :, :1] += v[:, ::2, None, None, None] * Tidentity
    for i in range(1, m):
        S[i] = polymatmul(S[i - 1][:, 1::2], T[i][::2])
        S[i][:, :, :, :, :S[i - 1].shape[-1]] += S[i - 1][:, ::2]
    result = polymatmul(S[m - 1][:, :, [1], :, :n-1], P_init).squeeze(1).squeeze(1).squeeze(1)
    return result


def chebyshev_transpose_mult_slow(v):
    """Naive multiplication P^T v where P is the matrix of coefficients of
    Chebyshev polynomials.
    Parameters:
        v: (batch_size, n)
    Return:
        P^T v: (batch_size, n)
    """
    n = v.shape[-1]
    # Construct the coefficient matrix P for Chebyshev polynomials
    P = np.zeros((n, n), dtype=np.float32)
    for i, coef in enumerate(np.eye(n)):
        P[i, :i + 1] = chebyshev.cheb2poly(coef)
    P = torch.tensor(P)
    return v @ P


def legendre_transpose_mult_slow(v):
    """Naive multiplication P^T v where P is the matrix of coefficients of
    Legendre polynomials.
    Parameters:
        v: (batch_size, n)
    Return:
        P^T v: (batch_size, n)
    """
    n = v.shape[-1]
    # Construct the coefficient matrix P for Legendre polynomials
    P = np.zeros((n, n), dtype=np.float32)
    for i, coef in enumerate(np.eye(n)):
        P[i, :i + 1] = legendre.leg2poly(coef)
    P = torch.tensor(P)
    return v @ P


def ops_transpose_mult_test():
    n = 8
    batch_size = 2
    v = torch.randn(batch_size, n)
    # Chebyshev polynomials
    result = ops_transpose_mult(2.0 * torch.ones(n), torch.zeros(n), -torch.ones(n), 1.0, (0.0, 1.0), v)
    result_slow = chebyshev_transpose_mult_slow(v)
    assert torch.allclose(result, result_slow)
    # Legendre polynomials
    n_range = torch.arange(n, dtype=torch.float)
    result = ops_transpose_mult((2 * n_range + 3) / (n_range + 2), torch.zeros(n), -(n_range + 1) / (n_range + 2), 1.0, (0.0, 1.0), v)
    result_slow = legendre_transpose_mult_slow(v)
    assert torch.allclose(result, result_slow)


if __name__ == '__main__':
    ops_transpose_mult_test()
