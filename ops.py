import numpy as np
import torch
from torch.nn import functional as F

from numpy.polynomial import chebyshev, legendre


def polymatmul(A, B):
    """Multiply two matrices of polynomials
    Parameters:
        A: (batchsize, n, m, d1)
        B: (batchsize, m, p, d2)
    Returns:
        AB: (batchsize, n, p, d1 + d2 - 1)
    """
    batchsize, n, m, d1 = A.shape
    batchsize_, m_, p, d2 = B.shape
    assert batchsize == batchsize_
    assert m == m_
    # Need to transpose B
    Bt = B.transpose(1, 2)
    # TODO: Figure out how to batch this. Maybe I'll need to use my own FFT-based multiplication
    result = torch.stack([
        F.conv1d(A_, Bt_.flip(-1), padding=Bt_.shape[-1] - 1)
        for A_, Bt_ in zip(A, Bt)
    ])
    return result


def ops_transpose_mult(a, b, c, p0, pm1, v):
    """Fast algorithm to multiply P^T v where P is the matrix of coefficients of
    OPs, specified by the coefficients a, b, c, and the starting polynomials p0,
    p_{-1}.
    In particular, the recurrence is P_{n+1}(x) = (a[n] x + b[n]) P_n(x) + c[n] P_{n-1}(x).
    Parameters:
        a: array of length n
        b: array of length n
        c: array of length n
        p0: real number representing P_0(x).
        pm1: pair of real numbers representing P_{-1}(x).
        v: array of length n
    Return:
        result: P^T v.
    """
    n = v.shape[0]
    m = int(np.log2(n))
    assert n == 1 << m, "Length n must be a power of 2."

    # Preprocessing: compute T_{i:j}, the transition matrix from p_i to p_j.
    T = [None] * (m + 1)
    # Lowest level, filled with T_{i:i+1}
    # n matrices, each 2 x 2, with coefficients being polynomials of degree <= 1
    T[0] = torch.zeros(n, 2, 2, 2)
    T[0][:, 0, 0, 1] = a[:n]
    T[0][:, 0, 0, 0] = b[:n]
    T[0][:, 0, 1, 0] = c[:n]
    T[0][:, 1, 0, 0] = 1.0
    for i in range(1, m + 1):
        T[i] = polymatmul(T[i - 1][1::2], T[i - 1][::2])
    # Check that T is computed correctly
    P_init = torch.tensor([[p0, 0], pm1], dtype=torch.float)  # [p_1, p_0]
    P_init = P_init.unsqueeze(0).unsqueeze(-2)
    # These should be the polynomials P_{n+1} and P_n
    Pnp1n = polymatmul(T[m], P_init).squeeze()

    # Bottom-up multiplication algorithm to avoid recursion
    S = [None] * m
    Tidentity = torch.eye(2).unsqueeze(0).unsqueeze(3)
    S[0] = v[1::2, None, None, None] * T[0][::2]
    S[0][:, :, :, :1] += v[::2, None, None, None] * Tidentity
    for i in range(1, m):
        S[i] = polymatmul(S[i - 1][1::2], T[i][::2])
        S[i][:, :, :, :S[i - 1].shape[-1]] += S[i - 1][::2]
    result = polymatmul(S[m - 1], P_init)[:, 0, :, :n].squeeze()
    return result


def chebyshev_transpose_mult_slow(v):
    """Naive multiplication P^T v where P is the matrix of coefficients of
    Chebyshev polynomials.
    Parameters:
        v: array of length n
    Return:
        P^T v: array of length n
    """
    n = v.shape[0]
    # Construct the coefficient matrix P for Chebyshev polynomials
    P = np.zeros((n, n), dtype=np.float32)
    for i, coef in enumerate(np.eye(n)):
        P[i, :i + 1] = chebyshev.cheb2poly(coef)
    P = torch.tensor(P)
    return P.t() @ v


def legendre_transpose_mult_slow(v):
    """Naive multiplication P^T v where P is the matrix of coefficients of
    Legendre polynomials.
    Parameters:
        v: array of length n
    Return:
        P^T v: array of length n
    """
    n = v.shape[0]
    # Construct the coefficient matrix P for Legendre polynomials
    P = np.zeros((n, n), dtype=np.float32)
    for i, coef in enumerate(np.eye(n)):
        P[i, :i + 1] = legendre.leg2poly(coef)
    P = torch.tensor(P)
    return P.t() @ v



def ops_transpose_mult_test():
    n = 8
    v = torch.randn(n)
    # Chebyshev polynomials
    result = ops_transpose_mult(2.0 * torch.ones(n), torch.zeros(n), -torch.ones(n), 1.0, (0.0, 1.0), v)
    result_slow = chebyshev_transpose_mult_slow(v)
    assert torch.allclose(result, result_slow)
    # Legendre polynomials
    n_range = torch.arange(n, dtype=torch.float)
    result = ops_transpose_mult((2 * n_range + 1) / (n_range + 1), torch.zeros(n), -n_range / (n_range + 1), 1.0, (0.0, 1.0), v)
    result_slow = legendre_transpose_mult_slow(v)
    assert torch.allclose(result, result_slow)
