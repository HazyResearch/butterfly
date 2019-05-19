"""Target matrices to factor: DFT, DCT, Hadamard, convolution, Legendre, Vandermonde.
Complex complex must be converted to real matrices with 2 as the last dimension
(for Pytorch's compatibility).
"""

import math

import numpy as np
from numpy.polynomial import legendre
import scipy.linalg as LA
from scipy.fftpack import dct, dst, fft2

# Copied from https://stackoverflow.com/questions/23869694/create-nxn-haar-matrix
def haar_matrix(n, normalized=False):
    # Allow only size n of power 2
    n = 2**np.ceil(np.log2(n))
    if n > 2:
        h = haar_matrix(n / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    # calculate upper haar part
    h_n = np.kron(h, [1, 1])
    # calculate lower haar part
    if normalized:
        h_i = np.sqrt(n/2)*np.kron(np.eye(len(h)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(h)), [1, -1])
    # combine parts
    h = np.vstack((h_n, h_i))
    return h


def hartley_matrix(n):
    """Matrix corresponding to the discrete Hartley transform.
    https://en.wikipedia.org/wiki/Discrete_Hartley_transform
    """
    range_ = np.arange(n)
    indices = np.outer(range_, range_)
    arg = indices * 2 * math.pi / n
    return np.cos(arg) + np.sin(arg)


def hilbert_matrix(n):
    """
    https://en.wikipedia.org/wiki/Hilbert_matrix
    """
    range_ = np.arange(n) + 1
    arg = range_[:, None] + range_ - 1
    return 1.0 / arg


def named_target_matrix(name, size):
    """
    Parameter:
        name: name of the target matrix
    Return:
        target_matrix: (n, n) numpy array for real matrices or (n, n, 2) for complex matrices.
    """
    if name == 'dft':
        return LA.dft(size, scale='sqrtn')[:, :, None].view('float64')
    elif name == 'idft':
        return np.ascontiguousarray(LA.dft(size, scale='sqrtn').conj().T)[:, :, None].view('float64')
    elif name == 'dft2':
        size_sr = int(math.sqrt(size))
        matrix = np.fft.fft2(np.eye(size_sr**2).reshape(-1, size_sr, size_sr), norm='ortho').reshape(-1, size_sr**2)
        # matrix1d = LA.dft(size_sr, scale='sqrtn')
        # assert np.allclose(np.kron(m1d, m1d), matrix)
        # return matrix[:, :, None].view('float64')
        from butterfly.utils import bitreversal_permutation
        br_perm = bitreversal_permutation(size_sr)
        br_perm2 = np.arange(size_sr**2).reshape(size_sr, size_sr)[br_perm][:, br_perm].reshape(-1)
        matrix = np.ascontiguousarray(matrix[:, br_perm2])
        return matrix[:, :, None].view('float64')
    elif name == 'dct':
        # Need to transpose as dct acts on rows of matrix np.eye, not columns
        # return dct(np.eye(size), norm='ortho').T
        return dct(np.eye(size)).T / math.sqrt(size)
    elif name == 'dst':
        return dst(np.eye(size)).T / math.sqrt(size)
    elif name == 'hadamard':
        return LA.hadamard(size) / math.sqrt(size)
    elif name == 'hadamard2':
        size_sr = int(math.sqrt(size))
        matrix1d = LA.hadamard(size_sr) / math.sqrt(size_sr)
        return np.kron(matrix1d, matrix1d)
    elif name == 'b2':
        size_sr = int(math.sqrt(size))
        import torch
        from butterfly import Block2x2DiagProduct
        b = Block2x2DiagProduct(size_sr)
        matrix1d = b(torch.eye(size_sr)).t().detach().numpy()
        return np.kron(matrix1d, matrix1d)
    elif name == 'convolution':
        np.random.seed(0)
        x = np.random.randn(size)
        return LA.circulant(x) / math.sqrt(size)
    elif name == 'hartley':
        return hartley_matrix(size) / math.sqrt(size)
    elif name == 'haar':
        return haar_matrix(size, normalized=True) / math.sqrt(size)
    elif name == 'legendre':
        grid = np.linspace(-1, 1, size + 2)[1:-1]
        return legendre.legvander(grid, size - 1).T / math.sqrt(size)
    elif name == 'hilbert':
        H = hilbert_matrix(size)
        return H / np.linalg.norm(H, 2)
    elif name == 'randn':
        np.random.seed(0)
        return np.random.randn(size, size) / math.sqrt(size)
    elif name == 'permutation':
        np.random.seed(0)
        perm = np.random.permutation(size)
        P = np.eye(size)[perm]
        return P
    elif name == 'low-rank-unnorm':
        np.random.seed(0)
        r = 1
        G = np.random.randn(size, r)
        H = np.random.randn(size, r)
        M = G @ H.T
        # M /= math.sqrt(size*r)
        return M
    elif name == 'low-rank':
        np.random.seed(0)
        r = 1
        G = np.random.randn(size, r)
        H = np.random.randn(size, r)
        M = G @ H.T
        M /= math.sqrt(size*r)
        return M
    else:
        assert False, 'Target matrix name not recognized or implemented'
