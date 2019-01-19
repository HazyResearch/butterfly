"""Target matrices to factor: DFT, DCT, Hadamard, convolution, Legendre, Vandermonde.
Complex complex must be converted to real matrices with 2 as the last dimension
(for Pytorch's compatibility).
"""

import math

import numpy as np
import scipy.linalg as LA
from scipy.fftpack import dct

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
    elif name == 'dct':
        # Need to transpose as dct acts on rows of matrix np.eye, not columns
        # return dct(np.eye(size), norm='ortho').T
        return dct(np.eye(size)).T / math.sqrt(size)
    elif name == 'hadamard':
        return LA.hadamard(size) / math.sqrt(size)
    elif name == 'convolution':
        np.random.seed(0)
        x = np.random.randn(size)
        return LA.circulant(x) / math.sqrt(size)
    elif name == 'haar':
        return haar_matrix(size, normalized=True) / math.sqrt(size)
    elif name == 'randn':
        np.random.seed(0)
        return np.random.randn(size, size) / math.sqrt(size)
    else:
        assert False, 'Target matrix name not recognized or implemented'
