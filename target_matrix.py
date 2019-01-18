"""Target matrices to factor: DFT, DCT, Hadamard, convolution, Legendre, Vandermonde.
Complex complex must be converted to real matrices with 2 as the last dimension
(for Pytorch's compatibility).
"""

import math

import numpy as np
import scipy.linalg as LA
from scipy.fftpack import dct


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
    else:
        assert False, 'Target matrix name not recognized or implemented'
