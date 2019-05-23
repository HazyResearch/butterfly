"""Target matrices to factor: DFT, DCT, Hadamard, convolution, Legendre, Vandermonde.
Complex complex must be converted to real matrices with 2 as the last dimension
(for Pytorch's compatibility).
"""

import math

import numpy as np
from numpy.polynomial import legendre
import scipy.linalg as LA
from scipy.fftpack import dct, dst, fft2
import scipy.sparse as sparse
from scipy.linalg import hadamard
import torch


import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

from butterfly import Butterfly

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

def krylov_construct(A, v, m):
    n = v.shape[0]
    assert A.shape == (n,n)
    d = np.diagonal(A, 0)
    subd = np.diagonal(A, -1)

    K = np.zeros(shape=(m,n))
    K[0,:] = v
    for i in range(1,m):
        K[i,1:] = subd*K[i-1,:-1]
    return K

def toeplitz_like(G, H):
    n = g.shape[0]
    r = g.shape[1]
    assert n == size and h.shape[0] == n and h.shape[1] == r
    A1 = np.diag(np.ones(size-1), -1)
    A1[0,n-1] = 1
    A1_ = np.diag(np.ones(size-1), -1)
    A1_[0,n-1] = -1
    rank1s = [krylov_construct(A1, G[:,i], n) @ krylov_construct(A1_, H[:i], n).T for i in range(r)]
    M = sum(ranks1s)
    return M


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
    elif name.startswith('rank-unnorm'):
        r = int(name[11:])
        np.random.seed(0)
        G = np.random.randn(size, r)
        H = np.random.randn(size, r)
        M = G @ H.T
        # M /= math.sqrt(size*r)
        return M
    elif name.startswith('rank'):
        r = int(name[4:])
        np.random.seed(0)
        G = np.random.randn(size, r)
        H = np.random.randn(size, r)
        M = G @ H.T
        M /= math.sqrt(size*r)
        return M
    elif name.startswith('sparse'):
        s = int(name[6:])
        # 2rn parameters
        np.random.seed(0)
        mask = sparse.random(size, size, density=s/size, data_rvs=np.ones)
        M = np.random.randn(size, size) * (mask.toarray())
        M /= math.sqrt(s)
        return M
    elif name.startswith('toeplitz'):
        r = int(name[8:])
        G = np.random.randn(size, r) / math.sqrt(size*r)
        H = np.random.randn(size, r) / math.sqrt(size*r)
        M = toeplitz_like(G, H)
        return M
    elif name == 'fastfood':
        n = size
        S = np.random.randn(n)
        G = np.random.randn(n)
        B = np.random.randn(n)
        # P = np.arange(n)
        P = np.random.permutation(n)
        H = hadamard(n)
        # SHGPHB
        # print(H)
        # print((H*B)[P,:])
        # print((H @ (G[:,np.newaxis] * (H * B)[P,:])))
        F = S[:,np.newaxis] * (H @ (G[:,np.newaxis] * (H * B)[P,:])) / n
        return F
        # x = np.random.randn(batch_size,n)
        # HB = hadamard_transform(B)
        # PHBx = HBx[:, P]
        # HGPHBx = hadamard_transform(G*PHBx)
        # return S*HGPHBx
    elif name == 'butterfly':
        # n (log n+1) params in the hierarchy
        b = Butterfly(in_size=size, out_size=size, bias=False, tied_weight=False, param='odo', nblocks=0)
        M = b(torch.eye(size))
        return M.cpu().detach().numpy()

    else:
        assert False, 'Target matrix name not recognized or implemented'

