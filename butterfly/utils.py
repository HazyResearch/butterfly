import math

import numpy as np

import torch


def bitreversal_permutation(n):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return perm.squeeze(0)


def twiddle_normal_to_fast_format(twiddle):
    """Convert twiddle stored in the normal format to the fast format.
    Parameters:
        twiddle: (nstack, log_n, n / 2, 2, 2)
    Returns:
        twiddle_fast: (nstack, log_n, 2, n)
    """
    twiddle = twiddle.clone()
    nstack = twiddle.shape[0]
    n = twiddle.shape[2] * 2
    m = int(math.log2(n))
    twiddle[:, :, :, 1] = twiddle[:, :, :, 1, [1, 0]]
    twiddle_list = []
    for i in range(m):
        stride = 1 << i
        new_twiddle = twiddle[:, i]
        new_twiddle = new_twiddle.reshape(nstack, n // 2 // stride, stride, 2, 2)
        new_twiddle = new_twiddle.permute(0, 1, 3, 2, 4)
        new_twiddle = new_twiddle.reshape(nstack, n, 2).transpose(1, 2)
        twiddle_list.append(new_twiddle)
    result = torch.stack(twiddle_list, dim=1)
    return result
