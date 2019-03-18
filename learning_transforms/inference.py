"""Convert BP model from Pytorch to Numpy for inference.
To compile Cython extension: python setup.py build_ext --inplace
"""
import numpy as np
import torch
from torch import nn

from timeit import default_timer as timer


from butterfly import Block2x2DiagProduct

from ABCD_mult import ABCD_mult, ABCD_mult_inplace, ABCD_mult_inplace_memview, ABCD_mult_inplace_complex, ABCD_mult_inplace_generic


def butterfly_mul_np(ABCDs, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    output = input_
    for ABCD in ABCDs[::-1]:
        output = output.reshape(output.shape[:-1] + (-1, 1, 2, ABCD.shape[-1]))
        output = (ABCD * output).sum(axis=-2).reshape(input_.shape)
    return output

# a = ABCD[..., 0][None]
# b = output.reshape(2048, 2, 1)
# b_cont = output.reshape(2048, 2).T
# c = output.reshape(2, 2048)
# %timeit a @ b
# %timeit a @ b_cont
# %timeit a @ c


def butterfly_mul_np_transpose(ABCDs_transpose, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy.
    Parameters:
        ABCDs_transpose: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    output = input_
    for ABCD in ABCDs_transpose[::-1]:
        # output = (output.reshape(-1, ABCD.shape[0], 1, 2) @ ABCD).reshape(input_.shape)
        # output = (output.reshape(ABCD.shape[0], -1, 2) @ ABCD).reshape(input_.shape)
        # output = (ABCD @ output.reshape(ABCD.shape[0], 2, -1)).reshape(input_.shape)
        # output = (ABCD @ output.reshape(ABCD.shape[0], 2, -1)).reshape(input_.shape)
        output = (ABCD @ output.reshape(ABCD.shape[0], 2, -1)).reshape(input_.shape)
        start = timer()
        [(ABCD @ output.reshape(ABCD.shape[0], 2, -1)).reshape(input_.shape) for _ in range(1000)]
        end = timer()
        print(end - start)
    return output


def butterfly_mul_cy(ABCDs, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy + Cython.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    assert input_.dtype == np.float32
    output = input_.copy()
    buffer = np.empty(input_.size, dtype=np.float32)
    for ABCD in ABCDs[::-1]:
        output = output.reshape((-1, 2, ABCD.shape[-1]))
        buffer = buffer.reshape(output.shape)
        ABCD_mult(ABCD, output, buffer)
        output, buffer = buffer, output
    return output.reshape(input_.shape)


def butterfly_mul_cy_inplace(ABCDs, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy + Cython.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    assert input_.dtype == np.float32
    output = input_.copy()
    for ABCD in ABCDs[::-1]:
        output = output.reshape((-1, 2, ABCD.shape[-1]))
        ABCD_mult_inplace(ABCD, output)

        # start = timer()
        # [ABCD_mult_inplace(ABCD, output.reshape((-1, 2, ABCD.shape[-1]))) for _ in range(1000)]
        # end = timer()
        # print(end - start)
    return output.reshape(input_.shape)


def butterfly_mul_cy_inplace_memview(ABCDs, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy + Cython.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    assert input_.dtype == np.float32
    output = input_.copy()
    for ABCD in ABCDs[::-1]:
        output = output.reshape((-1, 2, ABCD.shape[-1]))
        ABCD_mult_inplace_memview(ABCD, output)
    return output.reshape(input_.shape)


def butterfly_mul_cy_inplace_index(ABCDs, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy + Cython.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    assert input_.dtype == np.float32
    output = input_.copy()
    func = ABCD_mult_inplace_generic[float]
    # cython.floatcomplex, cython.doublecomplex
    for ABCD in ABCDs[::-1]:
        output = output.reshape((-1, 2, ABCD.shape[-1]))
        func(ABCD, output)
    return output.reshape(input_.shape)


def butterfly_mul_cy_inplace_complex(ABCDs, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy + Cython.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    assert input_.dtype == np.complex64
    output = input_.copy()
    for ABCD in ABCDs[::-1]:
        output = output.reshape((-1, 2, ABCD.shape[-1]))
        ABCD_mult_inplace_complex(ABCD, output)
    return output.reshape(input_.shape)


def butterfly_mul_cy_inplace_generic(ABCDs, input_):
    """Product of block 2x2 diagonal matrices, implemented in Numpy + Cython.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        input_: input_ vector as numpy array, (batch_size, n)
    """
    output = input_.copy()
    for ABCD in ABCDs[::-1]:
        output = output.reshape((-1, 2, ABCD.shape[-1]))
        ABCD_mult_inplace_generic(ABCD, output)
    return output.reshape(input_.shape)


def BP_mul_cy_inplace(ABCDs, perm, input_):
    """Product of block 2x2 diagonal matrices, with permutation, implemented in Numpy + Cython.
    Parameters:
        ABCDs: list of the ABCDs factors as used in Block2x2DiagProduct, in numpy array.
               we accept real and complex.
        perm: a permutation, (n, ) int numpy array
        input_: input_ vector as numpy array, (batch_size, n)
    """
    assert input_.dtype == np.float32
    output = input_[..., perm]
    for ABCD in ABCDs[::-1]:
        output = output.reshape((-1, 2, ABCD.shape[-1]))
        ABCD_mult_inplace(ABCD, output)
    return output.reshape(input_.shape)


def Block2x2DiagProduct_to_ABCDs(model):
    """Convert a model of the type Block2x2DiagProduct into list of ABCDs factors,
    ready for butterfly_mul_np.
    """
    assert isinstance(model, Block2x2DiagProduct)
    ABCDs = []
    if not model.complex:
        ABCDs = [factor.ABCD.detach().numpy() for factor in model.factors]
    else:
        ABCDs = [factor.ABCD.detach().numpy().view('complex64').squeeze(-1) for factor in model.factors]
    return ABCDs

# TODO: Turn these into tests

import os
os.environ['MKL_NUM_THREADS'] = '1'

n = 4096
batch_size = 1

x = torch.randn(batch_size, n)
B = Block2x2DiagProduct(n)
# B_matrix = B(torch.eye(n)).t().contiguous().detach().numpy()
B_matrix = B(torch.eye(n)).t().contiguous()
B_matrix_np = B_matrix.detach().numpy()
x_np = x.detach().numpy()

ABCDs = Block2x2DiagProduct_to_ABCDs(B)
# ABCDs_transpose = [a.T.copy() for a in ABCDs]
# TODO: need to tranpose for correct result, right now I'm just testing speed
ABCDs_transpose = [a.transpose(2, 0, 1).copy() for a in ABCDs]

# %timeit B_matrix @ x.t()
# %timeit B_matrix_np @ x_np.T
# %timeit B(x)
# %timeit butterfly_mul_np(ABCDs, x_np)
# %timeit butterfly_mul_np_transpose(ABCDs_transpose, x_np)
# %timeit butterfly_mul_cy(ABCDs, x_np)
# %timeit butterfly_mul_cy_inplace(ABCDs, x_np)
# %timeit butterfly_mul_cy_inplace_memview(ABCDs, x_np)
# %timeit butterfly_mul_cy_inplace_index(ABCDs, x_np)
# %timeit butterfly_mul_cy_inplace_generic(ABCDs, x_np)
# %timeit np.fft.fft(x_np)

# x = torch.randn(batch_size, n, 2)
# x_np = x.detach().numpy().view('complex64').squeeze(-1)
# B = Block2x2DiagProduct(n, complex=True)
# ABCDs = Block2x2DiagProduct_to_ABCDs(B)

# %timeit B(x)
# %timeit butterfly_mul_cy_inplace_complex(ABCDs, x_np)
# %timeit butterfly_mul_cy_inplace_generic(ABCDs, x_np)

# np.abs(B(x).detach().numpy().view('complex64').squeeze(-1) - butterfly_mul_cy_inplace_complex(ABCDs, x_np)).max()
