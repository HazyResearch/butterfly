import math
import operator
import functools

import torch
from torch import nn

from complex_utils import complex_mul, complex_matmul

# from sparsemax import SparsemaxFunction
# sm = SparsemaxFunction()


class Butterfly(nn.Module):
    """Butterfly matrix of size n x n where only the diagonal and the k-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, diagonal=1, complex=False, diag=None, subdiag=None, superdiag=None):
        """A butterfly matrix where only the diagonal and the k-th subdiagonal
        and superdiagonal are nonzero.
        Parameters:
            size: size of butterfly matrix
            diagonal: the k-th subdiagonal and superdiagonal that are nonzero.
            complex: real or complex matrix
            diag: initialization for the diagonal
            subdiag: initialization for the subdiagonal
            superdiag: initialization for the superdiagonal
        """
        super().__init__()
        assert size > diagonal, 'size must be larger than diagonal'
        self.size = size
        self.diagonal = diagonal
        self.complex = complex
        self.mul_op = complex_mul if complex else operator.mul
        diag_shape = (size, 2) if complex else (size, )
        superdiag_shape = subdiag_shape = (size - diagonal, 2) if complex else (size - diagonal,)
        if diag is None:
            self.diag = nn.Parameter(torch.randn(diag_shape))
            # self.diag = nn.Parameter(torch.ones(diag_shape))
        else:
            assert diag.shape == diag_shape, f'diag must have shape {diag_shape}'
            self.diag = diag
        if subdiag is None:
            self.subdiag = nn.Parameter(torch.randn(subdiag_shape))
            # self.subdiag = nn.Parameter(torch.ones(subdiag_shape))
        else:
            assert subdiag.shape == subdiag_shape, f'subdiag must have shape {subdiag_shape}'
            self.subdiag = subdiag
        if superdiag is None:
            self.superdiag = nn.Parameter(torch.randn(superdiag_shape))
            # self.superdiag = nn.Parameter(torch.ones(superdiag_shape))
        else:
            assert superdiag.shape == superdiag_shape, f'superdiag must have shape {superdiag_shape}'
            self.superdiag = superdiag

    def matrix(self):
        """Matrix form of the butterfly matrix
        """
        if not self.complex:
            return (torch.diag(self.diag)
                    + torch.diag(self.subdiag, -self.diagonal)
                    + torch.diag(self.superdiag, self.diagonal))
        else: # Use torch.diag_embed (available in Pytorch 1.0) to deal with complex case.
            return (torch.diag_embed(self.diag.t(), dim1=0, dim2=1)
                    + torch.diag_embed(self.subdiag.t(), -self.diagonal, dim1=0, dim2=1)
                    + torch.diag_embed(self.superdiag.t(), self.diagonal, dim1=0, dim2=1))

    def forward(self, input_):
        """
        Parameters:
            input_: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if not self.complex:
            output = self.diag * input_
            output[..., self.diagonal:] += self.subdiag * input_[..., :-self.diagonal]
            output[..., :-self.diagonal] += self.superdiag * input_[..., self.diagonal:]
        else:
            output = self.mul_op(self.diag, input_)
            output[..., self.diagonal:, :] += self.mul_op(self.subdiag, input_[..., :-self.diagonal, :])
            output[..., :-self.diagonal, :] += self.mul_op(self.superdiag, input_[..., self.diagonal:, :])
        # assert torch.allclose(output, input_ @ self.matrix().t())
        return output


class ButterflyProduct(nn.Module):
    """Product of butterfly matrices. The order are chosen by softmaxes, which
    are learnable.
    """

    def __init__(self, size, n_terms=None, complex=False, fixed_order=False):
        super().__init__()
        self.m = int(math.log2(size))
        assert size == 1 << self.m, "size must be a power of 2"
        if n_terms is None:
            n_terms = self.m
        self.n_terms = n_terms
        self.complex = complex
        self.matmul_op = complex_matmul if complex else operator.matmul
        self.butterflies = nn.ModuleList([Butterfly(size, diagonal=1 << i, complex=complex)
                                          for i in range(self.m)[::-1]])
        self.fixed_order = fixed_order
        if not fixed_order:
            self.logit = nn.Parameter(torch.randn((n_terms, self.m)))

    def matrix(self):
        if self.fixed_order:
            matrices = [butterfly.matrix() for butterfly in self.butterflies]
            return functools.reduce(self.matmul_op, matrices)
        else:
            prob = nn.functional.softmax(self.logit, dim=-1)
            # prob = sm(self.logit)
            # matrix = None
            # for i in range(self.n_terms):
            #     term = (torch.stack([butterfly.matrix() for butterfly in self.butterflies], dim=-1) * prob[i]).sum(dim=-1)
            #     matrix = term if matrix is None else term @ matrix
            stack = torch.stack([butterfly.matrix() for butterfly in self.butterflies], dim=-1)
            matrices = [(stack * prob[i]).sum(dim=-1) for i in range(self.n_terms)]
            # return torch.chain_matmul(matrices)  ## Doesn't work for complex
            return functools.reduce(self.matmul_op, matrices)

    def forward(self, input_):
        """
        Parameters:
            input_: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if self.fixed_order:
            output = input_
            for butterfly in self.butterflies[::-1]:
                output = self.butterflies(output)
            return output
        else:
            prob = nn.functional.softmax(self.logit, dim=-1)
            # prob = sm(self.logit)
            output = input_
            for i in range(self.n_terms)[::-1]:
                output = (torch.stack([butterfly(output) for butterfly in self.butterflies], dim=-1) * prob[i]).sum(dim=-1)
            return output


def test_butterfly():
    size = 4
    diag = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.float)
    subdiag = torch.tensor([[11, 12], [12, 13], [13, 14]], dtype=torch.float)
    model = Butterfly(size, diagonal=1, complex=True, diag=diag, subdiag=subdiag, superdiag=subdiag)
    matrix_real = torch.tensor([[ 1., 11.,  0.,  0.],
                                [11.,  2., 12.,  0.],
                                [ 0., 12.,  3., 13.],
                                [ 0.,  0., 13.,  4.]])
    matrix_imag = torch.tensor([[ 2., 12.,  0.,  0.],
                                [12.,  3., 13.,  0.],
                                [ 0., 13.,  4., 14.],
                                [ 0.,  0., 14.,  5.]])
    assert torch.allclose(model.matrix()[..., 0], matrix_real)
    assert torch.allclose(model.matrix()[..., 1], matrix_imag)

    batch_size = 3
    x = torch.randn((batch_size, size, 2))
    prod = torch.stack((x[..., 0] @ matrix_real.t() - x[..., 1] @ matrix_imag.t(),
                        x[..., 0] @ matrix_imag.t() + x[..., 1] @ matrix_real.t()), dim=-1)
    assert torch.allclose(model.forward(x), complex_matmul(x, model.matrix().transpose(0, 1)))


def test_butterfly_product():
    size = 4
    model = ButterflyProduct(size, complex=True)
    model.logit = nn.Parameter(torch.tensor([[1.0, float('-inf')], [float('-inf'), 1.0]]))
    assert torch.allclose(model.matrix(),
                          complex_matmul(model.butterflies[0].matrix(), model.butterflies[1].matrix()))

    batch_size = 3
    x = torch.randn((batch_size, size, 2))
    assert torch.allclose(model.forward(x), complex_matmul(x, model.matrix().transpose(0, 1)))


def main():
    test_butterfly()
    test_butterfly_product()


if __name__ == '__main__':
    main()
