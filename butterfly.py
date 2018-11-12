import math
import operator
import functools

import numpy as np
import torch
from torch import nn
from torch import optim

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
        diag_shape = (size, 2) if complex else (size, )
        superdiag_shape = subdiag_shape = (size - diagonal, 2) if complex else (size - diagonal,)
        if diag is None:
            self.diag = nn.Parameter(torch.randn(diag_shape))
        else:
            assert diag.shape == diag_shape, f'diag must have shape {diag_shape}'
            self.diag = diag
        if subdiag is None:
            self.subdiag = nn.Parameter(torch.randn(subdiag_shape))
        else:
            assert subdiag.shape == subdiag_shape, f'subdiag must have shape {subdiag_shape}'
            self.subdiag = subdiag
        if superdiag is None:
            self.superdiag = nn.Parameter(torch.randn(superdiag_shape))
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
        # TODO: Doesn't work for complex right now
        output = self.diag * input_
        output[..., self.diagonal:] += self.subdiag * input_[..., :-self.diagonal]
        output[..., :-self.diagonal] += self.superdiag * input_[..., self.diagonal:]
        # assert torch.allclose(output, input_ @ self.matrix().t())
        return output


class ButterflyProduct(nn.Module):
    """Product of butterfly matrices. The order are chosen by softmaxes, which
    are learnable.
    """

    def __init__(self, size, n_terms=None):
        super().__init__()
        self.m = int(math.log2(size))
        assert size == 1 << self.m, "size must be a power of 2"
        if n_terms is None:
            n_terms = self.m
        self.n_terms = n_terms
        self.butterflies = nn.ModuleList([Butterfly(size, diagonal=1 << i) for i in range(self.m)])
        self.presoftmaxes = nn.Parameter(torch.randn((n_terms, self.m)))

    def matrix(self):
        prob = nn.functional.softmax(self.presoftmaxes, dim=-1)
        # prob = sm(self.presoftmaxes)
        # matrix = None
        # for i in range(self.n_terms):
        #     term = (torch.stack([butterfly.matrix() for butterfly in self.butterflies], dim=-1) * prob[i]).sum(dim=-1)
        #     matrix = term if matrix is None else term @ matrix
        stack = torch.stack([butterfly.matrix() for butterfly in self.butterflies], dim=-1)
        matrices = [(stack * prob[i]).sum(dim=-1) for i in range(self.n_terms)]
        # matrix = torch.chain_matmul(matrices)  ## Only in Pytorch 1.0
        matrix = functools.reduce(operator.matmul, matrices)
        return matrix


    def forward(self, input_):
        """
        Parameters:
            input_: (..., size)
        Return:
            output: (..., size)
        """
        prob = nn.functional.softmax(self.presoftmaxes, dim=-1)
        # prob = sm(self.presoftmaxes)
        output = input_
        for i in range(self.n_terms)[::-1]:
            output = (torch.stack([butterfly(output) for butterfly in self.butterflies], dim=-1) * prob[i]).sum(dim=-1)
        return output

