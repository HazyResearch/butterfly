import copy
import itertools
import unittest

import torch

import torch_butterfly
from torch_butterfly.complex_utils import index_last_dim


class ButterflyComplexUtilsTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_index_last_dim(self):
        """Check that our index_last_dim backward is also correct for real input
        """
        sizes = (2, 3, 17)
        p = torch.randperm(sizes[-1])
        X = torch.randn(sizes, requires_grad=True)
        out_torch = X[..., p]
        out = index_last_dim(X, p)
        self.assertTrue(torch.allclose(out, out_torch))
        g = torch.randn_like(out)
        grad_x_torch, = torch.autograd.grad(out_torch, X, g)
        grad_x, = torch.autograd.grad(out, X, g)
        self.assertTrue(torch.allclose(grad_x, grad_x_torch))


if __name__ == "__main__":
    unittest.main()
