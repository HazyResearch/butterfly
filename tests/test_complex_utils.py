import copy
import itertools
import unittest

import torch

import torch_butterfly
from torch_butterfly.complex_utils import complex_matmul, index_last_dim


class ButterflyComplexUtilsTest(unittest.TestCase):

    def setUp(self):
        self.rtol = 1e-3
        self.atol = 1e-5

    def test_complex_matmul(self):
        """Check that our index_last_dim backward is also correct for real input
        """
        bs = (3, 5)
        for device in ['cpu', 'cuda']:
            X = torch.randn(*bs, 128, 16, dtype=torch.complex64, device=device, requires_grad=True)
            Y = torch.randn(*bs, 16, 32, dtype=torch.complex64, device=device, requires_grad=True)
            prod = complex_matmul(X, Y)
            prod_sum = (X.unsqueeze(-1) * Y.unsqueeze(-3)).sum(dim=-2)
            self.assertTrue(torch.allclose(prod, prod_sum, self.rtol, self.atol))
            g = torch.randn_like(prod)
            grad_X, grad_Y = torch.autograd.grad(prod, (X, Y), g)
            grad_X_sum, grad_Y_sum = torch.autograd.grad(prod_sum, (X, Y), g)
            self.assertTrue(torch.allclose(grad_X, grad_X_sum, self.rtol, self.atol))
            self.assertTrue(torch.allclose(grad_Y, grad_Y_sum, self.rtol, self.atol))

            X = torch.randn(5, 3, 32, 32, dtype=torch.complex64, device=device, requires_grad=True)
            Y = torch.randn(6, 3, 32, 32, dtype=torch.complex64, device=device, requires_grad=True)
            prod = complex_matmul(X.permute(2, 3, 0, 1), Y.permute(2, 3, 1, 0)).permute(2, 3, 0, 1)
            prod_sum = (X.unsqueeze(1) * Y).sum(dim=2)
            self.assertTrue(torch.allclose(prod, prod_sum, self.rtol, self.atol))
            g = torch.randn_like(prod)
            grad_X, grad_Y = torch.autograd.grad(prod, (X, Y), g)
            grad_X_sum, grad_Y_sum = torch.autograd.grad(prod_sum, (X, Y), g)
            self.assertTrue(torch.allclose(grad_X, grad_X_sum, self.rtol, self.atol))
            self.assertTrue(torch.allclose(grad_Y, grad_Y_sum, self.rtol, self.atol))

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
