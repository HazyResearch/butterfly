import math
import operator
import functools

import torch
from torch import nn

from complex_utils import complex_mul, complex_matmul
from ops import polymatmul, ops_transpose_mult_br
from sparsemax import sparsemax
from utils import bitreversal_permutation


class HstackDiag(nn.Module):
    """Horizontally stacked diagonal matrices of size n x 2n. Each entry in a 2x2
    matrix of polynomials.
    """

    def __init__(self, size, deg=0, diag1=None, diag2=None):
        """
        Parameters:
            size: size of diagonal matrix
            deg: degree of the polynomials
            diag1: initialization for the diagonal, should be n x 2 x 2 x (d + 1), where d is the degree of the polynomials
            diag2: initialization for the diagonal, should be n x 2 x 2 x (d + 1), where d is the degree of the polynomials
        """
        super().__init__()
        self.size = size
        self.diag1 = diag1 or nn.Parameter(torch.randn(size, 2, 2, deg + 1))
        self.diag2 = diag2 or nn.Parameter(torch.randn(size, 2, 2, deg + 1))
        assert self.diag1.shape == self.diag2.shape, 'The two diagonals must have the same shape'
        self.deg = self.diag1.shape[-1] - 1

    def forward(self, input_):
        """
        Parameters:
            input_: (b, 2 * size, 2, 2, d1)
        Return:
            output: (b, size, 2, 2, d1 + self.deg - 1)
        """
        output = polymatmul(input_[:, :self.size], self.diag1) + polymatmul(input_[:, self.size:], self.diag2)
        return output


class HstackDiagProduct(nn.Module):
    """Product of HstackDiag matrices.
    """

    def __init__(self, size):
        m = int(math.log2(size))
        assert size == 1 << m, "size must be a power of 2"
        super().__init__()
        self.size = size
        self.factors = nn.ModuleList([HstackDiag(size >> (i + 1), deg=(1 << i)) for i in range(m)[::-1]])
        self.P_init = nn.Parameter(torch.randn(1, 2, 1, 2))

    def forward(self, input_):
        """
        Parameters:
            input_: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input_
        for factor in self.factors[::-1]:
            output = factor(output)
        result = polymatmul(output[:, :, [1], :, :-1], self.P_init).squeeze(1).squeeze(1).squeeze(1)
        return result


def test_hstackdiag_product():
    size = 8
    model = HstackDiagProduct(size)

    # Legendre polynomials
    n = size
    m = int(np.log2(n))
    n_range = torch.arange(n, dtype=torch.float)
    a = (2 * n_range + 3) / (n_range + 2)
    b = torch.zeros(n)
    c = -(n_range + 1) / (n_range + 2)
    p0 = 1.0
    p1 = (0.0, 1.0)
    # Preprocessing: compute T_{i:j}, the transition matrix from p_i to p_j.
    T_br = [None] * m
    # Lowest level, filled with T_{i:i+1}
    # n matrices, each 2 x 2, with coefficients being polynomials of degree <= 1
    T_br[0] = torch.zeros(n, 2, 2, 2)
    T_br[0][:, 0, 0, 1] = a
    T_br[0][:, 0, 0, 0] = b
    T_br[0][:, 0, 1, 0] = c
    T_br[0][:, 1, 0, 0] = 1.0
    br_perm = bitreversal_permutation(n)
    T_br[0] = T_br[0][br_perm]
    for i in range(1, m):
        T_br[i] = polymatmul(T_br[i - 1][n >> i:], T_br[i - 1][:n >> i])

    P_init = torch.tensor([p1, [p0, 0.0]], dtype=torch.float)  # [p_1, p_0]
    P_init = P_init.unsqueeze(0).unsqueeze(-2)
    Tidentity = torch.eye(2).unsqueeze(0).unsqueeze(3)

    model.P_init = nn.Parameter(P_init)
    for i in range(m):
        factor = model.factors[m - i - 1]
        factor.diag1 = nn.Parameter(torch.cat((Tidentity.expand(factor.size, -1, -1, -1), torch.zeros(factor.size, 2, 2, factor.deg)), dim=-1))
        factor.diag2 = nn.Parameter(T_br[i][:factor.size])

    batch_size = 2
    x_original = torch.randn((batch_size, size))
    x = (x_original[:, :, None, None] * torch.eye(2)).unsqueeze(-1)
    output = model(x[:, br_perm])
    assert output.shape == (batch_size, size)
    assert torch.allclose(output, ops_transpose_mult_br(a, b, c, p0, p1, x_original))


def main():
    test_hstackdiag_product()


if __name__ == '__main__':
    main()
