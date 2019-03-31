import math

import torch
from torch import nn

from .complex_utils import complex_mul
from .utils import bitreversal_permutation


class ButterflyFactor(nn.Module):
    """Block matrix of size k n x k n of the form [[A, B], [C, D]] where each of A, B,
    C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, stack=1, complex=False, ABCD=None, n_blocks=1, tied_weight=True):
        """
        Parameters:
            size: input has shape (stack, ..., size)
            stack: number of stacked components, output has shape (stack, ..., size)
            complex: real or complex matrix
            ABCD: block of [[A, B], [C, D]], of shape (stack, 2, 2, size//2) if real or (stack, 2, 2, size//2, 2) if complex
            n_blocks: number of such blocks of ABCD
            tied_weight: whether the weights ABCD at different blocks are tied to be the same.
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.stack = stack
        self.complex = complex
        self.n_blocks = n_blocks
        self.tied_weight = tied_weight
        if tied_weight:
            ABCD_shape = (stack, 2, 2, size // 2) if not complex else (stack, 2, 2, size // 2, 2)
        else:
            ABCD_shape = (stack, n_blocks, 2, 2, size // 2) if not complex else (stack, n_blocks, 2, 2, size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            self.ABCD = nn.Parameter(torch.randn(ABCD_shape) * scaling)
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD

    def forward(self, input):
        """
        Parameters:
            input: (stack, ..., size) if real or (stack, ..., size, 2) if complex
            if not tied_weight: (stack, n_blocks, ..., size) if real or (stack, n_blocks, ..., size, 2) if complex
        Return:
            output: (stack, ..., size) if real or (stack, ..., size, 2) if complex
            if not tied_weight: (stack, n_blocks, ..., size) if real or (stack, n_blocks, ..., size, 2) if complex
        """
        if self.tied_weight:
            if not self.complex:
                return (self.ABCD.unsqueeze(1) * input.view(self.stack, -1, 1, 2, self.size // 2)).sum(dim=-2).view(input.shape)
            else:
                return complex_mul(self.ABCD.unsqueeze(1), input.view(self.stack, -1, 1, 2, self.size // 2, 2)).sum(dim=-3).view(input.shape)
        else:
            if not self.complex:
                return (self.ABCD.unsqueeze(2) * input.view(self.stack, self.n_blocks, -1, 1, 2, self.size // 2)).sum(dim=-2).view(input.shape)
            else:
                return complex_mul(self.ABCD.unsqueeze(2), input.view(self.stack, self.n_blocks, -1, 1, 2, self.size // 2, 2)).sum(dim=-3).view(input.shape)


class Butterfly(nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, in_size, out_size, complex=False, decreasing_size=True, tied_weight=True, bias=True):
        super().__init__()
        self.in_size = in_size
        m = int(math.ceil(math.log2(in_size)))
        self.in_size_extended = 1 << m  # Will zero-pad input if in_size is not a power of 2
        self.out_size = out_size
        self.stack = int(math.ceil(out_size / self.in_size_extended))
        self.complex = complex
        self.tied_weight = tied_weight
        in_sizes = [self.in_size_extended >> i for i in range(m)] if decreasing_size else [self.in_size_extended >> i for i in range(m)[::-1]]
        if tied_weight:
            self.factors = nn.ModuleList([ButterflyFactor(in_size_, stack=self.stack, complex=complex)
                                          for in_size_ in in_sizes])
        else:
            self.factors = nn.ModuleList([ButterflyFactor(in_size_, stack=self.stack, complex=complex, n_blocks=self.in_size_extended // in_size_, tied_weight=tied_weight)
                                          for in_size_ in in_sizes])
        if bias:
            if not self.complex:
                self.bias = nn.Parameter(torch.Tensor(out_size))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_size, 2))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if hasattr(self, 'bias'):
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (..., in_size) if real or (..., in_size, 2) if complex
        Return:
            output: (..., out_size) if real or (..., out_size, 2) if complex
        """
        output = input.contiguous()
        if self.in_size != self.in_size_extended:  # Zero-pad
            if not self.complex:
                output = torch.cat((output, torch.zeros(output.shape[:-1] + (self.in_size_extended - self.in_size, ), dtype=output.dtype, device=output.device)), dim=-1)
            else:
                output = torch.cat((output, torch.zeros(output.shape[:-2] + (self.in_size_extended - self.in_size, 2), dtype=output.dtype, device=output.device)), dim=-2)
        output = output.unsqueeze(0).expand((self.stack, ) + output.shape)
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view(output.shape[:-1] + (-1, factor.size))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        if not self.complex:
            output = output.permute(tuple(range(1, output.dim() - 1)) + (0, -1)).reshape(input.shape[:-1] + (self.stack * self.in_size_extended, ))[..., :self.out_size]
        else:
            output = output.permute(tuple(range(1, output.dim() - 2)) + (0, -2, -1)).reshape(input.shape[:-2] + (self.stack * self.in_size_extended, 2))[..., :self.out_size, :]
        if hasattr(self, 'bias'):
            output += self.bias
        return output


def test_buttefly_factor():
    batch_size = 3
    size = 8
    stack = 2
    model = ButterflyFactor(size, stack=stack)
    input = torch.randn((stack, batch_size, size))
    output = model(input)
    assert output.shape == (stack, batch_size, size)
    model = ButterflyFactor(size, stack=stack, complex=True)
    input = torch.randn((stack, batch_size, size, 2))
    output = model(input)
    assert output.shape == (stack, batch_size, size, 2)


def test_butterfly():
    batch_size = 3
    in_size = 7
    out_size = 15
    model = Butterfly(in_size, out_size)
    input = torch.randn((batch_size, in_size))
    output = model(input)
    assert output.shape == (batch_size, out_size)
    model = Butterfly(in_size, out_size, complex=True)
    input = torch.randn((batch_size, in_size, 2))
    output = model(input)
    assert output.shape == (batch_size, out_size, 2)


def test_butterfly_tied_weight():
    batch_size = 3
    in_size = 7
    out_size = 15
    model = Butterfly(in_size, out_size, tied_weight=False)
    input = torch.randn((batch_size, in_size))
    output = model(input)
    assert output.shape == (batch_size, out_size)
    model = Butterfly(in_size, out_size, complex=True, tied_weight=False)
    input = torch.randn((batch_size, in_size, 2))
    output = model(input)
    assert output.shape == (batch_size, out_size, 2)


def main():
    test_buttefly_factor()
    test_butterfly()
    test_butterfly_tied_weight()


if __name__ == '__main__':
    main()
