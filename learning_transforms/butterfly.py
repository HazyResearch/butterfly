import math
import operator
import functools

import torch
from torch import nn

from complex_utils import real_to_complex, complex_mul, complex_matmul
from sparsemax import sparsemax
from utils import bitreversal_permutation
from butterfly_factor import butterfly_factor_mult
from permutation_factor import permutation_factor_even_odd_mult, permutation_factor_reverse_mult


def sinkhorn(logit, n_iters=5):
    """Sinkhorn iterations.
    Parameters:
        logit: (..., n, n)
        n_iters: integer
    Return:
        (..., n, n) matrix that's close to a doubly stochastic matrix.
    """
    assert logit.dim() >= 2, 'logit must be at least a 2D tensor'
    assert logit.shape[-2] == logit.shape[-1], 'logit must be a square matrix'
    for _ in range(n_iters):
        logit = logit - torch.logsumexp(logit, dim=-1, keepdim=True)
        logit = logit - torch.logsumexp(logit, dim=-2, keepdim=True)
    return torch.exp(logit)


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

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if not self.complex:
            output = self.diag * input
            output[..., self.diagonal:] += self.subdiag * input[..., :-self.diagonal]
            output[..., :-self.diagonal] += self.superdiag * input[..., self.diagonal:]
        else:
            output = self.mul_op(self.diag, input)
            output[..., self.diagonal:, :] += self.mul_op(self.subdiag, input[..., :-self.diagonal, :])
            output[..., :-self.diagonal, :] += self.mul_op(self.superdiag, input[..., self.diagonal:, :])
        # assert torch.allclose(output, input @ self.matrix().t())
        return output


class MatrixProduct(nn.Module):
    """Product of matrices. The order are chosen by softmaxes, which are learnable.
    Each factor matrix must implement .matrix() function.
    """

    def __init__(self, factors, n_terms=None, complex=False, fixed_order=False, softmax_fn='softmax'):
        super().__init__()
        self.factors = nn.ModuleList(factors)
        if n_terms is None:
            n_terms = len(factors)
        self.n_terms = n_terms
        self.complex = complex
        self.matmul_op = complex_matmul if complex else operator.matmul
        self.fixed_order = fixed_order
        if not self.fixed_order:
            assert softmax_fn in ['softmax', 'sparsemax']
            self.logit = nn.Parameter(torch.randn((self.n_terms, len(factors))))
            if softmax_fn == 'softmax':
                self.softmax_fn = lambda logit: nn.functional.softmax(logit, dim=-1)
            else:
                self.softmax_fn = sparsemax

    def matrix(self, temperature=1.0):
        if self.fixed_order:
            matrices = [factor.matrix() for factor in self.factors]
            return functools.reduce(self.matmul_op, matrices)
        else:
            prob = self.softmax_fn(self.logit / temperature)
            stack = torch.stack([factor.matrix() for factor in self.factors])
            matrices = (prob @ stack.reshape(stack.shape[0], -1)).reshape((-1,) + stack.shape[1:])
            # Alternative: slightly slower but easier to understand
            # matrices = torch.einsum('ab, b...->a...', (prob, stack))
            # return torch.chain_matmul(*matrices)  ## Doesn't work for complex
            return functools.reduce(self.matmul_op, matrices)

    def forward(self, input, temperature=1.0):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if self.fixed_order:
            output = input
            for factor in self.factors[::-1]:
                output = factor(output)
            return output
        else:
            prob = self.softmax_fn(self.logit / temperature)
            output = input
            for i in range(self.n_terms)[::-1]:
                # output = (torch.stack([factor(output) for factor in self.factors], dim=-1) * prob[i]).sum(dim=-1)
                stack = torch.stack([factor(output) for factor in self.factors])
                output = (prob[i:i+1] @ stack.reshape(stack.shape[0], -1)).reshape(stack.shape[1:])
            return output


class ButterflyProduct(MatrixProduct):
    """Product of butterfly matrices. The order are chosen by softmaxes, which
    are learnable.
    """

    def __init__(self, size, n_terms=None, complex=False, fixed_order=False, softmax_fn='softmax', learn_perm=False):
        m = int(math.log2(size))
        assert size == 1 << m, "size must be a power of 2"
        self.size = size
        factors = [Butterfly(size, diagonal=1 << i, complex=complex) for i in range(m)[::-1]]
        super().__init__(factors, n_terms, complex, fixed_order, softmax_fn)
        self.learn_perm = learn_perm
        if learn_perm:
            self.perm_logit = nn.Parameter(torch.randn((size, size)))

    def matrix(self, temperature=1.0):
        matrix = super().matrix(temperature)
        if self.learn_perm:
            perm = sinkhorn(self.perm_logit / temperature)
            if not self.complex:
                matrix = matrix @ perm
            else:
                matrix = (matrix.transpose(-1, -2) @ perm).transpose(-1, -2)
        return matrix

    def forward(self, input, temperature=1.0):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if self.learn_perm:
            perm = sinkhorn(self.perm_logit / temperature)
            if not self.complex:
                input = input @ perm.t()
            else:
                input = (input.transpose(-1, -2) @ perm.t()).transpose(-1, -2)
        return super().forward(input, temperature)


class Block2x2Diag(nn.Module):
    """Block matrix of size n x n of the form [[A, B], [C, D]] where each of A, B,
    C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, complex=False, ABCD=None, ortho_init=False):
        """
        Parameters:
            size: size of butterfly matrix
            complex: real or complex matrix
            ABCD: block of [[A, B], [C, D]], of shape (2, 2, size//2) if real or (2, 2, size//2, 2) if complex
            ortho_init: whether the twiddle factors are initialized to be orthogonal (real) or unitary (complex)
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.complex = complex
        self.mul_op = complex_mul if complex else operator.mul
        ABCD_shape = (2, 2, size // 2) if not complex else (2, 2, size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            if not ortho_init:
                self.ABCD = nn.Parameter(torch.randn(ABCD_shape) * scaling)
            else:
                if not complex:
                    theta = torch.rand(size // 2) * math.pi * 2
                    c, s = torch.cos(theta), torch.sin(theta)
                    det = torch.randint(0, 2, (size // 2, ), dtype=c.dtype) * 2 - 1  # Rotation (+1) or reflection (-1)
                    self.ABCD = nn.Parameter(torch.stack((torch.stack((det * c, -det * s)),
                                                          torch.stack((s, c)))))
                else:
                    # Sampling from the Haar measure on U(2) is a bit subtle.
                    # Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
                    phi = torch.asin(torch.sqrt(torch.rand(size // 2)))
                    c, s = torch.cos(phi), torch.sin(phi)
                    alpha, psi, chi = torch.randn(3, size // 2) * math.pi * 2
                    phi = torch.randn(3, size // 2) * math.pi * 2
                    A = torch.stack((c * torch.cos(alpha + psi), c * torch.sin(alpha + psi)), dim=-1)
                    B = torch.stack((s * torch.cos(alpha + chi), s * torch.sin(alpha + chi)), dim=-1)
                    C = torch.stack((-s * torch.cos(alpha - chi), -s * torch.sin(alpha - chi)), dim=-1)
                    D = torch.stack((c * torch.cos(alpha - psi), c * torch.sin(alpha - psi)), dim=-1)
                    self.ABCD = nn.Parameter(torch.stack((torch.stack((A, B)),
                                                          torch.stack((C, D)))))
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if not self.complex:
            # return ((self.ABCD * input.view(input.shape[:-1] + (1, 2, self.size // 2))).sum(dim=-2)).view(input.shape)
            return butterfly_factor_mult(self.ABCD, input.view(-1, 2, self.size // 2)).view(input.shape)
        else:
            # return (self.mul_op(self.ABCD, input.view(input.shape[:-2] + (1, 2, self.size // 2, 2))).sum(dim=-3)).view(input.shape)
            return butterfly_factor_mult(self.ABCD, input.view(-1, 2, self.size // 2, 2)).view(input.shape)


class Block2x2DiagProduct(nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, size, complex=False, decreasing_size=True, ortho_init=False):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, "size must be a power of 2"
        self.size = size
        self.complex = complex
        sizes = [size >> i for i in range(m)] if decreasing_size else [size >> i for i in range(m)[::-1]]
        self.factors = nn.ModuleList([Block2x2Diag(size_, complex=complex, ortho_init=ortho_init) for size_ in sizes])

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input.contiguous()
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view(output.shape[:-1] + (-1, factor.size))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        return output


class Block2x2DiagRectangular(nn.Module):
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


class Block2x2DiagProductRectangular(nn.Module):
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
            self.factors = nn.ModuleList([Block2x2DiagRectangular(in_size_, stack=self.stack, complex=complex)
                                          for in_size_ in in_sizes])
        else:
            self.factors = nn.ModuleList([Block2x2DiagRectangular(in_size_, stack=self.stack, complex=complex, n_blocks=self.in_size_extended // in_size_, tied_weight=tied_weight)
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


class Block2x2DiagBmm(nn.Module):
    """Block matrix of size n x n of the form [[A, B], [C, D]] where each of A, B,
    C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, complex=False, ABCD=None):
        """
        Parameters:
            size: size of butterfly matrix
            complex: real or complex matrix
            ABCD: block of [[A, B], [C, D]], of shape (2, 2, size//2) if real or (2, 2, size//2, 2) if complex
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.complex = complex
        self.mul_op = complex_mul if complex else operator.mul
        ABCD_shape = (size // 2, 2, 2) if not complex else (2, 2, size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            self.ABCD = nn.Parameter(torch.randn(ABCD_shape) * scaling)
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD

    def forward(self, input):
        """
        Parameters:
            input: (size, batch_size) if real or (size, batch_size, 2) if complex
        Return:
            output: (size, batch_size) if real or (size, batch_size, 2) if complex
        """
        if not self.complex:
            # return ((self.ABCD * input.view(input.shape[:-1] + (1, 2, self.size // 2))).sum(dim=-2)).view(input.shape)
            # return butterfly_factor_mult(self.ABCD, input.view(-1, 2, self.size // 2)).view(input.shape)
            return (self.ABCD @ input.view(self.size // 2, 2, -1)).view(input.shape)
        else:
            # return (self.mul_op(self.ABCD, input.view(input.shape[:-2] + (1, 2, self.size // 2, 2))).sum(dim=-3)).view(input.shape)
            return butterfly_factor_mult(self.ABCD, input.view(-1, 2, self.size // 2, 2)).view(input.shape)


class Block2x2DiagProductBmm(nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, size, complex=False, decreasing_size=True):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, "size must be a power of 2"
        self.size = size
        self.complex = complex
        sizes = [size >> i for i in range(m)] if decreasing_size else [size >> i for i in range(m)[::-1]]
        self.factors = nn.ModuleList([Block2x2DiagBmm(size_, complex=complex) for size_ in sizes])
        self.br_perm = bitreversal_permutation(size)

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """

        output = input.t()[self.br_perm]
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view((factor.size, -1))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        return output[self.br_perm].t()


class BlockPerm(nn.Module):
    """Block permutation matrix of size n x n.
    """

    def __init__(self, size, logit=None, complex=False):
        """
        Parameters:
            size: size of permutation matrix
            complex: real of complex input
            logit: (3, ) nn.Parameter, containing logits for probability of
                   separating even and odd (logit[0]), probability of reversing
                   the first half (logit[1]), and probability of reversing the
                   second half (logit[2]).
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.complex = complex
        if logit is None:
            self.logit = nn.Parameter(torch.randn(3))
        else:
            self.logit = logit
        self.reverse_perm = nn.Parameter(torch.arange(self.size // 2 - 1, -1, -1), requires_grad=False)

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        prob = torch.sigmoid(self.logit)
        output = input
        if not self.complex:
            # There's a lot of complicated logic here buried under the reshape's and unsqueeze's and so on
            # First step: weighted mean of identity permutation and permutation that yields [even, odd]
            # output = ((1 - prob[0]) * output.view(-1, 2, self.size // 2) + prob[0] * output.view(-1, self.size // 2, 2).transpose(-1, -2)).view(-1, self.size)
            output = permutation_factor_even_odd_mult(prob[:1], output.view(-1, self.size))
            # output = output.view(-1, 2, self.size // 2)
            # Second step: weighted mean of identity permutation and permutation that reverses the first and the second half
            # output  = output.reshape(output.shape[:-1] + (2, self.size // 2))
            # output = (((1 - prob[1:]).unsqueeze(-1) * output + prob[1:].unsqueeze(-1) * output.flip(-1))).reshape(output.shape[:-2] + (self.size, ))
            # output = (((1 - prob[1:]).unsqueeze(-1) * output + prob[1:].unsqueeze(-1) * output[..., self.reverse_perm])).reshape(output.shape[:-2] + (self.size, ))
            output = permutation_factor_reverse_mult(prob[1:], output)
            # output = output.reshape(input.shape)
        else:
            # output = (1 - prob[0]) * output.reshape(output.shape[:-2] + (2, self.size // 2, 2)) + prob[0] * output.reshape(output.shape[:-2] + (self.size // 2, 2, 2)).transpose(-2, -3)
            output = permutation_factor_even_odd_mult(prob[:1], output.view(-1, self.size))
            # output = (((1 - prob[1:]).unsqueeze(-1).unsqueeze(-1) * output + prob[1:].unsqueeze(-1).unsqueeze(-1) * output.flip(-2))).reshape(output.shape[:-3] + (self.size, 2))
            output = permutation_factor_reverse_mult(prob[1:], output)
        return output.view(input.shape)

    def argmax(self):
        """
        Return:
            p: (self.size, ) array of int, the most probable permutation.
        """
        logit = nn.Parameter(torch.where(self.logit >= 0, torch.tensor(float('inf'), device=self.logit.device), torch.tensor(float('-inf'), device=self.logit.device)))
        argmax_instance = self.__class__(self.size, logit, complex=False)
        p = argmax_instance.forward(torch.arange(self.size, dtype=torch.float, device=self.logit.device)).round().long()
        return p


class BlockPermProduct(nn.Module):
    """Product of block permutation matrices.
    """

    def __init__(self, size, complex=False, share_logit=False, increasing_size=True):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, "size must be a power of 2"
        self.size = size
        self.complex = complex
        self.share_logit = share_logit
        # We don't need the permutation with size 2 since it's always the identity
        sizes = [size >> i for i in range(m - 1)[::-1]] if increasing_size else [size >> i for i in range(m - 1)]
        if share_logit:
            self.logit = nn.Parameter(torch.randn(3))
            self.factors = nn.ModuleList([BlockPerm(size_, self.logit, complex=complex) for size_ in sizes])
        else:
            self.factors = nn.ModuleList([BlockPerm(size_, complex=complex) for size_ in sizes])

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input.contiguous()
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(output.view(output.shape[:-1] + (-1, factor.size))).view(output.shape)
            else:
                output = factor(output.view(output.shape[:-2] + (-1, factor.size, 2))).view(output.shape)
        return output

    def argmax(self):
        """
        Return:
            p: (self.size, ) array of int, the most probable permutation.
        """
        p = torch.arange(self.size, device=self.factors[0].logit.device)
        for factor in self.factors[::-1]:
            p = p.reshape(-1, factor.size)[:, factor.argmax()].reshape(self.size)
        return p


class FixedPermutation(nn.Module):

    def __init__(self, permutation, complex=False):
        """Fixed permutation. Used to store argmax of BlockPerm and BlockPermProduct.
        Parameter:
            permutation: (n, ) tensor of ints
        """
        super().__init__()
        self.permutation = nn.Parameter(permutation, requires_grad=False)
        self.complex = complex

    def forward(self, input):
        return input[..., self.permutation] if not self.complex else input[..., self.permutation, :]



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
                          complex_matmul(model.factors[0].matrix(), model.factors[1].matrix()))

    batch_size = 3
    x = torch.randn((batch_size, size, 2))
    assert torch.allclose(model.forward(x), complex_matmul(x, model.matrix().transpose(0, 1)))


def test_butterfly_fft():
    # DFT matrix for n = 4
    size = 4
    DFT = torch.fft(real_to_complex(torch.eye(size)), 1)
    P = real_to_complex(torch.tensor([[1., 0., 0., 0.],
                                      [0., 0., 1., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 0., 1.]]))
    M0 = Butterfly(size,
                   diagonal=2,
                   complex=True,
                   diag=torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                   subdiag=torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True),
                   superdiag=torch.tensor([[1.0, 0.0], [0.0, -1.0]], requires_grad=True))
    M1 = Butterfly(size,
                   diagonal=1,
                   complex=True,
                   diag=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True),
                   subdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True),
                   superdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True))
    assert torch.allclose(complex_matmul(M0.matrix(), complex_matmul(M1.matrix(), P)), DFT)
    br_perm = torch.tensor(bitreversal_permutation(size))
    assert torch.allclose(complex_matmul(M0.matrix(), M1.matrix())[:, br_perm], DFT)
    D = complex_matmul(DFT, P.transpose(0, 1))
    assert torch.allclose(complex_matmul(M0.matrix(), M1.matrix()), D)


def test_butterfly_dct():
    from scipy.fftpack import dct
    # DCT matrix for n = 4
    size = 4
    # Need to transpose as dct acts on rows of matrix np.eye, not columns
    DCT = torch.tensor(dct(np.eye(size)).T, dtype=torch.float)
    M0diag=torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    M0subdiag=torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    M0superdiag=torch.tensor([[1.0, 0.0], [0.0, -1.0]], requires_grad=True)
    M0 = Butterfly(size, diagonal=2, complex=True, diag=M0diag, subdiag=M0subdiag, superdiag=M0superdiag)
    M1 = Butterfly(size,
                   diagonal=1,
                   complex=True,
                   diag=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], requires_grad=True),
                   subdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True),
                   superdiag=torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], requires_grad=True))
    arange_ = np.arange(size)
    dct_perm = np.concatenate((arange_[::2], arange_[::-2]))
    br_perm = bitreversal_permutation(size)
    perm = torch.arange(size)[dct_perm][br_perm]
    arange_ = torch.arange(size, dtype=torch.float)
    diag_real = 2 * torch.cos(-math.pi * arange_ / (2 * size))
    diag_imag = 2 * torch.sin(-math.pi * arange_ / (2 * size))
    diag = torch.stack((torch.diag(diag_real), torch.diag(diag_imag)), dim=-1)
    assert torch.allclose(complex_matmul(diag, complex_matmul(M0.matrix(), M1.matrix()))[:, perm, 0], DCT)
    D = torch.stack((diag_real, diag_imag), dim=-1)
    DM0 = Butterfly(size,
                    diagonal=2,
                    complex=True,
                    diag=complex_mul(D, M0diag),
                    subdiag=complex_mul(D[2:], M0subdiag),
                    superdiag=complex_mul(D[:2], M0superdiag))
    assert torch.allclose(complex_matmul(DM0.matrix(), M1.matrix())[:, perm, 0], DCT)


def test_block2x2diagproduct():
    # Factorization of the DFT matrix
    size = 4
    model = Block2x2DiagProduct(size, complex=True)
    model.factors[1].ABCD = nn.Parameter(torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]], [[[1.0, 0.0]], [[-1.0, 0.0]]]]))
    model.factors[0].ABCD = nn.Parameter(torch.tensor([[[[1.0, 0.0],
                                                         [1.0, 0.0]],
                                                        [[1.0, 0.0],
                                                         [0.0, -1.0]]],
                                                       [[[1.0, 0.0],
                                                         [1.0, 0.0]],
                                                        [[-1.0, 0.0],
                                                         [0.0, 1.0]]]]))
    input = torch.stack((torch.eye(size), torch.zeros(size, size)), dim=-1)
    assert torch.allclose(model(input[:, [0, 2, 1, 3]]), torch.fft(input, 1))


def test_block2x2diagrectangular():
    batch_size = 3
    size = 8
    stack = 2
    model = Block2x2DiagRectangular(size, stack=stack)
    input = torch.randn((stack, batch_size, size))
    output = model(input)
    assert output.shape == (stack, batch_size, size)
    model = Block2x2DiagRectangular(size, stack=stack, complex=True)
    input = torch.randn((stack, batch_size, size, 2))
    output = model(input)
    assert output.shape == (stack, batch_size, size, 2)


def test_block2x2diagproductrectangular():
    batch_size = 3
    in_size = 7
    out_size = 15
    model = Block2x2DiagProductRectangular(in_size, out_size)
    input = torch.randn((batch_size, in_size))
    output = model(input)
    assert output.shape == (batch_size, out_size)
    model = Block2x2DiagProductRectangular(in_size, out_size, complex=True)
    input = torch.randn((batch_size, in_size, 2))
    output = model(input)
    assert output.shape == (batch_size, out_size, 2)


def test_block2x2diagproductrectangular_tied_weight():
    batch_size = 3
    in_size = 7
    out_size = 15
    model = Block2x2DiagProductRectangular(in_size, out_size, tied_weight=False)
    input = torch.randn((batch_size, in_size))
    output = model(input)
    assert output.shape == (batch_size, out_size)
    model = Block2x2DiagProductRectangular(in_size, out_size, complex=True, tied_weight=False)
    input = torch.randn((batch_size, in_size, 2))
    output = model(input)
    assert output.shape == (batch_size, out_size, 2)


def test_blockpermproduct():
    size = 8
    input = torch.randn(3, size, 2)
    perm = BlockPermProduct(size, complex=True, share_logit=True)
    perm.logit[0] = float('inf')
    from utils import bitreversal_permutation
    assert torch.allclose(perm(input), input[:, bitreversal_permutation(size)])


def main():
    test_butterfly()
    test_butterfly_product()


if __name__ == '__main__':
    main()
