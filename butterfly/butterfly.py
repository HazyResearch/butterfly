import math
import torch
from torch import nn
import torch.nn.functional as F

from .utils import twiddle_normal_to_fast_format
from .butterfly_multiply import butterfly_mult, butterfly_mult_untied
from .butterfly_multiply import butterfly_ortho_mult_tied, bbt_ortho_mult_tied
from .butterfly_multiply import butterfly_ortho_mult_untied, butterfly_mult_untied_svd
from .butterfly_multiply import bbt_mult_untied, bbt_ortho_mult_untied
from .butterfly_multiply import odo_mult_untied

class Butterfly(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 1).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        param: The parameterization of the 2x2 butterfly factors, either 'regular', 'ortho', 'odo', or 'obdobt', or 'svd'.
            'ortho' and 'svd' only support real, not complex.
            'odo' means two orthogonal butterfly matrices and one diagonal matrix.
            'obdobt' means the building block is (OB D OBT)^nblocks, where OB means orthogonal butterfly.
        max_gain: (only for svd parameterization) controls the maximum and minimum singular values
            of the whole matrix (not of each factor).
            For example, max_gain=10.0 means that the singular values are in [0.1, 10.0].
        nblocks: number of (BB^T) blocks. If 0, it's just a butterfly. If > 0, ignore @increasing_stride.
        diag_constraint: whether to constrain the diagonal in ODO parameterization.
            None (no constraint), 'positive' (>= 0), 'bounded' (between [1/max_gain, max_gain]),
                'square' (use sigma^2 parameterization instead)
        expansion: the linear map is a sum of @expansion butterfly matrices
        diag_init: whether to initialize the diagonal in ODO with 1, or N(0, 1)
            'one', or 'normal'
        double: whether to double the input size, i.e. x -> [x, 0]
    """

    def __init__(self, in_size, out_size, bias=True, complex=False, tied_weight=True,
                 increasing_stride=True, ortho_init=False, param='regular', max_gain=10.0,
                 nblocks=0, diag_constraint=None, expansion=1, diag_init='normal', double=False, diag_bookends=False,
                 fast=True):
        super().__init__()
        self.double = double
        if double:
            in_size *= 2
            out_size *= 2
        self.in_size = in_size
        m = int(math.ceil(math.log2(in_size)))
        self.m = m
        size = self.in_size_extended = 1 << m  # Will zero-pad input if in_size is not a power of 2
        self.out_size = out_size
        self.nstack = int(math.ceil(out_size / self.in_size_extended))
        self.complex = complex
        self.tied_weight = tied_weight
        self.increasing_stride = increasing_stride
        self.ortho_init = ortho_init
        assert param in ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt', 'svd', 'ds', 'logit', 'ortho2']
        self.param = param
        self.max_gain_per_factor = max_gain ** (1 / m)
        self.nblocks = nblocks
        assert diag_constraint in [None, 'positive', 'bounded', 'square']
        self.diag_constraint = diag_constraint
        self.max_gain = max_gain
        self.expansion = expansion
        self.diag_init = diag_init
        self.diag_bookends = diag_bookends
        self.fast = fast
        self.nstack *= self.expansion
        if nblocks > 0:
            assert not complex, 'native BBT with complex is not supported, use two separate Butterflies (e.g. nn.Sequential)'
            if param not in  ['odo', 'odr', 'opdo']:  # Special case, we implement tied weight ODO with nblocks
                assert not tied_weight and param in ['regular', 'ortho', 'odo', 'odr', 'opdo', 'obdobt'], 'native BBT with tied_weight or complex or non-regular param is not supported, use two separate Butterflies'
        if tied_weight:
            twiddle_core_shape = (self.nstack, size - 1) if nblocks == 0 else (self.nstack, nblocks * 2, size - 1)
        else:
            twiddle_core_shape = (self.nstack, m, size // 2) if nblocks == 0 else (self.nstack, nblocks * 2 * m, size // 2)
        if param == 'regular':
            if not ortho_init:
                twiddle_shape = twiddle_core_shape + ((2, 2) if not complex else (2, 2, 2))
                scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
                self.twiddle = nn.Parameter(torch.randn(twiddle_shape) * scaling)
            else:
                if not complex:
                    theta = torch.rand(twiddle_core_shape) * math.pi * 2
                    c, s = torch.cos(theta), torch.sin(theta)
                    det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype) * 2 - 1  # Rotation (+1) or reflection (-1)
                    self.twiddle = nn.Parameter(torch.stack((torch.stack((det * c, -det * s), dim=-1),
                                                             torch.stack((s, c), dim=-1)), dim=-2))
                else:
                    # Sampling from the Haar measure on U(2) is a bit subtle.
                    # Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
                    phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
                    c, s = torch.cos(phi), torch.sin(phi)
                    alpha, psi, chi = torch.randn((3, ) + twiddle_core_shape) * math.pi * 2
                    A = torch.stack((c * torch.cos(alpha + psi), c * torch.sin(alpha + psi)), dim=-1)
                    B = torch.stack((s * torch.cos(alpha + chi), s * torch.sin(alpha + chi)), dim=-1)
                    C = torch.stack((-s * torch.cos(alpha - chi), -s * torch.sin(alpha - chi)), dim=-1)
                    D = torch.stack((c * torch.cos(alpha - psi), c * torch.sin(alpha - psi)), dim=-1)
                    self.twiddle = nn.Parameter(torch.stack((torch.stack((A, B), dim=-2),
                                                             torch.stack((C, D), dim=-2)), dim=-3))
            if fast:
                if nblocks == 0:
                    twiddle_fast = twiddle_normal_to_fast_format(self.twiddle)
                    if not increasing_stride:
                        twiddle_fast = twiddle_fast.flip(1)
                else:
                    twiddle_fast = []
                    for i, chunk in enumerate(self.twiddle.chunk(nblocks * 2, dim=1)):
                        chunk_fast = twiddle_normal_to_fast_format(chunk)
                        if i % 2 == 0:
                            chunk_fast = chunk_fast.flip(1)
                        twiddle_fast.append(chunk_fast)
                    twiddle_fast = torch.cat(twiddle_fast, dim=1)
                self.twiddle = nn.Parameter(twiddle_fast)
        else:
            assert not complex, 'orthogonal/svd parameterization is only implemented for real, not complex'
            if diag_init == 'normal':
                self.diag_gen = torch.randn # lambda s, n: torch.randn(s, n)
            elif diag_init == 'one':
                self.diag_gen = torch.ones
            elif diag_init == 'bernoulli':
                self.diag_gen = lambda shape: (torch.rand(shape) >= 0.5).float() * 2 - 1
            else: assert False, f"diag_init type {diag_init} not supported"

            if param == 'odo' and self.fast:
                self.twiddle = nn.Parameter(torch.rand(twiddle_core_shape) * math.pi * 2)
                # TODO: assuming that nblocks > 0, otherwise there's no diag
                # # Initialize diag away from zero for numerical stability
                # while True:  # Rejection sampling
                #     diag_temp = self.diag_gen(self.nstack * nblocks * size * 2)
                #     diag_temp = diag_temp[diag_temp.abs() > 0.2]
                #     if diag_temp.numel() >= self.nstack * nblocks * size:
                #         break
                # self.diag = nn.Parameter(diag_temp[:self.nstack * nblocks * size].view(self.nstack, nblocks, size))
                self.diag = nn.Parameter(self.diag_gen(self.nstack, nblocks, size))
                self.diag._is_structured = True
            elif param == 'ortho':
                assert not complex
                self.twiddle = nn.Parameter(torch.rand(twiddle_core_shape) * math.pi * 2)
            elif param == 'odo' or param == 'odr' or param == 'opdo':
                assert not complex
                if param == 'odr':
                    self.register_buffer('twiddle', torch.rand(twiddle_core_shape) * math.pi * 2)
                else:
                    self.twiddle = nn.Parameter(torch.rand(twiddle_core_shape) * math.pi * 2)
                self.twiddle1 = nn.Parameter(torch.rand(twiddle_core_shape) * math.pi * 2)
                self.twiddle1._is_structured = True
                # if diag_init == 'normal':
                #     self.diag = nn.Parameter(torch.randn(self.nstack, size) / math.sqrt(self.nstack))
                # else:
                #     self.diag = nn.Parameter(torch.ones(self.nstack, size) / math.sqrt(self.nstack))
                # self.diag = diag_gen(self.nstack, size) / math.sqrt(self.nstack)
                self.diag = nn.Parameter(self.diag_gen(self.nstack, size) / math.sqrt(self.nstack))
                self.diag._is_structured = True
                if self.diag_bookends:  # Extra diagonals on the left and right
                    # self.diag_left = nn.Parameter((torch.rand(self.nstack, size) >= 0.5).float() * 2 - 1)
                    # self.diag_right = nn.Parameter((torch.rand(self.nstack, size) >= 0.5).float() * 2 - 1)
                    self.diag_l = nn.Parameter(self.diag_gen(self.nstack, size))
                    self.diag_r = nn.Parameter(self.diag_gen(self.nstack, size))
                    self.diag_l._is_structured = True
                    self.diag_r._is_structured = True
            elif param == 'obdobt':
                assert not tied_weight and not complex
                self.twiddle = nn.Parameter(torch.rand(twiddle_core_shape) * math.pi * 2)
                self.twiddle1 = nn.Parameter(torch.rand(twiddle_core_shape) * math.pi * 2)
                self.twiddle1._is_structured = True
                # if diag_init == 'normal':
                #     self.diag = nn.Parameter(torch.randn(twiddle_core_shape[0], self.nstack, size))
                # else:
                #     self.diag = nn.Parameter(torch.ones(twiddle_core_shape[0], self.nstack, size))
                self.diag = nn.Parameter(self.diag_gen(twiddle_core_shape[0], self.nstack, size))
                self.diag._is_structured = True
            elif param == 'svd':
                assert not tied_weight, 'svd parameterization is only implemented for non-tied weight'
                theta_phi = torch.rand(twiddle_core_shape + (2, )) * math.pi * 2
                sigmas = torch.ones(twiddle_core_shape + (2, ), dtype=theta_phi.dtype) # Singular values
                self.twiddle = nn.Parameter(torch.stack((theta_phi, sigmas) , dim=-2))
            elif param == 'ds':
                self.twiddle = nn.Parameter(torch.rand(twiddle_core_shape))
            elif param == 'logit':
                self.twiddle = nn.Parameter(torch.rand(twiddle_core_shape)*2-1)
            elif param == 'ortho2':
                self.twiddle = nn.Parameter(torch.rand(twiddle_core_shape) * 2*math.pi)
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        if bias:
            if not self.double:
                bias_shape = (out_size, ) if not complex else (out_size, 2)
            else:
                bias_shape = (out_size // 2, ) if not complex else (out_size // 2, 2)
            self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size) if real or (batch, *, in_size, 2) if complex
        Return:
            output: (batch, *, out_size) if real or (batch, *, out_size, 2) if complex
        """
        output = self.pre_process(input)
        if self.param == 'regular':
            if self.tied_weight:
                output = butterfly_mult(self.twiddle, output, self.increasing_stride)
            else:
                output = butterfly_mult_untied(self.twiddle, output, self.increasing_stride, self.training, self.fast) if self.nblocks == 0 else bbt_mult_untied(self.twiddle, output, self.fast)
        elif self.param == 'ortho':
            if self.tied_weight:
                output = butterfly_ortho_mult_tied(self.twiddle, output, self.increasing_stride)
            else:
                output = butterfly_ortho_mult_untied(self.twiddle, output, self.increasing_stride) if self.nblocks == 0 else bbt_ortho_mult_untied(self.twiddle, output)
        elif self.param == 'odo' or self.param == 'odr' or self.param == 'opdo':
            diag = self.diag
            if self.diag_constraint == 'positive':
                with torch.no_grad():  # Projected SGD
                    diag.clamp_(min=0)
            elif self.diag_constraint == 'bounded':
                with torch.no_grad():  # Projected SGD
                    diag.clamp_(min=1 / self.max_gain, max=self.max_gain)
            elif self.diag_constraint == 'square':
                diag = diag * diag
            if self.param == 'odo' and self.fast:
                output = odo_mult_untied(self.twiddle, diag, output)
            else:
                if self.diag_bookends:
                    output = output * self.diag_r
                if self.tied_weight:
                    output = butterfly_ortho_mult_tied(self.twiddle, output, False) if self.nblocks == 0 else bbt_ortho_mult_tied(self.twiddle, output)
                else:
                    output = butterfly_ortho_mult_untied(self.twiddle, output, self.increasing_stride) if self.nblocks == 0 else bbt_ortho_mult_untied(self.twiddle, output)
                output = output * diag
                if self.param == 'opdo' and self.expansion > 1:
                    output = output.view(-1, self.expansion, output.shape[-1]).sum(dim=-2, keepdim=True).expand(-1, self.expansion, -1).reshape(output.shape)
                if self.tied_weight:
                    output = butterfly_ortho_mult_tied(self.twiddle1, output, True) if self.nblocks == 0 else bbt_ortho_mult_tied(self.twiddle1, output)
                else:
                    output = butterfly_ortho_mult_untied(self.twiddle1, output, not self.increasing_stride) if self.nblocks == 0 else bbt_ortho_mult_untied(self.twiddle1, output)
                if self.diag_bookends:
                    output = output * self.diag_l
        elif self.param == 'obdobt':
            for t, t1, d in zip(self.twiddle.split(self.m, dim=1), self.twiddle1.split(self.m, dim=1), self.diag):
                output = butterfly_ortho_mult_untied(t, output, False)
                output = output * d
                output = butterfly_ortho_mult_untied(t1, output, True)
        elif self.param == 'svd':
            with torch.no_grad():  # Projected SGD
                self.twiddle[..., 1, :].clamp_(min=1 / self.max_gain_per_factor, max=self.max_gain_per_factor)
            output = butterfly_mult_untied_svd(self.twiddle, output, self.increasing_stride)
        elif self.param == 'ds':
            p = self.twiddle
            twiddle = torch.stack((torch.stack((p, 1-p), dim=-1),
                                   torch.stack((1-p, p), dim=-1)), dim=-2)
            output = butterfly_mult(twiddle, output, self.increasing_stride) if self.tied_weight else butterfly_mult_untied(twiddle, output, self.increasing_stride)
        elif self.param == 'logit':
            p = 1.0/(1.0 + torch.exp(-self.twiddle))
            twiddle = torch.stack((torch.stack((p, 1-p), dim=-1),
                                   torch.stack((1-p, p), dim=-1)), dim=-2)
            output = butterfly_mult(twiddle, output, self.increasing_stride) if self.tied_weight else butterfly_mult_untied(twiddle, output, self.increasing_stride)
        elif self.param == 'ortho2':
            p = torch.cos(self.twiddle)**2
            # p = self.twiddle
            twiddle = torch.stack((torch.stack((p, 1-p), dim=-1),
                                   torch.stack((1-p, p), dim=-1)), dim=-2)
            output = butterfly_mult(twiddle, output, self.increasing_stride) if self.tied_weight else butterfly_mult_untied(twiddle, output, self.increasing_stride)
        return self.post_process(input, output)

    def pre_process(self, input):
        if self.complex:  # Reshape to (N, in_size, 2)
            output = input.view(-1, *input.size()[-2:])
        else:  # Reshape to (N, in_size)
            output = input.view(-1, input.size(-1))
        if self.double:
            output = F.pad(output, (0, output.shape[-1]))
        batch = output.shape[0]
        if self.in_size != self.in_size_extended:  # Zero-pad
            padding = (0, self.in_size_extended - self.in_size) if not self.complex else (0, 0, 0, self.in_size_extended - self.in_size)
            output = F.pad(output, padding)
        output = output.unsqueeze(1).expand((batch, self.nstack, self.in_size_extended) + (() if not self.complex else (2, )))
        return output

    def post_process(self, input, output):
        batch = output.shape[0]
        output = output.view((batch, self.nstack * self.in_size_extended) + (() if not self.complex else (2, )))
        out_size_extended = 1 << (int(math.ceil(math.log2(self.out_size))))
        if (self.nstack * self.in_size_extended // out_size_extended >= 2):  # Average instead of just take the top rows
            if not self.complex:
                output = output.view(batch, self.nstack * self.in_size_extended // out_size_extended, out_size_extended).sum(dim=1)
            else:
                output = output.view(batch, self.nstack * self.in_size_extended // out_size_extended, out_size_extended, 2).sum(dim=1)
        if self.double:
            output = output.view(batch, 2, out_size_extended // 2).sum(dim=1)
        if self.out_size != out_size_extended:  # Take top rows
            output = output[:, :self.out_size] if not self.double else output[:, :self.out_size // 2]
        if self.bias is not None:
            output = output + self.bias
        if self.complex:
            return output.view(*input.size()[:-2], self.out_size, 2) if not self.double else output.view(*input.size()[:-2], self.out_size // 2, 2)
        else:
            return output.view(*input.size()[:-1], self.out_size) if not self.double else output.view(*input.size()[:-1], self.out_size // 2)

    def extra_repr(self):
        s = 'in_size={}, out_size={}, bias={}, complex={}, tied_weight={}, increasing_stride={}, ortho_init={}, param={}, nblocks={}, expansion={}, diag_init={}, double={}'.format(
            self.in_size, self.out_size, self.bias is not None, self.complex, self.tied_weight, self.increasing_stride, self.ortho_init, self.param, self.nblocks, self.expansion, self.diag_init, self.double
        )
        if self.param == 'odo' or self.param == 'odr' or self.param == 'opdo':
            s += ', diag_constraint={}'.format('none' if self.diag_constraint is None else self.diag_constraint)
        return s

    def round_to_perm(self):
        if self.param in ['ds', 'logit', 'ortho2']:
            if self.param == 'ds':
                self.twiddle_round = self.twiddle.data
                self.twiddle.data = torch.where(self.twiddle_round > 0.5, torch.tensor(1.0), torch.tensor(0.0))
            elif self.param == 'logit':
                self.twiddle_round = 1.0/(1.0 + torch.exp(-self.twiddle.data))
                self.twiddle.data = torch.where(self.twiddle_round > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
            elif self.param == 'ortho2':
                self.twiddle_round = torch.cos(self.twiddle.data)**2
                self.twiddle.data = torch.where(self.twiddle_round > 0.5, torch.tensor(0.0), torch.tensor(math.pi/2.0))



class ButterflyBmm(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Perform batch matrix multiply.

    Parameters:
        in_size: size of input
        out_size: size of output
        matrix_batch: how many copies of the matrix
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        param: whether to parameterize the twiddle to always be orthogonal 2x2 matrices.
            Only implemented for real, not complex, for now.
        max_gain: (only for svd parameterization) controls the maximum and minimum singular values
            of the whole matrix (not of each factor).
            For example, max_gain=10.0 means that the singular values are in [0.1, 10.0].
        nblocks: number of (BB^T) blocks. If 0, it's just a butterfly. If > 0, ignore @increasing_stride.
        diag_constraint: whether to constrain the diagonal in ODO parameterization.
            None (no constraint), 'positive' (>= 0), 'bounded' (between [1/max_gain, max_gain]),
                'square' (use sigma^2 parameterization instead)
        expansion: the linear map is a sum of @expansion butterfly matrices
        double: whether to double the input size, i.e. x -> [x, 0]
    """

    def __init__(self, in_size, out_size, matrix_batch=1, bias=True, complex=False, tied_weight=True,
                 increasing_stride=True, ortho_init=False, param='regular', max_gain=10.0,
                 nblocks=0, diag_constraint=None, expansion=1, diag_init='one', double=False,
                 **extra_args):
        m = int(math.ceil(math.log2(in_size)))
        in_size_extended = 1 << m  # Will zero-pad input if in_size is not a power of 2
        nstack = int(math.ceil(out_size / in_size_extended))
        super().__init__(in_size_extended, in_size_extended * nstack * matrix_batch, bias, complex,
                         tied_weight, increasing_stride, ortho_init, param, max_gain, nblocks,
                         diag_constraint, expansion, diag_init, double, **extra_args)
        self.in_size = in_size if not double else in_size * 2
        self.out_size = out_size if not double else out_size * 2
        self.nstack = nstack * expansion
        self.matrix_batch = matrix_batch
        if self.bias is not None:
            with torch.no_grad():
                bias_reshape = self.bias.view((matrix_batch, in_size_extended * nstack) + (() if not self.complex else (2, )))
                self.bias = nn.Parameter(bias_reshape[:, :out_size].contiguous())

    def pre_process(self, input):
        batch = input.shape[0]
        output = input
        if self.in_size != self.in_size_extended:  # Zero-pad
            padding = (0, self.in_size_extended - self.in_size) if not self.complex else (0, 0, 0, self.in_size_extended - self.in_size)
            output = F.pad(output, padding)
        if self.double:
            output = F.pad(output, (0, output.shape[-1]))
        output = output.unsqueeze(2).expand((batch, self.matrix_batch, self.nstack, self.in_size_extended) + (() if not self.complex else (2, )))
        output = output.reshape((batch, self.matrix_batch * self.nstack, self.in_size_extended) + (() if not self.complex else (2, )))
        return output

    def post_process(self, input, output):
        batch = output.shape[0]
        output = output.view((batch, self.matrix_batch, self.nstack * self.in_size_extended) + (() if not self.complex else (2, )))
        out_size_extended = 1 << (int(math.ceil(math.log2(self.out_size))))
        if (self.nstack * self.in_size_extended // out_size_extended >= 2):  # Sum instead of just take the top rows
            if not self.complex:
                output = output.view(batch, self.matrix_batch, self.nstack * self.in_size_extended // out_size_extended, out_size_extended).sum(dim=2)
            else:
                output = output.view(batch, self.matrix_batch, self.nstack * self.in_size_extended // out_size_extended, out_size_extended, 2).sum(dim=2)
        if self.double:
            output = output.view(batch, self.matrix_batch, 2, out_size_extended // 2).mean(dim=2)
        if self.out_size != out_size_extended:  # Take top rows
            output = output[:, :, :self.out_size] if not self.double else output[:, :, :self.out_size // 2]
        return output if self.bias is None else output + self.bias

    def extra_repr(self):
        s = 'in_size={}, out_size={}, matrix_batch={}, bias={}, complex={}, tied_weight={}, increasing_stride={}, ortho_init={}, param={}, nblocks={}, expansion={}, diag_init={}, double={}'.format(
            self.in_size, self.out_size, self.matrix_batch, self.bias is not None, self.complex, self.tied_weight, self.increasing_stride, self.ortho_init, self.param, self.nblocks, self.expansion, self.diag_init, self.double
        )
        if self.param == 'odo':
            s += ', diag_constraint={}'.format('none' if self.diag_constraint is None else self.diag_constraint)
        return s
