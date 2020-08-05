import math
import torch
from torch import nn
import torch.nn.functional as F

from torch_butterfly.multiply import butterfly_multiply
from torch_butterfly.multiply import butterfly_multiply_torch
from torch_butterfly.complex_utils import real_dtype_to_complex, view_as_real, view_as_complex


class Butterfly(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, bias=True, complex=False,
                 increasing_stride=True, ortho_init=False, nblocks=1):
        super().__init__()
        self.in_size = in_size
        log_n = int(math.ceil(math.log2(in_size)))
        self.log_n = log_n
        size = self.in_size_extended = 1 << log_n  # Will zero-pad input if in_size is not a power of 2
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.in_size_extended))
        self.complex = complex
        self.increasing_stride = increasing_stride
        self.ortho_init = ortho_init
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype() if not complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_core_shape = (self.nstacks, nblocks, log_n, size // 2)
        if not ortho_init:
            twiddle_shape = twiddle_core_shape + (2, 2)
            # complex randn already has the correct scaling of stddev=1.0
            scaling = 1.0 / math.sqrt(2)
            self.twiddle = nn.Parameter(torch.randn(twiddle_shape, dtype=dtype) * scaling)
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
                A = torch.exp(1j * (alpha + psi)) * c
                B = torch.exp(1j * (alpha + chi)) * s
                C = -torch.exp(1j * (alpha - chi)) * s
                D = torch.exp(1j * (alpha - psi)) * c
                self.twiddle = nn.Parameter(torch.stack((torch.stack((A, B), dim=-1),
                                                         torch.stack((C, D), dim=-1)), dim=-2))
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # Pytorch 1.6 doesn't support torch.Tensor.add_(other, alpha) yet.
        # This is used in optimizers such as SGD.
        # So we have to store the parameters as real.
        if complex:
            self.twiddle = nn.Parameter(view_as_real(self.twiddle))
            if self.bias is not None:
                self.bias = nn.Parameter(view_as_real(self.bias))

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        # TODO: init the twiddle here, with copy_
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size)
        Return:
            output: (batch, *, out_size)
        """
        output = self.pre_process(input)
        twiddle = self.twiddle if not self.complex else view_as_complex(self.twiddle)
        output = butterfly_multiply(twiddle, output, self.increasing_stride)
        return self.post_process(input, output)

    def pre_process(self, input):
        # Reshape to (N, in_size)
        output = input.reshape(-1, input.size(-1))
        batch = output.shape[0]
        if self.in_size != self.in_size_extended:  # Zero-pad
            output = F.pad(output, (0, self.in_size_extended - self.in_size))
        output = output.unsqueeze(1).expand(batch, self.nstacks, self.in_size_extended)
        return output

    def post_process(self, input, output):
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * self.in_size_extended)
        out_size_extended = 1 << (int(math.ceil(math.log2(self.out_size))))
        if self.out_size != out_size_extended:  # Take top rows
            output = output[:, :self.out_size]
        if self.bias is not None:
            bias = self.bias if not self.complex else view_as_complex(self.bias)
            output = output + bias
        return output.view(*input.size()[:-1], self.out_size)

    def extra_repr(self):
        s = 'in_size={}, out_size={}, bias={}, complex={}, increasing_stride={}, ortho_init={}, nblocks={}'.format(
            self.in_size, self.out_size, self.bias is not None, self.complex, self.increasing_stride, self.ortho_init, self.nblocks,)
        return s
