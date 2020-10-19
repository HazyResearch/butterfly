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
        init: 'randn', 'ortho', or 'identity'. Whether the weight matrix should be initialized to
            from randn twiddle, or to be randomly orthogonal/unitary, or to be the identity matrix.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, bias=True, complex=False,
                 increasing_stride=True, init='randn', nblocks=1):
        super().__init__()
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = complex
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype() if not self.complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = (self.nstacks, nblocks, log_n, n // 2, 2, 2)
        assert init in ['randn', 'ortho', 'identity']
        self.init = init
        self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        # Pytorch 1.6 doesn't support torch.Tensor.add_(other, alpha) yet.
        # This is used in optimizers such as SGD.
        # So we have to store the parameters as real.
        if self.complex:
            self.twiddle = nn.Parameter(view_as_real(self.twiddle))
            if self.bias is not None:
                self.bias = nn.Parameter(view_as_real(self.bias))
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        twiddle = self.twiddle if not self.complex else view_as_complex(self.twiddle)
        if self.init == 'randn':
            # complex randn already has the correct scaling of stddev=1.0
            scaling = 1.0 / math.sqrt(2)
            with torch.no_grad():
                twiddle.copy_(torch.randn(twiddle.shape, dtype=twiddle.dtype) * scaling)
        elif self.init == 'ortho':
            twiddle_core_shape = twiddle.shape[:-2]
            if not self.complex:
                theta = torch.rand(twiddle_core_shape) * math.pi * 2
                c, s = torch.cos(theta), torch.sin(theta)
                det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype) * 2 - 1  # Rotation (+1) or reflection (-1)
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((det * c, -det * s), dim=-1),
                                               torch.stack((s, c), dim=-1)), dim=-2))
            else:
                # Sampling from the Haar measure on U(2) is a bit subtle.
                # Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
                phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
                c, s = torch.cos(phi), torch.sin(phi)
                alpha, psi, chi = torch.rand((3, ) + twiddle_core_shape) * math.pi * 2
                A = torch.exp(1j * (alpha + psi)) * c
                B = torch.exp(1j * (alpha + chi)) * s
                C = -torch.exp(1j * (alpha - chi)) * s
                D = torch.exp(1j * (alpha - psi)) * c
                with torch.no_grad():
                    twiddle.copy_(torch.stack((torch.stack((A, B), dim=-1),
                                               torch.stack((C, D), dim=-1)), dim=-2))
        elif self.init == 'identity':
            twiddle_new = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
            twiddle_new = twiddle_new.expand(*twiddle.shape).contiguous()
            with torch.no_grad():
                twiddle.copy_(twiddle_new)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.n in this case
        Return:
            output: (batch, *, out_size)
        """
        twiddle = self.twiddle if not self.complex else view_as_complex(self.twiddle)
        output = self.pre_process(input)
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = (twiddle[:, :, :log_n, :n // 2] if self.increasing_stride
                       else twiddle[:, :, -log_n:, :n // 2])
        if conjugate and self.complex:
            twiddle = twiddle.conj()
        if not transpose:
            output = butterfly_multiply(twiddle, output, self.increasing_stride)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.nblocks - 1) % 2 == 1)
            output = butterfly_multiply(twiddle, output, not last_increasing_stride)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))

    def pre_process(self, input):
        # Reshape to (N, in_size)
        input_size = input.size(-1)
        output = input.reshape(-1, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(1).expand(batch, self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * output.size(-1))
        out_size_extended = 1 << (int(math.ceil(math.log2(out_size))))
        if out_size != out_size_extended:  # Take top rows
            output = output[:, :out_size]
        if self.bias is not None:
            bias = self.bias if not self.complex else view_as_complex(self.bias)
            output = output + bias[:out_size]
        return output.view(*input.size()[:-1], out_size)

    def extra_repr(self):
        s = 'in_size={}, out_size={}, bias={}, complex={}, increasing_stride={}, init={}, nblocks={}'.format(
            self.in_size, self.out_size, self.bias is not None, self.complex, self.increasing_stride, self.init, self.nblocks,)
        return s


class ButterflyUnitary(Butterfly):
    """Same as Butterfly, but constrained to be unitary
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, bias=True, increasing_stride=True, nblocks=1):
        nn.Module.__init__(self)
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n  # Will zero-pad input if in_size is not a power of 2
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = True
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        complex_dtype = real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = (self.nstacks, nblocks, log_n, n // 2, 4)
        self.init = 'ortho'
        self.twiddle = nn.Parameter(torch.empty(twiddle_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_size, dtype=complex_dtype))
        else:
            self.register_parameter('bias', None)
        # Pytorch 1.6 doesn't support torch.Tensor.add_(other, alpha) yet.
        # This is used in optimizers such as SGD.
        # So we have to store the parameters as real.
        if self.bias is not None:
            self.bias = nn.Parameter(view_as_real(self.bias))
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        # Sampling from the Haar measure on U(2) is a bit subtle.
        # Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
        twiddle_core_shape = self.twiddle.shape[:-1]
        phi = torch.asin(torch.sqrt(torch.rand(twiddle_core_shape)))
        alpha, psi, chi = torch.rand((3, ) + twiddle_core_shape) * math.pi * 2
        with torch.no_grad():
            self.twiddle.copy_(torch.stack([phi, alpha, psi, chi], dim=-1))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.n in this case
        Return:
            output: (batch, *, out_size)
        """
        phi, alpha, psi, chi = torch.unbind(self.twiddle, -1)
        c, s = torch.cos(phi), torch.sin(phi)
        # Pytorch 1.6.0 doesn't support complex exp on GPU so we have to use cos/sin
        A = torch.stack((c * torch.cos(alpha + psi), c * torch.sin(alpha + psi)), dim=-1)
        B = torch.stack((s * torch.cos(alpha + chi), s * torch.sin(alpha + chi)), dim=-1)
        C = torch.stack((-s * torch.cos(alpha - chi), -s * torch.sin(alpha - chi)), dim=-1)
        D = torch.stack((c * torch.cos(alpha - psi), c * torch.sin(alpha - psi)), dim=-1)
        twiddle = torch.stack([torch.stack([A, B], dim=-2),
                               torch.stack([C, D], dim=-2)], dim=-3)
        twiddle = view_as_complex(twiddle)
        output = self.pre_process(input)
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = (twiddle[:, :, :log_n, :n // 2] if self.increasing_stride
                       else twiddle[:, :, -log_n:, :n // 2])
        if conjugate and self.complex:
            twiddle = twiddle.conj()
        if not transpose:
            output = butterfly_multiply(twiddle, output, self.increasing_stride)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.nblocks - 1) % 2 == 1)
            output = butterfly_multiply(twiddle, output, not last_increasing_stride)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))

    def extra_repr(self):
        s = 'in_size={}, out_size={}, bias={}, increasing_stride={}, nblocks={}'.format(
            self.in_size, self.out_size, self.bias is not None, self.increasing_stride, self.nblocks,)
        return s


class ButterflyBmm(Butterfly):
    """Same as Butterfly, but performs batched matrix multiply.
    Compatible with torch.nn.Linear.

    Parameters:
        in_size: size of input
        out_size: size of output
        matrix_batch: how many butterfly matrices
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: 'randn', 'ortho', or 'identity'. Whether the weight matrix should be initialized to
            from randn twiddle, or to be randomly orthogonal/unitary, or to be the identity matrix.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, matrix_batch=1, bias=True, complex=False,
                 increasing_stride=True, init='randn', nblocks=1):
        nn.Module.__init__(self)
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.matrix_batch = matrix_batch
        self.nstacks = int(math.ceil(out_size / self.n))
        self.complex = complex
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype() if not self.complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = (self.matrix_batch * self.nstacks, nblocks, log_n, n // 2, 2, 2)
        assert init in ['randn', 'ortho', 'identity']
        self.init = init
        self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.matrix_batch, out_size, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        # Pytorch 1.6 doesn't support torch.Tensor.add_(other, alpha) yet.
        # This is used in optimizers such as SGD.
        # So we have to store the parameters as real.
        if self.complex:
            self.twiddle = nn.Parameter(view_as_real(self.twiddle))
            if self.bias is not None:
                self.bias = nn.Parameter(view_as_real(self.bias))
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self.reset_parameters()

    def forward(self, input, transpose=False, conjugate=False):
        """
        Parameters:
            input: (batch, *, matrix_batch, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
        Return:
            output: (batch, *, matrix_batch, out_size)
        """
        return super().forward(input, transpose, conjugate, subtwiddle=False)

    def pre_process(self, input):
        # Reshape to (N, matrix_batch, in_size)
        input_size = input.size(-1)
        assert input.size(-2) == self.matrix_batch
        output = input.reshape(-1, self.matrix_batch, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(2).expand(batch, self.matrix_batch, self.nstacks, input_size)
        output = output.reshape(batch, self.matrix_batch * self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.matrix_batch, self.nstacks * output.size(-1))
        out_size_extended = 1 << (int(math.ceil(math.log2(out_size))))
        if out_size != out_size_extended:  # Take top rows
            output = output[:, :, :out_size]
        if self.bias is not None:
            bias = self.bias if not self.complex else view_as_complex(self.bias)
            output = output + bias[:, :out_size]
        return output.view(*input.size()[:-2], self.matrix_batch, self.out_size)

    def extra_repr(self):
        s = 'in_size={}, out_size={}, matrix_batch={}, bias={}, complex={}, increasing_stride={}, init={}, nblocks={}'.format(
            self.in_size, self.out_size, self.matrix_batch, self.bias is not None, self.complex, self.increasing_stride, self.init, self.nblocks,)
        return s
