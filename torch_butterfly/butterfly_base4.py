import math
import numbers

import torch
from torch import nn
import torch.nn.functional as F

from torch_butterfly import Butterfly
from torch_butterfly.multiply_base4 import butterfly_multiply_base4_torch
from torch_butterfly.multiply_base4 import twiddle_base2_to_base4
from torch_butterfly.complex_utils import real_dtype_to_complex, view_as_real, view_as_complex
from torch_butterfly.complex_utils import complex_matmul


class ButterflyBase4(Butterfly):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        twiddle = self.twiddle if not self.complex else view_as_complex(self.twiddle)
        twiddle4, twiddle2 = twiddle_base2_to_base4(twiddle, self.increasing_stride)
        self.twiddle4 = nn.Parameter(twiddle4)
        self.twiddle2 = nn.Parameter(twiddle2)
        del self.twiddle
        # Pytorch 1.6 doesn't support torch.Tensor.add_(other, alpha) yet.
        # This is used in optimizers such as SGD.
        # So we have to store the parameters as real.
        if self.complex:
            self.twiddle4 = nn.Parameter(view_as_real(self.twiddle4))
            self.twiddle2 = nn.Parameter(view_as_real(self.twiddle2))
        self.twiddle4._is_structured = True  # Flag to avoid weight decay
        self.twiddle2._is_structured = True  # Flag to avoid weight decay

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size)
        Return:
            output: (batch, *, out_size)
        """
        twiddle4 = self.twiddle4 if not self.complex else view_as_complex(self.twiddle4)
        twiddle2 = self.twiddle2 if not self.complex else view_as_complex(self.twiddle2)
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        output = butterfly_multiply_base4_torch(twiddle4, twiddle2, output,
                                                self.increasing_stride, output_size)
        return self.post_process(input, output)

    def __imul__(self, scale):
        """In-place multiply the whole butterfly matrix by some scale factor, by multiplying the
        twiddle.
        Scale must be nonnegative
        """
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        scale_per_entry = scale ** (1.0 / self.nblocks / self.log_n)
        self.twiddle4 *= scale_per_entry ** 2
        self.twiddle2 *= scale_per_entry
        return self
