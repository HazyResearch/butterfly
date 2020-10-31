import math
import numbers

import torch
from torch import nn
import torch.nn.functional as F

from torch_butterfly import Butterfly
from torch_butterfly.multiply_base4 import butterfly_multiply_base4_torch
from torch_butterfly.multiply_base4 import twiddle_base2_to_base4
from torch_butterfly.complex_utils import real_dtype_to_complex
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
        init = kwargs.get('init', None)
        if (isinstance(init, tuple) and len(init) == 2 and isinstance(init[0], torch.Tensor)
            and isinstance(init[1], torch.Tensor)):
            twiddle4, twiddle2 = init[0].clone(), init[1].clone()
            kwargs['init'] = 'empty'
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
            with torch.no_grad():
                twiddle4, twiddle2 = twiddle_base2_to_base4(self.twiddle, self.increasing_stride)
        del self.twiddle
        self.twiddle4 = nn.Parameter(twiddle4)
        self.twiddle2 = nn.Parameter(twiddle2)
        self.twiddle4._is_structured = True  # Flag to avoid weight decay
        self.twiddle2._is_structured = True  # Flag to avoid weight decay

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size)
        Return:
            output: (batch, *, out_size)
        """
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        output = butterfly_multiply_base4_torch(self.twiddle4, self.twiddle2, output,
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
