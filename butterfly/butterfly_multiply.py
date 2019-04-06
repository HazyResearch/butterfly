import math

import torch
from torch import nn

from .complex_utils import complex_mul

use_extension = True
try:
    from factor_multiply import butterfly_multiply_inplace, butterfly_multiply_inplace_backward
    from factor_multiply import butterfly_multiply_intermediate, butterfly_multiply_intermediate_backward
except:
    use_extension = False
    import warnings
    warnings.warn("C++/CUDA extension isn't installed. Will use butterfly multiply implemented in Pytorch, which is much slower.")


def butterfly_mult_torch(twiddle, input, return_intermediate=False):
    """
    Parameters:
        twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    nstack = twiddle.shape[0]
    assert twiddle.shape == (nstack, n - 1, 2, 2) if input.dim() == 2 else (nstack, n - 1, 2, 2, 2)
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    if input.dim() == 2:  # real
        output = input.contiguous().unsqueeze(1).expand(batch_size, nstack, n)
        intermediates = [output]
        for log_stride in range(m):
            stride = 1 << log_stride
            t = twiddle[:, (stride - 1):(2 * stride - 1)].permute(0, 2, 3, 1)  # shape (nstack, 2, 2, stride)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride)
            output = (t.unsqueeze(1) * output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n) if not return_intermediate else torch.stack(intermediates).view(m + 1, batch_size, nstack, n)
    else:  # complex
        output = input.contiguous().unsqueeze(1).expand(batch_size, nstack, n, 2)
        intermediates = [output]
        for log_stride in range(m):
            stride = 1 << log_stride
            t = twiddle[:, (stride - 1):(2 * stride - 1)].permute(0, 2, 3, 1, 4)  # shape (nstack, 2, 2, stride, 2)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride, 2)
            output = complex_mul(t.unsqueeze(1), output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n, 2) if not return_intermediate else torch.stack(intermediates).view(m + 1, batch_size, nstack, n, 2)


class ButterflyMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        """
        Parameters:
            twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
        Returns:
            output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        output_and_intermediate = butterfly_multiply_intermediate(twiddle, input)
        ctx.save_for_backward(twiddle, output_and_intermediate)
        return output_and_intermediate[-1]

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            output + intermediate values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
        Return:
            d_twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            d_input: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        twiddle, output_and_intermediate = ctx.saved_tensors
        d_coefficients, d_input = butterfly_multiply_intermediate_backward(grad, twiddle, output_and_intermediate)
        return d_coefficients, d_input

butterfly_mult = ButterflyMult.apply if use_extension else butterfly_mult_torch


class ButterflyMultInplace(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        """Experimental in-place implementation that does not store intermediate results.
        Instead, the intermediate results are computed from the output during the backward pass.
        Parameters:
            twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
        Returns:
            output: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        output = butterfly_multiply_inplace(twiddle, input)
        ctx.save_for_backward(twiddle, output)
        return output

    @staticmethod
    def backward(ctx, grad):
        twiddle, output = ctx.saved_tensors
        d_coefficients, d_input = butterfly_multiply_inplace_backward(grad, twiddle, output)
        return d_coefficients, d_input

butterfly_mult_inplace = ButterflyMultInplace.apply
