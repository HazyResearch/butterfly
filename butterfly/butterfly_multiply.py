import math

import torch
from torch import nn
import torch.nn.functional as F

from .complex_utils import complex_mul

use_extension = True
try:
    from factor_multiply import butterfly_multiply_intermediate, butterfly_multiply_intermediate_backward
    from factor_multiply import butterfly_multiply_untied, butterfly_multiply_untied_backward
    from factor_multiply import butterfly_multiply_untied_forward_backward
    from factor_multiply import butterfly_ortho_multiply_tied, butterfly_ortho_multiply_tied_backward
    from factor_multiply import butterfly_ortho_multiply_untied, butterfly_ortho_multiply_untied_backward
    from factor_multiply import bbt_multiply_untied, bbt_multiply_untied_forward_backward
    from factor_multiply import bbt_ortho_multiply_untied, bbt_ortho_multiply_untied_backward
    from factor_multiply import butterfly_multiply_untied_svd, butterfly_multiply_untied_svd_backward
    from factor_multiply import butterfly_multiply_untied_svd_forward_backward
    from factor_multiply import butterfly_multiply_inplace, butterfly_multiply_inplace_backward
    from factor_multiply import butterfly_factor_multiply, butterfly_factor_multiply_backward
    from factor_multiply import butterfly_conv2d, butterfly_conv2d_backward, butterfly_conv2d_forward_backward
    from factor_multiply import bbt_conv2d, bbt_conv2d_forward_backward
    from factor_multiply import butterfly_conv2d_svd, butterfly_conv2d_svd_forward_backward
    # from factor_multiply import butterfly_multiply_untied_eval

    import factor_multiply_fast as fmf
    from factor_multiply_fast import butterfly_multiply_untied_forward_fast
    from factor_multiply_fast import butterfly_multiply_untied_forward_backward_fast
    from factor_multiply_fast import butterfly_bbs_multiply_untied_forward_fast
    from factor_multiply_fast import butterfly_bbs_multiply_untied_forward_backward_fast
    from factor_multiply_fast import butterfly_odo_multiply_untied_forward_fast
    from factor_multiply_fast import butterfly_odo_multiply_untied_backward_fast
    from factor_multiply_fast import butterfly_odo_multiply_untied_forward_backward_fast
except:
    use_extension = False
    import warnings
    warnings.warn("C++/CUDA extension isn't installed properly. Will use butterfly multiply implemented in Pytorch, which is much slower.")

try:
    from apex import amp
    amp.register_float_function(fmf, 'butterfly_odo_multiply_untied_forward_fast')
    amp.register_float_function(fmf, 'butterfly_odo_multiply_untied_forward_backward_fast')
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex.")

def butterfly_mult_torch(twiddle, input, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
        input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 1).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
    """
    batch_size, nstack, n = input.shape[:3]
    assert twiddle.shape == (nstack, n - 1, 2, 2) if input.dim() == 2 else (nstack, n - 1, 2, 2, 2)
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    if input.dim() == 3:  # real
        output = input.contiguous()
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, (stride - 1):(2 * stride - 1)].permute(0, 2, 3, 1)  # shape (nstack, 2, 2, stride)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride)
            output = (t.unsqueeze(1) * output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n) for intermediate in intermediates])
    else:  # complex
        output = input.contiguous()
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, (stride - 1):(2 * stride - 1)].permute(0, 2, 3, 1, 4)  # shape (nstack, 2, 2, stride, 2)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride, 2)
            output = complex_mul(t.unsqueeze(1), output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n, 2) for intermediate in intermediates])


class ButterflyMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        # output_and_intermediate = butterfly_multiply_intermediate(twiddle, input, increasing_stride, True)
        # ctx.save_for_backward(twiddle, output_and_intermediate)
        output = butterfly_multiply_intermediate(twiddle, input, increasing_stride, False)
        ctx.save_for_backward(twiddle, input)
        ctx._increasing_stride = increasing_stride
        # return output_and_intermediate[-1]
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            output + intermediate values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
        Return:
            d_twiddle: (nstack, n - 1, 2, 2) if real or (nstack, n - 1, 2, 2, 2) if complex
            d_input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        # twiddle, output_and_intermediate = ctx.saved_tensors
        twiddle, input = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        output_and_intermediate = butterfly_multiply_intermediate(twiddle, input, increasing_stride, True)
        d_coefficients, d_input = butterfly_multiply_intermediate_backward(grad, twiddle, output_and_intermediate, increasing_stride)
        return d_coefficients, d_input, None  # Autograd requires 3 gradients

butterfly_mult = ButterflyMult.apply if use_extension else butterfly_mult_torch


def butterfly_mult_untied_torch(twiddle, input, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
        input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 1).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
    """
    batch_size, nstack, n = input.shape[:3]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    assert twiddle.shape == (nstack, m, n // 2, 2, 2) if input.dim() == 3 else (nstack, m, n // 2, 2, 2, 2)
    if input.dim() == 3:  # real
        output = input.contiguous()
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, log_stride].view(nstack, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)  # shape (nstack, n // (2 * stride, )2, 2, stride)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n) for intermediate in intermediates])
    else:  # complex
        output = input.contiguous()
        intermediates = [output]
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[:, log_stride].view(nstack, n // (2 * stride), stride, 2, 2, 2).permute(0, 1, 3, 4, 2, 5)  # shape (nstack, n // (2 * stride, )2, 2, stride, 2)
            output_reshape = output.view(batch_size, nstack, n // (2 * stride), 1, 2, stride, 2)
            output = complex_mul(t, output_reshape).sum(dim=4)
            intermediates.append(output)
        return output.view(batch_size, nstack, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, nstack, n, 2) for intermediate in intermediates])


class ButterflyMultUntied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True, is_training=True, fast=True):
        """
        Parameters:
            twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
            input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        # use optimized code for inference
        # if not is_training and not input.is_cuda and input.dim() == 3 and input.dtype == torch.float and input.shape[-1] > 8:
        if False:
            output = butterfly_multiply_untied_eval(twiddle, input, increasing_stride)
        else:
            if not fast:
                output = butterfly_multiply_untied(twiddle, input, increasing_stride, False)
            else:
                output = butterfly_multiply_untied_forward_fast(twiddle, input, increasing_stride)
        ctx.save_for_backward(twiddle, input)
        ctx._increasing_stride = increasing_stride
        ctx._fast = fast
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
            twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
            output + intermediate values for backward: (log n + 1, batch_size, nstack, n) if real or (log n + 1, batch_size, nstack, n, 2) if complex
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2) if real or (nstack, log n, n / 2, 2, 2, 2) if complex
            d_input: (batch_size, nstack, n) if real or (batch_size, nstack, n, 2) if complex
        """
        # twiddle, output_and_intermediate = ctx.saved_tensors
        twiddle, input = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        fast = ctx._fast
        n = input.shape[2]
        if input.dim() == 3 and n <= 1024 and input.is_cuda:
            if not fast:
                d_coefficients, d_input = butterfly_multiply_untied_forward_backward(twiddle, input, grad, increasing_stride)
            else:
                d_coefficients, d_input = butterfly_multiply_untied_forward_backward_fast(twiddle, input, grad, increasing_stride)
        else:
            output_and_intermediate = butterfly_multiply_untied(twiddle, input, increasing_stride, True)
            d_coefficients, d_input = butterfly_multiply_untied_backward(grad, twiddle, output_and_intermediate, increasing_stride)
        return d_coefficients, d_input, None, None, None  # Autograd requires 3 gradients

butterfly_mult_untied = ButterflyMultUntied.apply if use_extension else butterfly_mult_untied_torch


class ButterflyOrthoMultTied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, n - 1)
            input: (batch_size, nstack, n)
        Returns:
            output: (batch_size, nstack, n)
        """
        twiddle_cos, twiddle_sin = torch.cos(twiddle), torch.sin(twiddle)
        output = butterfly_ortho_multiply_tied(twiddle_cos, twiddle_sin, input, increasing_stride)
        ctx.save_for_backward(twiddle_cos, twiddle_sin, output)
        ctx._increasing_stride = increasing_stride
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n)
        Return:
            d_twiddle: (nstack, n - 1)
            d_input: (batch_size, nstack, n)
        """
        twiddle_cos, twiddle_sin, output = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        d_coefficients, d_input = butterfly_ortho_multiply_tied_backward(twiddle_cos, twiddle_sin, output, grad, increasing_stride)
        return d_coefficients, d_input, None


def butterfly_ortho_mult_tied(twiddle, input, increasing_stride):
    n = input.shape[2]
    if input.dim() == 3 and n <= 1024 and input.is_cuda:
        return ButterflyOrthoMultTied.apply(twiddle, input, increasing_stride)
    else:
        c, s = torch.cos(twiddle), torch.sin(twiddle)
        twiddle = torch.stack((torch.stack((c, -s), dim=-1),
                               torch.stack((s, c), dim=-1)), dim=-2)
        return butterfly_mult(twiddle, input, increasing_stride)


def butterfly_ortho_mult_tied_torch(twiddle, input, increasing_stride):
    c, s = torch.cos(twiddle), torch.sin(twiddle)
    twiddle = torch.stack((torch.stack((c, -s), dim=-1),
                           torch.stack((s, c), dim=-1)), dim=-2)
    return butterfly_mult_torch(twiddle, input, increasing_stride)


class ButterflyOrthoMultUntied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, log n, n / 2)
            input: (batch_size, nstack, n)
        Returns:
            output: (batch_size, nstack, n)
        """
        twiddle_cos, twiddle_sin = torch.cos(twiddle), torch.sin(twiddle)
        output = butterfly_ortho_multiply_untied(twiddle_cos, twiddle_sin, input, increasing_stride)
        ctx.save_for_backward(twiddle_cos, twiddle_sin, output)
        ctx._increasing_stride = increasing_stride
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n)
        Return:
            d_twiddle: (nstack, log n, n / 2)
            d_input: (batch_size, nstack, n)
        """
        twiddle_cos, twiddle_sin, output = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        d_coefficients, d_input = butterfly_ortho_multiply_untied_backward(twiddle_cos, twiddle_sin, output, grad, increasing_stride)
        return d_coefficients, d_input, None


def butterfly_ortho_mult_untied(twiddle, input, increasing_stride):
    n = input.shape[2]
    if input.dim() == 3 and n <= 1024 and input.is_cuda:
        return ButterflyOrthoMultUntied.apply(twiddle, input, increasing_stride)
    else:
        c, s = torch.cos(twiddle), torch.sin(twiddle)
        twiddle = torch.stack((torch.stack((c, -s), dim=-1),
                               torch.stack((s, c), dim=-1)), dim=-2)
        return butterfly_mult_untied(twiddle, input, increasing_stride, True, False)


def butterfly_ortho_mult_untied_torch(twiddle, input, increasing_stride):
    c, s = torch.cos(twiddle), torch.sin(twiddle)
    twiddle = torch.stack((torch.stack((c, -s), dim=-1),
                           torch.stack((s, c), dim=-1)), dim=-2)
    return butterfly_mult_untied_torch(twiddle, input, increasing_stride)


class BbtMultUntied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, fast=True):
        """
        Parameters:
            twiddle: (nstack, nblocks * 2 * log n, n / 2, 2, 2)
            input: (batch_size, nstack, n)
        Returns:
            output: (batch_size, nstack, n)
        """
        if not fast:
            output = bbt_multiply_untied(twiddle, input)
        else:
            output = butterfly_bbs_multiply_untied_forward_fast(twiddle, input)
        ctx.save_for_backward(twiddle, input)
        ctx._fast = fast
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n)
        Return:
            d_twiddle: (nstack, nblocks * 2 * log n, n / 2, 2, 2)
            d_input: (batch_size, nstack, n)
        """
        twiddle, input = ctx.saved_tensors
        fast = ctx._fast
        if not fast:
            d_coefficients, d_input = bbt_multiply_untied_forward_backward(twiddle, input, grad)
        else:
            d_coefficients, d_input = butterfly_bbs_multiply_untied_forward_backward_fast(twiddle, input, grad)
        return d_coefficients, d_input, None


def bbt_mult_untied(twiddle, input, fast=True):
    n = input.shape[2]
    m = int(math.log2(n))
    nblocks = twiddle.shape[1] // (2 * m)
    assert nblocks * 2 * m == twiddle.shape[1], 'twiddle must have shape (nstack, nblocks * 2 * log n, n / 2, 2, 2)'
    if n <= 1024 and input.is_cuda and nblocks <= 14:  # CUDA only supports nblocks <= 14
        return BbtMultUntied.apply(twiddle, input, fast)
    else:
        output = input
        reverse_idx = torch.arange(m - 1, -1, -1, device=twiddle.device)
        for t in twiddle.chunk(nblocks, dim=1):
            # output = butterfly_mult_untied(t[:, :m].flip(1), output, False)
            # flip is crazy slow, advanced indexing is slightly faster
            output = butterfly_mult_untied(t[:, reverse_idx], output, False, True, False)
            output = butterfly_mult_untied(t[:, m:], output, True, True, False)
        return output


def bbt_mult_untied_torch(twiddle, input):
    n = input.shape[2]
    m = int(math.log2(n))
    nblocks = twiddle.shape[1] // (2 * m)
    assert nblocks * 2 * m == twiddle.shape[1], 'twiddle must have shape (nstack, nblocks * 2 * log n, n / 2, 2, 2)'
    output = input
    for t in twiddle.chunk(nblocks, dim=1):
        output = butterfly_mult_untied_torch(t[:, :m].flip(1), output, False)
        output = butterfly_mult_untied_torch(t[:, m:], output, True)
    return output


def bbt_ortho_mult_tied(twiddle, input):
    n = input.shape[2]
    m = int(math.log2(n))
    nblocks = twiddle.shape[1] // 2
    assert nblocks * 2 == twiddle.shape[1], 'twiddle must have shape (nstack, nblocks * 2, n - 1)'
    output = input
    for t in twiddle.chunk(nblocks, dim=1):
        output = butterfly_ortho_mult_tied(t[:, 0], output, False)
        output = butterfly_ortho_mult_tied(t[:, 1], output, True)
    return output


class BbtOrthoMultUntied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        """
        Parameters:
            twiddle: (nstack, nblocks * 2 * log n, n / 2)
            input: (batch_size, nstack, n)
        Returns:
            output: (batch_size, nstack, n)
        """
        twiddle_cos, twiddle_sin = torch.cos(twiddle), torch.sin(twiddle)
        output = bbt_ortho_multiply_untied(twiddle_cos, twiddle_sin, input)
        ctx.save_for_backward(twiddle_cos, twiddle_sin, output)
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n)
        Return:
            d_twiddle: (nstack, nblocks * 2 * log n, n / 2)
            d_input: (batch_size, nstack, n)
        """
        twiddle_cos, twiddle_sin, output = ctx.saved_tensors
        d_coefficients, d_input = bbt_ortho_multiply_untied_backward(twiddle_cos, twiddle_sin, output, grad)
        return d_coefficients, d_input


def bbt_ortho_mult_untied(twiddle, input):
    n = input.shape[2]
    m = int(math.log2(n))
    nblocks = twiddle.shape[1] // (2 * m)
    if n <= 1024 and input.is_cuda:
        return BbtOrthoMultUntied.apply(twiddle, input)
    else:
        c, s = torch.cos(twiddle), torch.sin(twiddle)
        twiddle = torch.stack((torch.stack((c, -s), dim=-1),
                               torch.stack((s, c), dim=-1)), dim=-2)
        return bbt_mult_untied(twiddle, input)


def bbt_ortho_mult_untied_torch(twiddle, input):
    n = input.shape[2]
    c, s = torch.cos(twiddle), torch.sin(twiddle)
    twiddle = torch.stack((torch.stack((c, -s), dim=-1),
                            torch.stack((s, c), dim=-1)), dim=-2)
    return bbt_mult_untied_torch(twiddle, input)


class ODOMultUntied(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, diag, input):
        """
        Parameters:
            twiddle: (nstack, nblocks * 2 * log n, n / 2)
            diag: (nstack, nblocks, n)
            input: (batch_size, nstack, n)
        Returns:
            output: (batch_size, nstack, n)
        """
        twiddle_cos, twiddle_sin = torch.cos(twiddle), torch.sin(twiddle)
        output = fmf.butterfly_odo_multiply_untied_forward_fast(twiddle_cos, twiddle_sin, diag, input)
        # ctx.save_for_backward(twiddle_cos, twiddle_sin, diag, output)
        ctx.save_for_backward(twiddle_cos, twiddle_sin, diag, input)
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n)
        Return:
            d_twiddle: (nstack, nblocks * 2 * log n, n / 2)
            d_diag: (nstack, nblocks, n)
            d_input: (batch_size, nstack, n)
        """
        # twiddle_cos, twiddle_sin, diag, output = ctx.saved_tensors
        # d_coefficients, d_diag, d_input = butterfly_odo_multiply_untied_backward_fast(twiddle_cos, twiddle_sin, diag, output, grad)
        twiddle_cos, twiddle_sin, diag, input = ctx.saved_tensors
        d_coefficients, d_diag, d_input = fmf.butterfly_odo_multiply_untied_forward_backward_fast(twiddle_cos, twiddle_sin, diag, input, grad)
        if input.dtype == torch.float16:
            d_input = d_input.half()
        return d_coefficients, d_diag, d_input

odo_mult_untied = ODOMultUntied.apply


def twiddle_svd2regular(twiddle):
    """Convert SVD parameterization of twiddle to regular parameterization
    """
    cos_phi, sin_phi = torch.cos(twiddle[..., 0, 1]), torch.sin(twiddle[..., 0, 1])
    cos_theta, sin_theta = torch.cos(twiddle[..., 0, 0]), torch.sin(twiddle[..., 0, 0])
    sigmas = twiddle[..., 1, :]
    twiddle_phi = torch.stack((torch.stack((cos_phi, -sin_phi), dim=-1),
                               torch.stack((sin_phi, cos_phi), dim=-1)), dim=-2)
    twiddle_theta = torch.stack((torch.stack((cos_theta, -sin_theta), dim=-1),
                                 torch.stack((sin_theta, cos_theta), dim=-1)), dim=-2)
    return twiddle_theta @ (sigmas.unsqueeze(-1) * twiddle_phi)


def butterfly_mult_untied_svd_torch(twiddle, input, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, log n, n / 2, 2, 2)
        input: (batch_size, nstack, n) if real
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 1).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, nstack, n)
    """
    return butterfly_mult_untied_torch(twiddle_svd2regular(twiddle), input, increasing_stride, return_intermediates)


class ButterflyMultUntiedSvd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, log n, n / 2, 2, 2)
            input: (batch_size, nstack, n)
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, nstack, n)
        """
        # output_and_intermediate = butterfly_multiply_untied_svd(twiddle, input, increasing_stride)
        # ctx.save_for_backward(twiddle, output_and_intermediate)
        output = butterfly_multiply_untied_svd(twiddle, input, increasing_stride, False)
        ctx.save_for_backward(twiddle, input)
        ctx._increasing_stride = increasing_stride
        # return output_and_intermediate[-1]
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, nstack, n)
            twiddle: (nstack, log n, n / 2, 2, 2)
            output + intermediate values for backward: (log n + 1, batch_size, nstack, n)
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2)
            d_input: (batch_size, nstack, n)
        """
        # twiddle, output_and_intermediate = ctx.saved_tensors
        twiddle, input = ctx.saved_tensors
        increasing_stride = ctx._increasing_stride
        n = input.shape[2]
        if input.dim() == 3 and n <= 1024 and input.is_cuda:
            d_coefficients, d_input = butterfly_multiply_untied_svd_forward_backward(twiddle, input, grad, increasing_stride)
        else:
            output_and_intermediate = butterfly_multiply_untied_svd(twiddle, input, increasing_stride, True)
            d_coefficients, d_input = butterfly_multiply_untied_svd_backward(grad, twiddle, output_and_intermediate, increasing_stride)
        return d_coefficients, d_input, None  # Autograd requires 3 gradients

butterfly_mult_untied_svd = ButterflyMultUntiedSvd.apply if use_extension else butterfly_mult_untied_svd_torch


class ButterflyMultInplace(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input, increasing_stride=True):
        """Experimental in-place implementation that does not store intermediate results.
        Instead, the intermediate results are computed from the output during the backward pass.
        Parameters:
            twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
            input: (batch_size, n) if real or (batch_size, n, 2) if complex
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        Returns:
            output: (batch_size, n) if real or (batch_size, n, 2) if complex
        """
        assert increasing_stride, 'Decreasing stride not implemented'
        output = butterfly_multiply_inplace(twiddle, input)
        ctx.save_for_backward(twiddle, output)
        return output

    @staticmethod
    def backward(ctx, grad):
        twiddle, output = ctx.saved_tensors
        d_coefficients, d_input = butterfly_multiply_inplace_backward(grad, twiddle, output)
        return d_coefficients, d_input

butterfly_mult_inplace = ButterflyMultInplace.apply


def butterfly_mult_conv2d_torch(twiddle, input, kernel_size, padding, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, log n, n/2, 2, 2) where n = c_in
        input: (b_in, c_in, h_in, w_in)
        kernel_size: int, size of convolution kernel, currently only supports square kernels
        padding: amount of zero-padding around border of input
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed
    Returns:
        output: (b_in * h_out * w_out, nstack, c_in)
    """
    # unfold input to form the patches
    b_in, c_in, h_in, w_in = input.shape
    c_out = twiddle.size(0) // (kernel_size * kernel_size) * c_in
    assert c_in == 1 << int(math.log2(c_in)), "currently requires c_in to be a power of 2"
    assert c_out == 1 << int(math.log2(c_out)), "currently requires c_out to be a power of 2"
    h_out = h_in + 2 * padding - (kernel_size - 1)
    w_out = w_in + 2 * padding - (kernel_size - 1)
    matrix_batch = kernel_size * kernel_size
    c_out_ratio = c_out // c_in
    assert c_out_ratio >= 1, "only tested for c_out >= c_in"
    input_patches = F.unfold(input, kernel_size=kernel_size, dilation=1, padding=padding, stride=1).view(
        b_in, c_in, kernel_size * kernel_size, h_out * w_out)
    input_reshape = input_patches.permute(0, 3, 2, 1).reshape(b_in * h_out * w_out, matrix_batch, c_in)
    input_reshape = input_reshape.unsqueeze(2).expand(b_in * h_out * w_out, matrix_batch, c_out_ratio, c_in)
    input_reshape = input_reshape.reshape(b_in * h_out * w_out, matrix_batch * c_out_ratio, c_in)
    # perform matrix multiply
    return butterfly_mult_untied_torch(twiddle, input_reshape, increasing_stride, return_intermediates)


class ButterflyMultConv2d(torch.autograd.Function):
    # For fused unfolding, n <= 1024, CUDA only, real only
    # Assumes dilation=1, stride=1, and square kernels

    @staticmethod
    def forward(ctx, twiddle, input, kernel_size, padding, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, log n, n/2, 2, 2) where n = c_in
            input: (b_in, c_in, h_in, w_in)
            kernel_size: int, size of convolution kernel, currently only supports square kernels
            padding: amount of zero-padding around border of input
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                    decreasing stride (e.g., n/2, n/4, ..., 1).
                    Note that this only changes the order of multiplication, not how twiddle is stored.
                    In other words, twiddle[@log_stride] always stores the twiddle for @stride.
            return_intermediates: whether to return all the intermediate values computed
        Returns:
            output: (b_in * h_out * w_out, nstack, c_in)
        """
        output = butterfly_conv2d(twiddle, input, kernel_size,
                                  padding, increasing_stride, False)
        ctx.save_for_backward(twiddle, input)
        ctx._kernel_size = kernel_size
        ctx._padding = padding
        ctx._increasing_stride = increasing_stride
        ctx._input_size = input.size()
        ctx._b_in= input.size(0)
        ctx._c_in = input.size(1)
        ctx._h_in = input.size(2)
        ctx._w_in = input.size(3)
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (b_in * h_out * w_out, cin/cout * nstack, c_out)
            twiddle: (nstack, log n, n / 2, 2, 2) where n = c_in
            output + intermediate values for backward: (log n + 1, b_in * h_out * w_out,
                                                cin/cout * nstack, c_out)
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2)
            d_input: (b_in, c_in, h_in, w_in)
        """
        twiddle, input = ctx.saved_tensors
        # Save intermediates for backward pass
        # output_and_intermediate = butterfly_conv2d(twiddle, input,
            # ctx._kernel_size, ctx._padding, ctx._increasing_stride, True)
        # d_coefficients, d_input = butterfly_conv2d_backward(grad, twiddle,
            # output_and_intermediate, ctx._kernel_size, ctx._padding,
            # ctx._increasing_stride, ctx._b_in, ctx._c_in, ctx._h_in, ctx._w_in)
        d_coefficients, d_input = butterfly_conv2d_forward_backward(twiddle,
            input, grad, ctx._kernel_size, ctx._padding, ctx._increasing_stride)
        return d_coefficients, d_input, None, None, None
        # Autograd requires 5 gradients

butterfly_mult_conv2d = ButterflyMultConv2d.apply if use_extension else butterfly_mult_conv2d_torch


class BbtMultConv2d(torch.autograd.Function):
    # For fused unfolding, n <= 1024, CUDA only, real only
    # Assumes dilation=1, stride=1, and square kernels

    @staticmethod
    def forward(ctx, twiddle, input, kernel_size, padding):
        """
        Parameters:
            twiddle: (nstack, nblocks * 2 * log n, n/2, 2, 2) where n = c_in
            input: (b_in, c_in, h_in, w_in)
            kernel_size: int, size of convolution kernel, currently only supports square kernels
            padding: amount of zero-padding around border of input
        Returns:
            output: (b_in * h_out * w_out, nstack, c_in)
        """
        output = bbt_conv2d(twiddle, input, kernel_size, padding)
        ctx.save_for_backward(twiddle, input)
        ctx._kernel_size = kernel_size
        ctx._padding = padding
        ctx._input_size = input.size()
        ctx._b_in= input.size(0)
        ctx._c_in = input.size(1)
        ctx._h_in = input.size(2)
        ctx._w_in = input.size(3)
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (b_in * h_out * w_out, cin/cout * nstack, c_out)
            twiddle: (nstack, nblocks * 2 * log n, n / 2, 2, 2) where n = c_in
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2)
            d_input: (b_in, c_in, h_in, w_in)
        """
        twiddle, input = ctx.saved_tensors
        d_coefficients, d_input = bbt_conv2d_forward_backward(twiddle, input, grad, ctx._kernel_size, ctx._padding)
        return d_coefficients, d_input, None, None
        # Autograd requires 4 gradients

def bbt_mult_conv2d(twiddle, input, kernel_size, padding):
    n = input.shape[1]
    m = int(math.log2(n))
    nblocks = twiddle.shape[1] // (2 * m)
    assert nblocks * 2 * m == twiddle.shape[1], 'twiddle must have shape (nstack, nblocks * 2 * log n, n / 2, 2, 2)'
    if n <= 1024 and input.is_cuda and nblocks <= 14:  # CUDA only supports nblocks <= 14
        return BbtMultConv2d.apply(twiddle, input, kernel_size, padding)
    else:
        output = input
        reverse_idx = torch.arange(m - 1, -1, -1, device=twiddle.device)
        first = True
        for t in twiddle.chunk(nblocks, dim=1):
            # output = butterfly_mult_conv2d(t[:, :m].flip(1), output, False)
            # flip is crazy slow, advanced indexing is slightly faster
            if first:
                output = butterfly_mult_conv2d(t[:, reverse_idx], output, kernel_size, padding, False)
                first = False
            else:
                output = butterfly_mult_untied(t[:, reverse_idx], output, False, True, False)
            output = butterfly_mult_untied(t[:, m:], output, True, True, False)
        return output


def bbt_mult_conv2d_torch(twiddle, input, kernel_size, padding):
    n = input.shape[1]
    m = int(math.log2(n))
    nblocks = twiddle.shape[1] // (2 * m)
    assert nblocks * 2 * m == twiddle.shape[1], 'twiddle must have shape (nstack, nblocks * 2 * log n, n / 2, 2, 2)'
    output = input
    first = True
    for t in twiddle.chunk(nblocks, dim=1):
        if first:
            output = butterfly_mult_conv2d_torch(t[:, :m].flip(1), output, kernel_size, padding, False)
            first = False
        else:
            output = butterfly_mult_untied_torch(t[:, :m].flip(1), output, False)
        output = butterfly_mult_untied_torch(t[:, m:], output, True)
    return output



def butterfly_mult_conv2d_svd_torch(twiddle, input, kernel_size, padding, increasing_stride=True, return_intermediates=False):
    """
    Parameters:
        twiddle: (nstack, log n, n/2, 2, 2) where n = c_in
        input: (b_in, c_in, h_in, w_in)
        kernel_size: int, size of convolution kernel, currently only supports square kernels
        padding: amount of zero-padding around border of input
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                decreasing stride (e.g., n/2, n/4, ..., 1).
                Note that this only changes the order of multiplication, not how twiddle is stored.
                In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed
    Returns:
        output: (b_in * h_out * w_out, nstack, c_in)
    """
    return butterfly_mult_conv2d_torch(twiddle_svd2regular(twiddle), input, kernel_size, padding, increasing_stride, return_intermediates)


class ButterflyMultConv2dSvd(torch.autograd.Function):
    # For fused unfolding, n <= 1024, CUDA only, real only
    # Assumes dilation=1, stride=1, and square kernels

    @staticmethod
    def forward(ctx, twiddle, input, kernel_size, padding, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, log n, n/2, 2, 2) where n = c_in
            input: (b_in, c_in, h_in, w_in)
            kernel_size: int, size of convolution kernel, currently only supports square kernels
            padding: amount of zero-padding around border of input
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                    decreasing stride (e.g., n/2, n/4, ..., 1).
                    Note that this only changes the order of multiplication, not how twiddle is stored.
                    In other words, twiddle[@log_stride] always stores the twiddle for @stride.
            return_intermediates: whether to return all the intermediate values computed
        Returns:
            output: (b_in * h_out * w_out, nstack, c_in)
        """
        output = butterfly_conv2d_svd(twiddle, input, kernel_size,
                                  padding, increasing_stride, False)
        ctx.save_for_backward(twiddle, input)
        ctx._kernel_size = kernel_size
        ctx._padding = padding
        ctx._increasing_stride = increasing_stride
        ctx._input_size = input.size()
        ctx._b_in= input.size(0)
        ctx._c_in = input.size(1)
        ctx._h_in = input.size(2)
        ctx._w_in = input.size(3)
        return output

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (b_in * h_out * w_out, cin/cout * nstack, c_out)
            twiddle: (nstack, log n, n / 2, 2, 2) where n = c_in
            output + intermediate values for backward: (log n + 1, b_in * h_out * w_out,
                                                cin/cout * nstack, c_out)
        Return:
            d_twiddle: (nstack, log n, n / 2, 2, 2)
            d_input: (b_in, c_in, h_in, w_in)
        """
        twiddle, input = ctx.saved_tensors
        d_coefficients, d_input = butterfly_conv2d_svd_forward_backward(twiddle,
            input, grad, ctx._kernel_size, ctx._padding, ctx._increasing_stride)
        return d_coefficients, d_input, None, None, None
        # Autograd requires 5 gradients

butterfly_mult_conv2d_svd = ButterflyMultConv2dSvd.apply if use_extension else butterfly_mult_conv2d_svd_torch


class ButterflyFactorMult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, twiddle, input):
        """Multiply by a single factor.
        Parameters:
            twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
            input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        Returns:
            output: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        """
        ctx.save_for_backward(twiddle, input)
        return butterfly_factor_multiply(twiddle, input)

    @staticmethod
    def backward(ctx, grad):
        """
        Parameters:
            grad: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        Returns:
            d_twiddle: (2, 2, n) if real or (2, 2, n, 2) if complex
            d_input: (batch_size, 2, n) if real or (batch_size, 2, n, 2) if complex
        """
        twiddle, input = ctx.saved_tensors
        d_twiddle, d_input = butterfly_factor_multiply_backward(grad, twiddle, input)
        return d_twiddle, d_input

butterfly_factor_mult = ButterflyFactorMult.apply


def butterfly_mult_factors(twiddle, input, increasing_stride=True, return_intermediates=False):
    """Implementation that have separate kernels for each factor, for debugging.
    Parameters:
        twiddle: (n - 1, 2, 2) if real or (n - 1, 2, 2, 2) if complex
        input: (batch_size, n) if real or (batch_size, n, 2) if complex
        increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
            decreasing stride (e.g., n/2, n/4, ..., 1).
            Note that this only changes the order of multiplication, not how twiddle is stored.
            In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        return_intermediates: whether to return all the intermediate values computed, for debugging
    Returns:
        output: (batch_size, n) if real or (batch_size, n, 2) if complex
    """
    batch_size, n = input.shape[:2]
    m = int(math.log2(n))
    assert n == 1 << m, "size must be a power of 2"
    assert twiddle.shape == (n - 1, 2, 2) if input.dim() == 2 else (n - 1, 2, 2, 2)
    output = input.contiguous()
    intermediates = [output]
    if input.dim() == 2:  # real
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[(stride - 1):(2 * stride - 1)].permute(1, 2, 0)  # shape (2, 2, stride)
            output_reshape = output.view(batch_size * n // (2 * stride), 2, stride)
            output = butterfly_factor_mult(t, output_reshape)
            intermediates.append(output)
        return output.view(batch_size, n) if not return_intermediates else torch.stack([intermediate.view(batch_size, n) for intermediate in intermediates])
    else:  # complex
        for log_stride in range(m) if increasing_stride else range(m)[::-1]:
            stride = 1 << log_stride
            t = twiddle[(stride - 1):(2 * stride - 1)].permute(1, 2, 0, 3)  # shape (2, 2, stride, 2)
            output_reshape = output.view(batch_size * n // (2 * stride), 2, stride, 2)
            output = butterfly_factor_mult(t, output_reshape)
            intermediates.append(output)
        return output.view(batch_size, n, 2) if not return_intermediates else torch.stack([intermediate.view(batch_size, n, 2) for intermediate in intermediates])
