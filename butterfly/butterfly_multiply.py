import math

import torch
from torch import nn
import torch.nn.functional as F

from .complex_utils import complex_mul

use_extension = True
try:
    from factor_multiply import butterfly_multiply_intermediate, butterfly_multiply_intermediate_backward
    from factor_multiply import butterfly_multiply_untied, butterfly_multiply_untied_backward
    from factor_multiply import butterfly_multiply_untied_svd, butterfly_multiply_untied_svd_backward
    from factor_multiply import butterfly_multiply_inplace, butterfly_multiply_inplace_backward
    from factor_multiply import butterfly_factor_multiply, butterfly_factor_multiply_backward
    from factor_multiply import butterfly_conv2d, butterfly_conv2d_backward
except:
    use_extension = False
    import warnings
    warnings.warn("C++/CUDA extension isn't installed. Will use butterfly multiply implemented in Pytorch, which is much slower.")


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
    assert twiddle.shape == (nstack, m, n // 2, 2, 2) if input.dim() == 2 else (nstack, m, n // 2, 2, 2, 2)
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
    def forward(ctx, twiddle, input, increasing_stride=True):
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
        # output_and_intermediate = butterfly_multiply_untied(twiddle, input, increasing_stride)
        # ctx.save_for_backward(twiddle, output_and_intermediate)
        output = butterfly_multiply_untied(twiddle, input, increasing_stride, False)
        ctx.save_for_backward(twiddle, input)
        ctx._increasing_stride = increasing_stride
        # return output_and_intermediate[-1]
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
        output_and_intermediate = butterfly_multiply_untied(twiddle, input, increasing_stride, True)
        # print("untied: ", output_and_intermediate.size())
        # print("untied: ", output_and_intermediate)
        d_coefficients, d_input = butterfly_multiply_untied_backward(grad, twiddle, output_and_intermediate, increasing_stride)
        return d_coefficients, d_input, None  # Autograd requires 3 gradients

butterfly_mult_untied = ButterflyMultUntied.apply if use_extension else butterfly_mult_untied_torch


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

    cos_phi, sin_phi = torch.cos(twiddle[..., 0, 1]), torch.sin(twiddle[..., 0, 1])
    cos_theta, sin_theta = torch.cos(twiddle[..., 0, 0]), torch.sin(twiddle[..., 0, 0])
    sigmas = twiddle[..., 1, :]
    twiddle_phi = torch.stack((torch.stack((cos_phi, -sin_phi), dim=-1),
                               torch.stack((sin_phi, cos_phi), dim=-1)), dim=-2)
    twiddle_theta = torch.stack((torch.stack((cos_theta, -sin_theta), dim=-1),
                                 torch.stack((sin_theta, cos_theta), dim=-1)), dim=-2)
    twiddle_prod = twiddle_theta @ (sigmas.unsqueeze(-1) * twiddle_phi)
    return butterfly_mult_untied(twiddle_prod, input, increasing_stride)


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
    b_in = input.size(0) 
    c_in = input.size(1)
    h_in = input.size(2)
    w_in = input.size(3)
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
    # For n <= 1024, CUDA only, real only

    @staticmethod
    def forward(ctx, twiddle, input, kernel_size, padding, increasing_stride=True):
        """
        Parameters:
            twiddle: (nstack, log n, n/2, 2, 2) if real or (nstack, log n, n/2, 2, 2, 2) if complex
            input: (b_in, c_in, h_in, w_in)
            kernel_size: int, size of convolution kernel, currently only supports square kernels 
            padding: amount of zero-padding around border of input  
            increasing_stride: whether to multiply with increasing stride (e.g. 1, 4, ..., n/2) or
                    decreasing stride (e.g., n/2, n/4, ..., 1).
                    Note that this only changes the order of multiplication, not how twiddle is stored.
                    In other words, twiddle[@log_stride] always stores the twiddle for @stride.
            return_intermediates: whether to return all the intermediate values computed

        Returns:
            output: (b_in * h_out * w_out, nstack, c_out)
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
        # h_out = ctx._h_in + 2 * padding - (kernel_size - 1)
        # w_out = ctx._w_in + 2 * padding - (kernel_size - 1)
        # # twiddle nstack = out_channels/in_channels * matrix batach 
        # c_out = twiddle.size(0) // (kernel_size*kernel_size) * ctx._c_in
        # output = output.view(ctx._b_in*w_out*h_out, kernel_size*kernel_size,c_out)
        # output = output.mean(dim=1)
        # output = output.view(ctx._b_in, h_out * w_out, c_out).transpose(
        #     1, 2).view(ctx._b_in, c_out, h_out, w_out)
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
        output_and_intermediate = butterfly_conv2d(twiddle, input, 
            ctx._kernel_size, ctx._padding, ctx._increasing_stride, True) 
               
        # TODO: simplify/reduce these args if possible 
        print(grad.shape, output_and_intermediate.shape)
        d_coefficients, d_input = butterfly_conv2d_backward(grad, twiddle, 
            output_and_intermediate, ctx._kernel_size, ctx._padding, 
            ctx._increasing_stride, ctx._b_in, ctx._c_in, ctx._h_in, ctx._w_in)
        print(d_coefficients.shape, d_input.shape)
        return d_coefficients, d_input, None, None, None 
        # Autograd requires 5 gradients

butterfly_mult_conv2d = ButterflyMultConv2d.apply

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
