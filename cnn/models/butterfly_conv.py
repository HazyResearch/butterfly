import torch
from torch import nn
import torch.nn.functional as F

from butterfly import Butterfly
from butterfly.butterfly import ButterflyBmm


class Butterfly1x1Conv(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    """

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        # TODO: Only doing real for now
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)


class ButterflyConv2d(ButterflyBmm):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.

    Parameters:
        in_channels: size of input
        out_channels: size of output
        kernel_size: int or (int, int)
        stride: int or (int, int)
        padding; int or (int, int)
        dilation: int or (int, int)
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        param: The parameterization of the 2x2 butterfly factors, either 'regular' or 'ortho' or 'svd'.
            'ortho' and 'svd' only support real, not complex.
        max_gain: (only for svd parameterization) controls the maximum and minimum singular values
            of the whole matrix (not of each factor).
            For example, max_gain=10.0 means that the singular values are in [0.1, 10.0].
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 tied_weight=True, increasing_stride=True, ortho_init=False, param='regular', max_gain=10.0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        super().__init__(in_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], bias, False,
                         tied_weight, increasing_stride, ortho_init, param, max_gain)

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        # TODO: Only doing real for now
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        input_reshape = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1], c)
        output = super().forward(input_reshape).mean(dim=1)
        return output.view(batch, h_out * w_out, self.out_channels).transpose(1, 2).view(batch, self.out_channels, h_out, w_out)


class ButterflyConv2dBBT(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.

    Parameters:
        in_channels: size of input
        out_channels: size of output
        kernel_size: int or (int, int)
        stride: int or (int, int)
        padding; int or (int, int)
        dilation: int or (int, int)
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        param: The parameterization of the 2x2 butterfly factors, either 'regular' or 'ortho' or 'svd'.
            'ortho' and 'svd' only support real, not complex.
        max_gain: (only for svd parameterization) controls the maximum and minimum singular values
            of the whole BB^T matrix (not of each factor).
            For example, max_gain=10.0 means that the singular values are in [0.1, 10.0].
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 tied_weight=True, ortho_init=False, param='regular', max_gain=10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.layers = nn.Sequential(
            ButterflyBmm(in_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=False, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 2)),
            ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], bias, False, tied_weight, increasing_stride=True, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 2))
            )

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        # TODO: Only doing real for now
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        input_reshape = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1], c)
        output = self.layers(input_reshape).mean(dim=1)
        return output.view(batch, h_out * w_out, self.out_channels).transpose(1, 2).view(batch, self.out_channels, h_out, w_out)


class ButterflyConv2dBBTBBT(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.

    Parameters:
        in_channels: size of input
        out_channels: size of output
        kernel_size: int or (int, int)
        stride: int or (int, int)
        padding; int or (int, int)
        dilation: int or (int, int)
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        tied_weight: whether the weights in the butterfly factors are tied.
            If True, will have 4N parameters, else will have 2 N log N parameters (not counting bias)
         increasing_stride: whether to multiply with increasing stride (e.g. 1, 2, ..., n/2) or
             decreasing stride (e.g., n/2, n/4, ..., 1).
             Note that this only changes the order of multiplication, not how twiddle is stored.
             In other words, twiddle[@log_stride] always stores the twiddle for @stride.
        ortho_init: whether the weight matrix should be initialized to be orthogonal/unitary.
        param: The parameterization of the 2x2 butterfly factors, either 'regular' or 'ortho' or 'svd'.
            'ortho' and 'svd' only support real, not complex.
        max_gain: (only for svd parameterization) controls the maximum and minimum singular values
            of the whole BB^T BB^T matrix (not of each factor).
            For example, max_gain=10.0 means that the singular values are in [0.1, 10.0].
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 tied_weight=True, ortho_init=False, param='regular', max_gain=10.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.layers = nn.Sequential(
            ButterflyBmm(in_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=False, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4)),
            ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=True, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4)),
            ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], False, False, tied_weight, increasing_stride=False, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4)),
            ButterflyBmm(out_channels, out_channels, self.kernel_size[0] * self.kernel_size[1], bias, False, tied_weight, increasing_stride=True, ortho_init=ortho_init, param=param, max_gain=max_gain ** (1 / 4))
            )

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        # TODO: Only doing real for now
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        input_reshape = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1], c)
        output = self.layers(input_reshape).mean(dim=1)
        return output.view(batch, h_out * w_out, self.out_channels).transpose(1, 2).view(batch, self.out_channels, h_out, w_out)
