import math

import torch
from torch import nn
import torch.nn.functional as F


class LowRankConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, rank=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.rank = rank
        self.G = nn.Parameter(torch.Tensor(self.kernel_size[0] * self.kernel_size[1], self.rank, self.in_channels))
        self.H = nn.Parameter(torch.Tensor(self.kernel_size[0] * self.kernel_size[1], self.out_channels, self.rank))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Identical initialization to torch.nn.Linear
        fan_in, fan_out = self.in_channels, self.out_channels
        nn.init.uniform_(self.G, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        nn.init.uniform_(self.H, -1 / math.sqrt(self.rank), 1 / math.sqrt(self.rank))
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch, c, h, w = x.shape
        c_out = self.out_channels
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (h + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        # unfold x into patches and call batch matrix multiply
        input_patches = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride).view(
            batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        x = input_patches.permute(2, 0, 3, 1).reshape(self.kernel_size[0] * self.kernel_size[1], batch * h_out * w_out, c)
        output = x @ self.G.transpose(1, 2)
        output = output @ self.H.transpose(1, 2)
        # combine matrix batches
        output = output.mean(dim=0).view(batch, h_out * w_out, c_out).transpose(1, 2).view(batch, c_out, h_out, w_out)
        if self.bias is not None:
            output = output + self.bias.unsqueeze(-1).unsqueeze(-1)
        return output

