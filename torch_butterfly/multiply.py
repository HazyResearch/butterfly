import math
from typing import Tuple, Optional

import torch
from torch.nn import functional as F


@torch.jit.script
def butterfly_multiply_fw(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                          output_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_butterfly.butterfly_multiply_fw(twiddle, input, increasing_stride,
                                                           output_size)


@torch.jit.script
def butterfly_multiply_bw(twiddle: torch.Tensor, input: torch.Tensor, grad: torch.Tensor,
                          increasing_stride: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_butterfly.butterfly_multiply_bw(twiddle, input, grad, increasing_stride)


@torch.jit.script
def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                       output_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_butterfly.butterfly_multiply(twiddle, input, increasing_stride,
                                                        output_size)


def butterfly_multiply_torch(twiddle, input, increasing_stride=True, output_size=None):
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    # Pad or trim input to size n
    input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
    output_size = n if output_size is None else output_size
    assert output_size <= n
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_increasing_stride else log_n - 1 - idx
            stride = 1 << log_stride
            # shape (nstacks, n // (2 * stride), 2, 2, stride)
            t = twiddle[:, block, idx].view(
                nstacks, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(
                batch_size, nstacks, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, n)[:, :, :output_size]
