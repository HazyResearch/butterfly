import math
from typing import Tuple

import torch

from .complex_utils import complex_mul


@torch.jit.script
def butterfly_multiply_fw(twiddle: torch.Tensor, input: torch.Tensor,
                          increasing_stride: bool) -> torch.Tensor:
    return torch.ops.torch_butterfly.butterfly_multiply_fw(twiddle, input, increasing_stride)


@torch.jit.script
def butterfly_multiply_bw(twiddle: torch.Tensor, input: torch.Tensor,
                          grad: torch.Tensor, increasing_stride: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_butterfly.butterfly_multiply_bw(twiddle, input, grad, increasing_stride)


@torch.jit.script
def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor,
                       increasing_stride: bool) -> torch.Tensor:
    return torch.ops.torch_butterfly.butterfly_multiply(twiddle, input, increasing_stride)


def butterfly_multiply_torch(twiddle, input, increasing_stride=True):
    batch_size, nstacks, n = input.shape
    nblocks = twiddle.shape[1]
    log_n = int(math.log2(n))
    assert n == 1 << log_n, "size must be a power of 2"
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
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
            output = complex_mul(t, output_reshape).sum(dim=4)
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, n)
