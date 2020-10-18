import math

import torch

from torch_butterfly.complex_utils import complex_mul


def butterfly_multiply_base4_torch(twiddle4, twiddle2, input, increasing_stride=True):
    batch_size, nstacks, n = input.shape
    nblocks = twiddle4.shape[1]
    log_n = int(math.log2(n))
    assert n == 1 << log_n, "size must be a power of 2"
    if log_n // 2 > 0:
        assert twiddle4.shape == (nstacks, nblocks, log_n // 2, n // 4, 4, 4)
    if log_n % 2 == 1:
        assert twiddle2.shape == (nstacks, nblocks, 1, n // 2, 2, 2)
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n // 2):
            log2_stride = 2 * idx if cur_increasing_stride else log_n - 2 - 2 * idx
            stride = 1 << (log2_stride)
            # shape (nstacks, n // (4 * stride), 4, 4, stride)
            t = twiddle4[:, block, idx].view(
                nstacks, n // (4 * stride), stride, 4, 4).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(batch_size, nstacks, n // (4 * stride), 1, 4, stride)
            output = complex_mul(t, output_reshape).sum(dim=4)
        if log_n % 2 == 1:
            log2_stride = log_n - 1 if cur_increasing_stride else 0
            stride = 1 << log2_stride
            # shape (nstacks, n // (2 * stride), 2, 2, stride)
            t = twiddle2[:, block, 0].view(
                nstacks, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(batch_size, nstacks, n // (2 * stride), 1, 2, stride)
            output = complex_mul(t, output_reshape).sum(dim=4)
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, n)


def twiddle_base2_to_base4(twiddle, increasing_stride=True):
    nstacks, nblocks, log_n = twiddle.shape[:3]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    twiddle2 = (twiddle[:, :, -1:] if log_n % 2 == 1
                else torch.empty(nstacks, nblocks, 0, n // 2, 2, 2,
                                 dtype=twiddle.dtype, device=twiddle.device))
    twiddle4 = torch.empty(nstacks, nblocks, log_n // 2, n // 4, 4, 4,
                           dtype=twiddle.dtype, device=twiddle.device)
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n // 2):
            log2_stride = 2 * idx if cur_increasing_stride else log_n - 2 - 2 * idx
            stride = 1 << (log2_stride)
            # Warning: All this dimension manipulation (transpose and unsqueeze) is super tricky.
            # I'm not even sure why it works (I figured it out with trial and error).
            even = twiddle[:, block, 2 * idx].view(
                nstacks, n // (4 * stride), 2, stride, 2, 2).transpose(-3, -4)
            odd = twiddle[:, block, 2 * idx + 1].view(
                nstacks, n // (4 * stride), 2, stride, 2, 2).transpose(-3, -4)
            if cur_increasing_stride:
                prod = complex_mul(odd.transpose(-2, -3).unsqueeze(-1),
                                   even.transpose(-2, -3).unsqueeze(-4))
            else:
                prod = complex_mul(odd.unsqueeze(-2), even.permute(0, 1, 2, 4, 5, 3).unsqueeze(-3))
            prod = prod.reshape(nstacks, n // 4, 4, 4)
            twiddle4[:, block, idx].copy_(prod)
        cur_increasing_stride = not cur_increasing_stride
    return twiddle4, twiddle2
