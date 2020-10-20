import math

import torch
import torch.nn.functional as F

from torch_butterfly.multiply import butterfly_multiply

from benchmark_utils import benchmark, benchmark_fw_bw


batch_size = 2048
n = 512
log_n = int(math.log2(n))
assert n == 1 << log_n

input_size = n - 7
output_size = n - 5
input = torch.randn(batch_size, 1, input_size, device='cuda', requires_grad=True)
twiddle = torch.randn(1, 1, log_n, n // 2, 2, 2, device='cuda', requires_grad=True) / math.sqrt(2)


def fn(twiddle, input):
    return butterfly_multiply(twiddle, input, True, output_size)

def fn_padded(twiddle, input):
    input_padded = F.pad(input, (0, n - input_size))
    return butterfly_multiply(twiddle, input_padded, True)[:, :, :output_size]

print(benchmark_fw_bw(fn, (twiddle, input), 10000))
print(benchmark_fw_bw(fn_padded, (twiddle, input), 10000))
