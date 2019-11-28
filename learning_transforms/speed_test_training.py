import os
import time

import numpy as np
import torch
from torch import nn

from butterfly_factor import butterfly_factor_mult_intermediate
# from butterfly import Block2x2DiagProduct

# from test_factor_multiply import twiddle_list_concat

exps = np.arange(6, 14)
sizes = 1 << exps

batch_size = 256

ntrials = [100000, 100000, 10000, 10000, 10000, 10000, 10000, 10000]

dense_times = np.zeros(exps.size)
fft_times = np.zeros(exps.size)
butterfly_times = np.zeros(exps.size)
for idx_n, (n, ntrial) in enumerate(zip(sizes, ntrials)):
    print(n)
    # B = Block2x2DiagProduct(n).to('cuda')
    L = torch.nn.Linear(n, n, bias=False).to('cuda')
    x = torch.randn(batch_size, n, requires_grad=True).to('cuda')
    grad = torch.randn_like(x)
    # twiddle = twiddle_list_concat(B)

    # Dense multiply
    output = L(x)  # Do it once to initialize cuBlas handle and such
    torch.autograd.grad(output, (L.weight, x), grad)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ntrial):
        output = L(x)
        torch.autograd.grad(output, (L.weight, x), grad)
    torch.cuda.synchronize()
    end = time.perf_counter()
    dense_times[idx_n] = (end - start) / ntrial

    # FFT
    output = torch.rfft(x, 1)  # Do it once to initialize cuBlas handle and such
    grad_fft = torch.randn_like(output)
    torch.autograd.grad(output, x, grad_fft)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ntrial):
        output = torch.rfft(x, 1)
        torch.autograd.grad(output, x, grad_fft)
    torch.cuda.synchronize()
    end = time.perf_counter()
    fft_times[idx_n] = (end - start) / ntrial

    # Butterfly
    output = butterfly_factor_mult_intermediate(twiddle, x)
    torch.autograd.grad(output, (twiddle, x), grad)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ntrial):
        output = butterfly_factor_mult_intermediate(twiddle, x)
        torch.autograd.grad(output, (twiddle, x), grad)
    torch.cuda.synchronize()
    end = time.perf_counter()
    butterfly_times[idx_n] = (end-start) / ntrial

print(dense_times)
print(fft_times)
print(butterfly_times)

print(dense_times / butterfly_times)
print(dense_times / fft_times)


data = {
    'sizes': sizes,
    'speedup_fft': dense_times / fft_times,
    'speedup_butterfly': dense_times / butterfly_times,
}

import pickle
with open('speed_training_data.pkl', 'wb') as f:
    pickle.dump(data, f)
