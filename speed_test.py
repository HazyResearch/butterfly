import os
from timeit import default_timer as timer

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from butterfly import Block2x2DiagProduct, BlockPermProduct
from inference import Block2x2DiagProduct_to_ABCDs, BP_mul_cy_inplace

# We limit to 1 thread for reliable speed test
os.environ['MKL_NUM_THREADS'] = '1'

exps = np.arange(6, 14)
sizes = 1 << exps

ntrials = 10

dense_times = np.zeros(exps.size)
fft_times = np.zeros(exps.size)
bp_times = np.zeros(exps.size)
for idx_n, n in enumerate(sizes):
    print(n)
    x = np.random.random(n).astype(np.float32)
    B = Block2x2DiagProduct(n)
    P = BlockPermProduct(n)
    B_matrix = B(torch.eye(int(n))).t().contiguous()
    B_matrix_np = B_matrix.detach().numpy()

    ABCDs = Block2x2DiagProduct_to_ABCDs(B)
    perm = P.argmax().detach().numpy().astype(int)

    # Dense multiply
    start = timer()
    [B_matrix_np @ x for _ in range(ntrials)]
    end = timer()
    dense_times[idx_n] = (end-start)

    # FFT
    start = timer()
    [np.fft.fft(x) for _ in range(ntrials)]
    end = timer()
    fft_times[idx_n] = (end-start)

    # BP
    start = timer()
    [BP_mul_cy_inplace(ABCDs, perm, x) for _ in range(ntrials)]
    end = timer()
    bp_times[idx_n] = (end-start)

print(dense_times)
print(fft_times)
print(bp_times)

print(bp_times / fft_times)

plt.figure()
plt.semilogy(sizes, dense_times / fft_times, label='FFT')
plt.semilogy(sizes, dense_times / bp_times, label='BP')
plt.xscale('log', basex=2)
plt.xlabel("Dimension")
plt.ylabel("Speedup over GEMV")
plt.legend()
# plt.show()
plt.savefig('speed.pdf')

