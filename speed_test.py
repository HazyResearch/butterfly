import os
from timeit import default_timer as timer

import numpy as np
from scipy.fftpack import fft, dct, dst
import torch
from torch import nn

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from butterfly import Block2x2DiagProduct, BlockPermProduct
from inference import Block2x2DiagProduct_to_ABCDs, BP_mul_cy_inplace

from test_factor_multiply import twiddle_list_concat
from factor_multiply import butterfly_factor_multiply_inplace

# We limit to 1 thread for reliable speed test
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

exps = np.arange(6, 14)
sizes = 1 << exps

ntrials = [100000, 100000, 1000, 100, 100, 10, 10, 10]

dense_times = np.zeros(exps.size)
fft_times = np.zeros(exps.size)
scipyfft_times = np.zeros(exps.size)
dct_times = np.zeros(exps.size)
dst_times = np.zeros(exps.size)
bp_times = np.zeros(exps.size)
bp_all_times = np.zeros(exps.size)
for idx_n, (n, ntrial) in enumerate(zip(sizes, ntrials)):
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
    [B_matrix_np @ x for _ in range(ntrial)]
    end = timer()
    dense_times[idx_n] = (end-start) / ntrial

    # FFT
    start = timer()
    [np.fft.fft(x) for _ in range(ntrial)]
    end = timer()
    fft_times[idx_n] = (end-start) / ntrial

    # Scipy's FFT, it's about 1.5x slower than Numpy's FFT
    start = timer()
    [fft(x) for _ in range(ntrial)]
    end = timer()
    scipyfft_times[idx_n] = (end-start) / ntrial

    # DCT
    start = timer()
    [dct(x) for _ in range(ntrial)]
    end = timer()
    dct_times[idx_n] = (end-start) / ntrial

    # DST
    start = timer()
    [dst(x) for _ in range(ntrial)]
    end = timer()
    dst_times[idx_n] = (end-start) / ntrial

    # BP
    start = timer()
    [BP_mul_cy_inplace(ABCDs, perm, x) for _ in range(ntrial)]
    end = timer()
    bp_times[idx_n] = (end-start) / ntrial

    # BP_inplace_all
    twiddles = twiddle_list_concat(B)
    x_torch = torch.tensor(x).unsqueeze(0)
    # perm_torch = torch.tensor(perm)
    start = timer()
    [butterfly_factor_multiply_inplace(twiddles, x_torch) for _ in range(ntrial)]
    end = timer()
    bp_all_times[idx_n] = (end-start) / ntrial

print(dense_times)
print(fft_times)
print(scipyfft_times)
print(dct_times)
print(dst_times)
print(bp_times)
print(bp_all_times)

# print(bp_times / fft_times)
# print(bp_times / dct_times)
# print(bp_times / dst_times)
# print(bp_times / bp_all_times)

print(bp_all_times / fft_times)
print(bp_all_times / dct_times)
print(bp_all_times / dst_times)
print(bp_all_times / bp_times)


# plt.figure()
# plt.semilogy(sizes, dense_times / fft_times, label='FFT')
# # plt.semilogy(sizes, dense_times / scipyfft_times, label='scipy FFT')
# plt.semilogy(sizes, dense_times / dct_times, label='DCT')
# plt.semilogy(sizes, dense_times / dst_times, label='DST')
# plt.semilogy(sizes, dense_times / bp_times, label='BP')
# plt.xscale('log', basex=2)
# plt.xlabel("Dimension")
# plt.ylabel("Speedup over GEMV")
# plt.legend()
# # plt.show()
# plt.savefig('speed.pdf')

data = {
    'sizes': sizes,
    'speedup_fft': dense_times / fft_times,
    'speedup_dct': dense_times / dct_times,
    'speedup_dst': dense_times / dst_times,
    'speedup_bp': dense_times / bp_times,
}

# import pickle
# with open('speed_data.pkl', 'wb') as f:
#     pickle.dump(data, f)
