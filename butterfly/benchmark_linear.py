import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from butterfly import Butterfly
from butterfly.butterfly_multiply import butterfly_mult, butterfly_mult_untied, butterfly_mult_untied_svd, butterfly_mult_factors, butterfly_mult_inplace
from factor_multiply import butterfly_multiply_untied_batch, butterfly_multiply_untied_eval

# single threaded inference
torch.set_num_threads(1)

import time
nsteps = 2000

# batch_size = 22
# in_ = 512
# out_ = 512

def run(in_, out_, batch_size, verbose=False):
    # create multiple models so the weights aren't already loaded in the cache
    L = [torch.nn.Linear(in_, out_, bias=False) for _ in range(nsteps)]
    # sequence length
    x = torch.randn(batch_size, 1, in_, requires_grad=False)
    B_untied = [Butterfly(in_, out_, bias=False, tied_weight=False) for _ in range(nsteps)]
    twiddle_untied = [B_untied[i].twiddle for i in range(nsteps)]

    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = B_untied[i](x.contiguous().view(-1, x.shape[2])).view(x.shape[0], x.shape[1], -1)
    bfly_end = time.perf_counter()
    bfly_time = bfly_end - bfly_start
    print(bfly_time)

    B_untied = [i.eval() for i in B_untied]
    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = B_untied[i](x.contiguous().view(-1, x.shape[2])).view(x.shape[0], x.shape[1], -1)
    bfly_end = time.perf_counter()
    bfly_time = bfly_end - bfly_start
    print(bfly_time)

    # if verbose:
    #     print(f'Dim: {in_, out_} Batch Size: {batch_size}: {bfly_time}s')

    # gemm_start = time.perf_counter()
    # for i in range(nsteps):
    #     output = L[i](x)
    # gemm_end = time.perf_counter()
    # gemm_time = gemm_end - gemm_start

    # if verbose:
    #     print(f'Gemm forward: {gemm_time}s')

    # print(f'Dim: {in_, out_} Batch Size: {batch_size} Speedup: {gemm_time / bfly_time}x')

# without python overhead
def run_raw(in_, out_, batch_size, verbose=False, seed=1234):
    torch.manual_seed(seed)
    L = [torch.nn.Linear(in_, out_, bias=False) for _ in range(nsteps)]
    weights = [i.weight.t() for i in L]
    # sequence len
    x = torch.randn(batch_size, in_, requires_grad=False)
    x_torch = torch.randn(batch_size, 1, in_, requires_grad=False)
    x_stack = x.unsqueeze(1).expand((batch_size, max(1, out_//in_), in_))
    B_untied = [Butterfly(in_, out_, bias=False, tied_weight=False) for _ in range(nsteps)]
    twiddle_untied = [B_untied[i].twiddle for i in range(nsteps)]

    print("basic")
    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = butterfly_mult_untied(twiddle_untied[i], x_stack, True, True)
    bfly_end = time.perf_counter()
    bfly_time = bfly_end - bfly_start
    print(f'Dim: {in_, out_} Batch Size: {batch_size}: {bfly_time}s')

    print("optimized")
    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = butterfly_mult_untied(twiddle_untied[i], x_stack, True, False)
    bfly_end = time.perf_counter()
    bfly_time = bfly_end - bfly_start
    print(f'Dim: {in_, out_} Batch Size: {batch_size}: {bfly_time}s')

    # if batch_size >= 4:
    #     bfly_start = time.perf_counter()
    #     for i in range(nsteps):
    #         output = butterfly_multiply_untied_batch(twiddle_untied[i], x_stack, True)
    #     bfly_end = time.perf_counter()
    #     bfly_time = bfly_end - bfly_start
    #     if verbose:
    #         print(f'\nButterfly mult untied eval forward: {bfly_time}s')
    # else:
    #     bfly_start = time.perf_counter()
    #     for i in range(nsteps):
    #         output = butterfly_multiply_untied_eval(twiddle_untied[i], x_stack, True)
    #     bfly_end = time.perf_counter()
    #     bfly_time = bfly_end - bfly_start
    #     if verbose:
    #         print(f'\nButterfly mult untied eval forward: {bfly_time}s')

    gemm_start = time.perf_counter()
    for i in range(nsteps):
        output = x_torch.matmul(weights[i])
    gemm_end = time.perf_counter()
    gemm_time = gemm_end - gemm_start
    if verbose:
        print(f'Gemm forward: {gemm_time}s')

    # print(f'Dim: {in_, out_} Batch Size: {batch_size}')
    print(f'Speedup: {gemm_time / bfly_time}x')

# run(512, 512, 16, verbose=True)
# run(512, 1024, 16, verbose=True)
# run(1024, 512, 16, verbose=True)
# run(512, 512, 1, verbose=True)
# run(512, 1024, 1, verbose=True)
# run(1024, 512, 1, verbose=True)

# print("Call profile functions directly")
# run_raw(512, 512, 16, verbose=True)
# run_raw(512, 1024, 16, verbose=True)
# run_raw(1024, 512, 16, verbose=True)
# run_raw(512, 512, 1, verbose=True)
# run_raw(1024, 1024, 1, verbose=True)
# run_raw(512, 1024, 1, verbose=True)
run_raw(1024, 512, 1, verbose=True)