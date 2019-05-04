import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from butterfly import Butterfly
from butterfly.butterfly_multiply import butterfly_mult, butterfly_mult_untied, butterfly_mult_untied_svd, butterfly_mult_factors, butterfly_mult_inplace
from factor_multiply import butterfly_multiply_untied, butterfly_multiply_untied_eval

# single threaded inference
torch.set_num_threads(1)

import time
nsteps = 2000

# includes most of the python overhead of calling classes (for instance, minus reshaping the input)
def run(in_, out_, batch_size):
    # create multiple models so the weights aren't already loaded in the cache
    L = [torch.nn.Linear(in_, out_, bias=False) for _ in range(nsteps)]
    B_untied = [Butterfly(in_, out_, bias=False, tied_weight=False) for _ in range(nsteps)]
    twiddle_untied = [B_untied[i].twiddle for i in range(nsteps)]
    x = torch.randn(batch_size, in_, requires_grad=False)

    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = B_untied[i](x)
    bfly_end = time.perf_counter()
    bfly_time_train = bfly_end - bfly_start
    print(f'Butterfly Training Forward: {bfly_time_train}')

    B_untied = [i.eval() for i in B_untied]
    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = B_untied[i](x)
    bfly_end = time.perf_counter()
    bfly_time_eval = bfly_end - bfly_start
    print(f'Butterfly Inference Forward: {bfly_time_eval}')

    output = L[-1](x)
    gemm_start = time.perf_counter()
    for i in range(nsteps):
        output = L[i](x)
    gemm_end = time.perf_counter()
    gemm_time = gemm_end - gemm_start
    print(f'Linear Forward: {gemm_time}')

    print(f'Dim: {in_, out_} Batch Size: {batch_size} Speedup: {gemm_time / bfly_time_eval}x')

# call functions directly
def run_raw(in_, out_, batch_size):
    L = [torch.nn.Linear(in_, out_, bias=False) for _ in range(nsteps)]
    weights = [i.weight.t() for i in L]
    x = torch.randn(batch_size, in_, requires_grad=False)
    x_stack = x.unsqueeze(1).expand((batch_size, max(1, out_//in_), in_))
    B_untied = [Butterfly(in_, out_, bias=False, tied_weight=False) for _ in range(nsteps)]
    twiddle_untied = [B_untied[i].twiddle for i in range(nsteps)]

    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = butterfly_multiply_untied(twiddle_untied[i], x_stack, True, False)
    bfly_end = time.perf_counter()
    bfly_time_train = bfly_end - bfly_start
    print(f'Butterfly Training Forward: {bfly_time_train}')

    bfly_start = time.perf_counter()
    for i in range(nsteps):
        output = butterfly_multiply_untied_eval(twiddle_untied[i], x_stack, True)
    bfly_end = time.perf_counter()
    bfly_time_eval = bfly_end - bfly_start
    print(f'Butterfly Inference Forward: {bfly_time_eval}')

    gemm_start = time.perf_counter()
    for i in range(nsteps):
        output = x.matmul(weights[i])
    gemm_end = time.perf_counter()
    gemm_time = gemm_end - gemm_start
    print(f'Linear Forward: {gemm_time}')

    print(f'Dim: {in_, out_} Batch Size: {batch_size} Speedup: {gemm_time / bfly_time_eval}x')

print("Call functions with python class")
run(512, 512, 16)
run(512, 1024, 16)
run(1024, 512, 16)
run(512, 512, 1)
run(512, 1024, 1)
run(1024, 512, 1)

print("\nCall functions directly")
run_raw(512, 512, 16)
run_raw(512, 1024, 16)
run_raw(1024, 512, 16)
run_raw(512, 512, 1)
run_raw(512, 1024, 1)
run_raw(1024, 512, 1)