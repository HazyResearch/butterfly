import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from butterfly import Butterfly
from butterfly.butterfly_multiply import butterfly_mult, butterfly_mult_untied, butterfly_mult_untied_svd, butterfly_mult_factors, butterfly_mult_inplace
from factor_multiply import butterfly_multiply_untied_batch

# single threaded inference
torch.set_num_threads(1)

import time
nsteps = 1000

batch_size = 16
n = 512
# create multiple models so the weights aren't already loaded in the cache
L = [torch.nn.Linear(n, n, bias=False) for _ in range(nsteps)]
x = torch.randn(batch_size, n, requires_grad=True)
B_untied = [Butterfly(n, n, bias=False, tied_weight=False) for _ in range(nsteps)]
twiddle_untied = [B_untied[i].twiddle for i in range(nsteps)]

grad = torch.randn_like(x)

start = time.perf_counter()
for i in range(nsteps):
    output = butterfly_mult_untied(twiddle_untied[i], x.unsqueeze(1))
end = time.perf_counter()
print(f'Butterfly mult untied forward: {end - start}s')

start = time.perf_counter()
for i in range(nsteps):
    output = butterfly_multiply_untied_batch(twiddle_untied[i], x.unsqueeze(1), True)
end = time.perf_counter()
print(f'Butterfly mult untied batch forward: {end - start}s')

start = time.perf_counter()
for i in range(nsteps):
    output = L[i](x)
end = time.perf_counter()
print(f'Gemm forward: {end - start}s')