import torch

from butterfly import Block2x2DiagProduct, Block2x2DiagProductRectangular

from test_factor_multiply import twiddle_list_concat
from factor_multiply import butterfly_factor_multiply_inplace

batch_size = 100
n = 1024
# B = Block2x2DiagProduct(n).to('cuda')
B = Block2x2DiagProductRectangular(n, n, bias=False).to('cuda')
# W = torch.randn(n, n, requires_grad=False).to('cuda')
L = torch.nn.Linear(n, n, bias=False).to('cuda')
x = torch.randn(batch_size, n, requires_grad=True).to('cuda')
twiddle = twiddle_list_concat(B)


import time
nsteps = 1000
# nsteps = 1

grad = torch.randn_like(x)

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = B(x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly forward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = B(x)
    output.backward(gradient=grad)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly together: {end - start}s')

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = x.clone()
    butterfly_factor_multiply_inplace(twiddle, output)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly all forward: {end - start}s')


# output = x @ W.t()  # Do it once so that cuBlas handles are initialized, etc.
output = L(x)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    # output = x @ W.t()
    output = L(x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Gemm forward: {end - start}s')
output.backward(gradient=grad, retain_graph=True)  # Do it once just to be safe
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Gemm backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    # output = x @ W.t()
    output = L(x)
    output.backward(gradient=grad)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Gemm together: {end - start}s')

output = torch.rfft(x, 1)  # Do it once so that cuFFT plans are cached, etc.
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = torch.rfft(x, 1)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT forward: {end - start}s')
grad = torch.randn_like(output)  # Do it once just to be safe
output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = torch.rfft(x, 1)
    output.backward(gradient=grad)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT together: {end - start}s')

# output = B(x)
# output.backward(gradient=grad)
# output = L(x)
# output.backward(gradient=grad)
# output = torch.rfft(x, 1)
# output = x.clone()
# butterfly_factor_multiply_inplace(twiddle, output)

