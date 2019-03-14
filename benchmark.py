import torch

from butterfly import Block2x2DiagProduct, Block2x2DiagProductRectangular

from test_factor_multiply import twiddle_list_concat
from butterfly_factor import butterfly_factor_mult_inplace


batch_size = 256
n = 1024
B = Block2x2DiagProduct(n).to('cuda')
# B = Block2x2DiagProductRectangular(n, n).to('cuda')
# W = torch.randn(n, n, requires_grad=False).to('cuda')
L = torch.nn.Linear(n, n, bias=False).to('cuda')
x = torch.randn(batch_size, n, requires_grad=True).to('cuda')
twiddle = torch.randn_like(twiddle_list_concat(B), requires_grad=True)


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
    output = butterfly_factor_mult_inplace(twiddle, x)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly in-place forward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, (twiddle, x), grad, retain_graph=True)
    # output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly in-place backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = butterfly_factor_mult_inplace(twiddle, x)
    torch.autograd.grad(output, (twiddle, x), grad)
    # output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Butterfly in-place together: {end - start}s')

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
torch.autograd.grad(output, (L.weight, x), grad, retain_graph=True)
# output.backward(gradient=grad, retain_graph=True)  # Do it once just to be safe
output.backward(gradient=grad, retain_graph=True)  # Do it once just to be safe
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, (L.weight, x), grad, retain_graph=True)
    # output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'Gemm backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    # output = x @ W.t()
    output = L(x)
    torch.autograd.grad(output, (L.weight, x), grad)
    # output.backward(gradient=grad)
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
# output.backward(gradient=grad, retain_graph=True)
torch.autograd.grad(output, x, grad, retain_graph=True)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    torch.autograd.grad(output, x, grad, retain_graph=True)
    # output.backward(gradient=grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT backward: {end - start}s')
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    output = torch.rfft(x, 1)
    torch.autograd.grad(output, x, grad)
    # output.backward(gradient=grad)
torch.cuda.synchronize()
end = time.perf_counter()
print(f'CuFFT together: {end - start}s')

# output = B(x)
# output.backward(gradient=grad)
# output = L(x)
# output.backward(gradient=grad)
# output = torch.rfft(x, 1)
# output = butterfly_factor_mult_inplace(twiddle, x)
# output.backward(gradient=grad)
# # torch.autograd.grad(output, (twiddle, x), grad, retain_graph=True)


# x = torch.randn(3)
# w_init = torch.randn(3)
# w = torch.tensor(w_init, requires_grad=True)

# optimizer = torch.optim.SGD([w], lr=0.1)
# for i in range(10):
#     optimizer.zero_grad()
#     loss = x @ w
#     loss.backward()
#     optimizer.step()
# loss = x @ w
# print(loss.item())

# k = 2.0
# w_new = torch.tensor(w_init / k, requires_grad=True)
# optimizer = torch.optim.SGD([w_new], lr=0.1 / k**2)
# for i in range(10):
#     optimizer.zero_grad()
#     loss = x @ (k * w_new)
#     loss.backward()
#     optimizer.step()
# loss = x @ (k * w_new)
# print(loss.item())
