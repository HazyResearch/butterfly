import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch

from cnn.models.butterfly_conv import ButterflyConv2d
from butterfly.butterfly import ButterflyBmm
from butterfly.butterfly_multiply import butterfly_conv2d

import time
nsteps = 1000

in_planes = 128
out_planes = 128
kernel_size = 3
stride = 1
batch_size = 128
f_dim = 16
padding = 1

conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
    padding=1, bias=False).to('cuda')
bfly = ButterflyConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
    padding=1, bias=False, tied_weight=False, fused_unfold=False).to('cuda')
bfly_fused = ButterflyConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
    padding=1, bias=False, tied_weight=False, fused_unfold=True).to('cuda')
x = torch.randn(batch_size, in_planes, f_dim, f_dim, requires_grad=True).to('cuda')
grad = torch.randn_like(x)

# Conv2d
mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
output = conv1.forward(x)  # Do it once to initialize cuDNN handle and such
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	output = conv1.forward(x)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'Conv2d forward: {end - start}s {(mem2-mem1)/1e6}MB')

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.autograd.grad(output, (conv1.weight, x), grad, retain_graph=True)  # Do it once
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	torch.autograd.grad(output, (conv1.weight, x), grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'Conv2d backward: {end - start}s {(mem2-mem1)/1e6}MB')

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
output = conv1.forward(x)
torch.autograd.grad(output, (conv1.weight, x), grad)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	output = conv1.forward(x)
	torch.autograd.grad(output, (conv1.weight, x), grad)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'Conv2d together: {end - start}s {(mem2-mem1)/1e6}MB')

# Butterfly Conv

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	output = bfly.forward(x)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'ButterflyConv2d forward: {end - start}s {(mem2-mem1)/1e6}MB')

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	torch.autograd.grad(output, (bfly.twiddle, x), grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'ButterflyConv2d backward: {end - start}s {(mem2-mem1)/1e6}MB')

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	output = bfly.forward(x)
	torch.autograd.grad(output, (bfly.twiddle, x), grad)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'ButterflyConv2d together: {end - start}s {(mem2-mem1)/1e6}MB')

# Fused-unfold butterfly

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	output = bfly_fused.forward(x)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'ButterflyConv2d fused-unfold forward: {end - start}s {(mem2-mem1)/1e6}MB')

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	torch.autograd.grad(output, (bfly_fused.twiddle, x), grad, retain_graph=True)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'ButterflyConv2d fused-unfold backward: {end - start}s {(mem2-mem1)/1e6}MB')

mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
	output = bfly_fused.forward(x)
	torch.autograd.grad(output, (bfly_fused.twiddle, x), grad)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'ButterflyConv2d fused-unfold together: {end - start}s {(mem2-mem1)/1e6}MB')
