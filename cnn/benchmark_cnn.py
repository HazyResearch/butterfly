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
    padding=1, bias=False, tied_weight=False).to('cuda')
bfly_bmm = ButterflyBmm(in_planes, out_planes, kernel_size*kernel_size,
    bias=False).to('cuda')
x = torch.randn(batch_size, in_planes, f_dim, f_dim, requires_grad=True).to('cuda')
batched_x = torch.randn(batch_size * f_dim * f_dim, kernel_size * kernel_size, in_planes,
    requires_grad=True).to('cuda')
grad = torch.randn_like(x)
batched_grad = torch.randn_like(batched_x)

# Conv2d
# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.synchronize()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	output = conv1.forward(x)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'Conv2d forward: {end - start}s {(mem2-mem1)/1e6}MB')

# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.synchronize()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	torch.autograd.grad(output, (conv1.weight, x), grad, retain_graph=True)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'Conv2d backward: {end - start}s {(mem2-mem1)/1e6}MB')

# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.synchronize()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	output = conv1.forward(x)
# 	torch.autograd.grad(output, (conv1.weight, x), grad)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'Conv2d together: {end - start}s {(mem2-mem1)/1e6}MB')

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

h_out = (f_dim + 2 * padding - (kernel_size - 1) - 1)  + 1
w_out = (f_dim + 2 * padding - (kernel_size - 1) - 1)  + 1
mem1 = torch.cuda.memory_allocated()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(nsteps):
    # TODO: fuse folding into butterfly_conv2d? 
    output = butterfly_conv2d(bfly.twiddle, x, kernel_size, padding, False).mean(dim=1)
    output = output.view(
                batch_size, 
                h_out * w_out, out_planes).transpose(1, 2).view(batch_size, out_planes, 
                h_out, w_out)
torch.cuda.synchronize()
end = time.perf_counter()
mem2 = torch.cuda.max_memory_allocated()
print(f'ButterflyConv2d cuda forward: {end - start}s {(mem2-mem1)/1e6}MB')

# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.synchronize()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	torch.autograd.grad(output, (bfly.twiddle, x), grad, retain_graph=True)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'ButterflyConv2d backward: {end - start}s {(mem2-mem1)/1e6}MB')

# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.synchronize()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	output = bfly.forward(x)
# 	torch.autograd.grad(output, (bfly.twiddle, x), grad)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'ButterflyConv2d together: {end - start}s {(mem2-mem1)/1e6}MB')

# Butterfly BMM
# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# torch.cuda.synchronize()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	output = bfly_bmm.forward(batched_x)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'ButterflyBmm forward: {end - start}s {(mem2-mem1)/1e6}MB')

# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	torch.autograd.grad(output, (bfly_bmm.twiddle, batched_x), batched_grad,
# 		retain_graph=True)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'ButterflyBmm backward: {end - start}s {(mem2-mem1)/1e6}MB')

# mem1 = torch.cuda.memory_allocated()
# torch.cuda.reset_max_memory_allocated()
# start = time.perf_counter()
# for _ in range(nsteps):
# 	output = bfly_bmm.forward(batched_x)
# 	torch.autograd.grad(output, (bfly_bmm.twiddle, batched_x), batched_grad)
# torch.cuda.synchronize()
# end = time.perf_counter()
# mem2 = torch.cuda.max_memory_allocated()
# print(f'ButterflyBmm together: {end - start}s {(mem2-mem1)/1e6}MB')