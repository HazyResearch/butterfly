import torch
from torch import nn
from torch import optim

from butterfly_factor import butterfly_factor_mult
from permutation_factor import permutation_factor_even_odd_mult, permutation_factor_reverse_mult
from butterfly import Block2x2Diag, Block2x2DiagProduct, BlockPermProduct, Block2x2DiagProductBmm


def profile_butterfly_mult():
    nsteps = 10
    batch_size = 100
    n = 1024
    B = Block2x2DiagProduct(n)
    x = torch.randn(batch_size, n)
    # B(x)
    optimizer = optim.Adam(B.parameters(), lr=0.01)
    for _ in range(nsteps):
        optimizer.zero_grad()
        # output = B(x)
        # loss = nn.functional.mse_loss(output, x)
        output = x
        for factor in B.factors[::-1]:
            output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(x.shape)
        # output = output.reshape(x.shape)
        loss = output.sum()
        loss.backward()
        optimizer.step()


import os
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

nsteps = 50
# nsteps = 1
batch_size = 1000
# batch_size = 1
n = 1024
# n = 8
# B = Block2x2DiagProduct(n)
B = Block2x2DiagProduct(n).to('cuda')
# B = Block2x2DiagProductBmm(n).to('cuda')
# P = BlockPermProduct(n)
P = BlockPermProduct(n, complex=False).to('cuda')
model = nn.Sequential(P, B)
# model = nn.Sequential(B)
# x = torch.randn(batch_size, n, requires_grad=True)
x = torch.randn(batch_size, n, requires_grad=True).to('cuda')
# B = Block2x2DiagProduct(n, complex=True)
# x = torch.randn(batch_size, n, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# with torch.autograd.profiler.profile() as prof:
# # with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     for _ in range(nsteps):
#         optimizer.zero_grad()
#         output = model(x)
#         # output = x
#         # output_slow = x
#         # for factor in B.factors[::-1]:
#         #     output_prev = output
#         #     output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(x.shape)
#         #     output_slow = ((factor.ABCD * output_prev.view(-1, 1, 2, factor.size // 2)).sum(dim=-2)).view(output_prev.shape)
#         #     print((output - output_slow).abs().max().item())
#         #     grad = torch.randn_like(output)
#         #     d_twiddle, d_input = torch.autograd.grad(output, (factor.ABCD, output_prev), grad, retain_graph=True)
#         #     # d_twiddle_slow, d_input_slow = torch.autograd.grad(output_slow, (factor.ABCD, output_prev), grad, retain_graph=True)
#         #     d_twiddle_slow = (grad.view(-1, 2, 1, factor.size // 2) * output_prev.view(-1, 1, 2, factor.size // 2)).sum(dim=0)
#         #     d_input_slow = (factor.ABCD.transpose(0, 1) * grad.view(-1, 1, 2, factor.size // 2)).sum(dim=-2).view(output_prev.shape)
#         #     print((d_twiddle - d_twiddle_slow).abs().max().item())
#         #     print((d_input - d_input_slow).abs().max().item())
#         output = output.view(x.shape)
#         loss = nn.functional.mse_loss(output, x)
#         # loss = output.sum()
#         loss.backward()
#         optimizer.step()
# sorted_events = torch.autograd.profiler.EventList(sorted(prof.key_averages(), key=lambda event: event.cpu_time_total, reverse=True))
# print(sorted_events)



import time
# nsteps = 1000
nsteps = 1

grad = torch.randn_like(x)

# output = x
# torch.cuda.synchronize()
# start = time.perf_counter()
# for factor in B.factors[::-1]:
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         output_fast = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(x.shape)
#         # output = ((factor.ABCD * output.view(-1, 1, 2, factor.size // 2)).sum(dim=-2)).view(output.shape)
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         grad_fast = torch.autograd.grad(output_fast, (factor.ABCD, output), grad.view(output_fast.shape), retain_graph=True)
#         # output = ((factor.ABCD * output.view(-1, 1, 2, factor.size // 2)).sum(dim=-2)).view(output.shape)
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
# torch.cuda.synchronize()
# end = time.perf_counter()
# print('Total: ', end - start)

# output = x
# torch.cuda.synchronize()
# start = time.perf_counter()
# for factor in B.factors[::-1]:
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         # output_fast = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(x.shape)
#         output_slow = ((factor.ABCD * output.view(-1, 1, 2, factor.size // 2)).sum(dim=-2)).view(output.shape)
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
#     # torch.cuda.synchronize()
#     # start_micro = time.perf_counter()
#     # for _ in range(nsteps):
#     #     # output_fast = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(x.shape)
#     #     grad_slow = (output_slow.view(-1, 2, 1, factor.size // 2) * output_slow.view(-1, 1, 2, factor.size // 2)).sum(dim=0)
#     # torch.cuda.synchronize()
#     # end_micro = time.perf_counter()
#     # print(f'Size {factor.size}: {end_micro - start_micro}s')
# torch.cuda.synchronize()
# end = time.perf_counter()
# print('Total: ', end - start)

# a = torch.randn(batch_size * n // 2, 4, device='cuda')
# a = B.factors[-1].ABCD * x.view(-1, 1, 2, 1)
# print(a.shape)
# print(a.stride())
# b = a[:, ::2].sum(dim=-1)
# b = a.sum(dim=0)

output = x
prob = torch.zeros(3, device=x.device, requires_grad=True)
prob[0] = 0.7
prob[1] = 0.6
prob[2] = 0.4

# torch.cuda.synchronize()
# start = time.perf_counter()
# for factor in P.factors[::-1]:
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         output_fast = permutation_factor_even_odd_mult(prob[:1], output.view(-1, factor.size))
#         # output_slow = ((1 - prob[0]) * output.view(-1, 2, factor.size // 2) + prob[0] * output.view(-1, factor.size // 2, 2).transpose(-1, -2)).view(-1, factor.size)
#         # print((output_fast - output_slow).abs().max().item())
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         d_prob_fast, d_output_fast = torch.autograd.grad(output_fast, (prob, output), grad.view(output_fast.shape), retain_graph=True)
#         # d_prob_slow, d_output_slow = torch.autograd.grad(output_slow, (prob, output), grad.view(output_slow.shape), retain_graph=True)
#         # print((d_prob_fast))
#         # print((d_prob_slow))
#         # print((d_output_fast - d_output_slow).abs().max().item())
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
# torch.cuda.synchronize()
# end = time.perf_counter()
# print('Total: ', end - start)

# torch.cuda.synchronize()
# start = time.perf_counter()
# for factor in P.factors[::-1]:
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         output_slow = ((1 - prob[0]) * output.view(-1, 2, factor.size // 2) + prob[0] * output.view(-1, factor.size // 2, 2).transpose(-1, -2)).view(-1, factor.size)
#         # output = torch.add((1 - prob[0]) * output.view(-1, 2, factor.size // 2), prob[0], output.view(-1, factor.size // 2, 2).transpose(-1, -2)).view(-1, factor.size)
#         # output_slow = torch.lerp(output.view(-1, 2, factor.size // 2), output.view(-1, factor.size // 2, 2).transpose(-1, -2), prob[0]).view(-1, factor.size)
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         grad_slow = torch.autograd.grad(output_slow, (prob, output), grad.view(output_slow.shape), retain_graph=True)
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
# torch.cuda.synchronize()
# end = time.perf_counter()
# print('Total: ', end - start)

torch.cuda.synchronize()
start = time.perf_counter()
for factor in P.factors[::-1]:
    torch.cuda.synchronize()
    start_micro = time.perf_counter()
    for _ in range(nsteps):
        output_fast = permutation_factor_reverse_mult(prob[1:], output.view(-1, factor.size))
        # output_slow = ((1 - prob[1:]).unsqueeze(-1) * output.view(-1, 2, factor.size//2) + prob[1:].unsqueeze(-1) * output.view(-1, 2, factor.size//2).flip(-1)).view(-1, factor.size)
        # print((output_fast - output_slow).abs().max().item())
    torch.cuda.synchronize()
    end_micro = time.perf_counter()
    print(f'Size {factor.size}: {end_micro - start_micro}s')
    torch.cuda.synchronize()
    start_micro = time.perf_counter()
    for _ in range(nsteps):
        d_prob_fast, d_output_fast = torch.autograd.grad(output_fast, (prob, output), grad.view(output_fast.shape), retain_graph=True)
        # d_prob_slow, d_output_slow = torch.autograd.grad(output_slow, (prob, output), grad.view(output_slow.shape), retain_graph=True)
        # print((d_prob_fast))
        # print((d_prob_slow))
        # assert d_output_fast.shape == d_output_slow.shape
        # print((d_output_fast - d_output_slow).abs().max().item())
    torch.cuda.synchronize()
    end_micro = time.perf_counter()
    print(f'Size {factor.size}: {end_micro - start_micro}s')
torch.cuda.synchronize()
end = time.perf_counter()
print('Total: ', end - start)

# torch.cuda.synchronize()
# start = time.perf_counter()
# for factor in P.factors[::-1]:
#     reverse_idx = torch.arange(factor.size//2 - 1, -1, -1, device=output.device)
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         # output_slow = (((1 - prob[1:]).unsqueeze(-1) * output.view(-1, 2, factor.size//2) + prob[1:].unsqueeze(-1) * output.view(-1, 2, factor.size//2).flip(-1))).view(-1, factor.size)
#         output_slow = (((1 - prob[1:]).unsqueeze(-1) * output.view(-1, 2, factor.size//2) + prob[1:].unsqueeze(-1) * output.view(-1, 2, factor.size//2)[:, :, reverse_idx])).view(-1, factor.size)
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
#     torch.cuda.synchronize()
#     start_micro = time.perf_counter()
#     for _ in range(nsteps):
#         grad_slow = torch.autograd.grad(output_slow, (prob, output), grad.view(output_slow.shape), retain_graph=True)
#     torch.cuda.synchronize()
#     end_micro = time.perf_counter()
#     print(f'Size {factor.size}: {end_micro - start_micro}s')
# torch.cuda.synchronize()
# end = time.perf_counter()
# print('Total: ', end - start)
