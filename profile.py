import torch
from torch import nn
from torch import optim

from butterfly_factor import butterfly_factor_mult
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

nsteps = 50
batch_size = 100
n = 1024
B = Block2x2DiagProduct(n)
# B = Block2x2DiagProductBmm(n)
P = BlockPermProduct(n)
model = nn.Sequential(P, B)
# model = nn.Sequential(B)
x = torch.randn(batch_size, n)
# B = Block2x2DiagProduct(n, complex=True)
# x = torch.randn(batch_size, n, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

with torch.autograd.profiler.profile() as prof:
    for _ in range(nsteps):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, x)
        # output = x
        # for factor in B.factors[::-1]:
        #     output = butterfly_factor_mult(factor.ABCD, output.view(-1, 2, factor.size // 2)).view(x.shape)
        # output = output.reshape(x.shape)
        # loss = output.sum()
        loss.backward()
        optimizer.step()
sorted_events = torch.autograd.profiler.EventList(sorted(prof.key_averages(), key=lambda event: event.cpu_time_total, reverse=True))
print(sorted_events)

