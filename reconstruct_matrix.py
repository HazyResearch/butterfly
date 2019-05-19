import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from butterfly import Butterfly

# TODO..
parser = argparse.ArgumentParser('Butterfly matrix recovery')
parser.parse_args()

saved_model = torch.load('model_optimizer.pth')['model']
print('Loaded model')
mat = saved_model['layer3.0.conv2.weight']
mat = mat[:, :, 0, 0]  # start with just one filter matrix
numpy_mat = mat.detach().cpu().numpy()
m, n = numpy_mat.shape

butterfly = Butterfly(in_size=n, out_size=m, bias=False, tied_weight=False, param='odo', nblocks=1)
print(butterfly)
tot_param = sum([p.numel() for p in butterfly.parameters()])
print('Butterfly #parameters:', tot_param)
rank = int(np.ceil(tot_param / (m + n)))
print(f'Equivalent rank: {rank}; low-rank #parameters: {rank*(m+n)}')

u, s, v = np.linalg.svd(numpy_mat)
diff = (u * s) @ v - numpy_mat
print('Numerical reconstruction error:', np.sum(np.square(diff)))
s[rank:] = 0
diff = (u * s) @ v - numpy_mat
print('Low-rank reconstruction error:', np.sum(np.square(diff)))

butterfly = butterfly.cuda()
ident = torch.eye(n).cuda()

max_steps = 25000
lr_start = 5e-3
optim = torch.optim.LBFGS(butterfly.parameters(), lr=lr_start)
for i in range(max_steps):
    if i == 1000:
        for param_group in optim.param_groups:
            param_group['lr'] = 2e-3
    if i == 20000:
        for param_group in optim.param_groups:
            param_group['lr'] = 1e-3
    out = butterfly(ident)
    diff = out - mat
    loss = torch.sum(torch.mul(diff, diff))
    optim.zero_grad()
    loss.backward()
    optim.step(lambda: loss)
    if i % 10 == 0: print(i, loss.item())

torch.save(butterfly, 'lbfgs.pt')
