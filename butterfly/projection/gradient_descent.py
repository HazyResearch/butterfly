import math
import numpy as np
import torch

from butterfly.butterfly import Butterfly


def butterfly_GD(matrix, project_onto='B', lr=5e-3, max_iters=25000, tol=1e-8, plateau_length=1000, print_every=0):
    mat = torch.tensor(matrix).float()
    butterfly = Butterfly(*list(reversed(mat.size())), bias=False, tied_weight=False, increasing_stride=(project_onto == 'BT'))
    ident = torch.eye(butterfly.in_size)
    if torch.cuda.is_available():
        mat = mat.cuda()
        butterfly = butterfly.cuda()
        ident = ident.cuda()
    optim = torch.optim.SGD(butterfly.parameters(), lr=lr)
    last_losses = []
    for i in range(max_iters):
        out = butterfly(ident)
        diff = out - mat
        loss = torch.sum(torch.mul(diff, diff))
        optim.zero_grad()
        loss.backward()
        optim.step(lambda: loss)
        l = loss.item()
        if print_every and i % print_every == 0: print(i, l)
        if last_losses and l > last_losses[-1]:
            print('Decaying LR...')
            for pg in optim.param_groups: pg['lr'] /= 2
        if l < tol or (last_losses and np.amax(last_losses[-plateau_length:])-l < 1e-8):
            break
        last_losses.append(l)
    return butterfly, out.detach().cpu().numpy()
