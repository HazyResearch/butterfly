import math
import numpy as np
import torch

from butterfly.butterfly import Butterfly


def butterfly_GD(matrix, lr=5e-3, maxsteps=25000, tol=1e-8, print_every=0):
    mat = torch.tensor(matrix).float()
    butterfly = Butterfly(*list(reversed(mat.size())), bias=False, tied_weight=False)
    ident = torch.eye(butterfly.in_size)
    if torch.cuda.is_available():
        mat = mat.cuda()
        butterfly = butterfly.cuda()
        ident = ident.cuda()
    optim = torch.optim.SGD(butterfly.parameters(), lr=lr)
    for i in range(maxsteps):
        out = butterfly(ident)
        diff = out - mat
        loss = torch.sum(torch.mul(diff, diff))
        optim.zero_grad()
        loss.backward()
        optim.step(lambda: loss)
        l = loss.item()
        if print_every and i % print_every == 0: print(i, l)
        if l < tol:
            break
    return butterfly, out.detach().cpu().numpy()
