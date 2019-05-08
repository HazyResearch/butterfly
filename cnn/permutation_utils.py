import os, sys
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import torch
# import torchvision

def listperm2matperm(listperm):
    """Converts permutation list to matrix form.

    Args:
    listperm: (..., n) - tensor of list permutations of the set [n].

    Return:
    matperm: (..., n, n) - permutation matrices,
    matperm[t, i, listperm[t,i]] = 1
    """
    n = listperm.size(-1)
    P = torch.eye(n, dtype=torch.long, device=listperm.device)[listperm]
    return P

def matperm2listperm(matperm):
    """Converts permutation matrix to its enumeration (list) form.

    Args:
    matperm: (..., n, n)

    Returns:
    listperm: (..., n) - listperm[t,i] is the index of the only non-zero entry in matperm[t, i, :]
    """
    batch_size = matperm.size(0)
    n = matperm.size(-1)
    assert matperm.size(-2) == matperm.size(-1)

    #argmax is the index location of each maximum value found(argmax)
    # _, argmax = torch.max(matperm, dim=-1, keepdim=False)
    argmax = torch.argmax(matperm, dim=-1, keepdim=False)
    # argmax = argmax.view(batch_size, n_objects)
    return argmax

def invert_listperm(listperm):
    return matperm2listperm(torch.transpose(listperm2matperm(listperm), -1, -2))



def mse(perm, true):
    """ perm is matrix, true is list """
    return nn.functional.mse_loss(perm, listperm2matperm(true))

def nll(perm, true):
    """
    perm: (n, n) or (s, n, n)
    true: (n)
    """
    n = true.size(-1)
    # i = torch.arange(n, device=perm.device)
    # j = true.to(perm.device)
    # print("perm.nll:", perm.size(), true.size())
    elements = perm.cpu()[..., torch.arange(n), true]
    # elements = perm.cpu()[torch.arange(n), true]
    nll = -torch.sum(torch.log2(elements.to(perm.device)))
    if perm.dim() == 3: # normalize by number samples
        nll = nll / perm.size(0)
    # print("nll", nll)
    return nll

def dist(perm1, perm2, fn='nll'):
    """
    perm1: iterable of permutation tensors
           each tensor can have shape (n, n) or (s, n, n)
    perm2: iterable of permutation lists (n)
    """
    # TODO: is the scaling of this consistent across permutations of multiple "ranks"?
    loss = 0.0
    # if not isinstance(perm1, tuple):
    #     perm1, perm2 = (perm1,), (perm2,)
    if fn == 'nll':
        loss_fn = nll
    elif fn == 'mse':
        loss_fn = mse
    elif fn == 'was':
        loss_fn = transport
    else: assert False, f"perm.dist: fn {fn} not supported."

    for p1, p2 in zip(perm1, perm2):
        # print(p1.size(), p1.type())
        # print(p2.size(), p2.type())
        # print(p2, type(p2))
        loss = loss + loss_fn(p1, p2)
    # print(loss, loss.type())
    return loss

def entropy(p, reduction='mean'):
    """
    p: (..., n, n)
    Returns: avg
    Note: Max entropy of n x n matrix is n\log(n)
    """
    n = p.size(-1)
    entropy = -(p * torch.log2(p)).sum(dim=-1).sum(dim=-1) # can dim be list?
    if reduction is None:
        return entropy
    elif reduction == 'sum':
        return torch.sum(entropy)
    elif reduction == 'mean':
        return torch.mean(entropy) # entropy / p.view(-1, n, n).size(0)
    else: assert False, f"perm.entropy: reduction {reduction} not supported."

def transport(ds, p, reduction='mean'):
    """
    "Transport" distance between a doubly-stochastic matrix and a permutation
    ds: (..., n, n)
    p: (n)
    Returns: avg
    Note:
      uniform ds has transport distance (n^2-1)/3
      ds[...,i,p[i]] = 1 has transport 0
    """
    n = p.size(-1)
    dist = torch.arange(n).repeat(n,1).t() - p.repeat(n,1) # dist[i,j] = i - p[j]
    dist = torch.abs(dist).to(ds.device, dtype=torch.float)
    # dist = torch.tensor(dist, dtype=torch.float, device=ds.device)
    t1 = torch.sum(ds * dist, dim=[-2,-1])
    t2 = torch.sum(ds.transpose(-1,-2) * dist, dim=[-2,-1])
    print("transport: ", t1, t2)
    t = t1 + t2 # TODO: figure out right scaling for this. also transport between "rank 2" permutations

    if reduction is None:
        return t
    elif reduction == 'sum':
        return torch.sum(t)
    elif reduction == 'mean':
        return torch.mean(t)
    else: assert False, f"perm.transport: reduction {reduction} not supported."

def tv(x, norm=2, p=2):
    """ Image total variation
    x: (b, c, w, h)

    If D = (dx, dy) is the vector of differences at a given pixel,
    sum up ||D||_norm^p over image
    """
    # each pixel wants all channels as part of its delta vector
    x = x.transpose(-3, -2).transpose(-2, -1) # (b, w, h, c)
    dx = x[:, 1:, :, :] - x[:, :-1, :, :]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]

    # delta = torch.zeros_like(x)
    # delta[..., :-1, :] += torch.abs(dx) ** norm
    # delta[..., :, :-1] += torch.abs(dy) ** norm
    # delta = delta ** (1/norm)
    # tv = delta.sum() / x.size(0)
    delta = x.new_zeros(*x.size(), 2) # torch.zeros_like(x)
    delta[:, :-1, :, :, 0] = torch.abs(dx)
    delta[:, :, :-1, :, 1] = torch.abs(dy)
    delta = delta.flatten(-2, -1) # (b, w, h, 2*c)
    if norm == p:
        v = torch.sum(delta ** norm, dim=-1)
    else:
        v = torch.norm(torch.abs(delta), dim=-1, p=norm)
    tv = v.sum() / x.size(0)
    return tv
    # return torch.tensor(1.0, requires_grad=True, device=x.device)
