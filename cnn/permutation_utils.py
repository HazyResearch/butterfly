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

    loss1, loss2 = 0.0, 0.0
    for p1, p2 in zip(perm1, perm2):
        n = p2.size(-1)
        # print(p1.size(), p1.type())
        # print(p2.size(), p2.type())
        # print(p2, type(p2))
        if fn == 'was': # temporary casework
            l1, l2 = loss_fn(p1, p2)
            l1_, l2_ = loss_fn(p1, n-1-p2) # reversed permutation also good
            if l2_ < l2:
                loss1 += l1_
                loss2 += l2_
            else:
                loss1 += l1
                loss2 += l2
        else:
            loss = loss + loss_fn(p1, p2)
    # print(loss, loss.type())
    if fn == 'was':
        return loss1, loss2
    else:
        return loss

def entropy(p, reduction='mean'):
    """
    p: (..., n, n)
    Returns: avg
    Note: Max entropy of n x n matrix is n\log(n)
    """
    n = p.size(-1)
    eps = 1e-10
    entropy = -(p * torch.log2(eps+p)).sum(dim=[-2,-1])
    if reduction is None:
        return entropy
    elif reduction == 'sum':
        return torch.sum(entropy)
    elif reduction == 'mean':
        return torch.mean(entropy) # entropy / p.view(-1, n, n).size(0)
    else: assert False, f"perm.entropy: reduction {reduction} not supported."

def transport(ds, p, reduction='mean'):
    # TODO: figure out correct transport between "rank 2" permutations
    """
    "Transport" distance between a doubly-stochastic matrix and a permutation
    ds: (..., n, n)
    p: (n)
    Returns: avg
    Note:
      uniform ds has transport distance (n^2-1)/3
      ds[...,i,p[i]] = 1 has transport 0
    If distance raised to power p, average case becomes 2n^{p+1}/(p+1)(p+2)

    Scratchwork:
    true p permuted input with inp = orig[p], i.e. inp[i] = orig[p[i]]
    want to restore out[i] = orig[i] = inp[pinv[i]]
    model multiplies by input by some DS,
      out[i] = inp[j]ds[j,i] = inp[pinv[j]]ds[pinv[j], i]
    want ds[pinv[i],i] = 1, rest = 0
      define this matrix as P
    i.e. P[i, p[i]] = 1
    what's an acceptable error? can handle
      out[i] = orig[i+d] = inp[pinv[i+d]]
    i.e. ds[pinv[i+d], i] = 1
    i.e. ds[j, p[j]+-d] = 1
    so penalization function should be cost[i,j] = f(j - p[i])
    equivalent to optimal transport between rows of ds and P
    """
    n = p.size(-1)
    dist = torch.arange(n).repeat(n,1).t() - p.repeat(n,1) # dist[i,j] = i - p[j]
    # TODO transposing dist should switch t1 and t2
    # dist = torch.arange(n).repeat(n,1) - p.repeat(n,1).t() # dist[i,j] = j - p[i]
    dist = torch.abs(dist).to(ds.device, dtype=torch.float)
    # dist = torch.tensor(dist, dtype=torch.float, device=ds.device)
    t1 = torch.sum(ds * dist, dim=[-2,-1])
    t2 = torch.sum(ds.transpose(-1,-2) * dist, dim=[-2,-1])
    print("TRANSPORT: ", t1.cpu(), t2.cpu())
    t = t1+t2

    if reduction is None:
        return t
    elif reduction == 'sum':
        return torch.sum(t)
    elif reduction == 'mean':
        # return torch.mean(t)
        # QUICK DEBUG
        return (torch.mean(t1), torch.mean(t2))
    else: assert False, f"perm.transport: reduction {reduction} not supported."

def tv(x, norm=2, p=1, symmetric=False, reduction='mean'):
    """ Image total variation
    x: (b, c, w, h)

    If D = (dx, dy) is the vector of differences at a given pixel,
    sum up ||D||_norm^p over image

    Note that reduction='mean' only averages over the batch dimension
    """
    # each pixel wants all channels as part of its delta vector
    x = x.transpose(-3, -2).transpose(-2, -1) # (b, w, h, c)
    delta = x.new_zeros(*x.size(), 2)
    if not symmetric:
        dx = x[:, 1:, :, :] - x[:, :-1, :, :]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        delta[:, :-1, :, :, 0] = dx
        delta[:, :, :-1, :, 1] = dy
    else:
        dx = x[:, 2:, :, :] - x[:, :-2, :, :]
        dy = x[:, :, 2:, :] - x[:, :, :-2, :]
        delta[:, 1:-1, :, :, 0] = dx / 2.0
        delta[:, :, 1:-1, :, 1] = dy / 2.0
        # old symmetric version (4-sided)
    # delta = x.new_zeros(*x.size(), 4) # TODO do casework on symmetric to either 2 or 4?
    # dx_ =  x[:, :-1, :, :] - x[:, 1:, :, :]
    # dy_ =  x[:, :, :-1, :] - x[:, :, 1:, :]
    # delta[:, 1:, :, :, 0] = torch.abs(dx_)
    # delta[:, :, 1:, :, 1] = torch.abs(dy_)

    delta = delta.flatten(-2, -1) # (b, w, h, 2*c [or 4*c])
    if norm == p:
        v = torch.sum(torch.abs(delta) ** norm, dim=-1)
    else:
        v = torch.norm(torch.abs(delta), dim=-1, p=norm)
        if p != 1:
            v = v ** p

    if reduction is None:
        return v
    elif reduction == 'sum':
        return torch.sum(v)
    elif reduction == 'mean':
        return torch.sum(v) / v.size(0)
    else: assert False, f"perm.tv: reduction {reduction} not supported."
