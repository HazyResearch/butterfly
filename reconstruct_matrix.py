import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from butterfly import Butterfly
from eval_cifar import validate


def butterfly_compress(matrix, tot_param, **kwargs):
    butterfly = kwargs['butterfly'].cuda()
    nsteps = kwargs['steps']
    ident = torch.eye(butterfly.in_size).cuda()
    mat = torch.tensor(matrix).cuda()
    lr_start = 5e-3
    optim = torch.optim.LBFGS(butterfly.parameters(), lr=lr_start)
    for i in range(nsteps):
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
    return out.detach().cpu().numpy()


def lowrank_compress(matrix, tot_param):
    m, n = matrix.shape
    rank = int(np.ceil(tot_param / (m + n)))
    print(f'Equivalent rank: {rank}; low-rank #parameters: {rank*(m+n)}')
    u, s, vt = np.linalg.svd(matrix)
    u = u[:, :vt.shape[0]]
    diff = (u * s) @ vt - matrix
    print(f'Numerical reconstruction error: {np.sum(np.square(diff))}')
    s[rank:] = 0
    return (u * s) @ vt


def sparse_compress(matrix, tot_param):
    nth_largest = sorted(np.abs(matrix).flatten(), reverse=True)[tot_param]
    pruned = matrix * (np.abs(matrix) > nth_largest)
    return pruned


def compress(matrix, method, tot_param, **kwargs):
    if method == 'butterfly':
        compressed = butterfly_compress(matrix, tot_param, **kwargs)
    elif method == 'lowrank':
        compressed = lowrank_compress(matrix, tot_param)
    elif method == 'sparse':
        compressed = sparse_compress(matrix, tot_param)
    else:
        raise ValueError('Unknown method!')
    diff = compressed - matrix
    print(f'{method} error: {np.sum(np.square(diff))}')
    return compressed


def main():
    print('Initializing...')

    # TODO..
    parser = argparse.ArgumentParser('Butterfly matrix recovery')
    parser.add_argument('--method', type=str, choices=['none', 'butterfly', 'lowrank', 'sparse'], default='none', help='Compression method')
    parser.add_argument('--steps', type=int, default=25000, help='Number of training steps')
    args = parser.parse_args()

    state_dict = torch.load('model_optimizer.pth')['model']
    print('Loaded model')

    if args.method == 'none':
        print('Validating...')
        print(validate(state_dict))
        exit()

    for layer in range(4, 5):
        for block in range(2):
            for conv in range(1, 3):
                weight_name = f'layer{layer}.{block}.conv{conv}.weight'
                weight = state_dict[weight_name]
                print('Weight:', weight_name)
                for mat_id in range(weight[0,0].numel()):
                    mat_id_x = mat_id // weight[0,0,0].numel()
                    mat_id_y = mat_id % weight[0,0,0].numel()
                    matrix = weight[:,:,mat_id_x,mat_id_y].detach().cpu().numpy()
                    m, n = matrix.shape
                    butterfly = Butterfly(in_size=m, out_size=n, bias=False, tied_weight=False, param='odo', nblocks=1)
                    print('Matrix shape:', matrix.shape)
                    tot_param = sum([p.numel() for p in butterfly.parameters()])
                    print('#parameters:', tot_param)
                    compressed = compress(matrix, args.method, tot_param, butterfly=butterfly, steps=args.steps)
                    weight[:,:,mat_id_x,mat_id_y] = torch.tensor(compressed, device=weight.device)

    torch.save({'model': state_dict}, f'model_{args.method}.pth')
    print('Validating...')
    print(validate(state_dict))


if __name__ == '__main__':
    main()
