import numpy as np
import torch

from target_matrix import named_target_matrix

def baseline_rmse(name, size, param_fn):
    # dft = named_target_matrix('dft', 512)
    # dft = dft.view('complex128').squeeze(-1)
    # n, m = size, int(np.log(size)/np.log(2))
    n = size
    params = int(param_fn(n))
    # sparsity = 2 * n*m # n log n
    sparsity = params
    mat = named_target_matrix(name, n)
    # print(mat)
    # sparse
    entries = np.sort(mat.reshape(-1)**2)
    rmse_s = np.sqrt(np.sum(entries[:-sparsity])) # /n
    # low rank
    u, s, v = np.linalg.svd(mat)
    rank = params // (2*n)
    se = np.sum(s[rank:]**2) # rank log n
    rmse_lr = np.sqrt(se) # /n
    return rmse_s, rmse_lr

# transforms = ['dft', 'dct', 'dst', 'convolution', 'hadamard', 'hartley', 'legendre', 'hilbert', 'randn']
# sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
# bf_params = [lambda n: 2*n*np.log2(n)]
transforms = ['sparse1', 'rank1', 'butterfly', 'convolution', 'fastfood', 'randn']
bf_params = [
    lambda n: 4*n*(np.log2(n)+1),
    lambda n: n*(np.log2(n)+1),
    lambda n: n*(np.log2(n)+1),
    lambda n: 2*n*(np.log2(n)+1),
    lambda n: 2*n*(np.log2(n)+1),
    lambda n: n*(np.log2(n)+1),
]
sizes = [256]
print()
sparse_all_rmse = []
lr_all_rmse = []
for name, param_fn in zip(transforms, bf_params):
    sparse_rmse = []
    lr_rmse = []
    for N in sizes:
        if name == 'dft':  # Calculate by hand, does not support complex
            r1 = np.sqrt((N - np.log2(N)) / N)
            r2 = np.sqrt(N - np.log2(N)) / N
        else:
            r1, r2 = baseline_rmse(name, N, param_fn)
        print(f"{name:12} {r1:10.6} {r2:10.6}")
        sparse_rmse.append(r1)
        lr_rmse.append(r2)
    sparse_all_rmse.append(sparse_rmse)
    lr_all_rmse.append(lr_rmse)

import pickle

with open('sparse_rmse.pkl', 'wb') as f:
    pickle.dump(np.array(sparse_all_rmse), f)
with open('lr_rmse.pkl', 'wb') as f:
    pickle.dump(np.array(lr_all_rmse), f)
