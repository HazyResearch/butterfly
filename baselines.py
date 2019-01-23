import numpy as np

from target_matrix import named_target_matrix

def baseline_rmse(name, size):
    # dft = named_target_matrix('dft', 512)
    # dft = dft.view('complex128').squeeze(-1)
    n, m = size, int(np.log(size)/np.log(2))
    sparsity = 2 * n*m # n log n
    mat = named_target_matrix(name, n)
    # sparse
    entries = np.sort(mat.reshape(-1)**2)
    rmse_s = np.sqrt(np.sum(entries[:-sparsity]))/n
    # low rank
    u, s, v = np.linalg.svd(mat)
    se = np.sum(s[m:]**2) # rank log n
    rmse_lr = np.sqrt(se)/n
    return rmse_s, rmse_lr

transforms = ['dft', 'dct', 'dst', 'convolution', 'hadamard', 'hartley', 'legendre', 'hilbert', 'randn']
sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
print()
sparse_all_rmse = []
lr_all_rmse = []
for name in transforms:
    sparse_rmse = []
    lr_rmse = []
    for N in sizes:
        if name == 'dft':  # Calculate by hand, does not support complex
            r1 = np.sqrt((N - np.log2(N)) / N)
            r2 = np.sqrt(N - np.log2(N)) / N
        else:
            r1, r2 = baseline_rmse(name, N)
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
