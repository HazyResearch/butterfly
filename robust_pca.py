import itertools
import multiprocessing as mp
import os

import numpy as np

import cvxpy as cp

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from target_matrix import named_target_matrix

ntrials = 1

sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
# sizes = [8, 16, 32]
# transform_names = ['dft', 'dct', 'dst', 'convolution', 'hadamard', 'hartley', 'legendre', 'hilbert', 'randn']
transform_names = ['dct', 'dst', 'convolution', 'hadamard', 'hartley', 'legendre', 'randn']
model = {'dft': 'BP', 'dct': 'BP', 'dst': 'BP', 'convolution': 'BPBP', 'hadamard': 'BP', 'hartley': 'BP', 'legendre': 'BP', 'randn': 'BP'}

def sparse_lowrank_mse(name_size):
    name, size = name_size
    print(name, size)
    matrix = named_target_matrix(name, size)
    M = matrix
    lambda1 = cp.Parameter(nonneg=True)
    lambda2 = cp.Parameter(nonneg=True)
    L = cp.Variable((size, size))
    S = cp.Variable((size, size))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(M - L - S) / size**2 + lambda1 / size * cp.norm(L, 'nuc') + lambda2 / size**2 * cp.norm(S, 1)))

    result = []
    for _ in range(ntrials):
        l1 = np.exp(np.random.uniform(np.log(1e-2), np.log(1e4)))
        l2 = np.exp(np.random.uniform(np.log(1e-2), np.log(1e4)))
        lambda1.value = l1
        lambda2.value = l2
        try:
            prob.solve()
            nnz = (np.abs(S.value) >= 1e-7).sum()
            singular_values = np.linalg.svd(L.value, compute_uv=False)
            rank = (singular_values >= 1e-7).sum()
            n_params = nnz + 2 * rank * size
            mse = np.sum((matrix - L.value - S.value)**2) / size**2
            result.append((n_params, mse))
        except:
            pass
    budget = 2 * size * np.log2(size)
    if model[name] == 'BPBP':
        budget *= 2
    eligible = [res for res in result if res[0] <= budget]
    if eligible:
        mse = min(m for (n_params, m) in eligible)
    else:
        mse = np.sum(matrix**2) / size**2
    print(name, size, 'done')
    return (name, size, mse)

pool = mp.Pool()
mse = pool.map(sparse_lowrank_mse, list(itertools.product(transform_names, sizes)))

import pickle

with open('mse_robust_pca.pkl', 'wb') as f:
    pickle.dump(mse, f)

# with open('mse_robust_pca.pkl', 'rb') as f:
#     mse = pickle.load(f)
