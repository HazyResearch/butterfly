"""Compute the exact Fisher information matrix of a butterfly matrix.
For an n x n butterfly matrix, this has space complexity O(n^2 log^2 n), which is optimal, and
time complexity O(n^3 log^2 n).
The space is the bottleneck anyway.
"""
import math
from functools import partial
import numpy as np

import torch
import torch_butterfly

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random


# Avoid printing in scientific notation
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


def twiddle_factor_to_matrix(twiddle_factor, stride):
    """
    twiddle_factor: (n // 2, 2, 2)
    stride: int
    Return:
        (n, n)
    """
    n = twiddle_factor.shape[0] *  2
    assert twiddle_factor.shape == (n // 2, 2, 2)
    assert stride == 1 << int(math.log2(stride)), 'stride must be a power of 2'
    x = jnp.eye(n)
    t = jnp.moveaxis(twiddle_factor.reshape(n // (2 * stride), stride, 2, 2), -3, -1)
    y = x.reshape(n, n // (2 * stride), 1, 2, stride)
    y = (t * y).sum(axis=-2).reshape(n, n)
    return y.T


def twiddle_factor_perm(n, stride):
    """The indices in a n x n matrix that marks where the entries of a butterfly factors are.
    """
    # TODO: The logic here is more complicated than necessary
    # I don't have time rn to find a simpler way
    factor = jnp.arange(1, 1 + 2 * n).reshape(n // 2, 2, 2)
    matrix_flat = twiddle_factor_to_matrix(factor, stride).flatten()
    nonzero_locs, = jnp.nonzero(matrix_flat)
    perm = nonzero_locs[jnp.argsort(matrix_flat[nonzero_locs])]
    return perm


def butterfly_multiply_single(twiddle, x, increasing_stride=True, return_intermediates=False):
    """
    twiddle: (log_n, n / 2, 2, 2)
    x: (n)
    Return:
        (n)
    """
    log_n = twiddle.shape[0]
    n = 1 << log_n
    assert twiddle.shape == (log_n, n // 2, 2, 2)
    assert x.shape == (n,)
    y = x
    intermediates = [y]
    for idx in range(log_n):
        log_stride = idx if increasing_stride else log_n - 1 - idx
        stride = 1 << log_stride
        t = jnp.moveaxis(twiddle[idx].reshape(n // (2 * stride), stride, 2, 2), -3, -1)
        y = y.reshape(n // (2 * stride), 1, 2, stride)
        y = (t * y).sum(axis=-2).reshape(n)
        intermediates.append(y)
    return y if not return_intermediates else jnp.stack(intermediates)


butterfly_multiply = vmap(butterfly_multiply_single, in_axes=(None, 0))

torch.manual_seed(2357)
batch_size = 3
n = 32
log_n = int(math.log2(n))
twiddle_pt = torch.randn(1, 1, log_n, n // 2, 2, 2) / math.sqrt(2)
# twiddle_pt = torch.arange(1.0, 17.0).reshape(1, 1, log_n, n // 2, 2, 2)
x_pt = torch.randn(3, 1, n)
out_pt = torch_butterfly.butterfly_multiply(twiddle_pt, x_pt, increasing_stride=True).squeeze()
twiddle = jnp.array(twiddle_pt[0, 0].numpy())
x = jnp.array(x_pt[:, 0].numpy())
out = butterfly_multiply(twiddle, x)

key = random.PRNGKey(2357)

batch_size = 10000
key, key_x, key_y, key_true, key_y_t = random.split(key, 5)
x = random.normal(key_x, (batch_size, n))
true = random.normal(key_true, (n, n))
y = x @ true.T + 0.1 * random.normal(key_y, (batch_size, n))
loss = lambda twiddle, x, y: 0.5 * jnp.sum((butterfly_multiply_single(twiddle, x) - y)**2, axis=-1).mean()

factor_perms = jnp.stack([twiddle_factor_perm(n, 1 << i) for i in range(log_n)])
factor_row_perms = factor_perms // n
factor_col_perms = factor_perms % n
matrices = [twiddle_factor_to_matrix(twiddle[i], 1 << i) for i in range(log_n)]

def fisher_numerical(twiddle, x, key_y_t):
    """Compute Fisher information matrix numerically, using per-sample gradient
    """
    batch_size, n = x.shape
    y_t = butterfly_multiply(twiddle, x) + random.normal(key_y_t, (batch_size, n))
    grad_per_sample = vmap(grad(loss, 0), (None, 0, 0))(twiddle, x, y_t)
    grad_per_sample = grad_per_sample.swapaxes(-1, -2).reshape(batch_size, -1)
    fisher = (grad_per_sample.T @ grad_per_sample) / batch_size
    assert jnp.allclose(fisher, fisher.T)
    return fisher

def fisher_exact(twiddle, x, return_factor=False):
    # behind = [jnp.eye(n)]
    # for i in range(log_n - 1):
    #     behind.append(matrices[i] @ behind[-1])
    bmul_intermediate = vmap(partial(butterfly_multiply_single, return_intermediates=True),
                             (None, 0), 1)
    behind = bmul_intermediate(twiddle, jnp.eye(n)).swapaxes(-1, -2)[:-1]
    # ahead = [jnp.eye(n)]
    # for i in range(1, log_n)[::-1]:
    #     ahead.append(ahead[-1] @ matrices[i])
    # ahead = list(reversed(ahead))
    bmul_t_intermediate = vmap(partial(butterfly_multiply_single, increasing_stride=False,
                                       return_intermediates=True), (None, 0), 1)
    ahead = bmul_t_intermediate(twiddle[::-1].swapaxes(-1, -2), jnp.eye(n))[:-1][::-1]
    # fisher_exact_list = []
    # for i in range(log_n):
    #     fisher_exact_row = []
    #     for j in range(log_n):
    #         if j >= i:
    #             Fij = jnp.kron(behind[i] @ behind[j].T, ahead[i].T @ ahead[j])
    #             Fij_t = Fij[factor_perms[i]][:, factor_perms[j]]
    #         else:
    #             Fij_t = fisher_exact_list[j][i].T
    #         fisher_exact_row.append(Fij_t)
    #     fisher_exact_list.append(fisher_exact_row)
    # fisher_exact = jnp.block(fisher_exact_list)
    # A = jnp.stack([jnp.kron(behind[i], ahead[i].T) for i in range(log_n)])
    # PA = jnp.concatenate([jnp.kron(behind[i], ahead[i].T)[factor_perms[i]] for i in range(log_n)])
    # PA = vmap(lambda b, a, p: jnp.kron(b, a.T)[p])(behind, ahead, factor_perms).reshape(-1, n * n)
    # PA = vmap(
    #     lambda b, a, p: (jnp.repeat(b, n, 0)[p][:, :, None] * jnp.tile(a.T, (n, 1))[p][:, None, :])
    # )(behind, ahead, factor_perms).reshape(-1, n * n)
    # fisher_exact = PA @ PA.T
    PA = None
    # L = vmap(lambda b, p: jnp.repeat(b, n, 0)[p])(behind, factor_perms).reshape(-1, n)
    L = vmap(lambda b, p: b[p])(behind, factor_row_perms).reshape(-1, n)
    # R = vmap(lambda a, p: jnp.tile(a.T, (n, 1))[p])(ahead, factor_perms).reshape(-1, n)
    R = vmap(lambda a, p: a.T[p])(ahead, factor_col_perms).reshape(-1, n)
    fisher_exact = (L @ L.T) * (R @ R.T)
    return fisher_exact if not return_factor else fisher_exact, PA

F = fisher_numerical(twiddle, x, key_y_t)
F_exact, PA = fisher_exact(twiddle, x, return_factor=True)

print(jnp.linalg.norm(F - F_exact, 2) / jnp.linalg.norm(F, 2))
print(jnp.linalg.norm(F - F_exact, 'fro') / jnp.linalg.norm(F, 'fro'))

# for i in range(log_n):
#     for j in range(log_n):
#         print((i, j))
#         print(jnp.nonzero(fisher_exact_list[i][j]))

def check_pinv(A, A_pinv):
    AAp = A @ A_pinv
    ApA = A_pinv @ A
    return (jnp.linalg.norm(AAp @ A - A), jnp.linalg.norm(ApA @ A_pinv - A_pinv),
            jnp.linalg.norm(AAp.T - AAp), jnp.linalg.norm(ApA.T - ApA))


# F_exact_pinv = jnp.linalg.pinv(F_exact)
# U, S, _ = jnp.linalg.svd(PA, full_matrices=False)
# # (S > 1e-3).sum()
# # This seems to have rank (log_n + 1) n
# rank = (log_n + 1) * n
# # F_svd_pinv = U[:, :rank] @ jnp.diag(1.0 / S[:rank]**2) @ U.T[:rank]
# F_svd_pinv = (U[:, :rank] / S[:rank]**2) @ U.T[:rank]
# print([float(e) for e in check_pinv(F_exact, F_exact_pinv)])
# print([float(e) for e in check_pinv(F_exact, F_svd_pinv)])
