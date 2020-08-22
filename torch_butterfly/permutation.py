import math

import numpy as np

import torch
from torch import nn

from torch_butterfly import Butterfly
from torch_butterfly.complex_utils import index_last_dim


def bitreversal_permutation(n, pytorch_format=False):
    """Return the bit reversal permutation used in FFT.
    By default, the permutation is stored in numpy array.
    Parameter:
        n: integer, must be a power of 2.
        pytorch_format: whether the permutation is stored as numpy array or pytorch tensor.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(log_n):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    perm = perm.squeeze(0)
    return perm if not pytorch_format else torch.tensor(perm)


def wavelet_permutation(n, pytorch_format=False):
    """Return the bit reversal permutation used in discrete wavelet transform.
    Example: [0, 1, ..., 7] -> [0, 4, 2, 6, 1, 3, 5, 7]
    By default, the permutation is stored in numpy array.
    Parameter:
        n: integer, must be a power of 2.
        pytorch_format: whether the permutation is stored as numpy array or pytorch tensor.
    Return:
        perm: numpy array of size n
    """
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = np.arange(n)
    head, tail = perm[:], perm[:0]  # empty tail
    for i in range(log_n):
        even, odd = head[::2], head[1::2]
        head = even
        tail = np.hstack((odd, tail))
    perm = np.hstack((head, tail))
    return perm if not pytorch_format else torch.tensor(perm)


class FixedPermutation(nn.Module):

    def __init__(self, permutation: torch.Tensor) -> None:
        """Fixed permutation.
        Parameter:
            permutation: (n, ) tensor of ints
        """
        super().__init__()
        self.register_buffer('permutation', permutation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input: (batch, *, size)
        Return:
            output: (batch, *, size)
        """
        # return input[..., self.permutation]
        # Pytorch 1.6.0 doesn't have indexing_backward for complex on GPU.
        # So we use our own backward
        return index_last_dim(input, self.permutation)


def Identity(n, increasing_stride=True):
    b = Butterfly(n, n, bias=False, increasing_stride=increasing_stride)
    bd = b.twiddle.data
    bd.zero_()
    for block in bd.view(-1, 2, 2):
        block[0, 0] = block[1, 1] = 1
        block[0, 1] = block[1, 0] = 0
    return b

def perm_vec_to_mat(p, left=False):
    # Convert a permutation vector to a permutation matrix.
    # Mostly for debugging purposes.
    m = np.zeros((len(p), len(p)))
    for i in range(len(p)):
        m[p[i], i] = 1
    if left:
        # Left-multiplication by the resulting matrix
        # will result in the desired permutation.
        m = m.T
    return m

def perm_mat_to_vec(m, left=False):
    # Convert a permutation matrix to a permutation vector.
    # Mostly for debugging purposes.
    inp = np.arange(m.shape[0])
    if left:
        return m @ inp
    else:
        return m.T @ inp

def permute(v, perm):
    # Permute an input vector according to the desired permutation vector.
    # The i'th element of the output is simply v[perm[i]]. Equivalent to
    # multiplying by the corresponding permutation matrix.
    to_list = type(v) == list
    v = np.array(v)
    perm = list(np.array(perm, dtype=np.int32))
    if len(v.shape) == 1:
        v = v[perm]
    else:
        v = v[perm, :]
    if to_list:
        v = list(v)
    return v

def invert(perm):
    # Get the inverse of a given permutation vector.
    # Works with both numpy array and Pytorch Tensor
    assert isinstance(perm, (np.ndarray, torch.Tensor))
    n = perm.shape[-1]
    if isinstance(perm, np.ndarray):
        result = np.empty(n, dtype=int)
        result[perm] = np.arange(n, dtype=int)
    else:
        result = torch.empty(n, dtype=int, device=perm.device)
        result[perm] = torch.arange(n, dtype=int)
    return result

def block_diag(mat):
    # Check that each of the 4 blocks of a matrix is diagonal
    # (in other words, that the matrix is a butterfly factor).
    # Assumes that the matrix is square with even dimension.
    nh = mat.shape[0] // 2
    for i, j in np.ndindex((2, 2)):
        block = mat[i*nh:(i+1)*nh,j*nh:(j+1)*nh]
        if np.count_nonzero(block - np.diag(np.diagonal(block))):
            return False  # there's a nonzero off-diagonal entry
    return True

def is_butterfly(mat, k):
    # Checks whether "mat" is in B_k.
    assert (k > 1 and int(np.round(2**(np.log2(k)))) == k)
    n = mat.shape[0]
    assert (n >= k and int(np.round(2**(np.log2(n)))) == n)
    z = np.zeros(mat.shape)
    for i in range(n//k):
        # Iterate through each diagonal block of the matrix,
        # and check that it is a butterfly factor.
        block = mat[i*k:(i+1)*k,i*k:(i+1)*k]
        if not block_diag(block):
            return False
        z[i*k:(i+1)*k,i*k:(i+1)*k] = block
    # Check whether there are any nonzeros in off-diagonal blocks.
    return np.count_nonzero(mat - z) == 0

def to_butterfly(mat, k=None, logk=None, pytorch=False, check=False):
    # Converts a matrix to a butterfly factor B_k.
    # Assumes that it indeed has the correct sparsity pattern.
    assert ((k is not None) != (logk is not None))
    if logk is not None:
        k = 2**logk
    if check:
        assert (is_butterfly(mat, k))
    n = mat.shape[0]
    out = np.zeros((n//2, 2, 2))
    for block in range(n//2):
        base = (2*block//k) * k + (block % (k//2))
        for i, j in np.ndindex((2, 2)):
            out[block, i, j] = mat[base + j*k//2, base + i*k//2]
    if check:
        b = Identity(n)
        b.twiddle.data[0, 0, int(round(np.log2(k)))-1].copy_(torch.tensor(out))
        result = b(torch.eye(n))
        assert (torch.norm(result - torch.tensor(mat).float()).item() < 1e-6)
    if pytorch:
        out = torch.tensor(out).float()
    return out

class Node:
    def __init__(self, value):
        self.value = value
        self.in_edges = []
        self.out_edges = []
        self.locations = []

def half_balance(v):
    # Return the permutation vector that makes the permutation vector v
    # n//2-balanced. Directly follows the proof of Lemma D.2.
    assert (len(v) % 2 == 0)
    vh = len(v) // 2
    perm = list(range(len(v)))
    nodes = [Node(i) for i in range(vh)]
    # Build the graph
    for i in range(vh):
        # There is an edge from s to t
        s, t = nodes[v[i] % vh], nodes[v[i+vh] % vh]
        s.out_edges.append((t, i))
        t.in_edges.append((s, i+vh))
    while len(nodes):
        # Pick a random node.
        start_node, start_loc = nodes[-1], len(v) - 1
        next_node = None
        # Follow undirected edges until rereaching start_node.
        # As every node has undirected degree 2, this will find
        # all cycles in the graph. Reverse edges as needed to
        # make the cycle a directed cycle.
        while next_node != start_node:
            if next_node is None:
                next_node, next_loc = start_node, start_loc
            old_node, old_loc = next_node, next_loc
            if len(old_node.out_edges):
                # If there's an out-edge from old_node, follow it.
                next_node, old_loc = old_node.out_edges.pop()
                next_loc = old_loc + vh
                next_node.in_edges.remove((old_node, next_loc))
            else:
                # If there's no out-edge, there must be an in-edge.
                next_node, old_loc = old_node.in_edges.pop()
                next_loc = old_loc - vh
                next_node.out_edges.remove((old_node, next_loc))
                perm[old_loc], perm[next_loc] = perm[next_loc], perm[old_loc]
            nodes.remove(old_node)
    return perm

def modular_balance(v):
    # v is a permutation vector corresponding to a permutation matrix P.
    # Returns the sequence of permutations to transform v
    # into a modular-balanced matrix, as well as the resultant
    # modular-banced permutation vector. Directly follows the proof of
    # Lemma D.3.
    v = np.array(v, copy=True)
    t = n = len(v)
    perms = []
    while t >= 2:
        perm = np.arange(n)
        for c in range(n // t):
            # Balance each chunk of the vector (independently).
            chunk = v[c*t:(c+1)*t]
            chunk_p = half_balance(chunk)
            perm[c*t:(c+1)*t] = permute(perm[c*t:(c+1)*t], chunk_p)
            v[c*t:(c+1)*t] = permute(v[c*t:(c+1)*t], chunk_p)
        perms.append(perm)
        t //= 2
    return perms, v

def check_balanced(perm):
    if isinstance(perm, np.ndarray) and len(perm.shape) > 1:
        perm = perm_mat_to_vec(perm)
    n = len(perm)
    j = 2
    while j <= n:
        for chunk in range(n//j):
            mod_vals = set()
            for i in range(chunk*j, (chunk+1)*j):
                mod_vals.add(perm[i] % j)
            if len(mod_vals) != j:
                return False
        j *= 2
    return True

def to_butterflies(L):
    # Returns a sequence of butterflies that, when multiplied together,
    # create L. Assumptions: L is a modular-balanced permutation matrix.
    # Directly follows the proof of Lemma D.1. Optimized for readability,
    # not efficiency.
    if isinstance(L, list) or len(L.shape) == 1:
        L = perm_vec_to_mat(L)
    n = L.shape[0]
    if n == 2:
        return [L.copy()]  # L is its own inverse.
    nh = n//2
    perms = []
    L1, L2 = np.zeros((nh, nh)), np.zeros((nh, nh))
    for i, j in np.ndindex((nh, nh)):
        L1[i, j] = L[i, j] + L[i+nh, j]
        L2[i, j] = L[i, j+nh] + L[i+nh, j+nh]
    Lp = np.zeros((n, n))
    Lp[:nh, :nh] = L1
    Lp[nh:, nh:] = L2
    # By construction, Bn @ Lp = L.
    Bn = L @ Lp.T
    perms1 = to_butterflies(L1)
    perms2 = to_butterflies(L2)
    for p1, p2 in zip(perms1, perms2):
        # Combine the individual permutation matrices of size n/2
        # into a block-diagonal permutation matrix of size n.
        P = np.zeros((n, n))
        P[:nh,:nh] = p1
        P[nh:,nh:] = p2
        perms.append(P)
    perms.insert(0, Bn)
    return perms

def perm_to_butterfly_module(v, check=False):
    n = len(v)
    v = invert(v)
    Rinv_perms, L_vec = modular_balance(v)
    if check:
        assert (check_balanced(L_vec))
        v2 = np.copy(v)
        for p in Rinv_perms:
            v2 = permute(v2, p)
        assert (list(v2) == list(L_vec))
        lv2 = np.copy(L_vec)
        for p in reversed(Rinv_perms):
            lv2 = permute(lv2, invert(p))
        assert (list(lv2) == list(v))
    # Put in increasing_stride order
    R_perms = [perm_vec_to_mat(p).T for p in reversed(Rinv_perms)]
    # R_perms = [perm_vec_to_mat(p).T for p in Rinv_perms]
    if check:
        mat = perm_vec_to_mat(v, left=False)
        for p in reversed(R_perms):
            mat = mat @ p.T
        assert (np.linalg.norm(mat - perm_vec_to_mat(L_vec, left=False)) < 1e-6)
    L_perms = list(reversed(to_butterflies(L_vec)))
    L_module = Butterfly(n, n, bias=False, increasing_stride=True)
    R_module = Butterfly(n, n, bias=False, increasing_stride=False)
    for i, r in enumerate(R_perms):
        R_module.twiddle.data[0, 0, len(R_perms) - 1 - i].copy_(to_butterfly(r, logk=i+1, pytorch=True, check=check))
    for i, l in enumerate(L_perms):
        L_module.twiddle.data[0, 0, i].copy_(to_butterfly(l, logk=i+1, pytorch=True, check=check))
    module = torch.nn.Sequential(R_module, L_module)
    if check:
        result = module(torch.eye(n))
        assert (torch.norm(result - torch.tensor(perm_vec_to_mat(invert(v))).float()).item() < 1e-6)
    return module
