import math
from typing import List, Tuple, Union

import numpy as np
import scipy.linalg

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


def invert(perm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Get the inverse of a given permutation vector.
    Equivalent to converting a permutation vector from left-multiplication format to right
    multiplication format.
    Work with both numpy array and Pytorch Tensor.
    """
    assert isinstance(perm, (np.ndarray, torch.Tensor))
    n = perm.shape[-1]
    if isinstance(perm, np.ndarray):
        result = np.empty(n, dtype=int)
        result[perm] = np.arange(n, dtype=int)
    else:
        result = torch.empty(n, dtype=int, device=perm.device)
        result[perm] = torch.arange(n, dtype=int)
    return result


def perm_vec_to_mat(p: np.ndarray, left: bool = False) -> np.ndarray:
    """Convert a permutation vector to a permutation matrix.
    Parameters:
        p: a vector storing the permutation.
        left: whether it's in left- or right-multiplication format.
    """
    n = len(p)
    matrix = np.zeros((n, n), dtype=int)
    matrix[p, np.arange(n, dtype=int)] = 1
    # Left-multiplication by the resulting matrix will result in the desired permutation.
    return matrix if not left else matrix.T


def perm_mat_to_vec(m, left=False):
    """Convert a permutation matrix to a permutation vector.
    Parameters:
        p: a matrix storing the permutation.
        left: whether it's in left- or right-multiplication format.
    """
    input = np.arange(m.shape[0])
    return m @ input if left else m.T @ input


def is_2x2_block_diag(mat: np.ndarray) -> bool:
    """Check that each of the 4 blocks of a matrix is diagonal
    (in other words, that the matrix is a butterfly factor).
    Assumes that the matrix is square with even dimension.
    """
    nh = mat.shape[0] // 2
    for block in [mat[:nh, :nh], mat[:nh, nh:], mat[nh:, :nh], mat[nh:, nh:]]:
        if np.count_nonzero(block - np.diag(np.diagonal(block))):
            return False  # there's a nonzero off-diagonal entry
    return True


def is_butterfly_factor(mat: np.ndarray, k: int) -> bool:
    """Checks whether "mat" is in B_k.
    """
    assert k > 1 and k == 1 << int(math.log2(k))
    n = mat.shape[0]
    assert n >= k and n == 1 << int(math.log2(n))
    z = np.zeros(mat.shape)
    for i in range(n//k):
        # Iterate through each diagonal block of the matrix,
        # and check that it is a butterfly factor.
        block = mat[i * k:(i + 1) * k, i * k:(i + 1) * k]
        if not is_2x2_block_diag(block):
            return False
        z[i * k:(i + 1) * k, i * k:(i + 1) * k] = block
    # Check whether there are any nonzeros in off-diagonal blocks.
    return np.count_nonzero(mat - z) == 0


def matrix_to_butterfly_factor(mat, log_k, pytorch_format=False, check_input=False):
    """Converts a matrix to a butterfly factor B_k.
    Assumes that it indeed has the correct sparsity pattern.
    """
    k = 1 << log_k
    if check_input:
        assert is_butterfly_factor(mat, k)
    n = mat.shape[0]
    out = np.zeros((n // 2, 2, 2))
    for block in range(n // 2):
        base = (2 * block // k) * k + (block % (k // 2))
        for i, j in np.ndindex((2, 2)):
            out[block, i, j] = mat[base + i * k // 2, base + j * k//2]
    if pytorch_format:
        out = torch.tensor(out, dtype=torch.float32)
    return out


class Node:
    def __init__(self, value):
        self.value = value
        self.in_edges = []
        self.out_edges = []


def half_balance(
    v: np.ndarray, return_butterfly_factor: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Return the permutation vector that makes the permutation vector v
    n//2-balanced. Directly follows the proof of Lemma G.2.
    Parameters:
        v: the permutation as a vector, stored in right-multiplication format.
    """
    n = len(v)
    assert n % 2 == 0
    nh = n // 2
    nodes = [Node(i) for i in range(nh)]
    # Build the graph
    for i in range(nh):
        # There is an edge from s to t
        s, t = nodes[v[i] % nh], nodes[v[i + nh] % nh]
        s.out_edges.append((t, i))
        t.in_edges.append((s, i + nh))
    # Each node has undirected degree exactly 2
    assert all(len(node.in_edges) + len(node.out_edges) == 2 for node in nodes)
    swapped_low_locs = []
    swapped_high_locs = []
    while len(nodes):
        # Pick a random node.
        start_node, start_loc = nodes[-1], n - 1
        next_node = None
        # Follow undirected edges until rereaching start_node.
        # As every node has undirected degree 2, this will find
        # all cycles in the graph. Reverse edges as needed to
        # make the cycle a directed cycle.
        while next_node != start_node:
            if next_node is None:
                next_node, next_loc = start_node, start_loc
            old_node, old_loc = next_node, next_loc
            if old_node.out_edges:
                # If there's an out-edge from old_node, follow it.
                next_node, old_loc = old_node.out_edges.pop()
                next_loc = old_loc + nh
                next_node.in_edges.remove((old_node, next_loc))
            else:
                # If there's no out-edge, there must be an in-edge.
                next_node, old_loc = old_node.in_edges.pop()
                next_loc = old_loc - nh
                next_node.out_edges.remove((old_node, next_loc))
                swapped_low_locs.append(next_loc)
                swapped_high_locs.append(old_loc)
            nodes.remove(old_node)
    if not return_butterfly_factor:
        perm = np.arange(n, dtype=int)
        perm[swapped_low_locs], perm[swapped_high_locs] = swapped_high_locs, swapped_low_locs
        return perm
    else:
        twiddle = torch.eye(2).expand(n // 2, 2, 2).contiguous()
        swap_matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
        twiddle[swapped_low_locs] = swap_matrix.unsqueeze(0)
        return twiddle


def modular_balance(v: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Returns the sequence of permutations to transform permutation vector v
    into a modular-balanced matrix, as well as the resultant
    modular-balanced permutation vector. Directly follows the proof of
    Lemma G.3.
    Parameters:
        v: a permutation vector corresponding to a permutation matrix P, stored in
            right-multiplication format.
    """
    t = n = len(v)
    perms = []
    while t >= 2:
        # Balance each chunk of the vector independently.
        chunks = np.split(v, n // t)
        # Adjust indices of the permutation
        swap_perm = np.hstack([half_balance(chunk) + chunk_idx * t
                               for chunk_idx, chunk in enumerate(chunks)])
        v = v[swap_perm]
        perms.append(swap_perm)
        t //= 2
    return perms, v


def is_modular_balanced(perm):
    """Corresponds to Definition G.1 in the paper.
    perm is stored in right-multiplication format, either as a vector or a matrix.
    """
    if isinstance(perm, np.ndarray) and len(perm.shape) > 1:
        perm = perm_mat_to_vec(perm)
    n = len(perm)
    log_n = int(math.log2(n))
    assert n == 1 << log_n
    for j in (1 << k for k in range(1, log_n + 1)):
        for chunk in range(n // j):
            mod_vals = set(perm[i] % j for i in range(chunk * j, (chunk + 1) * j))
            if len(mod_vals) != j:
                return False
    return True


def modular_balanced_to_butterfly_factor(L: np.ndarray) -> List[np.ndarray]:
    """Returns a sequence of butterfly factors that, when multiplied together, create L.
    Assumptions: L is a modular-balanced permutation matrix.
    Directly follows the proof of Lemma G.1.
    Optimized for readability, not efficiency.
    Parameters:
        L: a modular-balanced permutation matrix, stored in the right-multiplication format.
            (i.e. applying L to a vector x is equivalent to x @ L).
            Can also be stored as a vector (again in right-multiplication format).
    Return:
        butterflies: a list of butterfly factors, stored as matrices (not in twiddle format).
            The matrices are permutation matrices stored in right-multiplication format.
    """
    if isinstance(L, list) or len(L.shape) == 1:
        L = perm_vec_to_mat(L)
    n = L.shape[0]
    if n == 2:
        return [L.copy()]  # L is its own inverse, and is already a butterfly.
    nh = n//2
    L1 = L[:nh, :nh] + L[nh:, :nh]
    L2 = L[:nh, nh:] + L[nh:, nh:]
    perms = []
    Lp = scipy.linalg.block_diag(L1, L2)
    # By construction, Bn @ Lp = L.
    Bn = L @ Lp.T
    perms1 = modular_balanced_to_butterfly_factor(L1)
    perms2 = modular_balanced_to_butterfly_factor(L2)
    # Combine the individual permutation matrices of size n/2
    # into a block-diagonal permutation matrix of size n.
    return [Bn] + [scipy.linalg.block_diag(p1, p2) for p1, p2 in zip(perms1, perms2)]


def perm2butterfly(v: Union[np.ndarray, torch.Tensor],
                   increasing_stride: bool = False) -> Butterfly:
    """
    Parameter:
        v: a permutation, stored as a vector, in left-multiplication format.
            (i.e., applying v to a vector x is equivalent to x[p])
        increasing_stride: whether the returned Butterfly should have increasing_stride=False or
            True. False corresponds to Lemma G.3 and True corresponds to Lemma G.6.
    Return:
        b: a Butterfly that performs the same permutation as v.
    """
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    n = len(v)
    log_n = int(math.ceil(math.log2(n)))
    if n < 1 << log_n:  # Pad permutation to the next power-of-2 size
        v = np.concatenate([v, np.arange(n, 1 << log_n)])
    if increasing_stride:  # Follow proof of Lemma G.6
        br = bitreversal_permutation(1 << log_n)
        b = perm2butterfly(br[v[br]], increasing_stride=False)
        b.increasing_stride=True
        br_half = bitreversal_permutation((1 << log_n) // 2, pytorch_format=True)
        with torch.no_grad():
            b.twiddle.copy_(b.twiddle[:, :, :, br_half])
        b.in_size = b.out_size = n
        return b
    # modular_balance expects right-multiplication format so we convert the format of v.
    Rinv_perms, L_vec = modular_balance(invert(v))
    L_perms = list(reversed(modular_balanced_to_butterfly_factor(L_vec)))
    R_perms = [perm_vec_to_mat(invert(p), left=True) for p in reversed(Rinv_perms)]
    # Stored in increasing_stride=True twiddle format.
    # Need to take transpose because the matrices are in right-multiplication format.
    L_twiddle = torch.stack([matrix_to_butterfly_factor(l.T, log_k=i+1, pytorch_format=True)
                             for i, l in enumerate(L_perms)])
    # Stored in increasing_stride=False twiddle format so we need to flip the order
    R_twiddle = torch.stack([matrix_to_butterfly_factor(r, log_k=i+1, pytorch_format=True)
                             for i, r in enumerate(R_perms)]).flip([0])
    b = Butterfly(n, n, bias=False, increasing_stride=False, nblocks=2)
    with torch.no_grad():
        b.twiddle.copy_(torch.stack([R_twiddle, L_twiddle]).unsqueeze(0))
    return b
