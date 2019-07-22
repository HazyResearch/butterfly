from collections import defaultdict
import cvxpy as cp
import numpy as np
import torch

from butterfly.butterfly import Butterfly


## Helper functions.

def gen_power_matrix(n):
    '''Suppose we have butterfly factors B_2,B_4,...,B_n.
    Let the (2n log2 n)-dimensional vector x denote the nonzero entries
    of the factors, in order. Let v be the matrix product (B_nB_{n/2}...B_2),
    flattened into a vector in row-major order. Note that each entry of v
    is a monomial in the entries of x.
    The matrix that this function generates has a 1 in entry (i,j) if v_i
    depends on x_j, and a 0 otherwise.
    '''
    logn = int(round(np.log2(n)))
    A = np.zeros((n**2, 2*n*logn))
    for s in range(n**2):
        for k in range(logn):
            i = s // n
            j = s % n
            kp = 2**(k+1)
            ind = 2*n*k + 2*(i%kp + 1 + kp*(j//kp)) + (j % kp)//(2**k) - 1
            A[s, ind-1] = 1
    return A.astype(np.int)

def f2lineq(A, b, suppress=False):
    '''Solve Ax = b for x over the field F_2.'''
    b = b.reshape(len(b), 1)
    m_a, n_a = A.shape
    aa = np.hstack((A, b))
    aa_flat = aa.flatten()
    m_aa, n_aa = aa.shape

    row_idx = 0
    col_idx = 0
    row_store = []
    col_store = []

    while (row_idx < m_aa) and (col_idx < n_aa-1):
        while aa[row_idx,col_idx] == 0 and col_idx < n_aa-1:
            idx = np.where(aa[row_idx:m_aa, col_idx] != 0)[0]
            if len(idx) == 0:
                col_idx += 1
            else:
                idx2 = np.where(aa[row_idx:m_aa, col_idx] == 1)[0]
                idx = np.hstack((idx2, idx))
                idx = idx[0]
                r1 = aa[row_idx,:].copy()
                r2 = aa[row_idx+idx,:].copy()
                aa[row_idx,:], aa[row_idx+idx,:] = r2, r1
        if aa[row_idx,col_idx] != 0 and col_idx < n_aa:
            row_store.append(row_idx)
            col_store.append(col_idx)
            idx = np.where(aa[:,col_idx] != 0)[0]
            for i in idx:
                if i != row_idx:
                    aa[i,:] = (aa[i,:] + aa[row_idx,:] * (2 - aa[i,col_idx])) % 2
            col_idx += 1
        row_idx += 1

    if (np.linalg.matrix_rank(aa) > np.linalg.matrix_rank(aa[:,:n_a])):
        if not suppress: print('Warning: No solution')
        x = np.array([])
    else:
        x = np.zeros((n_a, 1))
        x[col_store,0] = aa[row_store,n_aa-1]
    return x.flatten()

def find_connected_components(A):
    '''
    The input A is a binary mxn matrix.
    Any two numbers i,j such that A(k,i) = A(k,j) = 1 for some k
    are connected by an edge.
    This function finds all connected components of A. It returns
    the node set of each component as well as the associated rows of A.
    '''
    class Node:
        def __init__(self, id):
            self.id = id
            self.neighbors = set()
            self.rows = set()
    
    m, n = A.shape
    
    all_nodes = [Node(i) for i in range(n)]
    for k in range(m):
        nonzeros = np.nonzero(A[k])[0]
        for i, nz1 in enumerate(nonzeros):
            all_nodes[nz1].rows.add(k)
            for nz2 in nonzeros[i+1:]:
                all_nodes[nz1].neighbors.add(nz2)
                all_nodes[nz2].neighbors.add(nz1)

    components = []
    rows = []
    untraversed = set(range(n))
    while untraversed:
        v = all_nodes[untraversed.pop()]
        cur_comp = [v]
        ids = set()
        cur_rows = set()
        while cur_comp:
            v = cur_comp.pop()
            ids.add(v.id)
            untraversed.discard(v.id)
            cur_rows.update(v.rows)
            cur_comp.extend([all_nodes[n] for n in v.neighbors if n not in ids])
        components.append(sorted(list(ids)))
        rows.append(sorted(list(cur_rows)))
    return components, rows

def solve_SOS_SDP(pows, M):
    # Set up and solve the sum-of-squares SDP, using the given set of monomials
    # represented by the matrix "pows," and the given flat matrix M.

    zero_constraints = defaultdict(list)
    for i in range(len(pows)):
        for j in range(i, len(pows)):
            total_pow = tuple(pows[i] + pows[j])
            if i != j:  # All cross terms should be zero.
                zero_constraints[total_pow].append((i+1, j+1))
    zero_constraints = zero_constraints.values()

    N = pows.shape[0] + 1
    frobsq = np.sum(M**2)
    Q = cp.Variable((N,N), symmetric=True)
    gamma = cp.Variable()
    constraints = [Q >> 0, Q[0,0] == frobsq-gamma]
    for zc in zero_constraints:
        # Constrain any monomial that does not appear in f to equal 0.
        constraints.append(sum([Q[t] for t in zc]) == 0)
    for i in range(1, N):
        # Now, iterate through all monomials that appear in the product matrix.
        # The coefficient of the square of the monomial should just be 1,
        # while the coefficient of the monomial itself should be -2M_{ij}.
        # Accounting for symmetry, this means that the corresponding off-diagonal
        # matrix entries should both be -M_{ij}.
        constraints.append(Q[i, i] == 1)
    constraints.append(Q[0, 1:] == -M)
    
    prob = cp.Problem(cp.Maximize(gamma), constraints)
    prob.solve(eps=1e-8, max_iters=25000)
    return prob, Q.value, gamma.value

def butterfly_recover(pows, vals, tol=1e-5):
    '''Given monomials represented by the matrix "pows" and their values,
       recover the values of the actual variables.
       To do this, we take the logarithm of the absolute value of each equation
       and solve a linear system (with "pows" as the matrix in question)
       for the absolute values. We also solve a linear system over F_2
       for the signs of the variables.
       Zero variables (within the specified tolerance) are handled separately.'''
    v = vals
    x = np.zeros(pows.shape[1])
    # Exclude zero entries (corresponding variables will also be 0)
    nonzero_rows = np.where(np.abs(v) > tol)[0]
    v = v[nonzero_rows]
    pows = pows[nonzero_rows, :]
    nonzero_cols = np.where(np.sum(pows, axis=0) > 0)[0]
    pows = pows[:, nonzero_cols]
    pow_sum = np.sum(pows, axis=1)
    assert (np.all(pow_sum == pow_sum[0]))  # All monomials should have the same degree
    pow_sum = pow_sum[0]

    # Solve for the magnitudes of the butterfly factor entries.
    rhs = np.log(np.abs(v))
    # Note: by construction, "rhs" is in the range of "pows".
    abs_x = np.exp(np.linalg.lstsq(pows, rhs, rcond=None)[0])

    # Solve for the signs of the butterfly factor entries.
    sign_rhs = ((np.sign(v)+1)/2 + 1 + pow_sum) % 2
    s = f2lineq(pows, sign_rhs)*2 - 1
    # Compute the butterfly factor entries from their magnitudes and signs.
    x_restricted = abs_x * s
    x[nonzero_cols] = x_restricted
    return x

def bit_reversal_perm(n):
    # Adapted from "Fast Bit-Reversal Algorithms" by Anne Cathrine Elster
    t = int(round(np.log2(n)))
    assert (2 ** t == n)
    L = 1
    p = [0]*n
    for q in range(t):
        n = n // 2
        for j in range(L):
            p[L+j] = p[j] + n
        L *= 2
    return p

def gen_rand_butterfly(n):
    # Generate a random n x n butterfly matrix.
    B = np.eye(n)
    logn = int(round(np.log2(n)))
    factors = []
    for k in range(1, logn+1):
        # Generate a random block-2x2-diagonal matrix
        M = np.zeros((n, n))
        for b in range(n//2):
            M[b*2:(b+1)*2,b*2:(b+1)*2] = np.random.randn(2, 2)
        m = 2**k
        bitrev = np.array(bit_reversal_perm(m)*(n//m))
        for b in range(n//m):
            bitrev[b*m:(b+1)*m] += b*m
        M = M[bitrev,:]
        M = M[:,bitrev]
        factors.append(M)
        B = M @ B
    return B, factors

def vector_to_butterfly_factors(x, n, transpose=False):
    '''Copy the NumPy vector 'x' into the weights of a butterfly module.'''
    # Flip factor signs so that there are as few negatives as possible (aesthetic reasons only)
    logn = int(round(np.log2(n)))
    last_factors = x[2*n*(logn%2):]
    num_neg, num_pos = np.count_nonzero(last_factors < 0), np.count_nonzero(last_factors > 0)
    if num_neg > num_pos: last_factors *= -1

    butterfly = Butterfly(n, n, bias=False, tied_weight=False, increasing_stride=transpose)
    weight = butterfly.twiddle.data[0]
    w_flat = weight.view(weight.size(0), -1)
    for i, factor in enumerate(w_flat):
        x_section = x[i*2*n:]
        stride = 1 << i
        for j in range(factor.numel()):
            block_idx = j // (stride*4)
            inner_idx = j % (stride*4) // 4
            corner = j % 4
            if transpose and 1 <= corner <= 2:
                corner = 3 - corner  # switch upper right, lower left
            index = block_idx*stride*4 + inner_idx*2 + (corner%2)*(stride*2) + (corner//2)
            factor[j] = x_section[index]
    nonzero_factors = 0
    max_gm = 1  # Geometric mean of max magnitudes of nonzero factors
    for factor in weight:
        f_max = torch.norm(factor, float('inf'))
        if f_max:  # Compute geometric mean of max values
            nonzero_factors += 1
            max_gm *= f_max

    if nonzero_factors: max_gm = max_gm**(1./nonzero_factors)
    for factor in weight:
        # Scale each factor so that the max entry magnitude is the same.
        # Not necessary, but nicer.
        f_max = torch.norm(factor, float('inf'))
        if f_max:
            factor.mul_(max_gm / f_max)

    return butterfly


# Main function.
def butterfly_SOS(M, project_onto='B', verbose=True):
    '''
    Let f(B_2,...,B_n) = ||B_n...B_2 - M||_F^2, where  
    We try to find the maximum value 'gamma' such that (f-gamma) is a
    sum of squares. This implies f >= gamma. We do this by solving an
    SDP: we try to maximize gamma subject to the requirement f = z^T (Q-gamma*E1) z,
    where E1 is the matrix that is 1 in the upper left corner and 0 elsewhere,
    (Q-gamma E1) is constrained to be symmetric and positive semidefinite, and
    z is a vector of monomials in the entries of B_2,...,B_n.
    The monomials in z are precisely the monomials which appear in the
    product matrix B_n...B_2, in row-major order
    [along with the constant monomial 1, which is the first entry of z].
    '''
    assert (len(M.shape) == 2 and M.shape[0] == M.shape[1])
    n = M.shape[0]
    logn = int(round(np.log2(n)))
    assert (2 ** logn == n)
    m_max = np.amax(np.abs(M))
    if m_max > 1:
        # Normalize by the max entry (seems to help convergence if entries are large)
        M = M / m_max
    if project_onto == 'BT': M = M.T
    pows_full = gen_power_matrix(n)
    components, rows = find_connected_components(pows_full)
    x_full = np.zeros(2*n*logn)

    for eqn, (component, rowset) in enumerate(zip(components, rows)):
        # Restrict the matrix to the current set of entries.
        pows_restricted = pows_full[rowset,:]
        pows_restricted = pows_restricted[:,component]

        M_flat = M.flatten()[rowset]
        frobsq = np.sum(M_flat**2)
        if np.count_nonzero(M_flat) == 0:
            continue  # all entries 0

        prob, Q, gamma = solve_SOS_SDP(pows_restricted, M_flat)
        if verbose:
            niter = prob.solver_stats.num_iters
            print(f'SDP #{eqn+1} optimal value is {gamma:.3e} ({niter} iterations taken)')
        
        '''
        We've now found the maximum gamma such that
        Q-gamma*E1 is SPSD and f = z^T(Q-gamma*E1)z;
        gamma is an upper bound on the lower bound of f.
        Now, we want to extract a value of z such that
        f attains the value gamma. This is simply the zero eigenvector
        of Q, normalized so that the first entry is 1.
        This gives us the values of the entries of the product matrix.
        '''
        D, V = np.linalg.eig(Q)
        zero_ev_ind = np.argmin(D)  # should be the index of the zero eigenvalue
        zero_ev = V[:,zero_ev_ind]
        zero_ev /= zero_ev[0]
        v = zero_ev[1:]
        # Get the factor entries from the product matrix entries.
        x = butterfly_recover(pows_restricted, v)
        x_full[component] = x

    # Rescale appropriately.
    if m_max > 1:
        M *= m_max
        x_full = x_full * (m_max**(1./np.log2(n)))
    butterfly = vector_to_butterfly_factors(x_full, n, transpose=(project_onto == 'BT'))
    if project_onto == 'BT': M = M.T  # undo transposition

    M_recovered = butterfly(torch.eye(n)).detach().cpu().numpy()  # This will be the closest butterfly matrix to 'M'.

    return butterfly, M_recovered
