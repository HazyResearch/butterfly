import copy

import torch
from torch import nn
from torch.nn import functional as F

from torch_butterfly import Butterfly
from torch_butterfly.permutation import FixedPermutation
from torch_butterfly.complex_utils import view_as_real, view_as_complex


def diagonal_butterfly(butterfly: Butterfly,
                       diagonal: torch.Tensor,
                       diag_first: bool,
                       inplace: bool = True) -> Butterfly:
    """
    Combine a Butterfly and a diagonal into another Butterfly.
    Only support nstacks==1 for now.
    Parameters:
        butterfly: Butterfly(in_size, out_size)
        diagonal: size (in_size,) if diag_first, else (out_size,). Should be of type complex
            if butterfly.complex == True.
        diag_first: If True, the map is input -> diagonal -> butterfly.
            If False, the map is input -> butterfly -> diagonal.
        inplace: whether to modify the input Butterfly
    """
    assert butterfly.nstacks == 1
    assert butterfly.bias is None
    twiddle = (butterfly.twiddle.clone() if not butterfly.complex else
               view_as_complex(butterfly.twiddle).clone())
    n = 1 << twiddle.shape[2]
    if diagonal.shape[-1] < n:
        diagonal = F.pad(diagonal, (0, n - diagonal.shape[-1]), value=1)
    if diag_first:
        if butterfly.increasing_stride:
            twiddle[:, 0, 0, :, :, 0] *= diagonal[::2].unsqueeze(-1)
            twiddle[:, 0, 0, :, :, 1] *= diagonal[1::2].unsqueeze(-1)
        else:
            n = diagonal.shape[-1]
            twiddle[:, 0, 0, :, :, 0] *= diagonal[:n // 2].unsqueeze(-1)
            twiddle[:, 0, 0, :, :, 1] *= diagonal[n // 2:].unsqueeze(-1)
    else:
        # Whether the last block is increasing or decreasing stride
        increasing_stride = butterfly.increasing_stride != ((butterfly.nblocks - 1) % 2 == 1)
        if increasing_stride:
            n = diagonal.shape[-1]
            twiddle[:, -1, -1, :, 0, :] *= diagonal[:n // 2].unsqueeze(-1)
            twiddle[:, -1, -1, :, 1, :] *= diagonal[n // 2:].unsqueeze(-1)
        else:
            twiddle[:, -1, -1, :, 0, :] *= diagonal[::2].unsqueeze(-1)
            twiddle[:, -1, -1, :, 1, :] *= diagonal[1::2].unsqueeze(-1)
    out_butterfly = butterfly if inplace else copy.deepcopy(butterfly)
    with torch.no_grad():
        out_butterfly.twiddle.copy_(twiddle if not butterfly.complex else view_as_real(twiddle))
    return out_butterfly


def butterfly_product(butterfly1: Butterfly, butterfly2: Butterfly) -> Butterfly:
    """
    Combine product of two butterfly matrices into one Butterfly.
    """
    assert butterfly1.bias is None and butterfly2.bias is None
    assert butterfly1.complex == butterfly2.complex
    assert butterfly1.nstacks == butterfly2.nstacks
    assert butterfly1.log_n == butterfly2.log_n
    b1_end_increasing_stride = butterfly1.increasing_stride != (butterfly1.nblocks % 2 == 1)
    if b1_end_increasing_stride != butterfly2.increasing_stride:
        # Need to insert an Identity block
        identity = Butterfly(butterfly1.in_size, butterfly1.out_size, bias=False,
                             complex=butterfly1.complex,
                             increasing_stride=b1_end_increasing_stride, init='identity')
        butterfly1 = butterfly_product(butterfly1, identity)
    b = Butterfly(1 << butterfly1.log_n, 1 << butterfly1.log_n, bias=False,
                  complex=butterfly1.complex, increasing_stride=butterfly1.increasing_stride,
                  nblocks=butterfly1.nblocks + butterfly2.nblocks).to(butterfly1.twiddle.device)
    b.in_size = butterfly1.in_size
    b.out_size = butterfly2.out_size
    with torch.no_grad():
        # Don't need view_as_complex here since all the twiddles are stored in real.
        b.twiddle.copy_(torch.cat((butterfly1.twiddle, butterfly2.twiddle), dim=1))
    return b


class TensorProduct(nn.Module):

    def __init__(self, map1, map2) -> None:
        """Perform map1 on the last dimension of the input and then map2 on the next
        to last dimension.
        """
        super().__init__()
        self.map1 = map1
        self.map2 = map2

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameter:
            input: (*, n2, n1)
        Return:
            output: (*, n2, n1)
        """
        out = self.map1(input)
        return self.map2(out.transpose(-1, -2)).transpose(-1, -2)


class Flatten2D(nn.Module):
    """Combine the last 2 dimensions of the input. """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameter:
            input: (*, n2, n1)
        Return:
            output: (*, n2 * n1)
        """
        return input.reshape(*input.shape[:-2], input.shape[-2] * input.shape[-1])


class Unflatten2D(nn.Module):
    """Reshape the last dimension of the input into 2 dimensions. """

    def __init__(self, last_dim: int) -> None:
        super().__init__()
        self.last_dim = last_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameter:
            input: (*, n2 * n1)
        Return:
            output: (*, n2, n1)
        """
        return input.reshape(*input.shape[:-1], -1, self.last_dim)


def butterfly_kronecker(butterfly1: Butterfly, butterfly2: Butterfly) -> Butterfly:
    """Combine two butterflies of size n1 and n2 into their Kronecker product of size n1 * n2.
    They must both have increasing_stride=True or increasing_stride=False.
    If butterfly1 or butterfly2 has padding, then the kronecker product (after flattening input)
    will not produce the same result unless the input is padding in the same way before flattening.

    Only support nstacks==1, nblocks==1 for now.
    """
    assert butterfly1.increasing_stride == butterfly2.increasing_stride
    assert butterfly1.complex == butterfly2.complex
    assert not butterfly1.bias and not butterfly2.bias
    assert butterfly1.nstacks == 1 and butterfly2.nstacks == 1
    assert butterfly1.nblocks == 1 and butterfly2.nblocks == 1
    increasing_stride = butterfly1.increasing_stride
    complex = butterfly1.complex
    log_n1 = butterfly1.twiddle.shape[2]
    log_n2 = butterfly2.twiddle.shape[2]
    log_n = log_n1 + log_n2
    n = 1 << log_n
    twiddle1 = butterfly1.twiddle if not complex else view_as_complex(butterfly1.twiddle)
    twiddle2 = butterfly2.twiddle if not complex else view_as_complex(butterfly2.twiddle)
    twiddle1 = twiddle1.repeat(1, 1, 1, 1 << log_n2, 1, 1)
    twiddle2 = twiddle2.repeat_interleave(1 << log_n1, dim=3)
    twiddle = (torch.cat((twiddle1, twiddle2), dim=2) if increasing_stride else
               torch.cat((twiddle2, twiddle1), dim=2))
    b = Butterfly(n, n, bias=False, complex=complex,
                  increasing_stride=increasing_stride).to(twiddle.device)
    b.in_size = butterfly1.in_size * butterfly2.in_size
    b.out_size = butterfly1.out_size * butterfly2.out_size
    with torch.no_grad():
        b_twiddle = b.twiddle if not complex else view_as_complex(b.twiddle)
        b_twiddle.copy_(twiddle)
    return b


def permutation_kronecker(perm1: FixedPermutation, perm2: FixedPermutation) -> FixedPermutation:
    """Combine two permutations of size n1 and n2 into their Kronecker product of size n1 * n2.
    """
    n1, n2 = perm1.permutation.shape[-1], perm2.permutation.shape[-1]
    x = torch.arange(n2 * n1, device=perm1.permutation.device).reshape(n2, n1)
    perm = perm2(perm1(x).t()).t().reshape(-1)
    return FixedPermutation(perm)
