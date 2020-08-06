import copy

import torch
from torch import nn

from torch_butterfly import Butterfly
from torch_butterfly.permutation import FixedPermutation
from torch_butterfly.complex_utils import view_as_complex


def diagonal_butterfly(butterfly: Butterfly,
                       diagonal: torch.Tensor,
                       diag_first: bool = True,
                       inplace: bool = True) -> Butterfly:
    """
    Combine a Butterfly and a diagonal into another Butterfly
    Parameters:
        butterfly: Butterfly of size n
        diagonal: size (n, ). Should be of type complex if butterfly.complex == True.
        diag_first: If True, the map is input -> diagonal -> butterfly.
            If False, the map is input -> butterfly -> diagonal.
        inplace: whether to modify the input Butterfly
    """
    # TODO: test
    out_butterfly = butterfly if inplace else copy.deepcopy(butterfly)
    twiddle = out_butterfly.twiddle
    if out_butterfly.complex:
        twiddle = view_as_complex(out_butterfly.twiddle)
    if diag_first:
        if butterfly.increasing_stride:
            twiddle[:, :, 0, :, 0, :] *= diagonal[::2].unsqueeze(-1)
            twiddle[:, :, 0, :, 1, :] *= diagonal[1::2].unsqueeze(-1)
        else:
            n = diagonal.shape[-1]
            twiddle[:, :, 0, :, :, 0] *= diagonal[:n // 2].unsqueeze(-1)
            twiddle[:, :, 0, :, :, 1] *= diagonal[n // 2:].unsqueeze(-1)
    else:
        # Whether the last block is increasing or decreasing stride
        increasing_stride = butterfly.increasing_stride != ((butterfly.nblocks - 1) % 2 == 1)
        if increasing_stride:
            n = diagonal.shape[-1]
            twiddle[:, :, 0, :, :, 0] *= diagonal[:n // 2].unsqueeze(-1)
            twiddle[:, :, 0, :, :, 1] *= diagonal[n // 2:].unsqueeze(-1)
        else:
            twiddle[:, :, -1, :, 0, :] *= diagonal[::2].unsqueeze(-1)
            twiddle[:, :, -1, :, 1, :] *= diagonal[1::2].unsqueeze(-1)
    return out_butterfly


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
