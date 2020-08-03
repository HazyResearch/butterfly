import math

import torch
from torch import nn
from torch.nn import functional as F

from torch_butterfly.butterfly import Butterfly
from torch_butterfly.permutation import FixedPermutation, bitreversal_permutation
from torch_butterfly.diagonal import Diagonal


def fft(n, normalized=False, br_first=True, with_br_perm=True):
    """ Construct an nn.Module based on Butterfly that exactly performs the FFT.
    Parameters:
        n: size of the FFT. Must be a power of 2.
        normalized: if True, corresponds to the unitary FFT (i.e. multiplied by 1/sqrt(n))
        br_first: which decomposition of FFT. br_first=True corresponds to decimation-in-time.
                  br_first=False corresponds to decimation-in-frequency.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    factors = []
    for log_size in range(1, log_n + 1):
        size = 1 << log_size
        exp = torch.exp(-1j * torch.arange(size // 2, dtype=torch.float) / size * 2 * math.pi)
        o = torch.ones_like(exp)
        twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                      torch.stack((o, -exp), dim=-1)), dim=-2)
        factors.append(twiddle_factor.repeat(n // size, 1, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:  # Take conjugate transpose of the BP decomposition of ifft
        twiddle = twiddle.transpose(-1, -2).flip(dims=(2,))
    if normalized:  # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
        twiddle /= math.sqrt(2)
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first)
    with torch.no_grad():
        b.twiddle.copy_(twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def ifft(n, normalized=False, br_first=True, with_br_perm=True):
    """ Construct an nn.Module based on Butterfly that exactly performs the inverse FFT.
    Parameters:
        n: size of the iFFT. Must be a power of 2.
        normalized: if True, corresponds to the unitary iFFT (i.e. multiplied by 1/sqrt(n), not 1/n)
        br_first: which decomposition of iFFT. br_first=True corresponds to decimation-in-frequency.
                  br_first=False corresponds to decimation-in-time.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    factors = []
    for log_size in range(1, log_n + 1):
        size = 1 << log_size
        exp = torch.exp(1j * torch.arange(size // 2, dtype=torch.float) / size * 2 * math.pi)
        o = torch.ones_like(exp)
        twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                      torch.stack((o, -exp), dim=-1)), dim=-2)
        factors.append(twiddle_factor.repeat(n // size, 1, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:  # Take conjugate transpose of the BP decomposition of fft
        twiddle = twiddle.transpose(-1, -2).flip(dims=(2,))
    if normalized:  # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
        twiddle /= math.sqrt(2)
    else:
        twiddle /= 2
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first)
    with torch.no_grad():
        b.twiddle.copy_(twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def circulant(col, transposed=False, separate_diagonal=True):
    """ Construct an nn.Module based on Butterfly that exactly performs circulant matrix
    multiplication.
    Parameters:
        col: torch.Tensor of size (n, ). The first column of the circulant matrix.
        transposed: if True, then the circulant matrix is transposed, i.e. col is the first *row*
                    of the matrix.
        separate_diagonal: if True, the returned nn.Module is Butterfly, Diagonal, Butterfly.
                           if False, the diagonal is combined into the Butterfly part.
    """
    assert col.dim() == 1, 'Vector col must have dimension 1'
    n = col.shape[0]
    log_n = int(math.ceil(math.log2(n)))
    # For non-power-of-2, maybe there's a way to only pad up to size 1 << log_n?
    # I've only figured out how to pad to size 1 << (log_n + 1).
    # e.g., [a, b, c] -> [a, b, c, 0, 0, a, b, c]
    n_extended = n if n == 1 << log_n else 1 << (log_n + 1)
    b_fft = fft(n_extended, normalized=True, br_first=False, with_br_perm=False)
    b_fft.in_size = n
    b_ifft = ifft(n_extended, normalized=True, br_first=True, with_br_perm=False)
    b_ifft.out_size = n
    if n < n_extended:
        col_0 = F.pad(col, (0, 2 * ((1 << log_n) - n)))
        col = torch.cat((col_0, col))
    if not col.is_complex():
        float_dtype_to_complex = {torch.float32: torch.complex64, torch.float64: torch.complex128}
        dtype = float_dtype_to_complex[col.dtype]
        col = col.to(dtype)
    # This fft must have normalized=False for the correct scaling. These are the eigenvalues of the
    # circulant matrix.
    col_f = torch.view_as_complex(torch.fft(torch.view_as_real(col),
                                            signal_ndim=1, normalized=False))
    if transposed:
        # We could have just transpose the iFFT * Diag * FFT to get FFT * Diag * iFFT.
        # Instead we use the fact that row is the reverse of col, but the 0-th element stays put.
        # This corresponds to the same reversal in the frequency domain.
        # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Time_and_frequency_reversal
        # col_f = np.fft.fft(row.numpy())
        col_f = torch.cat((col_f[:1], col_f[1:].flip(dims=(0,))))
    br_perm = (bitreversal_permutation(n_extended, pytorch_format=True))
    diag = col_f[..., br_perm]
    if separate_diagonal:
        return nn.Sequential(b_fft, Diagonal(n_extended, diag), b_ifft)
    else:
        # Combine the diagonal with the last twiddle factor of b_fft
        with torch.no_grad():
            b_fft.twiddle[:, :, -1, :, 0, :] *= diag[::2].unsqueeze(-1)
            b_fft.twiddle[:, :, -1, :, 1, :] *= diag[1::2].unsqueeze(-1)
        # Combine the b_fft and b_ifft into one Butterfly (with nblocks=2).
        # Need to force the internal twiddle to have size n_extended.
        b = Butterfly(n_extended, n_extended, bias=False, complex=True,
                      increasing_stride=False, nblocks=2)
        b.in_size = n
        b.out_size = n
        with torch.no_grad():
            b.twiddle.copy_(torch.cat((b_fft.twiddle, b_ifft.twiddle), dim=1))
        return b
