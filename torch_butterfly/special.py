import math

import torch
from torch import nn
from torch.nn import functional as F

from torch_butterfly.butterfly import Butterfly
from torch_butterfly.permutation import FixedPermutation, bitreversal_permutation
from torch_butterfly.diagonal import Diagonal
from torch_butterfly.complex_utils import view_as_real, view_as_complex
from torch_butterfly.complex_utils import complex_mul, real2complex, Real2Complex, Complex2Real


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
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    if normalized:
        twiddle /= math.sqrt(2)
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first)
    with torch.no_grad():
        view_as_complex(b.twiddle).copy_(twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def ifft(n, normalized=False, br_first=True, with_br_perm=True):
    """ Construct an nn.Module based on Butterfly that exactly performs the inverse FFT.
    Parameters:
        n: size of the iFFT. Must be a power of 2.
        normalized: if True, corresponds to unitary iFFT (i.e. multiplied by 1/sqrt(n), not 1/n)
        br_first: which decomposition of iFFT. True corresponds to decimation-in-frequency.
                  False corresponds to decimation-in-time.
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
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    if normalized:
        twiddle /= math.sqrt(2)
    else:
        twiddle /= 2
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first)
    with torch.no_grad():
        view_as_complex(b.twiddle).copy_(twiddle)
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
    complex = col.is_complex()
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
        col = real2complex(col)
    # This fft must have normalized=False for the correct scaling. These are the eigenvalues of the
    # circulant matrix.
    col_f = view_as_complex(torch.fft(view_as_real(col), signal_ndim=1, normalized=False))
    if transposed:
        # We could have just transposed the iFFT * Diag * FFT to get FFT * Diag * iFFT.
        # Instead we use the fact that row is the reverse of col, but the 0-th element stays put.
        # This corresponds to the same reversal in the frequency domain.
        # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Time_and_frequency_reversal
        col_f = torch.cat((col_f[:1], col_f[1:].flip(dims=(0,))))
    br_perm = (bitreversal_permutation(n_extended, pytorch_format=True))
    diag = col_f[..., br_perm]
    if separate_diagonal:
        if not complex:
            return nn.Sequential(Real2Complex(), b_fft, Diagonal(diagonal_init=diag), b_ifft,
                                 Complex2Real())
        else:
            return nn.Sequential(b_fft, Diagonal(diagonal_init=diag), b_ifft)
    else:
        # Combine the diagonal with the last twiddle factor of b_fft
        with torch.no_grad():
            twiddle = view_as_complex(b_fft.twiddle)
            twiddle[:, :, -1, :, 0, :] *= diag[::2].unsqueeze(-1)
            twiddle[:, :, -1, :, 1, :] *= diag[1::2].unsqueeze(-1)
        # Combine the b_fft and b_ifft into one Butterfly (with nblocks=2).
        # Need to force the internal twiddle to have size n_extended.
        b = Butterfly(n_extended, n_extended, bias=False, complex=True,
                      increasing_stride=False, nblocks=2)
        b.in_size = n
        b.out_size = n
        with torch.no_grad():
            # Don't need view_as_complex here since all the twiddles are stored in real.
            b.twiddle.copy_(torch.cat((b_fft.twiddle, b_ifft.twiddle), dim=1))
        return b if complex else nn.Sequential(Real2Complex(), b, Complex2Real())


def hadamard(n, normalized=False, increasing_stride=True):
    """ Construct an nn.Module based on Butterfly that exactly performs the Hadamard transform.
    Parameters:
        n: size of the Hadamard transform. Must be a power of 2.
        normalized: if True, corresponds to the orthogonal Hadamard transform
                    (i.e. multiplied by 1/sqrt(n))
        increasing_stride: whether the returned Butterfly has increasing stride.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    twiddle = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
    if normalized:
        twiddle /= math.sqrt(2)
    twiddle = twiddle.reshape(1, 1, 1, 1, 2, 2).expand((1, 1, log_n, n // 2, 2, 2))
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    b = Butterfly(n, n, bias=False, increasing_stride=increasing_stride)
    with torch.no_grad():
        b.twiddle.copy_(twiddle)
    return b


def conv1d_circular_singlechannel(n, weight, separate_diagonal=True):
    """ Construct an nn.Module based on Butterfly that exactly performs nn.Conv1d
    with a single in-channel and single out-channel, with circular padding.
    The output of nn.Conv1d must have the same size as the input (i.e. kernel size must be 2k + 1,
    and padding k for some integer k).
    Parameters:
        n: size of the input.
        weight: torch.Tensor of size (1, 1, kernel_size). Kernel_size must be odd, and smaller than
                n. Padding is assumed to be (kernel_size - 1) // 2.
        separate_diagonal: if True, the returned nn.Module is Butterfly, Diagonal, Butterfly.
                           if False, the diagonal is combined into the Butterfly part.
    """
    assert weight.dim() == 3, 'Weight must have dimension 3'
    kernel_size = weight.shape[-1]
    assert kernel_size < n
    assert kernel_size % 2 == 1, 'Kernel size must be odd'
    assert weight.shape[:2] == (1, 1), 'Only support single in-channel and single out-channel'
    padding = (kernel_size - 1) // 2
    col = F.pad(weight.flip(dims=(-1,)), (0, n - kernel_size)).roll(-padding, dims=-1)
    return circulant(col.squeeze(1).squeeze(0), separate_diagonal=separate_diagonal)


def conv1d_circular_multichannel(n, weight):
    """ Construct an nn.Module based on Butterfly that exactly performs nn.Conv1d
    with multiple in/out channels, with circular padding.
    The output of nn.Conv1d must have the same size as the input (i.e. kernel size must be 2k + 1,
    and padding k for some integer k).
    Parameters:
        n: size of the input.
        weight: torch.Tensor of size (out_channels, in_channels, kernel_size). Kernel_size must be
                odd, and smaller than n. Padding is assumed to be (kernel_size - 1) // 2.
    """
    assert weight.dim() == 3, 'Weight must have dimension 3'
    kernel_size = weight.shape[-1]
    assert kernel_size < n
    assert kernel_size % 2 == 1, 'Kernel size must be odd'
    out_channels, in_channels = weight.shape[:2]
    padding = (kernel_size - 1) // 2
    col = F.pad(weight.flip(dims=(-1,)), (0, n - kernel_size)).roll(-padding, dims=-1)
    # From here we mimic the circulant construction, but the diagonal multiply is replaced with
    # multiply and then sum across the in-channels.
    complex = col.is_complex()
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
        col = torch.cat((col_0, col), dim=-1)
    if not col.is_complex():
        col = real2complex(col)
    # This fft must have normalized=False for the correct scaling. These are the eigenvalues of the
    # circulant matrix.
    col_f = view_as_complex(torch.fft(view_as_real(col), signal_ndim=1, normalized=False))
    br_perm = (bitreversal_permutation(n_extended, pytorch_format=True))
    col_f = col_f[..., br_perm]
    # We just want (input_f.unsqueeze(1) * col_f).sum(dim=2).
    # This can be written as matrix multiply but Pytorch 1.6 doesn't yet support complex matrix
    # multiply.

    # We write this as an nn.Module just to use nn.Sequential
    class DiagonalMultiplySum(nn.Module):
        def __init__(self, diagonal_init):
            """
            Parameters:
                diagonal_init: (out_channels, in_channels, size)
            """
            super().__init__()
            self.diagonal = nn.Parameter(diagonal_init.detach().clone())
            self.complex = self.diagonal.is_complex()
            if self.complex:
                self.diagonal = nn.Parameter(view_as_real(self.diagonal))

        def forward(self, input):
            """
            Parameters:
                input: (batch, in_channels, size)
            Return:
                output: (batch, out_channels, size)
            """
            diagonal = self.diagonal if not self.complex else view_as_complex(self.diagonal)
            return complex_mul(input.unsqueeze(1), diagonal).sum(dim=2)

    if not complex:
        return nn.Sequential(Real2Complex(), b_fft, DiagonalMultiplySum(col_f), b_ifft,
                             Complex2Real())
    else:
        return nn.Sequential(b_fft, DiagonalMultiplySum(col_f), b_ifft)
