import math
from functools import reduce

import torch
from torch import nn
from torch.nn import functional as F

from torch_butterfly.butterfly import Butterfly
from torch_butterfly.permutation import FixedPermutation, bitreversal_permutation
from torch_butterfly.permutation import wavelet_permutation
from torch_butterfly.diagonal import Diagonal
from torch_butterfly.complex_utils import view_as_real, view_as_complex
from torch_butterfly.complex_utils import complex_mul, real2complex, Real2Complex, Complex2Real
from torch_butterfly.combine import diagonal_butterfly, TensorProduct, butterfly_product
from torch_butterfly.combine import butterfly_kronecker, permutation_kronecker
from torch_butterfly.combine import Flatten2D, Unflatten2D


def fft(n, normalized=False, br_first=True, with_br_perm=True) -> nn.Module:
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


def ifft(n, normalized=False, br_first=True, with_br_perm=True) -> nn.Module:
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


def circulant(col, transposed=False, separate_diagonal=True) -> nn.Module:
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
    b_fft = fft(n_extended, normalized=True, br_first=False, with_br_perm=False).to(col.device)
    b_fft.in_size = n
    b_ifft = ifft(n_extended, normalized=True, br_first=True, with_br_perm=False).to(col.device)
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
    br_perm = bitreversal_permutation(n_extended, pytorch_format=True).to(col.device)
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
            b_fft = diagonal_butterfly(b_fft, diag, diag_first=False, inplace=True)
        # Combine the b_fft and b_ifft into one Butterfly (with nblocks=2).
        b = butterfly_product(b_fft, b_ifft)
        b.in_size = n
        b.out_size = n
        return b if complex else nn.Sequential(Real2Complex(), b, Complex2Real())


def hadamard(n, normalized=False, increasing_stride=True) -> Butterfly:
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


def hadamard_diagonal(diagonals: torch.Tensor, normalized=False,
                      increasing_stride=True) -> Butterfly:
    """ Construct an nn.Module based on Butterfly that performs multiplication by H D H D ... H D,
    where H is the Hadamard matrix and D is a diagonal matrix
    Parameters:
        diagonals: (k, n), where k is the number of diagonal matrices and n is the dimension of the
            Hadamard transform.
        normalized: if True, corresponds to the orthogonal Hadamard transform
                    (i.e. multiplied by 1/sqrt(n))
        increasing_stride: whether the returned Butterfly has increasing stride.
    """
    k, n = diagonals.shape
    butterflies = []
    for i, diagonal in enumerate(diagonals.unbind()):
        cur_increasing_stride = increasing_stride != (i % 2 == 1)
        h = hadamard(n, normalized, cur_increasing_stride)
        butterflies.append(diagonal_butterfly(h, diagonal, diag_first=True))
    return reduce(butterfly_product, butterflies)


def conv1d_circular_singlechannel(n, weight, separate_diagonal=True) -> nn.Module:
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


def conv1d_circular_multichannel(n, weight) -> nn.Module:
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
    b_fft = fft(n_extended, normalized=True, br_first=False, with_br_perm=False).to(col.device)
    b_fft.in_size = n
    b_ifft = ifft(n_extended, normalized=True, br_first=True, with_br_perm=False).to(col.device)
    b_ifft.out_size = n
    if n < n_extended:
        col_0 = F.pad(col, (0, 2 * ((1 << log_n) - n)))
        col = torch.cat((col_0, col), dim=-1)
    if not col.is_complex():
        col = real2complex(col)
    # This fft must have normalized=False for the correct scaling. These are the eigenvalues of the
    # circulant matrix.
    col_f = view_as_complex(torch.fft(view_as_real(col), signal_ndim=1, normalized=False))
    br_perm = bitreversal_permutation(n_extended, pytorch_format=True).to(col.device)
    col_f = col_f[..., br_perm]
    # We just want (input_f.unsqueeze(1) * col_f).sum(dim=2).
    # This can be written as matrix multiply but Pytorch 1.6 doesn't yet support complex matrix
    # multiply.

    if not complex:
        return nn.Sequential(Real2Complex(), b_fft, DiagonalMultiplySum(col_f), b_ifft,
                             Complex2Real())
    else:
        return nn.Sequential(b_fft, DiagonalMultiplySum(col_f), b_ifft)


def fft2d(n1: int, n2: int, normalized: bool = False, br_first: bool = True,
          with_br_perm: bool = True, flatten=False) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs the 2D FFT.
    Parameters:
        n1: size of the FFT on the last input dimension. Must be a power of 2.
        n2: size of the FFT on the second to last input dimension. Must be a power of 2.
        normalized: if True, corresponds to the unitary FFT (i.e. multiplied by 1/sqrt(n))
        br_first: which decomposition of FFT. br_first=True corresponds to decimation-in-time.
                  br_first=False corresponds to decimation-in-frequency.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
        flatten: whether to combine the 2 butterflies into 1 with Kronecker product.
    """
    b_fft1 = fft(n1, normalized=normalized, br_first=br_first, with_br_perm=False)
    b_fft2 = fft(n2, normalized=normalized, br_first=br_first, with_br_perm=False)
    b = TensorProduct(b_fft1, b_fft2) if not flatten else butterfly_kronecker(b_fft1, b_fft2)
    if with_br_perm:
        br_perm1 = FixedPermutation(bitreversal_permutation(n1, pytorch_format=True))
        br_perm2 = FixedPermutation(bitreversal_permutation(n2, pytorch_format=True))
        br_perm = (TensorProduct(br_perm1, br_perm2)
                   if not flatten else permutation_kronecker(br_perm1, br_perm2))
        if not flatten:
            return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
        else:
            return (nn.Sequential(Flatten2D(), br_perm, b, Unflatten2D(n1)) if br_first
                    else nn.Sequential(Flatten2D(), b, br_perm, Unflatten2D(n1)))
    else:
        return b if not flatten else nn.Sequential(Flatten2D(), b, Unflatten2D(n1))


def ifft2d(n1: int, n2: int, normalized: bool = False, br_first: bool = True,
           with_br_perm: bool = True, flatten=False) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs the 2D iFFT.
    Parameters:
        n1: size of the iFFT on the last input dimension. Must be a power of 2.
        n2: size of the iFFT on the second to last input dimension. Must be a power of 2.
        normalized: if True, corresponds to the unitary iFFT (i.e. multiplied by 1/sqrt(n))
        br_first: which decomposition of iFFT. True corresponds to decimation-in-frequency.
                  False corresponds to decimation-in-time.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
        flatten: whether to combine the 2 butterflies into 1 with Kronecker product.
    """
    b_ifft1 = ifft(n1, normalized=normalized, br_first=br_first, with_br_perm=False)
    b_ifft2 = ifft(n2, normalized=normalized, br_first=br_first, with_br_perm=False)
    b = TensorProduct(b_ifft1, b_ifft2) if not flatten else butterfly_kronecker(b_ifft1, b_ifft2)
    if with_br_perm:
        br_perm1 = FixedPermutation(bitreversal_permutation(n1, pytorch_format=True))
        br_perm2 = FixedPermutation(bitreversal_permutation(n2, pytorch_format=True))
        br_perm = (TensorProduct(br_perm1, br_perm2)
                   if not flatten else permutation_kronecker(br_perm1, br_perm2))
        if not flatten:
            return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
        else:
            return (nn.Sequential(Flatten2D(), br_perm, b, Unflatten2D(n1)) if br_first
                    else nn.Sequential(Flatten2D(), b, br_perm, Unflatten2D(n1)))
    else:
        return b if not flatten else nn.Sequential(Flatten2D(), b, Unflatten2D(n1))


def conv2d_circular_multichannel(n1: int, n2: int, weight: torch.Tensor,
                                 flatten: bool=False) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs nn.Conv2d
    with multiple in/out channels, with circular padding.
    The output of nn.Conv2d must have the same size as the input (i.e. kernel size must be 2k + 1,
    and padding k for some integer k).
    Parameters:
        n1: size of the last dimension of the input.
        n2: size of the second to last dimension of the input.
        weight: torch.Tensor of size (out_channels, in_channels, kernel_size2, kernel_size1).
            Kernel_size must be odd, and smaller than n1/n2. Padding is assumed to be
            (kernel_size - 1) // 2.
        flatten: whether to internally flatten the last 2 dimensions of the input. Only support n1
            and n2 being powers of 2.
    """
    assert weight.dim() == 4, 'Weight must have dimension 4'
    kernel_size2, kernel_size1 = weight.shape[-2], weight.shape[-1]
    assert kernel_size1 < n1, kernel_size2 < n2
    assert kernel_size1 % 2 == 1 and kernel_size2 % 2 == 1, 'Kernel size must be odd'
    out_channels, in_channels = weight.shape[:2]
    padding1 = (kernel_size1 - 1) // 2
    padding2 = (kernel_size2 - 1) // 2
    col = F.pad(weight.flip(dims=(-1,)), (0, n1 - kernel_size1)).roll(-padding1, dims=-1)
    col = F.pad(col.flip(dims=(-2,)), (0, 0, 0, n2 - kernel_size2)).roll(-padding2, dims=-2)
    # From here we mimic the circulant construction, but the diagonal multiply is replaced with
    # multiply and then sum across the in-channels.
    complex = col.is_complex()
    log_n1 = int(math.ceil(math.log2(n1)))
    log_n2 = int(math.ceil(math.log2(n2)))
    if flatten:
        assert n1 == 1 << log_n1, n2 == 1 << log_n2
    # For non-power-of-2, maybe there's a way to only pad up to size 1 << log_n1?
    # I've only figured out how to pad to size 1 << (log_n1 + 1).
    # e.g., [a, b, c] -> [a, b, c, 0, 0, a, b, c]
    n_extended1 = n1 if n1 == 1 << log_n1 else 1 << (log_n1 + 1)
    n_extended2 = n2 if n2 == 1 << log_n2 else 1 << (log_n2 + 1)
    b_fft = fft2d(n_extended1, n_extended2, normalized=True, br_first=False,
                  with_br_perm=False, flatten=flatten).to(col.device)
    if not flatten:
        b_fft.map1.in_size = n1
        b_fft.map2.in_size = n2
    else:
        b_fft = b_fft[1]  # Ignore the Flatten2D and Unflatten2D
    b_ifft = ifft2d(n_extended1, n_extended2, normalized=True, br_first=True,
                    with_br_perm=False, flatten=flatten).to(col.device)
    if not flatten:
        b_ifft.map1.out_size = n1
        b_ifft.map2.out_size = n2
    else:
        b_ifft = b_ifft[1]  # Ignore the Flatten2D and Unflatten2D
    if n1 < n_extended1:
        col_0 = F.pad(col, (0, 2 * ((1 << log_n1) - n1)))
        col = torch.cat((col_0, col), dim=-1)
    if n2 < n_extended2:
        col_0 = F.pad(col, (0, 0, 0, 2 * ((1 << log_n2) - n2)))
        col = torch.cat((col_0, col), dim=-2)
    if not col.is_complex():
        col = real2complex(col)
    # This fft must have normalized=False for the correct scaling. These are the eigenvalues of the
    # circulant matrix.
    col_f = view_as_complex(torch.fft(view_as_real(col), signal_ndim=2, normalized=False))
    br_perm1 = bitreversal_permutation(n_extended1, pytorch_format=True).to(col.device)
    br_perm2 = bitreversal_permutation(n_extended2, pytorch_format=True).to(col.device)
    # col_f[..., br_perm2, br_perm1] would error "shape mismatch: indexing tensors could not be
    # broadcast together"
    col_f = col_f[..., br_perm2, :][..., br_perm1]
    if flatten:
        col_f = col_f.reshape(*col_f.shape[:-2], col_f.shape[-2] * col_f.shape[-1])
    # We just want (input_f.unsqueeze(1) * col_f).sum(dim=2).
    # This can be written as matrix multiply but Pytorch 1.6 doesn't yet support complex matrix
    # multiply.
    if not complex:
        if not flatten:
            return nn.Sequential(Real2Complex(), b_fft, DiagonalMultiplySum(col_f), b_ifft,
                                Complex2Real())
        else:
            return nn.Sequential(Real2Complex(), Flatten2D(), b_fft, DiagonalMultiplySum(col_f),
                                 b_ifft, Unflatten2D(n1), Complex2Real())
    else:
        if not flatten:
            return nn.Sequential(b_fft, DiagonalMultiplySum(col_f), b_ifft)
        else:
            return nn.Sequential(Flatten2D(), b_fft, DiagonalMultiplySum(col_f), b_ifft,
                                 Unflatten2D(n1))


def wavelet_haar(n, with_perm=True) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs the multilevel discrete
    wavelet transform with the Haar wavelet.
    Parameters:
        n: size of the discrete wavelet transform. Must be a power of 2.
        with_perm: whether to return both the butterfly and the wavelet rearrangement permutation.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    factors = []
    for log_size in range(1, log_n + 1):
        size = 1 << log_size
        factor = torch.tensor([[1, 1], [1, -1]], dtype=torch.float).reshape(1, 2, 2) / math.sqrt(2)
        identity = torch.eye(2).reshape(1, 2, 2)
        num_identity = size // 2 - 1
        twiddle_factor = torch.cat((factor, identity.expand(num_identity, 2, 2)))
        factors.append(twiddle_factor.repeat(n // size, 1, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    b = Butterfly(n, n, bias=False, increasing_stride=True)
    with torch.no_grad():
        b.twiddle.copy_(twiddle)
    if with_perm:
        perm = FixedPermutation(wavelet_permutation(n, pytorch_format=True))
        return nn.Sequential(b, perm)
    else:
        return b
