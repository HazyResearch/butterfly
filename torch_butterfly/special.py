import math
from functools import reduce

import torch
from torch import nn
from torch.nn import functional as F
import torch.fft

from torch_butterfly.butterfly import Butterfly, ButterflyUnitary
from torch_butterfly.permutation import FixedPermutation, bitreversal_permutation, invert
from torch_butterfly.permutation import wavelet_permutation
from torch_butterfly.diagonal import Diagonal
from torch_butterfly.complex_utils import real2complex, Real2Complex, Complex2Real
from torch_butterfly.complex_utils import index_last_dim
from torch_butterfly.combine import diagonal_butterfly, TensorProduct, butterfly_product
from torch_butterfly.combine import butterfly_kronecker, permutation_kronecker


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
        exp = torch.exp(-2j * math.pi * torch.arange(0.0, size // 2) / size)
        o = torch.ones_like(exp)
        twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                      torch.stack((o, -exp), dim=-1)), dim=-2)
        factors.append(twiddle_factor.repeat(n // size, 1, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:  # Take conjugate transpose of the BP decomposition of ifft
        twiddle = twiddle.transpose(-1, -2).flip([2])
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    if normalized:
        twiddle /= math.sqrt(2)
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first, init=twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def fft_unitary(n, br_first=True, with_br_perm=True) -> nn.Module:
    """ Construct an nn.Module based on ButterflyUnitary that exactly performs the FFT.
    Since it's unitary, it corresponds to normalized=True.
    Parameters:
        n: size of the FFT. Must be a power of 2.
        br_first: which decomposition of FFT. br_first=True corresponds to decimation-in-time.
                  br_first=False corresponds to decimation-in-frequency.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    factors = []
    for log_size in range(1, log_n + 1):
        size = 1 << log_size
        angle = -2 * math.pi * torch.arange(0.0, size // 2) / size
        phi = torch.ones_like(angle) * math.pi / 4
        alpha = angle / 2 + math.pi / 2
        psi = -angle / 2 - math.pi / 2
        if br_first:
            chi = angle / 2 - math.pi / 2
        else:
            # Take conjugate transpose of the BP decomposition of ifft, which works out to this,
            # plus the flip later.
            chi = -angle / 2 - math.pi / 2
        twiddle_factor = torch.stack([phi, alpha, psi, chi], dim=-1)
        factors.append(twiddle_factor.repeat(n // size, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:
        twiddle = twiddle.flip([2])
    b = ButterflyUnitary(n, n, bias=False, increasing_stride=br_first)
    with torch.no_grad():
        b.twiddle.copy_(twiddle)
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
        exp = torch.exp(2j * math.pi * torch.arange(0.0, size // 2) / size)
        o = torch.ones_like(exp)
        twiddle_factor = torch.stack((torch.stack((o, exp), dim=-1),
                                      torch.stack((o, -exp), dim=-1)), dim=-2)
        factors.append(twiddle_factor.repeat(n // size, 1, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:  # Take conjugate transpose of the BP decomposition of fft
        twiddle = twiddle.transpose(-1, -2).flip([2])
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    if normalized:
        twiddle /= math.sqrt(2)
    else:
        twiddle /= 2
    b = Butterfly(n, n, bias=False, complex=True, increasing_stride=br_first, init=twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def ifft_unitary(n, br_first=True, with_br_perm=True) -> nn.Module:
    """ Construct an nn.Module based on ButterflyUnitary that exactly performs the iFFT.
    Since it's unitary, it corresponds to normalized=True.
    Parameters:
        n: size of the iFFT. Must be a power of 2.
        br_first: which decomposition of iFFT. br_first=True corresponds to decimation-in-time.
                  br_first=False corresponds to decimation-in-frequency.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    log_n = int(math.ceil(math.log2(n)))
    assert n == 1 << log_n, 'n must be a power of 2'
    factors = []
    for log_size in range(1, log_n + 1):
        size = 1 << log_size
        angle = 2 * math.pi * torch.arange(0.0, size // 2) / size
        phi = torch.ones_like(angle) * math.pi / 4
        alpha = angle / 2 + math.pi / 2
        psi = -angle / 2 - math.pi / 2
        if br_first:
            chi = angle / 2 - math.pi / 2
        else:
            # Take conjugate transpose of the BP decomposition of fft, which works out to this,
            # plus the flip later.
            chi = -angle / 2 - math.pi / 2
        twiddle_factor = torch.stack([phi, alpha, psi, chi], dim=-1)
        factors.append(twiddle_factor.repeat(n // size, 1))
    twiddle = torch.stack(factors, dim=0).unsqueeze(0).unsqueeze(0)
    if not br_first:
        twiddle = twiddle.flip([2])
    b = ButterflyUnitary(n, n, bias=False, increasing_stride=br_first)
    with torch.no_grad():
        b.twiddle.copy_(twiddle)
    if with_br_perm:
        br_perm = FixedPermutation(bitreversal_permutation(n, pytorch_format=True))
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


def dct(n: int, type: int = 2, normalized: bool = False) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs the DCT.
    Parameters:
        n: size of the DCT. Must be a power of 2.
        type: either 2, 3, or  4. These are the only types supported. See scipy.fft.dct's notes.
        normalized: if True, corresponds to the orthogonal DCT (see scipy.fft.dct's notes)
    """
    assert type in [2, 3, 4]
    # Construct the permutation before the FFT: separate the even and odd and then reverse the odd
    # e.g., [0, 1, 2, 3] -> [0, 2, 3, 1].
    perm = torch.arange(n)
    perm = torch.cat((perm[::2], perm[1::2].flip([0])))
    br = bitreversal_permutation(n, pytorch_format=True)
    postprocess_diag = 2 * torch.exp(-1j * math.pi * torch.arange(0.0, n) / (2 * n))
    if type in [2, 4]:
        b = fft(n, normalized=normalized, br_first=True, with_br_perm=False)
        if type == 4:
            even_mul = torch.exp(-1j * math.pi / (2 * n) * (torch.arange(0.0, n, 2) + 0.5))
            odd_mul = torch.exp(1j * math.pi / (2 * n) * (torch.arange(1.0, n, 2) + 0.5))
            preprocess_diag = torch.stack((even_mul, odd_mul), dim=-1).flatten()
            # This proprocess_diag is before the permutation.
            # To move it after the permutation, we have to permute the diagonal
            b = diagonal_butterfly(b, preprocess_diag[perm[br]], diag_first=True)
        if normalized:
            if type in [2, 3]:
                postprocess_diag[0] /= 2.0
                postprocess_diag[1:] /= math.sqrt(2)
            elif type == 4:
                postprocess_diag /= math.sqrt(2)
        b = diagonal_butterfly(b, postprocess_diag, diag_first=False)
        return nn.Sequential(FixedPermutation(perm[br]), Real2Complex(), b, Complex2Real())
    else:
        assert type == 3
        b = ifft(n, normalized=normalized, br_first=False, with_br_perm=False)
        postprocess_diag[0] /= 2.0
        if normalized:
            postprocess_diag[1:] /= math.sqrt(2)
        else:
            # We want iFFT with the scaling of 1.0 instead of 1 / n
            with torch.no_grad():
                b.twiddle *= 2
        b = diagonal_butterfly(b, postprocess_diag.conj(), diag_first=True)
        perm_inverse = invert(perm)
        return nn.Sequential(Real2Complex(), b, Complex2Real(), FixedPermutation(br[perm_inverse]))


def dst(n: int, type: int = 2, normalized: bool = False) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs the DST.
    Parameters:
        n: size of the DST. Must be a power of 2.
        type: either 2 or 4. These are the only types supported. See scipy.fft.dst's notes.
        normalized: if True, corresponds to the orthogonal DST (see scipy.fft.dst's notes)
    """
    assert type in [2, 4]
    b = fft(n, normalized=normalized, br_first=True, with_br_perm=False)
    # Construct the permutation before the FFT: separate the even and odd and then reverse the odd
    # e.g., [0, 1, 2, 3] -> [0, 2, 3, 1].
    perm = torch.arange(n)
    perm = torch.cat((perm[::2], perm[1::2].flip([0])))
    br = bitreversal_permutation(n, pytorch_format=True)
    if type == 2:
        even_mul = torch.exp(-1j * math.pi * torch.arange(0.0, n, 2) / n)
        odd_mul = -torch.exp(1j * math.pi * (torch.arange(1.0, n, 2) + 1) / n)
    elif type == 4:
        even_mul = torch.exp(-1j * math.pi * torch.arange(0.0, n, 2) / (2 * n))
        odd_mul = -torch.exp(1j * math.pi * (torch.arange(1.0, n, 2) + 1) / (2 * n))
    preprocess_diag = torch.stack((even_mul, odd_mul), dim=-1).flatten()
    # This proprocess_diag is before the permutation.
    # To move it after the permutation, we have to permute the diagonal
    b = diagonal_butterfly(b, preprocess_diag[perm[br]], diag_first=True)
    if type == 2:
        postprocess_diag = 2j * torch.exp(-1j * math.pi * (torch.arange(0.0, n) + 1) / (2 * n))
    elif type == 4:
        postprocess_diag = 2j * torch.exp(-1j * math.pi * (torch.arange(0.0, n) + 0.5) / (2 * n))
    if normalized:
        if type == 2:
            postprocess_diag[0] /= 2.0
            postprocess_diag[1:] /= math.sqrt(2)
        elif type == 4:
            postprocess_diag /= math.sqrt(2)
    b = diagonal_butterfly(b, postprocess_diag, diag_first=False)
    return nn.Sequential(FixedPermutation(perm[br]), Real2Complex(), b, Complex2Real())


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
    col_f = torch.fft.fft(col, norm=None)
    if transposed:
        # We could have just transposed the iFFT * Diag * FFT to get FFT * Diag * iFFT.
        # Instead we use the fact that row is the reverse of col, but the 0-th element stays put.
        # This corresponds to the same reversal in the frequency domain.
        # https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Time_and_frequency_reversal
        col_f = torch.cat((col_f[:1], col_f[1:].flip([0])))
    br_perm = bitreversal_permutation(n_extended, pytorch_format=True).to(col.device)
    # diag = col_f[..., br_perm]
    diag = index_last_dim(col_f, br_perm)
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


def toeplitz(col, row=None, separate_diagonal=True) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs Toeplitz matrix
    multiplication.
    Parameters:
        col: torch.Tensor of size (n, ). The first column of the Toeplitz matrix.
        row: torch.Tensor of size (n, ). The first row of the Toeplitz matrix. If None, assume
            row == col.conj(). The first element of row will be ignored.
        separate_diagonal: if True, the returned nn.Module is Butterfly, Diagonal, Butterfly.
                           if False, the diagonal is combined into the Butterfly part.
    """
    if row is None:
        row = col.conj()
    assert col.dim() == 1, 'Vector col must have dimension 1'
    complex = col.is_complex()
    n, = col.shape
    m, = row.shape
    log_n_m = int(math.ceil(math.log2(n + m - 1)))
    n_extended = 1 << log_n_m
    # Extend to a circulant matrix
    if n + m - 1 < n_extended:
        col = F.pad(col, (0, n_extended - (n + m - 1)))
    col = torch.cat([col, row[1:].flip([0])])
    b = circulant(col, separate_diagonal=separate_diagonal)
    # Adjust in_size = m and out_size = n
    if separate_diagonal:
        if not complex:
            b[1].in_size = m
            b[3].out_size = n
        else:
            b[0].in_size = m
            b[2].out_size = n
    else:
        if not complex:
            b[1].in_size = m
            b[1].out_size = n
        else:
            b.in_size = m
            b.out_size = n
    return b


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
    # Divide the whole transform by sqrt(n) by dividing each factor by n^(1/2 log_n) = sqrt(2)
    if normalized:
        twiddle /= math.sqrt(2)
    twiddle = twiddle.reshape(1, 1, 1, 1, 2, 2).expand((1, 1, log_n, n // 2, 2, 2))
    b = Butterfly(n, n, bias=False, increasing_stride=increasing_stride, init=twiddle)
    return b


def hadamard_diagonal(diagonals: torch.Tensor, normalized: bool = False,
                      increasing_stride: bool = True, separate_diagonal: bool = True) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that performs multiplication by H D H D ... H D,
    where H is the Hadamard matrix and D is a diagonal matrix
    Parameters:
        diagonals: (k, n), where k is the number of diagonal matrices and n is the dimension of the
            Hadamard transform.
        normalized: if True, corresponds to the orthogonal Hadamard transform
                    (i.e. multiplied by 1/sqrt(n))
        increasing_stride: whether the returned Butterfly has increasing stride.
        separate_diagonal: if False, the diagonal is combined into the Butterfly part.
    """
    k, n = diagonals.shape
    if not separate_diagonal:
        butterflies = []
        for i, diagonal in enumerate(diagonals.unbind()):
            cur_increasing_stride = increasing_stride != (i % 2 == 1)
            h = hadamard(n, normalized, cur_increasing_stride)
            butterflies.append(diagonal_butterfly(h, diagonal, diag_first=True))
        return reduce(butterfly_product, butterflies)
    else:
        modules = []
        for i, diagonal in enumerate(diagonals.unbind()):
            modules.append(Diagonal(diagonal_init=diagonal))
            cur_increasing_stride = increasing_stride != (i % 2 == 1)
            h = hadamard(n, normalized, cur_increasing_stride)
            modules.append(h)
        return nn.Sequential(*modules)


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
    col = F.pad(weight.flip([-1]), (0, n - kernel_size)).roll(-padding, dims=-1)
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

    def forward(self, input):
        """
        Parameters:
            input: (batch, in_channels, size)
        Return:
            output: (batch, out_channels, size)
        """
        return (input.unsqueeze(1) * self.diagonal).sum(dim=2)


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
    col = F.pad(weight.flip([-1]), (0, n - kernel_size)).roll(-padding, dims=-1)
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
    col_f = torch.fft.fft(col, norm=None)
    br_perm = bitreversal_permutation(n_extended, pytorch_format=True).to(col.device)
    # col_f = col_f[..., br_perm]
    col_f = index_last_dim(col_f, br_perm)
    # We just want (input_f.unsqueeze(1) * col_f).sum(dim=2).
    # This can be written as a complex matrix multiply as well.

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
            return (nn.Sequential(nn.Flatten(start_dim=-2), br_perm, b, nn.Unflatten(-1, (n2, n1)))
                    if br_first else nn.Sequential(nn.Flatten(start_dim=-2), b, br_perm,
                                                   nn.Unflatten(-1, (n2, n1))))
    else:
        return b if not flatten else nn.Sequential(nn.Flatten(start_dim=-2), b,
                                                   nn.Unflatten(-1, (n2, n1)))


def fft2d_unitary(n1: int, n2: int, br_first: bool = True,
                  with_br_perm: bool = True) -> nn.Module:
    """ Construct an nn.Module based on ButterflyUnitary that exactly performs the 2D FFT.
    Corresponds to normalized=True.
    Does not support flatten for now.
    Parameters:
        n1: size of the FFT on the last input dimension. Must be a power of 2.
        n2: size of the FFT on the second to last input dimension. Must be a power of 2.
        br_first: which decomposition of FFT. br_first=True corresponds to decimation-in-time.
                  br_first=False corresponds to decimation-in-frequency.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    b_fft1 = fft_unitary(n1, br_first=br_first, with_br_perm=False)
    b_fft2 = fft_unitary(n2, br_first=br_first, with_br_perm=False)
    b = TensorProduct(b_fft1, b_fft2)
    if with_br_perm:
        br_perm1 = FixedPermutation(bitreversal_permutation(n1, pytorch_format=True))
        br_perm2 = FixedPermutation(bitreversal_permutation(n2, pytorch_format=True))
        br_perm = TensorProduct(br_perm1, br_perm2)
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


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
            return (nn.Sequential(nn.Flatten(start_dim=-2), br_perm, b, nn.Unflatten(-1, (n2, n1)))
                    if br_first else nn.Sequential(nn.Flatten(start_dim=-2), b, br_perm,
                                                   nn.Unflatten(-1, (n2, n1))))
    else:
        return b if not flatten else nn.Sequential(nn.Flatten(start_dim=-2), b,
                                                   nn.Unflatten(-1, (n2, n1)))


def ifft2d_unitary(n1: int, n2: int, br_first: bool = True,
                   with_br_perm: bool = True) -> nn.Module:
    """ Construct an nn.Module based on ButterflyUnitary that exactly performs the 2D iFFT.
    Corresponds to normalized=True.
    Does not support flatten for now.
    Parameters:
        n1: size of the iFFT on the last input dimension. Must be a power of 2.
        n2: size of the iFFT on the second to last input dimension. Must be a power of 2.
        br_first: which decomposition of iFFT. True corresponds to decimation-in-frequency.
                  False corresponds to decimation-in-time.
        with_br_perm: whether to return both the butterfly and the bit reversal permutation.
    """
    b_ifft1 = ifft_unitary(n1, br_first=br_first, with_br_perm=False)
    b_ifft2 = ifft_unitary(n2, br_first=br_first, with_br_perm=False)
    b = TensorProduct(b_ifft1, b_ifft2)
    if with_br_perm:
        br_perm1 = FixedPermutation(bitreversal_permutation(n1, pytorch_format=True))
        br_perm2 = FixedPermutation(bitreversal_permutation(n2, pytorch_format=True))
        br_perm = TensorProduct(br_perm1, br_perm2)
        return nn.Sequential(br_perm, b) if br_first else nn.Sequential(b, br_perm)
    else:
        return b


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
    col = F.pad(weight.flip([-1]), (0, n1 - kernel_size1)).roll(-padding1, dims=-1)
    col = F.pad(col.flip([-2]), (0, 0, 0, n2 - kernel_size2)).roll(-padding2, dims=-2)
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
        b_fft = b_fft[1]  # Ignore the nn.Flatten and nn.Unflatten
    b_ifft = ifft2d(n_extended1, n_extended2, normalized=True, br_first=True,
                    with_br_perm=False, flatten=flatten).to(col.device)
    if not flatten:
        b_ifft.map1.out_size = n1
        b_ifft.map2.out_size = n2
    else:
        b_ifft = b_ifft[1]  # Ignore the nn.Flatten and nn.Unflatten
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
    col_f = torch.fft.fftn(col, dim=(-1, -2), norm=None)
    br_perm1 = bitreversal_permutation(n_extended1, pytorch_format=True).to(col.device)
    br_perm2 = bitreversal_permutation(n_extended2, pytorch_format=True).to(col.device)
    # col_f[..., br_perm2, br_perm1] would error "shape mismatch: indexing tensors could not be
    # broadcast together"
    # col_f = col_f[..., br_perm2, :][..., br_perm1]
    col_f = torch.view_as_complex(torch.view_as_real(col_f)[..., br_perm2, :, :][..., br_perm1, :])
    if flatten:
        col_f = col_f.reshape(*col_f.shape[:-2], col_f.shape[-2] * col_f.shape[-1])
    # We just want (input_f.unsqueeze(1) * col_f).sum(dim=2).
    # This can be written as a complex matrix multiply as well.
    if not complex:
        if not flatten:
            return nn.Sequential(Real2Complex(), b_fft, DiagonalMultiplySum(col_f), b_ifft,
                                Complex2Real())
        else:
            return nn.Sequential(Real2Complex(), nn.Flatten(start_dim=-2), b_fft,
                                 DiagonalMultiplySum(col_f), b_ifft, nn.Unflatten(-1, (n2, n1)),
                                 Complex2Real())
    else:
        if not flatten:
            return nn.Sequential(b_fft, DiagonalMultiplySum(col_f), b_ifft)
        else:
            return nn.Sequential(nn.Flatten(start_dim=-2), b_fft, DiagonalMultiplySum(col_f),
                                 b_ifft, nn.Unflatten(-1, (n2, n1)))


def fastfood(diag1: torch.Tensor, diag2: torch.Tensor, diag3: torch.Tensor,
             permutation: torch.Tensor, normalized: bool = False,
             increasing_stride: bool = True, separate_diagonal: bool = True) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that performs Fastfood multiplication:
    x -> Diag3 @ H @ Diag2 @ P @ H @ Diag1,
    where H is the Hadamard matrix and P is a permutation matrix.
    Parameters:
        diag1: (n,), where n is a power of 2.
        diag2: (n,)
        diag3: (n,)
        permutation: (n,)
        normalized: if True, corresponds to the orthogonal Hadamard transform
                    (i.e. multiplied by 1/sqrt(n))
        increasing_stride: whether the first Butterfly in the sequence has increasing stride.
        separate_diagonal: if False, the diagonal is combined into the Butterfly part.
    """
    n, = diag1.shape
    assert diag2.shape == diag3.shape == permutation.shape == (n,)
    h1 = hadamard(n, normalized, increasing_stride)
    h2 = hadamard(n, normalized, not increasing_stride)
    if not separate_diagonal:
        h1 = diagonal_butterfly(h1, diag1, diag_first=True)
        h2 = diagonal_butterfly(h2, diag2, diag_first=True)
        h2 = diagonal_butterfly(h2, diag3, diag_first=False)
        return nn.Sequential(h1, FixedPermutation(permutation), h2)
    else:
        return nn.Sequential(Diagonal(diagonal_init=diag1), h1, FixedPermutation(permutation),
                             Diagonal(diagonal_init=diag2), h2, Diagonal(diagonal_init=diag3))


def acdc(diag1: torch.Tensor, diag2: torch.Tensor, dct_first: bool = True,
         separate_diagonal: bool = True) -> nn.Module:
    """ Construct an nn.Module based on Butterfly that exactly performs either the multiplication:
        x -> diag2 @ iDCT @ diag1 @ DCT @ x
    or
        x -> diag2 @ DCT @ diag1 @ iDCT @ x.
    In the paper [1], the math describes the 2nd type while the implementation uses the 1st type.
    Note that the DCT and iDCT are normalized.
    [1] Marcin Moczulski, Misha Denil, Jeremy Appleyard, Nando de Freitas.
    ACDC: A Structured Efficient Linear Layer.
    http://arxiv.org/abs/1511.05946
    Parameters:
        diag1: (n,), where n is a power of 2.
        diag2: (n,), where n is a power of 2.
        dct_first: if True, uses the first type above; otherwise use the second type.
        separate_diagonal: if False, the diagonal is combined into the Butterfly part.
    """
    n, = diag1.shape
    assert diag2.shape == (n,)
    assert n == 1 << int(math.ceil(math.log2(n))), 'n must be a power of 2'
    # Construct the permutation before the FFT: separate the even and odd and then reverse the odd
    # e.g., [0, 1, 2, 3] -> [0, 2, 3, 1].
    # This permutation is actually in B (not just B^T B or B B^T). This can be checked with
    # perm2butterfly.
    perm = torch.arange(n)
    perm = torch.cat((perm[::2], perm[1::2].flip([0])))
    perm_inverse = invert(perm)
    br = bitreversal_permutation(n, pytorch_format=True)
    postprocess_diag = 2 * torch.exp(-1j * math.pi * torch.arange(0.0, n) / (2 * n))
    # Normalize
    postprocess_diag[0] /= 2.0
    postprocess_diag[1:] /= math.sqrt(2)
    if dct_first:
        b_fft = fft(n, normalized=True, br_first=False, with_br_perm=False)
        b_ifft = ifft(n, normalized=True, br_first=True, with_br_perm=False)
        b1 = diagonal_butterfly(b_fft, postprocess_diag[br], diag_first=False)
        b2 = diagonal_butterfly(b_ifft, postprocess_diag.conj()[br], diag_first=True)
        if not separate_diagonal:
            b1 = diagonal_butterfly(b_fft, diag1[br], diag_first=False)
            b2 = diagonal_butterfly(b2, diag2[perm], diag_first=False)
            return nn.Sequential(FixedPermutation(perm),
                                 Real2Complex(), b1, Complex2Real(),
                                 Real2Complex(), b2, Complex2Real(),
                                 FixedPermutation(perm_inverse))
        else:
            return nn.Sequential(FixedPermutation(perm),
                                 Real2Complex(), b1, Complex2Real(),
                                 Diagonal(diagonal_init=diag1[br]),
                                 Real2Complex(), b2, Complex2Real(),
                                 Diagonal(diagonal_init=diag2[perm]),
                                 FixedPermutation(perm_inverse))
    else:
        b_fft = fft(n, normalized=True, br_first=True, with_br_perm=False)
        b_ifft = ifft(n, normalized=True, br_first=False, with_br_perm=False)
        b1 = diagonal_butterfly(b_ifft, postprocess_diag.conj(), diag_first=True)
        b2 = diagonal_butterfly(b_fft, postprocess_diag, diag_first=False)
        if not separate_diagonal:
            b1 = diagonal_butterfly(b1, diag1[perm][br], diag_first=False)
            b2 = diagonal_butterfly(b_fft, diag2, diag_first=False)
            return nn.Sequential(Real2Complex(), b1, Complex2Real(),
                                 Real2Complex(), b2, Complex2Real())
        else:
            return nn.Sequential(Real2Complex(), b1, Complex2Real(),
                                 Diagonal(diagonal_init=diag1[perm][br]),
                                 Real2Complex(), b2, Complex2Real(),
                                 Diagonal(diagonal_init=diag2))


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
    b = Butterfly(n, n, bias=False, increasing_stride=True, init=twiddle)
    if with_perm:
        perm = FixedPermutation(wavelet_permutation(n, pytorch_format=True))
        return nn.Sequential(b, perm)
    else:
        return b
