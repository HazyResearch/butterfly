import math
import numpy as np
import torch
from torch import nn

from butterfly.complex_utils import complex_mul, conjugate


def toeplitz_krylov_transpose_multiply(v, u, f=0.0):
    """Multiply Krylov(Z_f, v_i)^T @ u.
    Parameters:
        v: (nstack, rank, n)
        u: (batch_size, n)
        f: real number
    Returns:
        product: (batch, nstack, rank, n)
    """
    _, n = u.shape
    _, _, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    if f != 0.0:  # cycle version
        # Computing the roots of f
        mod = abs(f) ** (torch.arange(n, dtype=u.dtype, device=u.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=u.dtype, device=u.device),
                               torch.zeros(n, dtype=u.dtype, device=u.device)), dim=-1)
        else:  # Find primitive roots of -1
            angles = torch.arange(n, dtype=u.dtype, device=u.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)
        u_f = torch.ifft(eta_inverse * u[..., np.newaxis], 1)
        v_f = torch.fft(eta * v.unsqueeze(-1), 1)
        uv_f = complex_mul(u_f.unsqueeze(1).unsqueeze(1), v_f)
        uv = torch.fft(uv_f, 1)
        # We only need the real part of complex_mul(eta, uv)
        return eta[..., 0] * uv[..., 0] - eta[..., 1] * uv[..., 1]
    else:
        u_f = torch.rfft(torch.cat((u.flip(1), torch.zeros_like(u)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        uv_f = complex_mul(u_f.unsqueeze(1).unsqueeze(1), v_f)
        return torch.irfft(uv_f, 1, signal_sizes=(2 * n, ))[..., :n].flip(3)


def toeplitz_krylov_multiply(v, w, f=0.0):
    """Multiply \sum_i Krylov(Z_f, v_i) @ w_i.
    Parameters:
        v: (nstack, rank, n)
        w: (batch_size, nstack, rank, n)
        f: real number
    Returns:
        product: (batch, nstack, n)
    """
    _, nstack, rank, n = w.shape
    nstack_, rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    assert nstack == nstack_, 'w and v must have the same nstack'
    if f != 0.0:  # cycle version
        # Computing the roots of f
        mod = abs(f) ** (torch.arange(n, dtype=w.dtype, device=w.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=w.dtype, device=w.device),
                               torch.zeros(n, dtype=w.dtype, device=w.device)), dim=-1)
        else:  # Find primitive roots of -1
            angles = torch.arange(n, dtype=w.dtype, device=w.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)
        w_f = torch.fft(eta * w[..., np.newaxis], 1)
        v_f = torch.fft(eta * v[..., np.newaxis], 1)
        wv_sum_f = complex_mul(w_f, v_f).sum(dim=2)
        wv_sum = torch.ifft(wv_sum_f, 1)
        # We only need the real part of complex_mul(eta_inverse, wv_sum)
        return eta_inverse[..., 0] * wv_sum[..., 0] - eta_inverse[..., 1] - wv_sum[..., 1]
    else:
        w_f = torch.rfft(torch.cat((w, torch.zeros_like(w)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        wv_sum_f = complex_mul(w_f, v_f).sum(dim=2)
        return torch.irfft(wv_sum_f, 1, signal_sizes=(2 * n, ))[..., :n]


def toeplitz_mult(G, H, x, cycle=True):
    """Multiply \sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Parameters:
        G: Tensor of shape (nstack, rank, n)
        H: Tensor of shape (nstack, rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, nstack, n)
    """
    # f = (1,-1) if cycle else (1,1)
    f = (1, -1) if cycle else (0, 0)
    transpose_out = toeplitz_krylov_transpose_multiply(H, x, f[1])
    return toeplitz_krylov_multiply(G, transpose_out, f[0])


class ToeplitzlikeLinear(nn.Module):

    def __init__(self, in_size, out_size, rank=4, bias=True, corner=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.nstack = int(math.ceil(out_size / self.in_size))
        self.rank = rank
        assert not corner, 'corner not currently supported'
        self.corner = corner
        init_stddev = math.sqrt(1. / (rank * in_size))
        self.G = nn.Parameter(torch.randn(self.nstack, rank, in_size) * init_stddev)
        self.H = nn.Parameter(torch.randn(self.nstack, rank, in_size) * init_stddev)
        self.G._is_structured = True  # Flag to avoid weight decay
        self.H._is_structured = True
        self.register_buffer('reverse_idx', torch.arange(in_size - 1, -1, -1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size)
        Return:
            output: (batch, *, out_size)
        """
        u = input.view(np.prod(input.size()[:-1]), input.size(-1))
        batch = u.shape[0]
        # output = toeplitz_mult(self.G, self.H, input, self.corner)
        # return output.reshape(batch, self.nstack * self.size)
        n = self.in_size
        v = self.H
        # u_f = torch.rfft(torch.cat((u.flip(1), torch.zeros_like(u)), dim=-1), 1)
        u_f = torch.rfft(torch.cat((u[:, self.reverse_idx], torch.zeros_like(u)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        uv_f = complex_mul(u_f.unsqueeze(1).unsqueeze(1), v_f)
        # transpose_out =  torch.irfft(uv_f, 1, signal_sizes=(2 * n, ))[..., :n].flip(3)
        transpose_out =  torch.irfft(uv_f, 1, signal_sizes=(2 * n, ))[..., self.reverse_idx]
        v = self.G
        w = transpose_out
        w_f = torch.rfft(torch.cat((w, torch.zeros_like(w)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        wv_sum_f = complex_mul(w_f, v_f).sum(dim=2)
        output = torch.irfft(wv_sum_f, 1, signal_sizes=(2 * n, ))[..., :n]
        output = output.reshape(batch, self.nstack * self.in_size)[:, :self.out_size]
        if self.bias is not None:
            output = output + self.bias
        return output.view(*input.size()[:-1], self.out_size)

    def extra_repr(self):
        return 'in_size={}, out_size={}, bias={}, rank={}, corner={}'.format(
            self.in_size, self.out_size, self.bias is not None, self.rank, self.corner
        )


class Toeplitzlike1x1Conv(ToeplitzlikeLinear):

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w)
        Return:
            output: (batch, nstack * c, h, w)
        """
        # TODO: this is for old code with square Toeplitzlike, need to be updated
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)
