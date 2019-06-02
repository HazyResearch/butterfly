import math
import torch
from torch import nn

from butterfly.complex_utils import complex_mul


class CirculantLinear(nn.Module):

    def __init__(self, size, nstack=1):
        super().__init__()
        self.size = size
        self.nstack = nstack
        init_stddev = math.sqrt(1. / self.size)
        c = torch.randn(nstack, size) * init_stddev
        self.c_f = nn.Parameter(torch.rfft(c, 1))
        self.c_f._is_structured = True  # Flag to avoid weight decay

    def forward(self, input):
        """
        Parameters:
            input: (batch, size)
        Return:
            output: (batch, nstack * size)
        """
        batch = input.shape[0]
        input_f = torch.rfft(input, 1)
        prod = complex_mul(self.c_f, input_f.unsqueeze(1))
        return torch.irfft(prod, 1, signal_sizes=(self.size, )).view(batch, self.nstack * self.size)


class Circulant1x1Conv(CirculantLinear):

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w)
        Return:
            output: (batch, nstack * c, h, w)
        """
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)

# Code below is for testing different implementations of circulant multiply
from butterfly.complex_utils import *

class CirculantMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, x):
        n = x.shape[-1]
        x_f = torch.rfft(x, 1)
        c_f = torch.rfft(c, 1)
        ctx.save_for_backward(c_f, x_f)
        # prod = complex_mul(c_f, x_f)
        # prod = cupy2torch((torch2cupy(c_f).view('complex64') * torch2cupy(x_f).view('complex64')).view('float32'))
        prod = torch.empty_like(x_f)
        cp.multiply(torch2cupy(c_f).view('complex64'), torch2cupy(x_f).view('complex64'), out=torch2cupy(prod).view('complex64'))
        return torch.irfft(prod, 1, signal_sizes=(n, ))

    @staticmethod
    def backward(ctx, grad):
        n = grad.shape[-1]
        c_f, x_f = ctx.saved_tensors
        grad_f = torch.rfft(grad, 1)
        # dx_f = complex_mul(grad_f, conjugate(c_f))
        grad_f_cp = torch2cupy(grad_f).view('complex64')
        # dx_f = cupy2torch((torch2cupy(c_f).view('complex64').conj() * grad_f_cp).view('float32'))
        dx_f = torch.empty_like(x_f)
        cp.multiply(torch2cupy(c_f).view('complex64').conj(), grad_f_cp, out=torch2cupy(dx_f).view('complex64'))
        # dc_f = complex_mul(grad_f, conjugate(x_f)).sum(dim=0)
        # dc_f = cupy2torch((torch2cupy(x_f).view('complex64').conj() * grad_f_cp).view('float32')).sum(dim=0)
        dc_f = torch.empty_like(x_f)
        cp.multiply(torch2cupy(x_f).view('complex64').conj(), grad_f_cp, out=torch2cupy(dc_f).view('complex64'))
        dc_f = dc_f.sum(dim=0)
        # t1 = torch2cupy(x_f).view('complex64').conj().squeeze(-1)
        # t2 = grad_f_cp.squeeze(-1)
        # temp = (t1.T[:, np.newaxis] @ t2.T[..., np.newaxis]).squeeze()
        dx = torch.irfft(dx_f, 1, signal_sizes=(n, ))
        dc = torch.irfft(dc_f, 1, signal_sizes=(n, ))
        return dc, dx

circulant_custom_backward = CirculantMult.apply

def circulant_fft(c, x):
    n = x.shape[-1]
    x_f = torch.rfft(x, 1)
    c_f = torch.rfft(c, 1)
    prod = complex_mul(c_f, x_f)
    return torch.irfft(prod, 1, signal_sizes=(n, ))

def circulant_indexing(c, x):
    n = x.shape[-1]
    a = torch.arange(n, device=c.device)
    b = -a
    indices = a + b.unsqueeze(-1)
    C = c[indices]
    return x @ C

def anticirculant_as_strided(c, x):
    n = x.shape[-1]
    c_ext = torch.cat((c, c), dim=-1)
    C = c_ext.as_strided((n, n), (1, 1))
    return x @ C.contiguous().t()

def circulant_as_strided(c, x):
    n = x.shape[-1]
    reverse_idx = torch.arange(n - 1, -1, -1, device=c.device)
    c_rev = c[reverse_idx]
    c_ext = torch.cat((c_rev, c_rev), dim=-1)
    C = c_ext.as_strided((n, n), (1, 1))[:, reverse_idx]
    return x @ C



if __name__ == '__main__':

    import time

    nsteps = 1000
    n = 512
    batch_size = 128
    x = torch.randn(batch_size, n, device='cuda', requires_grad=True)
    c = torch.randn(n, device='cuda', requires_grad=True)
    grad = torch.randn_like(x)

    output = circulant_fft(c, x)
    torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_fft(c, x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'Circulant_fft forward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'Circulant_fft backward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_fft(c, x)
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'Circulant_fft together: {end - start}s')

    output = circulant_custom_backward(c, x)
    torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_custom_backward(c, x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'Circulant_custom_backward forward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'Circulant_custom_backward backward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_custom_backward(c, x)
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'Circulant_custom_backward together: {end - start}s')

    output = circulant_indexing(c, x)
    torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_indexing(c, x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'circulant_indexing forward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'circulant_indexing backward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_indexing(c, x)
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'circulant_indexing together: {end - start}s')

    output = circulant_as_strided(c, x)
    torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_as_strided(c, x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'circulant_as_strided forward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'circulant_as_strided backward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = circulant_as_strided(c, x)
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'circulant_as_strided together: {end - start}s')

    output = anticirculant_as_strided(c, x)
    torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = anticirculant_as_strided(c, x)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'anticirculant_as_strided forward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'anticirculant_as_strided backward: {end - start}s')
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(nsteps):
        output = anticirculant_as_strided(c, x)
        torch.autograd.grad(output, (c, x), grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f'anticirculant_as_strided together: {end - start}s')
