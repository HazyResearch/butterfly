# encoding: utf8

"""
From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label
Classification. André F. T. Martins, Ramón Fernandez Astudillo
In: Proc. of ICML 2016, https://arxiv.org/abs/1602.02068

Code adapted from https://github.com/vene/sparse-structured-attention
"""

import torch


def project_simplex(v, z=1.0):
    """Project a vector v onto the simplex.
    That is, return argmin_w ||w - v||^2 where w >= 0 elementwise and sum(w) = z.
    Parameters:
        v: Tensor of shape (batch_size, n)
        z: real number
    Return:
        Projection of v on the simplex, along the last dimension: (batch_size, n)
    """
    v_sorted, _ = v.sort(dim=-1, descending=True)
    range_ = torch.arange(1.0, 1 + v.shape[-1])
    cumsum_divided = (v_sorted.cumsum(dim=-1) - z) / range_
    # rho = (v_sorted - cumsum_divided > 0).nonzero()[-1]
    cond = (v_sorted - cumsum_divided > 0).type(v.dtype)
    rho = (cond * range_).argmax(dim=-1)
    tau = cumsum_divided[range(v.dim()), rho]
    return torch.clamp(v - tau.unsqueeze(-1), min=0)


def sparsemax_grad(output, grad):
    support = output > 0
    support_f = support.type(grad.dtype)
    s = (grad * support_f).sum(dim=-1) / support_f.sum(dim=-1)
    return support_f * (grad - s.unsqueeze(-1))
    # temp = (grad - s.unsqueeze(-1))[support]
    # result = torch.zeros_like(grad)
    # result[support] = temp


class Sparsemax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, v):
        output = project_simplex(v)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad):
        output,  = ctx.saved_tensors
        return sparsemax_grad(output, grad)


sparsemax = Sparsemax.apply
