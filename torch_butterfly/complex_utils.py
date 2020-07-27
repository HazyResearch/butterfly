import torch


# Autograd for complex isn't implemented yet so we have to manually write the backward
class ComplexMul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X, Y)
        return X * Y

    @staticmethod
    def backward(ctx, grad):
        X, Y = ctx.saved_tensors
        grad_X, grad_Y = None, None
        if ctx.needs_input_grad[0]:
            grad_X = (grad * Y.conj()).sum_to_size(*X.shape)
        if ctx.needs_input_grad[1]:
            grad_Y = (grad * X.conj()).sum_to_size(*Y.shape)
        return grad_X, grad_Y


complex_mul = ComplexMul.apply
