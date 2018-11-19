import torch
from torch import nn


def semantic_loss_exactly_one(prob, dim=-1):
    """Semantic loss to encourage the multinomial probability to be "peaked",
    i.e. only one class is picked.
    The loss has the form -log sum_{i=1}^n p_i prod_{j=1, j!=i}^n (1 - p_j).
    Paper: http://web.cs.ucla.edu/~guyvdb/papers/XuICML18.pdf
    Code: https://github.com/UCLA-StarAI/Semantic-Loss/blob/master/semi_supervised/semantic.py
    Parameters:
        prob: probability of a multinomial distribution, shape (n, )
        dim: dimension to sum over
    Returns:
        semantic_loss: shape (1, )
    """
    # This is probably not the most numerically stable way to implement the
    # loss. Maybe it's better to compute from log softmax. The difficulty is to
    # compute log(1 - p) from log(p). Pytorch's logsumexp doesn't support
    # weight yet (as of Pytorch 1.0), unlike scipy's logsumexp, so we can't do
    # subtraction in log scale.
    return -((1 - prob).log().sum(dim=dim) + (prob / (1 - prob)).sum(dim=dim).log())


def test_semantic_loss_exactly_one():
    m = 5
    logit = torch.randn(m)
    p = nn.functional.softmax(logit, dim=-1)
    # Compute manually
    result = 0.0
    for i in range(m):
        prod = p[i].clone()
        for j in range(m):
            if j != i:
                prod *= 1 - p[j]
        result += prod
    result = -torch.log(result)
    result1 = -torch.logsumexp(torch.log(1 - p).sum() + torch.log(p / (1 - p)), dim=-1)
    result2 = semantic_loss_exactly_one(p)
    assert torch.allclose(result, result1)
    assert torch.allclose(result, result2)


if __name__ == '__main__':
    test_semantic_loss_exactly_one()
