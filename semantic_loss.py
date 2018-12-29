import torch
from torch import nn


# def semantic_loss_exactly_one(prob, dim=-1):
#     """Semantic loss to encourage the multinomial probability to be "peaked",
#     i.e. only one class is picked.
#     The loss has the form -log sum_{i=1}^n p_i prod_{j=1, j!=i}^n (1 - p_j).
#     Paper: http://web.cs.ucla.edu/~guyvdb/papers/XuICML18.pdf
#     Code: https://github.com/UCLA-StarAI/Semantic-Loss/blob/master/semi_supervised/semantic.py
#     Parameters:
#         prob: probability of a multinomial distribution, shape (n, )
#         dim: dimension to sum over
#     Returns:
#         semantic_loss: shape (1, )
#     """
    # This is probably not the most numerically stable way to implement the
    # loss. Maybe it's better to compute from log softmax. The difficulty is to
    # compute log(1 - p) from log(p). Pytorch's logsumexp doesn't support
    # weight yet (as of Pytorch 1.0), unlike scipy's logsumexp, so we can't do
    # subtraction in log scale.

    # loss =  -((1 - prob).log().sum(dim=dim) + (prob / (1 - prob)).sum(dim=dim).log())
    # Hacky way to avoid NaN when prob is very peaked, but doesn't work because the gradient is still NaN
    # loss[torch.isnan(loss)] = 0.0
    # Another hacky way: clamp the result instead of return inf - inf, doesn't work either
    # loss =  -(torch.clamp((1 - prob).log().sum(dim=dim), min=-torch.finfo(prob.dtype).max) + torch.clamp((prob / (1 - prob)).sum(dim=dim).log(), max=torch.finfo(prob.dtype).max))
    # TODO: This only works when dim=-1 and prob is 2 dimensional
    # loss = torch.zeros(prob.shape[0])
    # prob_not_one = torch.all(prob != 1.0, dim=-1)
    # loss[prob_not_one] = -((1 - prob[prob_not_one]).log().sum(dim=dim) + (prob[prob_not_one] / (1 - prob[prob_not_one])).sum(dim=dim).log())
    # return loss


def semantic_loss_exactly_one(log_prob):
    """Semantic loss to encourage the multinomial probability to be "peaked",
    i.e. only one class is picked.
    The loss has the form -log sum_{i=1}^n p_i prod_{j=1, j!=i}^n (1 - p_j).
    Paper: http://web.cs.ucla.edu/~guyvdb/papers/XuICML18.pdf
    Code: https://github.com/UCLA-StarAI/Semantic-Loss/blob/master/semi_supervised/semantic.py
    Parameters:
        log_prob: log probability of a multinomial distribution, shape (batch_size, n)
    Returns:
        semantic_loss: shape (batch_size)
    """
    _, argmaxes = torch.max(log_prob, dim=-1)
    # Compute log(1-p) separately for the largest probabilities, by doing
    # logsumexp on the rest of the log probabilities.
    log_prob_temp = log_prob.clone()
    log_prob_temp[range(log_prob.shape[0]), argmaxes] = torch.tensor(float('-inf'))
    log_1mprob_max = torch.logsumexp(log_prob_temp, dim=-1)
    # Compute log(1-p) normally for the rest of the probabilities
    log_1mprob = torch.log1p(-torch.exp(log_prob_temp))
    log_1mprob[range(log_prob.shape[0]), argmaxes] = log_1mprob_max
    loss = -(log_1mprob.sum(dim=-1) + torch.logsumexp(log_prob - log_1mprob, dim=-1))
    return loss


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
    result2 = semantic_loss_exactly_one(p.unsqueeze(0)).squeeze()
    assert torch.allclose(result, result1)
    assert torch.allclose(result, result2)


if __name__ == '__main__':
    test_semantic_loss_exactly_one()
