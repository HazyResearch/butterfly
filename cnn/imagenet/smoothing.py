import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class KnowledgeDistillationLoss(nn.Module):
    """
    Loss with knowledge distillation.
    """
    def __init__(self, original_loss, temperature=1.0, alpha_ce=0.5):
        super().__init__()
        self.original_loss = original_loss
        self.temperature = temperature
        self.alpha_ce = alpha_ce

    def forward(self, s_logit, t_logit, target):
        # Adapted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/distillation/distiller.py
        # Scaled by temperature^2 to balance the soft and hard loss
        # See https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
        # or https://github.com/stanford-futuredata/lit-code/blob/master/cifar10/distillation_loss.py
        loss_kd = F.kl_div(F.log_softmax(s_logit / self.temperature, dim=-1),
                           F.softmax(t_logit / self.temperature, dim=-1),
                           reduction='batchmean') * (self.temperature)**2
        loss_og = self.original_loss(s_logit, target)
        return (1 - self.alpha_ce) * loss_og + self.alpha_ce * loss_kd
