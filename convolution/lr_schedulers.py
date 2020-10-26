import torch


def LeNetScheduler(optimizer, nepochs, **kwargs):

    def sched(epoch):
        if epoch < int(nepochs * 0.5):
            return 1.0
        elif epoch < int(nepochs * 0.75):
            return 0.5
        else:
            return 0.1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: sched(epoch))
