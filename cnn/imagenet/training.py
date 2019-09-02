import sys
from pathlib import Path
project_root = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from . import logger as log
# from . import resnet as models
from . import utils

from cnn.mobilenet_imagenet import MobileNet
from cnn.mobilenet_imagenet import Butterfly1x1Conv

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class ModelAndLoss(nn.Module):
    def __init__(self, arch, loss, pretrained_weights=None, cuda=True, fp16=False,
                 width=1.0, n_struct_layers=0, struct='D', softmax_struct='D'):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))
        # model = models.build_resnet(arch[0], arch[1])
        if arch == 'mobilenetv1':
            model = MobileNet(width_mult=width, structure=[struct] * n_struct_layers,
                              softmax_structure=softmax_struct)
            # if args.distilled_param_path:
            #     model.load_state_dict(model.mixed_model_state_dict(args.full_model_path, args.distilled_param_path))
        else:
            model = models.__dict__[arch]()
        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda()
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        if hasattr(self, '_teacher_model'):
            with torch.no_grad():
                teacher_output = self._teacher_model(data)
            loss = self.loss(output, teacher_output, target)
        else:
            loss = self.loss(output, target)

        return loss, output

    def distributed(self):
        self.model = DDP(self.model)

    def load_model_state(self, state):
        if not state is None:
            self.model.load_state_dict(state)


def get_optimizer(parameters, fp16, lr, momentum, weight_decay,
                  nesterov=False,
                  state=None,
                  static_loss_scale=1., dynamic_loss_scale=False,
                  bn_weight_decay = False):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        structured_params = [v for n, v in parameters if getattr(v, '_is_structured', False)]
        no_wd_params = [v for n, v in parameters if getattr(v, '_no_wd', False)]
        unstructured_params = [v for n, v in parameters
                               if not getattr(v, '_is_structured', False) and not getattr(v, '_no_wd', False)]
        params_dict = [{'params': structured_params, 'weight_decay': 0.0},
                       {'params': no_wd_params, 'weight_decay': 0.0},
                       {'params': unstructured_params}]
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        structured_params = [v for n, v in parameters if getattr(v, '_is_structured', False)]
        no_wd_params = [v for n, v in parameters if getattr(v, '_no_wd', False) or 'bn' in n]
        unstructured_params = [v for n, v in parameters
                               if not getattr(v, '_is_structured', False) and not getattr(v, '_no_wd', False) and 'bn' not in n]
        params_dict = [{'params': structured_params, 'weight_decay': 0.0},
                       {'params': no_wd_params, 'weight_decay': 0.0},
                       {'params': unstructured_params}]
    optimizer = torch.optim.SGD(params_dict, lr,
                                momentum=momentum,
                                weight_decay=weight_decay,
                                nesterov = nesterov)
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale,
                                   verbose=False)

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr', log.IterationMeter(), log_level=1)
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, epoch_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            # lr = base_lr * (epoch + 1) / warmup_length
            lr = base_lr * (1 + iteration + epoch * epoch_length) / (warmup_length * epoch_length)
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, epoch_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            # lr = base_lr * (epoch + 1) / warmup_length
            lr = base_lr * (1 + iteration + epoch * epoch_length) / (warmup_length * epoch_length)
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1-(e/es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, epoch_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            # lr = base_lr * (epoch + 1) / warmup_length
            lr = base_lr * (1 + iteration + epoch * epoch_length) / (warmup_length * epoch_length)
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier)/es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)



def get_train_step(model_and_loss, optimizer, fp16, use_amp = False, batch_size_multiplier = 1):
    def _step(input, target, optimizer_step=True, sync_metrics=False):
        input_var = input
        target_var = target
        loss, output = model_and_loss(input_var, target_var)
        # prec1, prec5 = torch.zeros(1), torch.zeros(1)
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
        reduced_loss = loss.data

        # For best performance, it doesn't make sense to sync these metrics every
        # iteration, since they incur an allreduce and some host<->device syncs.
        if sync_metrics:
            if torch.distributed.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.data)
                prec1 = utils.reduce_tensor(prec1)
                prec5 = utils.reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if optimizer_step:
            opt = optimizer.optimizer if isinstance(optimizer, FP16_Optimizer) else optimizer
            for param_group in opt.param_groups:
                for param in param_group['params']:
                    param.grad /= batch_size_multiplier

            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def train(train_loader, model_and_loss, optimizer, lr_scheduler, fp16, logger, epoch, print_freq,
          use_amp=False, prof=-1, batch_size_multiplier=1, register_metrics=True):

    if register_metrics and logger is not None:
        logger.register_metric('train.top1', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.top5', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.loss', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.total_time', log.AverageMeter(), log_level=0)
        logger.register_metric('train.compute_ips', log.AverageMeter(), log_level=1)
        logger.register_metric('train.total_ips', log.AverageMeter(), log_level=0)
        logger.register_metric('train.data_time', log.AverageMeter(), log_level=1)
        logger.register_metric('train.compute_time', log.AverageMeter(), log_level=1)

    step = get_train_step(model_and_loss, optimizer, fp16, use_amp = use_amp, batch_size_multiplier = batch_size_multiplier)

    model_and_loss.train()
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        if prof > 0:
            if i >= prof:
                break

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, prec1, prec5 = step(input, target, optimizer_step = optimizer_step,
                                  sync_metrics=i % print_freq == 0)

        it_time = time.time() - end

        if logger is not None:
            logger.log_metric('train.top1', to_python_float(prec1))
            logger.log_metric('train.top5', to_python_float(prec5))
            logger.log_metric('train.loss', to_python_float(loss))
            logger.log_metric('train.total_time', it_time)
            logger.log_metric('train.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('train.total_ips', calc_ips(bs, it_time))
            logger.log_metric('train.data_time', data_time)
            logger.log_metric('train.compute_time', it_time - data_time)

        end = time.time()



def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = input
        target_var = target

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader, model_and_loss, fp16, logger, epoch, prof=-1, register_metrics=True):
    if register_metrics and logger is not None:
        logger.register_metric('val.loss',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.top1',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.top5',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.compute_ips',  log.AverageMeter(), log_level = 1)
        logger.register_metric('val.total_ips',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.data_time',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.compute_time', log.AverageMeter(), log_level = 1)

    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)

    for i, (input, target) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end
        if prof > 0:
            if i > prof:
                break

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric('val.top1', to_python_float(prec1))
            logger.log_metric('val.top5', to_python_float(prec5))
            logger.log_metric('val.loss', to_python_float(loss))
            logger.log_metric('val.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('val.total_ips', calc_ips(bs, it_time))
            logger.log_metric('val.data_time', data_time)
            logger.log_metric('val.compute_time', it_time - data_time)

        end = time.time()

    return top1.get_val()

# Train loop {{{
def calc_ips(batch_size, time):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    tbs = world_size * batch_size
    return tbs/time

def train_loop(model_and_loss, optimizer, lr_scheduler, train_loader, val_loader, epochs, fp16, logger,
               should_backup_checkpoint, print_freq, use_amp=False,
               batch_size_multiplier = 1,
               best_prec1 = 0, start_epoch = 0, prof = -1, skip_training = False, skip_validation = False, save_checkpoints = True, checkpoint_dir='./'):

    prec1 = -1

    epoch_iter = range(start_epoch, epochs)
    if logger is not None:
        epoch_iter = logger.epoch_generator_wrapper(epoch_iter)
    for epoch in epoch_iter:
        if not skip_training:
            train(train_loader, model_and_loss, optimizer, lr_scheduler, fp16, logger, epoch, print_freq,
                  use_amp = use_amp, prof = prof, register_metrics=epoch==start_epoch, batch_size_multiplier=batch_size_multiplier)

        if not skip_validation:
            prec1 = validate(val_loader, model_and_loss, fp16, logger, epoch, prof = prof, register_metrics=epoch==start_epoch)

        if save_checkpoints and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if should_backup_checkpoint(epoch):
                backup_filename = 'checkpoint-{}.pth.tar'.format(epoch + 1)
            else:
                backup_filename = None
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_and_loss.arch,
                'state_dict': model_and_loss.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_dir=checkpoint_dir, backup_filename=backup_filename)
# }}}

def get_input_cov(model, train_loader, layer_names, max_batches=None):
    # hook to capture intermediate inputs
    def hook(module, input):
        x, = input
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        if not hasattr(module, '_count'):
            module._count = 1
        else:
            module._count += 1
        # Compute the first moment E[X], averaged over batches.
        current_mean = x.mean(dim=0)
        if not hasattr(module, '_mean'):
            module._mean = current_mean
        else:
            module._mean += (current_mean - module._mean) / module._count
        # Compute the covariance (actually 2nd moment) E[X^T X], averaged over batches.
        current_cov = (x.t() @ x) / x.shape[0]
        if not hasattr(module, '_cov'):
            module._cov = current_cov
        else:
            module._cov += (current_cov - module._cov) / module._count

    module_dict = dict(model.named_modules())
    hook_handles = []
    for layer_name in layer_names:
        hook_handles.append(module_dict[layer_name].register_forward_pre_hook(hook))

    model.eval()
    data_iter = enumerate(train_loader)
    for batch_idx, (input, _) in data_iter:
        with torch.no_grad():
            model(input)
        if max_batches is not None and batch_idx >= max_batches:
            # DALI iterator doesn't support resetting before finishing the epoch
            # if hasattr(train_loader, 'dalipipeline'):
            #     train_loader.dalipipeline.reset()
            break
    for h in hook_handles:
        h.remove()
    # mean = {layer_name + '.mean': module_dict[layer_name]._mean for layer_name in layer_names}
    cov = {layer_name: module_dict[layer_name]._cov for layer_name in layer_names}
    if torch.distributed.is_initialized():
        cov = {layer_name: utils.reduce_tensor(c.data) for layer_name, c in cov.items()}
    return cov


def butterfly_projection_cov(teacher_module, input_cov, butterfly_structure='odo_1',
                             n_Adam_steps=20000, n_LBFGS_steps=50):
    try:
        in_channels = teacher_module.in_channels
        out_channels = teacher_module.out_channels
    except:
        raise ValueError("Only convolutional layers currently supported.")
    param = butterfly_structure.split('_')[0]
    nblocks = 0 if len(butterfly_structure.split('_')) <= 1 else int(butterfly_structure.split('_')[1])
    residual = False if len(butterfly_structure.split('_')) <= 2 else (butterfly_structure.split('_')[2] == 'res')
    student_module = Butterfly1x1Conv(in_channels, out_channels,
        bias=False, tied_weight=False, ortho_init=True,
        param=param, nblocks=nblocks)
    student_module = student_module.to(input_cov.device)

    with torch.no_grad():
        Sigma, U = torch.symeig(input_cov, eigenvectors=True)
        Sigma = Sigma.clamp(0)  # avoid small negative eigenvalues
        input = torch.diag(Sigma.sqrt()) @ U.t()
        input = input.reshape(in_channels, in_channels, 1, 1)  # To be compatible with conv2d
        target = teacher_module(input)
        # Normalize input so that output has MSE 1.0
        input /= (target ** 2).mean().sqrt()
        target = teacher_module(input)

    def loss_fn():
        output = student_module(input)
        if residual:
            if output.shape[1] == 2 * input.shape[1]:
                b, c, h, w = input.shape
                output = (output.reshape(b, 2, c, h, w) + input.reshape(b, 1, c, h, w)).reshape(b, 2 * c, h, w)
            else:
                output = output + input
        return F.mse_loss(output, target)

    optimizer = optim.Adam(student_module.parameters())
    student_module.train()
    for _ in range(n_Adam_steps):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()

    optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, student_module.parameters()),
                            tolerance_grad=1e-7,  # Pytorch 1.2 sets this too high https://github.com/pytorch/pytorch/pull/25240
                            line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss
    for i in range(n_LBFGS_steps):
        loss = optimizer.step(closure)

    if torch.distributed.is_initialized():
        # Get the model from the process with the lowest loss.
        # Losses could be different due to different initialization of student_module.
        all_losses = [torch.empty_like(loss) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(all_losses, loss)
        best_rank = min(range(len(all_losses)), key=lambda i: all_losses[i])
        loss = all_losses[best_rank]
        for p in student_module.parameters():
            torch.distributed.broadcast(p, best_rank)

    return student_module, loss.item()
