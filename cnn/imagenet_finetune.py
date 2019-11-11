import argparse
import os
import shutil
import time
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import imagenet.logger as log
from imagenet.smoothing import LabelSmoothing, KnowledgeDistillationLoss
from imagenet.mixup import NLLMultiLabelSmooth, MixUpWrapper
from imagenet.dataloaders import DATA_BACKEND_CHOICES
from imagenet.dataloaders import get_pytorch_train_loader, get_pytorch_val_loader
from imagenet.dataloaders import get_dali_train_loader, get_dali_val_loader
from imagenet.training import ModelAndLoss, get_optimizer, train_loop
from imagenet.training import lr_step_policy, lr_cosine_policy, lr_linear_policy
from imagenet.training import get_input_cov, butterfly_projection_cov
from imagenet.utils import should_backup_checkpoint, save_checkpoint


def add_parser_arguments(parser):
    custom_model_names = ['mobilenetv1']
    model_names = sorted(name for name in models.__dict__
                        if name.islower() and not name.startswith("__")
                        and callable(models.__dict__[name])) + custom_model_names

    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--data-backend', metavar='BACKEND', default='dali-cpu',
                        choices=DATA_BACKEND_CHOICES)

    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

    parser.add_argument('--struct', metavar='STRUCT', default='odo_4',
                        type=str,
                        help='structure for 1x1 conv: ' + ' (default: odo_4)')
    parser.add_argument('--softmax-struct', metavar='SMSTRUCT', default='D',
                        type=str,
                        help='structure for softmax layer: ' + ' (default: D)')
    parser.add_argument('--sm-pooling', metavar='SMPOOL', default=1,
                        type=int,
                        help='pooling before the softmax layer: ' + ' (default: 1)')
    parser.add_argument('--n-struct-layers', default=0, type=int,
                        metavar='NSL', help='Number of structured layer (default 7)')
    parser.add_argument('--width', default=1.0, type=float,
                        metavar='WIDTH', help='Width multiplier of the CNN (default 1.0)')
    parser.add_argument('--distilled-param-path', default='', type=str, metavar='PATH',
                        help='path to distilled parameters (default: none)')
    parser.add_argument('--full-model-path', default='', type=str, metavar='PATH',
                        help='path to full model checkpoint (default: none)')

    parser.add_argument("--temperature", default=1., type=float,
                        help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha-ce", default=0.0, type=float,
                        help="Linear weight for the distillation loss. Must be >=0.")

    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 5)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256) per gpu')

    parser.add_argument('--optimizer-batch-size', default=-1, type=int,
                        metavar='N', help='size of a total batch size, for simulating bigger batches')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr-schedule', default='step', type=str, metavar='SCHEDULE', choices=['step','linear','cosine'])

    parser.add_argument('--warmup', default=5, type=int,
                        metavar='E', help='number of warmup epochs')

    parser.add_argument('--label-smoothing', default=0.0, type=float,
                        metavar='S', help='label smoothing')
    parser.add_argument('--mixup', default=0.0, type=float,
                        metavar='ALPHA', help='mixup alpha')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--bn-weight-decay', action='store_true',
                        help='use weight_decay on batch normalization learnable parameters, default: false)')
    parser.add_argument('--nesterov', action='store_true',
                        help='use nesterov momentum, default: false)')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained-weights', default='', type=str, metavar='PATH',
                        help='load weights from here')

    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')
    parser.add_argument('--prof', type=int, default=-1,
                        help='Run only N iterations')
    parser.add_argument('--amp', action='store_true',
                        help='Run model AMP (automatic mixed precision) mode.')

    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument('--seed', default=None, type=int,
                        help='random seed used for np and pytorch')

    parser.add_argument('--gather-checkpoints', action='store_true',
                        help='Gather checkpoints throughout the training')

    parser.add_argument('--raport-file', default='experiment_raport.json', type=str,
                        help='file in which to store JSON experiment raport')

    parser.add_argument('--final-weights', default='model.pth.tar', type=str,
                        help='file in which to store final model weights')

    parser.add_argument('--evaluate', action='store_true', help='evaluate checkpoint/model')
    parser.add_argument('--training-only', action='store_true', help='do not evaluate')

    parser.add_argument('--no-checkpoints', action='store_false', dest='save_checkpoints')

    parser.add_argument('--workspace', type=str, default='./')


def main(args):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.amp and args.fp16:
        print("Please use only one of the --fp16/--amp flags")
        exit(1)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print("Warning: simulated batch size {} is not divisible by actual batch size {}".format(args.optimizer_batch_size, tbs))
        batch_size_multiplier = int(args.optimizer_batch_size/ tbs)
        print("BSM: {}".format(batch_size_multiplier))

    pretrained_weights = None
    if args.pretrained_weights:
        if os.path.isfile(args.pretrained_weights):
            print("=> loading pretrained weights from '{}'".format(args.pretrained_weights))
            pretrained_weights = torch.load(args.pretrained_weights)
        else:
            print("=> no pretrained weights found at '{}'".format(args.resume))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model_state = checkpoint['state_dict']
            # Strip names to be compatible with Pytorch 1.2, i.e. 'module.conv1.weight' -> 'conv1.weight'
            model_state = {name.replace('module.', ''): weight for name, weight in model_state.items()}
            optimizer_state = checkpoint['optimizer']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None


    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)
    if args.alpha_ce > 0.0:
        loss_og = loss()
        loss = lambda: KnowledgeDistillationLoss(loss_og, args.temperature, args.alpha_ce)

    # Create data loaders and optimizers as needed
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()

    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, 1000, args.mixup > 0.0, workers=args.workers, fp16=args.fp16)
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, 1000, train_loader)
    train_loader._len = train_loader_len

    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, 1000, False, workers=args.workers, fp16=args.fp16)
    val_loader._len = val_loader_len

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger = log.Logger(
                args.print_freq,
                [
                    log.JsonBackend(os.path.join(args.workspace, args.raport_file), log_level=1),
                    log.StdOut1LBackend(train_loader_len, val_loader_len, args.epochs * args.n_struct_layers, log_level=0),
                ])

        for k, v in args.__dict__.items():
            logger.log_run_tag(k, v)
    else:
        logger = None

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.

    if args.lr_schedule == 'step':
        lr_policy = lr_step_policy(args.lr, [15,20], 0.1, args.warmup, train_loader_len, logger=logger)
    elif args.lr_schedule == 'cosine':
        lr_policy = lr_cosine_policy(args.lr, args.warmup, args.epochs, train_loader_len, logger=logger)
    elif args.lr_schedule == 'linear':
        lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs, train_loader_len, logger=logger)

    model_and_loss = ModelAndLoss(
            args.arch,
            loss,
            pretrained_weights=pretrained_weights,
            cuda = True, fp16 = args.fp16,
            width=args.width, n_struct_layers=0,
            struct=args.struct, softmax_struct=args.softmax_struct, sm_pooling=args.sm_pooling)
    checkpoint = torch.load(args.full_model_path, map_location = lambda storage, loc: storage.cuda(args.gpu))
    model_state = checkpoint['state_dict']
    # Strip names to be compatible with Pytorch 1.2, i.e. 'module.conv1.weight' -> 'conv1.weight'
    model_state = {name.replace('module.', ''): weight for name, weight in model_state.items()}
    model_and_loss.model.load_state_dict(model_state)
    if args.distributed:
        model_and_loss.distributed()
    teacher_model = model_and_loss.model
    layers_to_replace = ['layers.12.conv2', 'layers.11.conv2', 'layers.10.conv2', 'layers.9.conv2',
                         'layers.8.conv2', 'layers.7.conv2', 'layers.6.conv2']
    for n_struct_layers in range(1, args.n_struct_layers + 1):
        print(f'Fine-tuning the {n_struct_layers}-th to last layer.')
        layer_name = layers_to_replace[n_struct_layers - 1]
        # if args.local_rank == 0:
        #     print(dict(model_and_loss.model.named_modules()).keys())
        start = time.perf_counter()
        # layer name seems to have 'module.' x n_struct_layers for some reason
        layer_name_prefix = ''.join(['module.'] * (1 if n_struct_layers == 1 else 2))
        input_cov = get_input_cov(model_and_loss.model, train_loader, [layer_name_prefix + layer_name], max_batches=100)[layer_name_prefix + layer_name]
        end = time.perf_counter()
        print(f'Got input_cov in {end - start}s')
        teacher_module = dict(model_and_loss.model.named_modules())[layer_name_prefix + layer_name]
        start = time.perf_counter()
        student_module, _ = butterfly_projection_cov(teacher_module, input_cov, args.struct)
        end = time.perf_counter()
        print(f'Butterfly projection in {end - start}s')

        prev_state_dict = model_and_loss.model.state_dict()
        prev_state_dict = {name.replace('module.', ''): weight for name, weight in prev_state_dict.items()}
        structured_params = student_module.state_dict()
        model_and_loss = ModelAndLoss(
                args.arch,
                loss,
                pretrained_weights=pretrained_weights,
                cuda = True, fp16 = args.fp16,
                width=args.width, n_struct_layers=n_struct_layers,
                struct=args.struct, softmax_struct=args.softmax_struct, sm_pooling=args.sm_pooling)
        current_state_dict_keys = model_and_loss.model.state_dict().keys()
        state_dict = {name: param for name, param in prev_state_dict.items() if name in current_state_dict_keys}
        state_dict.update({layer_name + '.' + name: param for name, param in structured_params.items()})
        model_and_loss.model.load_state_dict(state_dict)
        if args.distributed:
            model_and_loss.distributed()

        optimizer = get_optimizer(list(model_and_loss.model.named_parameters()),
                args.fp16,
                args.lr, args.momentum, args.momentum, args.weight_decay,
                nesterov = args.nesterov,
                bn_weight_decay = args.bn_weight_decay,
                state=optimizer_state,
                static_loss_scale = args.static_loss_scale,
                dynamic_loss_scale = args.dynamic_loss_scale)

        if args.amp:
            model_and_loss, optimizer = amp.initialize(
                    model_and_loss, optimizer,
                    opt_level="O1",
                    loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale)

        if args.distributed:
            model_and_loss.distributed()
        if args.alpha_ce > 0.0:
            model_and_loss._teacher_model = teacher_model
            model_and_loss._teacher_model.eval()

        train_loop(
            model_and_loss, optimizer,
            lr_policy,
            train_loader, val_loader, args.epochs,
            args.fp16, logger, should_backup_checkpoint(args), args.print_freq, use_amp=args.amp,
            batch_size_multiplier = batch_size_multiplier,
            start_epoch = args.start_epoch, best_prec1 = best_prec1, prof=args.prof,
            skip_training = args.evaluate, skip_validation = args.training_only,
            save_checkpoints=args.save_checkpoints and not args.evaluate, checkpoint_dir=args.workspace)

    exp_duration = time.time() - exp_start_time
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.end()
    print("Experiment ended")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True

    main(args)
