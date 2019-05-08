import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import argparse, shutil, time, warnings
import subprocess
from datetime import datetime
from pathlib import Path
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
import models.resnet_imagenet as models # only use imagenet models
from distributed import DistributedDataParallel as DDP
import logging

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--save-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='Print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--small', action='store_true', help='start with smaller images')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--dp', action='store_true', help='Use nn.DataParallel instead of fastai DistributedDataParallel.')
    parser.add_argument('--sz',       default=224, type=int, help='Size of transformed image.')
    parser.add_argument('--decay-int', default=30, type=int, help='Decay LR by 10 every decay-int epochs')
    parser.add_argument('--loss-scale', type=float, default=1,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--prof', dest='prof', action='store_true', help='Only run a few iters for profiling.')

    parser.add_argument('--dist-url', default='file://sync.file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--distributed-init-method')
    parser.add_argument('--world-size', default=1, type=int,
                        help='Number of GPUs to use. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                        'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--num_structured_layers', default=0, type=int,
                        help='Number of layers starting from the back that are structured')
    parser.add_argument('--structure_type', default='B', type=str, choices=['B', 'BBT', 'BBTBBT'],
                        help='Structure of butterflies')
    parser.add_argument('--nblocks', default=1, type=int, help='Number of blocks for each butterfly')
    parser.add_argument('--param', default='regular', type=str, help='Parametrization of butterfly factors')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
logging.basicConfig(
level=logging.INFO,
handlers=[
    logging.FileHandler(f'{args.save_dir}/main_thread.log'),
    logging.StreamHandler()
])
logger = logging.getLogger()
logger.info(args) 
label = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode('ascii').strip()
logger.info(f'Git hash: {label}')

def get_loaders(traindir, valdir, use_val_sampler=True, min_scale=0.08):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_tfm = [transforms.ToTensor(), normalize]

    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(args.sz, scale=(min_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ] + tensor_tfm))
    val_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(int(args.sz*1.14)),
            transforms.CenterCrop(args.sz),
        ] + tensor_tfm))

    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None)
    val_sampler = (torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(args.batch_size), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler if use_val_sampler else None)

    return train_loader,val_loader,train_sampler,val_sampler


def main():
    start_time = datetime.now()
    args.distributed = True #args.world_size > 1
    args.gpu = 0
    if args.distributed:
        import socket
        args.gpu = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        logger.info('| distributed init (rank {}): {}'.format(args.rank, args.distributed_init_method))
        dist.init_process_group( backend=args.dist_backend, init_method=args.distributed_init_method,
            world_size=args.world_size, rank=args.rank, )
        logger.info('| initialized host {} as rank {}'.format(socket.gethostname(), args.rank))
        #args.gpu = args.rank % torch.cuda.device_count()
        #torch.cuda.set_device(args.gpu)
        #logger.info('initializing...')
        #dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size)
        #logger.info('initialized')

    # create model
    if args.pretrained: model = models.__dict__[args.arch](pretrained=True)
    else: model = models.__dict__[args.arch](num_structured_layers=args.num_structured_layers,
            structure_type=args.structure_type, nblocks=args.nblocks, param=args.param)
    model = model.cuda()
    n_dev = torch.cuda.device_count()
    logger.info('Created model')
    if args.distributed: model = DDP(model)
    elif args.dp:
        model = nn.DataParallel(model)
        args.batch_size *= n_dev
    logger.info('Set up data parallel')

    global structured_params
    global unstructured_params
    structured_params = filter(lambda p: hasattr(p, '_is_structured') and p._is_structured, model.parameters())
    unstructured_params = filter(lambda p: not (hasattr(p, '_is_structured') and p._is_structured), model.parameters())

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD([{'params': structured_params, 'weight_decay': 0.0},
            {'params': unstructured_params}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logger.info('Created optimizer')
    best_acc1 = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else: logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.small:
        traindir = os.path.join(args.data+'-sz/160', 'train')
        valdir = os.path.join(args.data+'-sz/160', 'val')
        args.sz = 128
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        args.sz = 224

    train_loader,val_loader,train_sampler,val_sampler = get_loaders(traindir, valdir, use_val_sampler=True)
    logger.info('Loaded data')
    if args.evaluate: return validate(val_loader, model, criterion, epoch, start_time)

    logger.info(model)
    logger.info('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    logger.info('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f'Epoch {epoch}')
        adjust_learning_rate(optimizer, epoch)
        if epoch==int(args.epochs*0.4+0.5):
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            args.sz = 224
            train_loader,val_loader,train_sampler,val_sampler = get_loaders( traindir, valdir)
        if epoch==int(args.epochs*0.92+0.5):
            args.sz=288
            args.batch_size=128
            train_loader,val_loader,train_sampler,val_sampler = get_loaders(
                traindir, valdir, use_val_sampler=False, min_scale=0.5)

        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            train(train_loader, model, criterion, optimizer, epoch)

        if args.prof: break
        acc1 = validate(val_loader, model, criterion, epoch, start_time)

        if args.rank == 0:
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({
                'epoch': epoch + 1, 'arch': args.arch, 'state_dict': model.state_dict(),
                'best_acc1': best_acc1, 'optimizer' : optimizer.state_dict(),
            }, is_best)

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

class data_prefetcher():
    def __init__(self, loader, prefetch=True):
        self.loader,self.prefetch = iter(loader),prefetch
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        if not self.prefetch:
            input,target = next(self.loader)
            return input.cuda(non_blocking=True),target.cuda(non_blocking=True)

        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader, prefetch=True)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1
        if args.prof and (i > 200): break
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(acc1), input.size(0))
        top5.update(to_python_float(acc5), input.size(0))

        loss = loss*args.loss_scale
        # compute gradient and do SGD step

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        end = time.time()
        input, target = prefetcher.next()

        if args.rank == 0 and i % args.print_freq == 0 and i > 1:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, epoch, start_time):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = -1
    while input is not None:
        i += 1

        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        reduced_loss = reduce_tensor(loss.data)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        reduced_acc1 = reduce_tensor(acc1)
        reduced_acc5 = reduce_tensor(acc5)

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(acc1), input.size(0))
        top5.update(to_python_float(acc5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

        input, target = prefetcher.next()

    time_diff = datetime.now()-start_time
    logger.info(f'Epoch {epoch}: {float(time_diff.total_seconds())} sec')
    logger.info(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n')

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{args.save_dir}/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every few epochs"""
    if   epoch<4 : lr = args.lr/(4-epoch)
    elif epoch<int(args.epochs*0.47+0.5): lr = args.lr/1
    elif epoch<int(args.epochs*0.78+0.5): lr = args.lr/10
    elif epoch<int(args.epochs*0.95+0.5): lr = args.lr/100
    else         : lr = args.lr/1000
    for param_group in optimizer.param_groups: param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the acccuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__': main()

