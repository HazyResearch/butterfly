import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import io
import argparse, shutil, time, warnings
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet_imagenet as models # only use imagenet models
from distributed import DistributedDataParallel as DDP
from models.butterfly_conv import ButterflyConv2d, ButterflyConv2dBBT
import logging
from distributed import DistributedDataParallel as DDP

import torchvision.models as torch_models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--output-dir', type=str,
        default=Path.cwd(), help='Directory to save intermediates.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='Print frequency (default: 10)')
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
    parser.add_argument('--write_freq', default=10, type=int, help='Frequency to write activations' +
                        'to disk. Affected by number of activations collected at a time' +
                        'and size of available RAM.')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(args)

def get_size(input):
    if isinstance(input, tuple):
        input_sizes = input[0].size()
        assert len(input) == 1, "input tuple size greater than 1"
    else:
        input_sizes = input.size()
    return input_sizes

def get_teacher_intermediates(teacher_model, train_loader, layers_to_replace):

    num_batches = len(train_loader)

    # define hook to capture intermediate inputs/activations and intermediate outputs
    # serialize outputs
    # keep dictionary to io objects for different layers
    teacher_inputs = {}
    teacher_outputs = {}
    teacher_input_size = {}
    teacher_output_size = {}

    def make_mmap_file(path, size):
        view_size = torch.Size([num_batches] + list(size))
        # shared needs to be true for file to be created
        return torch.from_file(path, size=int(np.prod(view_size)),
            shared=True).view(view_size)

    # store ids of modules to names -- can't figure out
    # other way to get same name of module within hook
    name_map = {}
    for name, module in teacher_model.named_modules():
        name_map[id(module)] = name

    global batch_idx
    batch_idx = 0
    def hook(module, input, output):
        mod_id = id(module)
        if mod_id not in teacher_inputs:
            teacher_inputs[mod_id] = make_mmap_file(f'{args.output_dir}/{name_map[mod_id]}_input.pt', get_size(input))
            teacher_outputs[mod_id] = make_mmap_file(f'{args.output_dir}/{name_map[mod_id]}_output.pt', get_size(output))
        if mod_id not in teacher_input_size:
            teacher_input_size[mod_id] = get_size(input)
            teacher_output_size[mod_id] = get_size(output)
        # input is a tuple?
        teacher_inputs[mod_id][batch_idx] = input[0].cpu().detach()
        teacher_outputs[mod_id][batch_idx] = output.cpu().detach()

    teacher_model.eval()
    end = time.time()

    batch_time = AverageMeter()
    for name, module in teacher_model.named_modules():
        if name in layers_to_replace:
            module.register_forward_hook(hook)

    batch_time = AverageMeter()
    prefetcher = data_prefetcher(train_loader)
    input, _ = prefetcher.next()
    batch_idx = 0

    while input is not None:
        input = input.cuda()
        teacher_model(input)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        input, _ = prefetcher.next()

        if args.rank == 0 and batch_idx % args.print_freq == 0:
            logger.info('{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                   batch_idx, len(train_loader), batch_time=batch_time))

        batch_idx += 1

    logging.info("Computed teacher intermediates. ")

    # save in-memory io objects to disk for distilling
    # at later time
    for layer_id in teacher_inputs:
        layer_name = name_map[layer_id]
        # write sizes
        with open(f'{args.output_dir}/{layer_name}_input_sz.pt', 'wb') as f:
            input_size = torch.Size([batch_idx] + list(teacher_input_size[layer_id]))
            torch.save(input_size, f)
        with open(f'{args.output_dir}/{layer_name}_output_sz.pt', 'wb') as f:
            output_size = torch.Size([batch_idx] + list(teacher_output_size[layer_id]))
            torch.save(output_size, f)


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

def main():
    # load pretrained teacher model from torchvision
    teacher_model = torch_models.__dict__[args.arch](pretrained=True)

    teacher_model.cuda()

    # filter modules to save inputs and outputs as candidates for
    # butterfly replacement
    # tradeoff in potentially needing to rerun v. needing to support
    # loading and saving more activations
    # TODO: better name filtering?
    # print([name for name, _ in teacher_model.named_modules()])
    # layers_to_replace = [name for name, _ in teacher_model.named_modules()
    #     if 'conv' in name or 'downsample' in name]
    layers_to_replace = ['layer4.1.conv2']

    # print(layers_to_replace)
    # fill in candidates here
    # layers_to_replace = ["module.layer3.1.conv1"]
    logger.info(layers_to_replace)

    # for layer_name in layers_to_replace:
    #     if (os.path.exists(f'{args.output_dir}/{layer_name}_input.pt') or
    #         os.path.exists(f'{args.output_dir}/{layer_name}_output.pt')):
    #         raise ValueError("Teacher input/outputs already exist.")

    # load data
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    tensor_tfm = [transforms.ToTensor(), normalize]
    # train on standard ImageNet size images
    train_dataset = datasets.ImageFolder(
        traindir, transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
        ] + tensor_tfm))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    logger.info('Loaded data')

    get_teacher_intermediates(teacher_model, train_loader, layers_to_replace)

if __name__ == '__main__': main()

