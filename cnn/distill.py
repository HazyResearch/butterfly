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
import math
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet_imagenet as imagenet_models # only use imagenet models
import models
from models.butterfly_conv import ButterflyConv2d, ButterflyConv2dBBT
from models.low_rank_conv import LowRankConv2d
import logging
import torch.utils.data as data_utils
import torchvision.models as torch_models
from train_utils import AverageMeter
import torch.nn.functional as F

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_parser():
    parser = argparse.ArgumentParser(description='Learn mapping from pretrained input to output for structured layer.')
    parser.add_argument('--output-dir', type=str, default=Path.cwd(), help='Directory to save logs and models.')
    parser.add_argument('--input-dir', type=str, default=Path.cwd(), help='Directory to load intermediates.')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='Print frequency (default: 10)')
    parser.add_argument('--decay-int', default=30, type=int, help='Decay LR by 10 every decay-int epochs')
    parser.add_argument('--structure_type', default='B', type=str, choices=['B', 'BBT', 'LR'],
                        help='Structure of matrices')
    parser.add_argument('--nblocks', default=1, type=int, help='Number of blocks for the butterfly')
    parser.add_argument('--param', default='regular', type=str, help='Parametrization of butterfly factors')
    parser.add_argument('--layer', required=True, type=str, help='Layer to replace with a structured layer')
    parser.add_argument('--resume', type=str, help='Structued layer to continue distilling')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Convergence tolerance to stop training')
    parser.add_argument('--dataset', type=str, help='Dataset name for selecting correct model.', required=True)
    parser.add_argument('--size', action='store_true', help='Return the size of the weights')
    return parser

args = get_parser().parse_args()
os.makedirs(args.output_dir, exist_ok=True)
file_tag = f'{args.output_dir}/structure_nlayers_{args.layer}_{args.structure_type}_blocks_{args.nblocks}_{args.param}'
logging.basicConfig(
level=logging.INFO,
handlers=[
    logging.StreamHandler(),
    logging.FileHandler(f'{file_tag}.log', 'a')
])
logger = logging.getLogger()
logger.info(args)

def eval_student(structured_layer, train_loader, criterion, epoch):
    structured_layer.eval()
    start = time.time()
    batch_time = AverageMeter()
    loss = 0
    for i, (teacher_input, teacher_output) in enumerate(train_loader):
        teacher_output = teacher_output.cuda()
        teacher_input = teacher_input.cuda()
        data_time = time.time() - start
        student_output = structured_layer(teacher_input)
        loss += F.mse_loss(student_output, teacher_output).item()
        batch_time.update(time.time() - start)
        start = time.time()
    training_loss = loss / len(train_loader)
    logging.info(f'Epoch[{epoch}]\t'
                f'Training Loss: {training_loss:.5f}')
    return training_loss

def train_student(structured_layer, train_loader, criterion, optimizer, epoch):
    structured_layer.train()
    start = time.time()
    batch_time = AverageMeter()
    for i, (teacher_input, teacher_output) in enumerate(train_loader):
        teacher_output = teacher_output.cuda()
        teacher_input = teacher_input.cuda()
        data_time = time.time() - start
        student_output = structured_layer(teacher_input)
        loss = criterion(student_output, teacher_output)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - start)
        start = time.time()
        if i % args.print_freq == 0 and i > 0:
            logging.info(f'Epoch[{epoch}]: {i}/{len(train_loader)}\t'
                        f'Time Data: {data_time:.3f}, Batch Avg: {batch_time.avg:.3f}\t'
                        f'Loss: {loss:.5f}')

def get_mmap_files(traindir):
    input_size = torch.load(f'{traindir}/{args.layer}_input_sz.pt')
    output_size = torch.load(f'{traindir}/{args.layer}_output_sz.pt')
    logging.info(f'Input size: {input_size}')
    logging.info(f'Output size: {output_size}')
    teacher_input = torch.from_file(f'{traindir}/{args.layer}_input.pt',
        size=int(np.prod(input_size))).view(input_size).pin_memory()
    teacher_output = torch.from_file(f'{traindir}/{args.layer}_output.pt',
        size=int(np.prod(output_size))).view(output_size).pin_memory()
    return teacher_input, teacher_output

def load_teacher(traindir):
    teacher_input, teacher_output = get_mmap_files(traindir)
    train = data_utils.TensorDataset(teacher_input, teacher_output)
    train_loader = data_utils.DataLoader(train, batch_size=256, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    return train_loader

def main():
    # load pretrained teacher model from torchvision
    if args.dataset == 'cifar10':
        teacher_model = models.ResNet18()
    elif args.dataset == 'imagenet':
        teacher_model = imagenet_models.resnet18()

    modules = set([name for name, _ in teacher_model.named_modules()])
    assert args.layer in modules, f"{args.layer} not in network"

    if args.resume:
        ckpt = torch.load(args.resume)
        structured_layer = ckpt['model']
        start_epoch = ckpt['epoch'] + 1
        logging.info("Loading existing structured layer.")

    else:
        logging.info("Creating new structured layer.")
        start_epoch = 0
        # get parameters from layer to replace to use in butterfly
        for name, module in teacher_model.named_modules():
            if name == args.layer:
                try:
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                except:
                    raise ValueError("Only convolutional layers currently supported.")

        # create butterfly for specific layer and train
        if args.structure_type == 'B':
            structured_layer = ButterflyConv2d(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, tied_weight=False, ortho_init=True,
                param=args.param)
        elif args.structure_type == 'BBT' or args.nblocks > 1:
            structured_layer = ButterflyConv2dBBT(in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                bias=False, nblocks=args.nblocks, tied_weight=False,
                ortho_init=True, param=args.param)
        elif args.structure_type == 'LR':
            if args.nblocks == 0:
                rank = int(math.log2(out_channels))
            else:
                rank = int(math.log2(out_channels))*args.nblocks * 2
            structured_layer =  LowRankConv2d(in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False, rank=rank)
            print("HERE")
        else:
            raise ValueError("Invalid structure!")

    if args.size:
        total_elems= 0
        for p in structured_layer.parameters():
            total_elems += p.numel()
            logging.info(p.size())
        logging.info(f"Number of elements: {total_elems}")
        return

    structured_layer.cuda()

    # load data
    traindir = args.input_dir
    train_loader = load_teacher(traindir)

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda()
    # optimizer = torch.optim.SGD(structured_layer.parameters(), lr=args.lr, momentum=args.momentum)

    optimizer = torch.optim.Adam(structured_layer.parameters(), lr=args.lr)
    logger.info('Created optimizer')

    ckpt_file = f'{file_tag}.pt'
    prev_training_loss = float('inf')
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_student(structured_layer, train_loader, criterion, optimizer, epoch)
        training_loss = eval_student(structured_layer, train_loader, criterion, epoch)
        # save structured_layer weights to be loaded in weight surgery to pretrained model
        torch.save({'model': structured_layer, 'epoch': epoch}, ckpt_file)
        # convergence condition
        if abs(prev_training_loss - training_loss) < args.tolerance:
            logging.info("Converged!")
            break
        prev_training_loss = training_loss

if __name__ == '__main__': main()

