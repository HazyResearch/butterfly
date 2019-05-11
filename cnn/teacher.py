import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import io
import argparse, shutil, time, warnings
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet_imagenet as models # only use imagenet models
from models.butterfly_conv import ButterflyConv2d, ButterflyConv2dBBT
import logging
import torchvision.models as torch_models
from train_utils import AverageMeter, data_prefetcher, strip_prefix_if_present
import models.resnet_imagenet as models # only use imagenet models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_parser():
    parser = argparse.ArgumentParser(description='Save PyTorch ImageNet intermediate activations.')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--output-dir', type=str,
        default=Path.cwd(), help='Directory to save intermediates.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='Print frequency (default: 10)')
    parser.add_argument('--layers', type=str, required=True,
                        help='Layers to save inputs and outputs for.'
                             'Use as a list: e.g. layer1,layer2,layer3.'
                             'More layers takes longer.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path for teacher model.')
    parser.add_argument('--max_batches', type=int, help='Maximum number of batches'
                        'to collect activations for')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(args)

def get_size(input):
    # don't include batch size
    if isinstance(input, tuple):
        input_sizes = input[0].size()[1:]
        assert len(input) == 1, "input tuple size greater than 1"
    else:
        input_sizes = input.size()[1:]
    return input_sizes

def get_teacher_intermediates(teacher_model, train_loader, layers_to_replace):
    if args.max_batches is not None:
        num_batches = args.max_batches
    else:
        num_batches = len(train_loader)

    # define hook to capture intermediate inputs/activations and intermediate outputs
    # serialize outputs
    # keep dictionary to io objects for different layers
    teacher_inputs = {}
    teacher_outputs = {}
    teacher_input_size = {}
    teacher_output_size = {}

    # store ids of modules to names -- can't figure out
    # other way to get same name of module within hook
    name_map = {}
    for name, module in teacher_model.named_modules():
        name_map[id(module)] = name

    batch_size = args.batch_size
    # use memory mapping to manage large activations
    def make_mmap_file(path, input_size):
        view_size = torch.Size([num_batches * args.batch_size]) + input_size
        # shared needs to be true for file to be created
        return torch.from_file(path, size=int(np.prod(view_size)),
                shared=True).view(view_size)

    batch_idx = 0
    data_idx = 0
    # TODO: store only inputs or outputs (otherwise we may be storing duplicate info
    # if we already stored neighboring layer)
    # won't cause answers to be wrong, but could be wasteful
    def hook(module, input, output):
        current_batch_size = output.size(0)
        mod_id = id(module)
        input_size = get_size(input)
        output_size = get_size(output)
        if mod_id not in teacher_inputs:
            teacher_inputs[mod_id] = make_mmap_file(
                f'{args.output_dir}/{name_map[mod_id]}_input.pt', input_size)
            teacher_outputs[mod_id] = make_mmap_file(
                f'{args.output_dir}/{name_map[mod_id]}_output.pt', output_size)
        if mod_id not in teacher_input_size:
            teacher_input_size[mod_id] = input_size
            teacher_output_size[mod_id] = output_size

        # save inputs to memory mapped files
        # TODO: input always a length-1 tuple?
        teacher_inputs[mod_id][data_idx:data_idx+current_batch_size] = input[0].cpu().detach()
        teacher_outputs[mod_id][data_idx:data_idx+current_batch_size] = output.cpu().detach()

    teacher_model.eval()
    for name, module in teacher_model.named_modules():
        if name in layers_to_replace:
            module.register_forward_hook(hook)

    batch_time = AverageMeter()
    prefetcher = data_prefetcher(train_loader)
    input, _ = prefetcher.next()
    end = time.time()
    while input is not None:
        input = input.cuda()
        teacher_model(input)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_freq == 0:
            logger.info('Batch:{0}/{1}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                   batch_idx, num_batches, batch_time=batch_time))
        batch_idx += 1
        data_idx += input.size(0) # for variable size batches
        if args.max_batches is not None and batch_idx == args.max_batches:
            break
        input, _ = prefetcher.next()
    logging.info("Computed teacher intermediates. ")

    # write sizes to disk for easy loading of memory maps at a later time
    for layer_id in teacher_inputs:
        layer_name = name_map[layer_id]
        # write sizes
        with open(f'{args.output_dir}/{layer_name}_input_sz.pt', 'wb') as f:
            input_size = torch.Size([data_idx]) + teacher_input_size[layer_id]
            torch.save(input_size, f)
        with open(f'{args.output_dir}/{layer_name}_output_sz.pt', 'wb') as f:
            output_size = torch.Size([data_idx]) + teacher_output_size[layer_id]
            torch.save(output_size, f)

def main():
    # load pretrained teacher model from torchvision
    # teacher_model = torch_models.__dict__[args.arch](pretrained=True)
    teacher_model = models.__dict__[args.arch]()
    loaded_state_dict = torch.load(args.model_path)['state_dict']
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    teacher_model.load_state_dict(loaded_state_dict)

    teacher_model.cuda()
    logging.info(teacher_model)
    modules = set([name for name, _ in teacher_model.named_modules()])

    # filter modules to save inputs and outputs as candidates for
    # butterfly replacement
    # tradeoff in potentially needing to rerun v. needing to support
    # loading and saving more activations
    layers_to_replace = [layer.strip() for layer in args.layers.split(',')]
    for layer in layers_to_replace:
        assert layer in modules, "Layer not in network"
    logger.info(layers_to_replace)

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
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    logger.info('Loaded data')

    get_teacher_intermediates(teacher_model, train_loader, layers_to_replace)

if __name__ == '__main__': main()