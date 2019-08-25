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
import torchvision.models as models
import logging

from mobilenet_imagenet import MobileNet

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from imagenet.dataloaders import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ['mobilenetv1', 'mobilenetv1_struct']

def get_parser():
    parser = argparse.ArgumentParser(description='Save PyTorch ImageNet covariance of activations.')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--output-dir', type=str,
        default=Path.cwd(), help='Directory to save intermediates.')
    parser.add_argument('--data-backend', metavar='BACKEND', default='dali-cpu',
                            choices=DATA_BACKEND_CHOICES)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--width', default=1.0, type=float,
                        metavar='WIDTH', help='Width multiplier of the CNN (default 1.0)')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='Print frequency (default: 10)')
    parser.add_argument('--layers', nargs='+', type=str, required=True,
                        help='Layers to save inputs and outputs for.'
                             'Use as a list: e.g. layer1 layer2 layer3.'
                             'More layers takes longer.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path for teacher model.')
    parser.add_argument('--max_batches', type=int, help='Maximum number of batches'
                        'to collect activations for')
    # parser.add_argument('--dataset', type=str, help='Dataset name for selecting train loader and augmentation.')
    return parser

cudnn.benchmark = True
args = get_parser().parse_args()
os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info(args)

def get_teacher_intermediates(teacher_model, train_loader, layers_to_replace):
    if args.max_batches is not None:
        num_batches = args.max_batches
    else:
        num_batches = train_loader._len

    # hook to capture intermediate inputs
    def hook(module, input):
        x, = input
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        # Compute the covariance (actually 2nd moment) E[X^T X].
        # Average over batches.
        if not hasattr(module, '_count'):
            module._count = 1
        else:
            module._count += 1
        current_cov = x.t() @ x
        if not hasattr(module, '_cov'):
            module._cov = current_cov
        else:
            module._cov += (current_cov - module._cov) / module._count

    module_dict = dict(teacher_model.named_modules())
    for layer_name in layers_to_replace:
        module_dict[layer_name].register_forward_pre_hook(hook)

    teacher_model.eval()
    batch_time = AverageMeter()
    end = time.time()
    data_iter = enumerate(train_loader)
    for batch_idx, (input, _) in data_iter:
        input = input.cuda()
        with torch.no_grad():
            teacher_model(input)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_freq == 0:
            logger.info('Batch:{0}/{1}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                   batch_idx, num_batches, batch_time=batch_time))
        if args.max_batches is not None and batch_idx == args.max_batches:
            break
    logging.info("Computed teacher intermediates. ")
    saved_cov = {layer_name: module_dict[layer_name]._cov for layer_name in layers_to_replace}
    with open(f'{args.output_dir}/input_cov.pt', 'wb') as f:
        torch.save(saved_cov, f)

def main():
    # resnet models are different for imagenet
    if args.arch == 'mobilenetv1':
        teacher_model = MobileNet(width_mult=args.width)
    else:
        teacher_model = models.__dict__[args.arch]()
    print(teacher_model)
    loaded_state_dict = torch.load(args.model_path)['state_dict']
    # Strip names to be compatible with Pytorch 1.2, i.e. 'module.conv1.weight' -> 'conv1.weight'
    loaded_state_dict = {name.replace('module.', ''): weight for name, weight in loaded_state_dict.items()}
    teacher_model.load_state_dict(loaded_state_dict)

    teacher_model = teacher_model.cuda()
    logging.info(teacher_model)
    modules = [name for name, _ in teacher_model.named_modules()]
    logging.info(modules)

    # filter modules to save inputs and outputs as candidates for
    # butterfly replacement
    # tradeoff in potentially needing to rerun v. needing to support
    # loading and saving more activations
    layers_to_replace = args.layers
    for layer in layers_to_replace:
        assert layer in modules, f"{layer} not in network"
    logger.info(layers_to_replace)

    # load data
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
    elif args.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
    elif args.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, 1000, False, workers=args.workers, fp16=False)
    train_loader._len = train_loader_len

    logger.info('Loaded data')

    get_teacher_intermediates(teacher_model, train_loader, layers_to_replace)


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


if __name__ == '__main__': main()
