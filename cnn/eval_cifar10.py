import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')

import math
from pathlib import Path
import pickle
import random

import numpy as np

import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import models
import dataset_utils

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--num_structured_layers', default=0, type=int,
                        help='Number of layers starting from the back that are structured')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='MobileNet')
    parser.add_argument('--structure_type', default='B', type=str, choices=['B', 'BBT', 'LR'],
                        help='Structure of butterflies')
    parser.add_argument('--nblocks', default=1, type=int, help='Number of blocks for each butterfly')
    parser.add_argument('--param', default='regular', type=str, help='Parametrization of butterfly factors')
    parser.add_argument('--path', type=str, help='Path to state dict for model')
    return parser

def _test(model, test_loader):
    model.eval()
    model.to(device)
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.data.view_as(pred)).long().cpu().sum()
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct.item() / len(test_loader.dataset)
    print(test_loss, accuracy, len(test_loader.dataset))
    return {"mean_loss": test_loss,
            "mean_accuracy": accuracy}

args = get_parser().parse_args()
device = 'cuda'
config = {'name': 'CIFAR10'}
train_loader, valid_loader, test_loader = dataset_utils.get_dataset(config)
model = models.__dict__[args.arch](num_structured_layers=args.num_structured_layers,
    structure_type=args.structure_type, nblocks=args.nblocks, param=args.param)
if args.num_structured_layers != 0:
    model.load_state_dict(torch.load(args.path))
else:
    model.load_state_dict(torch.load(args.path)['model'])

_test(model, test_loader)
