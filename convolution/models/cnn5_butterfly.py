'''Baseline CNN in PyTorch.'''
# Adapted from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

import torch.nn as nn
import torch.nn.functional as F

from .cnn5 import CNN5
from .kops import KOP2d


class CNN5Butterfly(CNN5):
    name = 'cnn5butterfly'

    def __init__(self, num_channels=32, num_classes=10, **kwargs):
        nn.Module.__init__(self)
        self.num_channels = num_channels
        in_size = 32
        self.conv1 = KOP2d(in_size, 3, num_channels, 3, **kwargs)
        self.bn1   = nn.BatchNorm2d(num_channels)
        self.conv2 = KOP2d(in_size // 2, num_channels, num_channels * 2, 3, **kwargs)
        self.bn2   = nn.BatchNorm2d(num_channels * 2)
        self.conv3 = KOP2d(in_size // 4, num_channels * 2, num_channels * 4, 3, **kwargs)
        self.bn3   = nn.BatchNorm2d(num_channels * 4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4 * 4 * num_channels * 4, num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(num_channels * 4)
        self.fc2 = nn.Linear(num_channels * 4, num_classes)

