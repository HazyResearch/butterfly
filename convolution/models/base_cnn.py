'''Baseline CNN in PyTorch.'''
# Adapted from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py

import torch.nn as nn
import torch.nn.functional as F


class CNN5(nn.Module):
    name = 'cnn5'

    def __init__(self, num_channels=32, num_classes=10):
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, num_channels, 3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels * 2, 3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(num_channels * 2)
        self.conv3 = nn.Conv2d(num_channels * 2, num_channels * 4, 3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(num_channels * 4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4 * 4 * num_channels * 4, num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(num_channels * 4)
        self.fc2 = nn.Linear(num_channels * 4, num_classes)

    def forward(self, x):
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        x = self.bn1(self.conv1(x))                         # batch_size x num_channels x 32 x 32
        x = F.relu(F.max_pool2d(x, 2))                      # batch_size x num_channels x 16 x 16
        x = self.bn2(self.conv2(x))                         # batch_size x num_channels*2 x 16 x 16
        x = F.relu(F.max_pool2d(x, 2))                      # batch_size x num_channels*2 x 8 x 8
        x = self.bn3(self.conv3(x))                         # batch_size x num_channels*4 x 8 x 8
        x = F.relu(F.max_pool2d(x, 2))                      # batch_size x num_channels*4 x 4 x 4

        # flatten the output for each image
        x = x.view(-1, 4*4*self.num_channels*4)             # batch_size x 4*4*num_channels*4

        # apply 2 fully connected layers
        x = F.relu(self.fcbn1(self.fc1(x)))                 # batch_size x self.num_channels*4
        x = self.fc2(x)                                     # batch_size x num_classes
        return x
