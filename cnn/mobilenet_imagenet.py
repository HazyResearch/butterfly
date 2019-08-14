'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from butterfly import Butterfly


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Butterfly1x1Conv(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    """

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, structure='D'):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        if structure == 'D':
            self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            param = structure.split('_')[0]
            nblocks = 0 if len(structure.split('_')) <= 1 else int(structure.split('_')[1])
            self.conv2 = Butterfly1x1Conv(in_planes, out_planes, bias=False, tied_weight=False, param=param, nblocks=nblocks)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=1000, width_mult=1.0, round_nearest=8, structure=None):
        """
        structure: list of string
        """
        super(MobileNet, self).__init__()
        self.width_mult = width_mult
        self.round_nearest = round_nearest
        self.structure = [] if structure is None else structure
        self.n_structure_layer = len(self.structure)
        self.structure = ['D'] * (len(self.cfg) - self.n_structure_layer) + self.structure
        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.layers = self._make_layers(in_planes=input_channel)
        self.last_channel = _make_divisible(1024 * width_mult, round_nearest)
        self.linear = nn.Linear(self.last_channel, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x, struct in zip(self.cfg, self.structure):
            out_planes = _make_divisible((x if isinstance(x, int) else x[0]) * self.width_mult, self.round_nearest)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, structure=struct))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layers(out)
        out = out.mean([2, 3])
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
