'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .butterfly_conv import Butterfly1x1Conv
from .circulant1x1conv import Circulant1x1Conv
from .toeplitzlike1x1conv import Toeplitzlike1x1Conv
from .low_rank_conv import LowRankConv2d

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, is_structured=False, structure_type='B',
        nblocks=0, param='regular'):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        if is_structured:
            # if structure_type == 'B':
            #     self.conv2 = Butterfly1x1Conv(in_planes, out_planes, bias=False, tied_weight=False, ortho_init=True)
            if structure_type == 'B':
                self.conv2 = ButterflyConv2d(in_planes, out_planes,
                    kernel_size=1, bias=False, nblocks=nblocks, tied_weight=False,
                    ortho_init=True, param=param, stride=1, padding=0)
            elif structure_type == 'Circulant' and out_planes % in_planes == 0:
                self.conv2 = Circulant1x1Conv(in_planes, out_planes // in_planes)
            elif structure_type == 'Toeplitzlike' and out_planes % in_planes == 0:
                self.conv2 = Toeplitzlike1x1Conv(in_planes, out_planes // in_planes)
            elif structure_type == 'LR':
                # Low rank should match the number of parameters of butterfly
                rank = int(math.log2(out_planes)) if nblocks == 0 else nblocks * 2 * int(math.log2(out_planes))
                self.conv2 = LowRankConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                bias=False, rank=rank)
        else:
            self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10, num_structured_layers=0, structure_type='B', nblocks=0, param='regular'):
        assert structure_type in ['B', 'LR', 'Circulant', 'Toeplitzlike']
        assert num_structured_layers <= len(self.cfg)
        super(MobileNet, self).__init__()
        self.structure_type = structure_type
        self.param = param
        self.nblocks = nblocks
        self.is_structured = [False] * (len(self.cfg) - num_structured_layers) + [True] * num_structured_layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x, is_structured in zip(self.cfg, self.is_structured):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, is_structured, structure_type=self.structure_type,
                param=self.param, nblocks=self.nblocks))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
