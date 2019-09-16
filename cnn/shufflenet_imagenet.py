'''ShuffleNet in PyTorch.

See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from cnn.mobilenet_imagenet import _make_divisible
from cnn.mobilenet_imagenet import Butterfly1x1Conv


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, grouped_conv_1st_layer=True, shuffle='P'):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.shuffle = shuffle

        mid_planes = _make_divisible(out_planes // 4, groups)
        if stride == 2:  # Reduce out_planes due to concat
            out_planes -= in_planes
        g = groups if grouped_conv_1st_layer else 1  # No grouped conv for the first layer of stage 2
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        if shuffle == 'P':
            self.shuffle0 = nn.Identity()
            self.shuffle1 = ShuffleBlock(groups=g)
        else:
            param = shuffle.split('_')[0]
            nblocks = 0 if len(shuffle.split('_')) <= 1 else int(shuffle.split('_')[1])
            self.shuffle0 = Butterfly1x1Conv(in_planes, in_planes, bias=False, tied_weight=False, ortho_init=True, param=param, nblocks=nblocks)
            self.shuffle1 = Butterfly1x1Conv(mid_planes, mid_planes, bias=False, tied_weight=False, ortho_init=True, param=param, nblocks=nblocks)

        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.conv2.weight._no_wd = True
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.shuffle0(x))), inplace=True)
        out = self.shuffle1(out)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1), inplace=True) if self.stride==2 else F.relu(out+res, inplace=True)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000, groups=8, width_mult=1.0, shuffle='P'):
        super(ShuffleNet, self).__init__()
        num_blocks = [4, 8, 4]
        groups_to_outplanes = {1: [144, 288, 576],
                               2: [200, 400, 800],
                               3: [240, 480, 960],
                               4: [272, 544, 1088],
                               8: [384, 768, 1536]}
        out_planes = groups_to_outplanes[groups]
        out_planes = [_make_divisible(p * width_mult, groups) for p in out_planes]

        input_channel = _make_divisible(24 * width_mult, groups)
        self.conv1 = nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.in_planes = input_channel
        self.stage2 = self._make_layer(out_planes[0], num_blocks[0], groups, grouped_conv_1st_layer=False, shuffle=shuffle)
        self.stage3 = self._make_layer(out_planes[1], num_blocks[1], groups, shuffle=shuffle)
        self.stage4 = self._make_layer(out_planes[2], num_blocks[2], groups, shuffle=shuffle)
        self.linear = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups, grouped_conv_1st_layer=True, shuffle='P'):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(Bottleneck(self.in_planes, out_planes, stride=stride, groups=groups,
                                     grouped_conv_1st_layer=grouped_conv_1st_layer, shuffle=shuffle))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.maxpool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = out.mean([2, 3])
        out = self.linear(out)
        return out


def test():
    net = ShuffleNet()
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    print(y)

# test()
