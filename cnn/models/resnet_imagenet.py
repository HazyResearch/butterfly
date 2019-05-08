# modified from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/models/resnet.py

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from .layers import Flatten

from .butterfly_conv import ButterflyConv2d, ButterflyConv2dBBT

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def butterfly3x3(in_planes, planes, stride=1, structure_type='B', nblocks=1,
        param='regular'):
    if structure_type == 'B':
        bfly = ButterflyConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, tied_weight=False, ortho_init=True, param=param)
    elif structure_type == 'BBT':
        bfly = ButterflyConv2dBBT(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nblocks=nblocks, tied_weight=False, ortho_init=True, param=param)
    else:
        raise ValueError("Structure type isn't supported.")
    return bfly


def butterfly1x1(in_planes, planes, stride=1, structure_type='B', nblocks=1,
        param='regular'):
    if structure_type == 'B':
        bfly = ButterflyConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, tied_weight=False, ortho_init=True, param=param)
    elif structure_type == 'BBT':
        bfly = ButterflyConv2dBBT(in_planes, planes, kernel_size=1, stride=stride, bias=False, nblocks=nblocks, tied_weight=False, ortho_init=True, param=param)
    else:
        raise ValueError("Structure type isn't supported.")
    return bfly

def bn1(planes):
    m = nn.BatchNorm1d(planes)
    m.weight.data.fill_(1)
    m.bias.data.zero_()
    return m

def bn(planes, init_zero=False):
    m = nn.BatchNorm2d(planes)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_structured=False, structure_type='B', nblocks=1,
            param='regular'):
        super().__init__()
        if is_structured:
            self.conv1 = butterfly3x3(inplanes, planes, stride=stride, structure_type=structure_type,
                    nblocks=nblocks, param=param)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        if is_structured:
            self.conv2 = butterfly3x3(planes, planes, structure_type=structure_type,
                    nblocks=nblocks, param=param)
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)

        out += residual
        out = self.relu(out)
        out = self.bn2(out)

        return out

class BottleneckFinal(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual
        out = self.bn3(out)
        out = self.relu(out)

        return out

class BottleneckZero(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4, init_zero=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        if self.downsample is not None: residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, k=1, vgg_head=False,
            num_structured_layers=0, structure_type='B', nblocks=1, param='regular'):
        assert num_structured_layers <= 4
        assert structure_type in ['B', 'BBT', 'BBTBBT']
        super().__init__()
        self.is_structured = [False] * (4 - num_structured_layers) + [True] * num_structured_layers
        self.inplanes = 64

        features = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            , bn(64) , nn.ReLU(inplace=True) , nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            , self._make_layer(block, int(64*k), layers[0], is_structured=self.is_structured[0],
                structure_type=structure_type, nblocks=nblocks, param=param)
            , self._make_layer(block, int(128*k), layers[1], stride=2, is_structured=self.is_structured[1],
                structure_type=structure_type, nblocks=nblocks, param=param)
            # Only stacking butterflies in the 3rd layer for now
            , self._make_layer(block, int(256*k), layers[2], stride=2, is_structured=self.is_structured[2],
                structure_type=structure_type, nblocks=nblocks, param=param)
            , self._make_layer(block, int(512*k), layers[3], stride=2, is_structured=self.is_structured[3],
                structure_type=structure_type, nblocks=nblocks, param=param)]
        out_sz = int(512*k) * block.expansion

        if vgg_head:
            features += [nn.AdaptiveAvgPool2d(3), Flatten()
                , nn.Linear(out_sz*3*3, 4096), nn.ReLU(inplace=True), bn1(4096), nn.Dropout(0.25)
                , nn.Linear(4096,   4096), nn.ReLU(inplace=True), bn1(4096), nn.Dropout(0.25)
                , nn.Linear(4096, num_classes)]
        else: features += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(out_sz, num_classes)]

        self.features = nn.Sequential(*features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, is_structured=False,
            structure_type='B', nblocks=1, param='regular'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if is_structured:
                downsample = nn.Sequential(
                    butterfly1x1(self.inplanes, planes * block.expansion, stride=stride, structure_type=structure_type,
                        nblocks=nblocks, param=param),
                    bn(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                    bn(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            is_structured=is_structured, structure_type=structure_type,
            nblocks=nblocks, param=param))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(self.inplanes, planes,
            is_structured=is_structured, structure_type=structure_type,
            nblocks=nblocks, param=param))
        return nn.Sequential(*layers)

    def forward(self, x): return self.features(x)

# resnet50 does not support currently support structure
# def resnet50(**kwargs):
#    raise ValueError('resnet50
#    model = ResNet(Bottleneck, [3, 4, 6, 3])
#    return model

def resnet18(num_structured_layers=0, structure_type='B', nblocks=1, param='regular'):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_structured_layers=num_structured_layers,
            structure_type=structure_type, nblocks=nblocks, param=param)
    return model
