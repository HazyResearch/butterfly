import os, sys
import math
import random

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')
from butterfly import Butterfly
from butterfly.butterfly_multiply import butterfly_mult_untied
import permutation_utils as perm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PResNet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_classes=10, zero_init_residual=False, **perm_args):
        super().__init__()

        self.block              = block
        self.layers             = layers
        self.num_classes        = num_classes
        self.zero_init_residual = zero_init_residual

        self.permute = TensorPermutation(32, 32, **perm_args)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * self.block.expansion, 512 * self.block.expansion)

        self.logits = nn.Linear(512 * self.block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.size())
        # print(x)
        batch = x.size(0)
        x = self.permute(x)
        x = x.view(-1, 3, 32, 32)

        # print(x.size())
        x = self.conv1(x)
        # print(x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())

        # x = self.avgpool(x)
        x = F.avg_pool2d(x, 4)
        # print(x.size())
        x = x.view(x.size(0), -1)
        # print(x.size())
        # print(x.size())
        # x = F.relu(self.fc(x))
        x = self.logits(x)
        # print(x.size())

        # x = x.view(-1, batch, self.num_classes)

        return x

class TensorPermutation(nn.Module):
    def __init__(self, w, h, method='identity', rank=2, train=True, **kwargs):
        super().__init__()
        self.w = w
        self.h = h

        if method == 'linear':
            self.perm_type = LinearPermutation
        elif method == 'butterfly':
            self.perm_type = ButterflyPermutation
        elif method == 'identity':
            self.perm_type = IdentityPermutation
        else:
            assert False, f"Permutation method {method} not supported."

        self.rank = rank
        if self.rank == 1:
            self.permute = nn.ModuleList([self.perm_type(w*h, **kwargs)])
        elif self.rank == 2:
            self.permute = nn.ModuleList([self.perm_type(w, **kwargs), self.perm_type(h, **kwargs)])
            # self.permute2 = self.perm_type(h, **kwargs)
        else:
            assert False, "prank must be 1 or 2"
        # TODO: maybe it makes sense to set ._is_perm_param here

        # if stochastic:
        #     self.perm_fn = self.perm_type.sample_soft_perm
        # else:
        #     self.perm_fn =self.perm_type.mean_perm
        # elif acqfn == 'mean':
        #     self.perm_fn =self.perm_type.mean_perm
        # elif acqfn == 'sample':
        #     self.perm_fn = self.perm_type.sample_soft_perm
        # else:
        #     assert False, f"Permutation acquisition function {acqfn} not supported."

        if train == False:
            for p in self.parameters():
                p.requires_grad = False


    def forward(self, x, perm=None):
        if perm is None:
            perm_fn = self.perm_type.generate_perm
        elif perm == 'mean':
            perm_fn = self.perm_type.mean_perm
        elif perm == 'mle':
            perm_fn = self.perm_type.mle_perm
        elif perm == 'sample':
            perm_fn = self.perm_type.sample_perm
        else: assert False, f"Permutation type {perm} not supported."

        if self.rank == 1:
            perm = perm_fn(self.permute[0])
            x = x.view(-1, self.w*self.h)
            x = x @ perm
            x = x.view(-1, 3, self.w, self.h) # TODO make this channel agnostic
        elif self.rank == 2:
            x = x.transpose(-1, -2)
            perm2 = perm_fn(self.permute[1])
            x = x @ perm2.unsqueeze(-3).unsqueeze(-3) # unsqueeze to explicitly call matmul, can use einsum too
            x = x.transpose(-1, -2)
            perm1 = perm_fn(self.permute[0])
            x = x @ perm1.unsqueeze(-3).unsqueeze(-3)
            # collapse samples with batch
            x = x.view(-1, 3, self.w, self.h)
        return x

    def get_permutations(self, perm=None):
        if perm is None:
            perm_fn = self.perm_type.generate_perm
        elif perm == 'mean':
            perm_fn = self.perm_type.mean_perm
        elif perm == 'mle':
            perm_fn = self.perm_type.mle_perm
        elif perm == 'sample':
            perm_fn = self.perm_type.sample_perm
        else: assert False, f"Permutation type {perm} not supported."
        # return shape (rank, s, n, n)
        perms = torch.stack([perm_fn(p) for p in self.permute], dim=0)
        # print("get_permutations:", perms.shape)
        return perms

    def entropy(self, p):
        ents = torch.stack([perm.entropy(p) for perm in self.permute], dim=0) # (rank,)
        return torch.mean(ents)



class Permutation(nn.Module):

    def forward(self, x, samples=1):
        soft_perms = self.sample_soft_perm((samples, x.size(0)))
        return x.unsqueeze(0) @ soft_perms

    def mean_perm(self):
        pass
    def sample_soft_perm(self, sample_shape=()):
        """ Return soft permutation of shape sample_shape + (size, size) """
        pass

class IdentityPermutation(Permutation):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def generate_perm(self):
        return torch.eye(self.size, device=device)
    def mean_perm(self):
        return torch.eye(self.size, device=device)
    def mle_perm(self):
        return torch.eye(self.size, device=device)
    def sample_perm(self):
        return torch.eye(self.size, device=device)

class LinearPermutation(Permutation):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.W = nn.Parameter(torch.empty(size, size))
        self.W.is_perm_param = True
        nn.init.kaiming_uniform_(self.W)

    def generate_perm(self):
        return self.W
    def mean_perm(self):
        return self.W
    def mle_perm(self):
        return self.W
    def sample_perm(self):
        return self.W

    # def hard_perm(self):
    #     return self.W

    # def sample_soft_perm(self, sample_shape=()):
    #     return self.W.view(*([1]*len(sample_shape)), size, size)



class ButterflyPermutation(Permutation):
    def __init__(self, size, sig='BT1', param='ortho2', stochastic=False, temp=1.0, samples=1, sample_method='gumbel', hard=False):
        super().__init__()
        self.size = size
        self.sig = sig
        self.param = param
        self.stochastic = stochastic # TODO align this block
        self.temp = temp
        self.samples = samples
        self.sample_method = sample_method
        self.hard = hard
        self.m = int(math.ceil(math.log2(size)))
        assert size == (1<<self.m), "ButterflyPermutation: Only power of 2 supported."

        if self.stochastic:
            self.mean_temp = 1.0
            self.sample_temp = temp
            if hard:
                self.generate_fn = self.sample_hard_perm
            else:
                self.generate_fn = self.sample_soft_perm # add this attr for efficiency (avoid casing in every call to generate())
            # self.sample_method = 'gumbel'
        else:
            self.mean_temp = temp
            self.generate_fn = self.mean_perm
            # no sample_temp; soft perm shouldn't be called in the non-stochastic case
        self.hard_temp = 0.02
        self.hard_iters = int(1./self.hard_temp)


        # assume square matrices so 'nstack' is always 1
        if sig[:2] == 'BT' and (sig[2:]).isdigit(): # TODO: empty number indicates 1
            depth = int(sig[2:])
            self.twiddle_core_shape = (2*depth, 1, self.m, self.size//2)
            self.strides = [0,1] * depth # 1 for increasing, 0 for decreasing
        elif sig[0] == 'B' and (sig[1:]).isdigit():
            depth = int(sig[1:])
            self.twiddle_core_shape = (depth, 1, self.m, self.size//2)
            self.strides = [1] * depth # 1 for increasing, 0 for decreasing
        elif sig[0] == 'T' and (sig[1:]).isdigit():
            depth = int(sig[1:])
            self.twiddle_core_shape = (depth, 1, self.m, self.size//2)
            self.strides = [0] * depth # 1 for increasing, 0 for decreasing
        else:
            assert False, f"ButterflyPermutation: signature {sig} not supported."
        # self.twiddle has shape (depth, 1, log n, n/2)
        self.depth = self.twiddle_core_shape[0]

        margin = 1e-3
        # sample from [margin, 1-margin]
        init = (1-2*margin)*(torch.rand(self.twiddle_core_shape)) + margin
        if self.param == 'ds':
            self.twiddle = nn.Parameter(init)
        elif self.param == 'logit':
            # self.twiddle = nn.Parameter(torch.rand(self.twiddle_core_shape)*2-1)
            init = perm.sample_gumbel(self.twiddle_core_shape) - perm.sample_gumbel(self.twiddle_core_shape)
            # init_temp = random.uniform(0.2, 0.4)
            # init_temp = random.uniform(0.5, )
            init_temp = 1.0 / self.depth
            # init_temp = random.uniform(0.1, 0.2)
            # init_temp = 0.2
            self.twiddle = nn.Parameter(init / init_temp)
            # self.twiddle = nn.Parameter(init)
            # self.twiddle = nn.Parameter(torch.log(init / (1.-init)))
            # logits = torch.log(init / (1.-init))
            # self.twiddle = nn.Parameter( logits / temp)
            # breakpoint()
        elif param == 'ortho2':
        # TODO change initialization for this type
            # self.twiddle = nn.Parameter(torch.rand(self.twiddle_core_shape) * 2*math.pi)
            self.twiddle = nn.Parameter(torch.acos(torch.sqrt(init)))
        else:
            assert False, f"ButterflyPermutation: Parameter type {self.param} not supported."
        self.twiddle._is_perm_param = True


    def entropy(self, p=None):
        """ TODO: How does this compare to the matrix entropy of the expanded mean matrix? """
        if p == 'logit':
            assert self.param=='logit'
            def binary_ent(p):
                eps = 1e-10
                return -(p * torch.log2(eps+p) + (1-p)*torch.log2(1-p+eps))
            _twiddle = self.map_twiddle(self.twiddle)
            ent1 = torch.sum(binary_ent(_twiddle))
            return ent1
            # could be better to not map at all
            x = torch.exp(-self.twiddle)
            ent2 = torch.log2(1. + x) + self.twiddle * (x/(1.+x))
            ent2 = torch.sum(ent2)
            print(ent1-ent2)
            return ent2

        if p is None:
            perms = self.generate_perm()
        elif p == 'mean':
            perms = self.mean_perm()
        elif p == 'mle':
            perms = self.mle_perm()
        elif p == 'sample':
            perms = self.sample_perm()
        else: assert False, f"Permutation type {p} not supported."
        return perm.entropy(perms, reduction='mean')

    def generate_perm(self):
        """ Generate (a batch of) permutations for training """
        # TODO add the extra dimension even with mean for consistency
        return self.generate_fn()


    def map_twiddle(self, twiddle): # TODO static
        if self.param=='ds':
            return twiddle
        elif self.param=='logit':
            return 1.0/(1.0 + torch.exp(-twiddle))
        elif self.param=='ortho2':
            return torch.cos(twiddle)**2
        else:
            assert False, f"Unreachable"

    def compute_perm(self, twiddle, strides, squeeze=True):
        """
        # twiddle: (depth, 1, log n, n/2)
        twiddle: (depth, samples, log n, n/2)
        strides: (depth,) bool

        Returns: (samples, n, n)
        """
        samples = twiddle.size(1)
        # print("compute_perm twiddle REQUIRES GRAD: ", twiddle.requires_grad)
        P = torch.eye(self.size, device=twiddle.device).unsqueeze(1).repeat((1,samples,1)) # (n, s, n) : put samples in the 'nstack' parameter of butterfly_mult
        # print("compute_perm REQUIRES GRAD: ", P.requires_grad)
        for t, stride in zip(twiddle, strides):
            twiddle_factor_mat = torch.stack((torch.stack((t, 1-t), dim=-1),
                                              torch.stack((1-t, t), dim=-1)), dim=-2) # TODO efficiency by stacking other order?
            P = butterfly_mult_untied(twiddle_factor_mat, P, stride, self.training)
            # print("REQUIRES GRAD: ", P.requires_grad)

        P = P.transpose(0, 1) # (s, n, n)
        return P.squeeze() if squeeze else P
        # return P.view(self.size, self.size) # (n, n)

    def mean_perm(self):
        # TODO isn't scaling mean by temperature
        # print("mean_perm twiddle REQUIRES GRAD: ", self.twiddle.requires_grad)
        _twiddle = self.map_twiddle(self.twiddle)
        p = self.compute_perm(_twiddle, self.strides)
        # print("mean_perm REQUIRES GRAD: ", p.requires_grad)
        return p

    def mle_perm(self):
        _twiddle = self.map_twiddle(self.twiddle)
        hard_twiddle = torch.where(_twiddle > 0.5, torch.tensor(1.0, device=_twiddle.device), torch.tensor(0.0, device=_twiddle.device))
        p = self.compute_perm(hard_twiddle, self.strides)
        return p

    def sample_perm(self, sample_shape=()):
        if self.stochastic:
            return self.sample_soft_perm()
        else:
            return self.sample_hard_perm()

    def sample_soft_perm(self, sample_shape=()):
        sample_shape = (self.samples,)

        if self.param == 'logit':
            # # TODO use pytorch's gumbel distribution...
            # assert torch.all(self.twiddle == self.twiddle), "NANS FOUND"
            # logits = torch.stack((self.twiddle, torch.zeros_like(self.twiddle)), dim=-1) # (depth, 1, log n, n/2, 2)
            # assert torch.all(logits == logits), "NANS FOUND"
            # logits_noise = perm.add_gumbel_noise(logits, sample_shape)
            # assert torch.all(logits_noise == logits_noise), "NANS FOUND"
            # sample_twiddle = torch.softmax(logits_noise / self.sample_temp, dim=-1)[..., 0] # shape (s, depth, 1, log n, n/2)
            # assert torch.all(sample_twiddle == sample_twiddle), "NANS FOUND"
            logits = torch.stack((self.twiddle, torch.zeros_like(self.twiddle)), dim=-1) # (depth, 1, log n, n/2, 2)
            shape = logits.size()
            # noise = perm.sample_gumbel((logits.size(0), self.samples)+logits.size()[2:])
            # logits_noise = logits + noise.to(logits.device) # (d, s, log n, n/2, 2)
            noise = perm.sample_gumbel((logits.size(0), self.samples)+logits.size()[2:], device=logits.device)
            logits_noise = logits + noise # (d, s, log n, n/2, 2)
            sample_twiddle = torch.softmax(logits_noise / self.sample_temp, dim=-1)[..., 0] # (depth, s, log n, n/2)
            perms = self.compute_perm(sample_twiddle, self.strides, squeeze=False)
            return perms
        else: # TODO make this case batched over samples too
            _twiddle = self.map_twiddle(self.twiddle)

            if self.sample_method == 'gumbel':
                # TODO: Can't take log!! multiply by exponential instead
                logits = torch.stack((torch.log(_twiddle), torch.log(1.-_twiddle)), dim=-1) # (depth, 1, log n, n/2, 2)
                logits_noise = perm.add_gumbel_noise(logits, sample_shape) # alternate way of doing this: sample one uniform parameter instead of two gumbel
                sample_twiddle = torch.softmax(logits_noise / self.sample_temp, dim=-1)[..., 0] # shape (s, depth, 1, log n, n/2)
            elif self.sample_method == 'uniform':
                r = torch.rand(_twiddle.size())
                _twiddle = _twiddle - r
                sample_twiddle = 1.0 / (1.0 + torch.exp(-_twiddle / self.sample_temp))
            else: assert False, "sample_method {self.sample_method} not supported"

        perms = torch.stack([self.compute_perm(twiddle, self.strides) for twiddle in sample_twiddle], dim=0) # (s, n, n)
        return perms

    def sample_hard_perm(self, sample_shape=()):
        sample_shape = (self.samples,)
        _twiddle = self.map_twiddle(self.twiddle)

        r = torch.rand(_twiddle.size(), device=_twiddle.device)
        _twiddle = _twiddle - r
        # sample_twiddle = 1.0 / (1.0 + torch.exp(-_twiddle / self.sample_temp))
        # hard_twiddle = torch.where(_twiddle>0, torch.tensor(1.0, device=_twiddle.device), torch.tensor(0.0, device=_twiddle.device)) # shape (s, depth, 1, log n, n/2)
        sample_twiddle = _twiddle.repeat(*sample_shape, *([1]*_twiddle.dim())) # TODO try expand
        hard_twiddle = torch.where(sample_twiddle>0,
                                   torch.ones_like(sample_twiddle),
                                   torch.zeros_like(sample_twiddle)
                                   ) # shape (s, depth, 1, log n, n/2)
        # print("HARD_TWIDDLE SHAPE", hard_twiddle.shape)
        # sample_twiddle = _twiddle.expand(sample_shape+_twiddle.shape)
        sample_twiddle.data = hard_twiddle # straight through estimator
        if self.training: assert sample_twiddle.requires_grad
        # TODO can make this a lot faster

        perms = torch.stack([self.compute_perm(twiddle, self.strides) for twiddle in sample_twiddle], dim=0) # (s, n, n)
        return perms

        # logits = torch.stack((torch.log(tw), torch.zeros_like(tw)), dim=-1) # (depth, 1, log n, n/2, 2)
        # logits_noise = perm.add_gumbel_noise(logits, sample_shape) # alternate way of doing this: sample one uniform parameter instead of two gumbel
        # logits_noise = logits_noise[..., 0] - logits_noise[..., 1]
        # sample_twiddle = torch.where(logits_noise>0, torch.tensor(1.0, device=_twiddle.device), torch.tensor(0.0, device=_twiddle.device)) # shape (s, depth, 1, log n, n/2)
        # return sample_twiddle



def PResNet18(pretrained=False, **kwargs):
    model = PResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def PResNet34(pretrained=False, **kwargs):
    model = PResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def PResNet50(pretrained=False, **kwargs):
    model = PResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def PResNet101(pretrained=False, **kwargs):
    model = PResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def PResNet152(pretrained=False, **kwargs):
    model = PResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
