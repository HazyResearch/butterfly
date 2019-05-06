import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

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

    def __init__(self, block=BasicBlock, layers=[2,2,2,2], num_classes=10, zero_init_residual=False, method='linear', prank=2, stochastic=False, temp=1.0):
        super().__init__()

        self.block              = block
        self.layers             = layers
        self.num_classes        = num_classes
        self.zero_init_residual = zero_init_residual

        self.permute = TensorPermutation(32, 32, method=method, rank=prank, stochastic=stochastic, temp=temp)

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
        x = self.permute(x)

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

        return x

class TensorPermutation(nn.Module):
    def __init__(self, w, h, method='linear', rank=1, stochastic=False, temp=1.0):
        super().__init__()
        if method == 'linear':
            self.perm_type = LinearPermutation
        elif method == 'sinkhorn':
            self.perm_type = SinkhornPermutation
        self.perm_fn = self.perm_type.sample_soft_perm if stochastic else self.perm_type.mean_perm

        self.rank = rank
        self.w = w
        self.h = h
        if self.rank == 1:
            self.permute = self.perm_type(w*h, temp=temp)
        elif self.rank == 2:
            self.permute1 = self.perm_type(w, temp=temp)
            self.permute2 = self.perm_type(h, temp=temp)
        else:
            assert False, "prank must be 1 or 2"



    def forward(self, x):
        if self.rank == 1:
            # perm = self.permute.sample_soft_perm()
            perm = self.perm_fn(self.permute)
            x = x.view(-1, self.w*self.h)
            x = x @ perm
            x = x.view(-1, 3, self.w, self.h) # TODO make this channel agnostic
        elif self.rank == 2:
            x = x.transpose(-1, -2)
            # perm2 = self.permute2.sample_soft_perm()
            perm2 = self.perm_fn(self.permute2)
            x = x @ perm2
            x = x.transpose(-1, -2)
            # perm1 = self.permute1.sample_soft_perm()
            perm1 = self.perm_fn(self.permute1)
            x = x @ perm1
        return x


class Permutation(nn.Module):

    def forward(self, x, samples=1):
        soft_perms = self.sample_soft_perm((samples, x.size(0)))
        return x.unsqueeze(0) @ soft_perms

    def mean_perm(self):
        pass
    def sample_soft_perm(self, sample_shape=()):
        """ Return soft permutation of shape sample_shape + (size, size) """
        pass

class LinearPermutation(Permutation):
    def __init__(self, size, temp=1.0):
        super().__init__()
        self.size = size
        self.W = nn.Parameter(torch.empty(size, size))
        nn.init.kaiming_uniform_(self.W)

    def mean_perm(self):
        return self.W

    def sample_soft_perm(self, sample_shape=()):
        return self.W # TODO do this properly

class SinkhornPermutation(Permutation):
    def __init__(self, size, temp=1.0):
        super().__init__()
        self.size = size
        self.temp = temp
        self.log_alpha = nn.Parameter(torch.zeros(size, size))
        nn.init.kaiming_uniform_(self.log_alpha)
        # TODO: test effect of random initialization

    def mean_perm(self):
        return self.sinkhorn(self.log_alpha, n_iters=20)

    def sample_soft_perm(self, sample_shape=()):
        log_alpha_noise = self.add_gumbel_noise(self.log_alpha, sample_shape)
        soft_perms = self.sinkhorn(log_alpha_noise, self.temp)
        return soft_perms

    def hard_perm(self):
        """ Round to nearest permutation (in this case, MLE) """
        l = self.log_alpha.detach()
        P = self.sinkhorn(l, temp=0.01, n_iters=100)
        return P

    def sinkhorn(self, log_alpha, temp=1.0, n_iters=20):
        """
        Performs incomplete Sinkhorn normalization

        log_alpha: a tensor with shape ending in (n, n)
        n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)

        Returns:
        A 3D tensor of close-to-doubly-stochastic matrices over the last two dimensions
        """
        log_alpha = log_alpha / temp
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        return torch.exp(log_alpha)

    def sample_gumbel(self, shape, eps=1e-10):
        U = torch.rand(shape, dtype=torch.float, device=device)
        return -torch.log(eps - torch.log(U + eps))

    def add_gumbel_noise(self, log_alpha, sample_shape=()):
        """
        Args:
        log_alpha: shape (N, N)
        temp: temperature parameter, a float.
        sample_shape: represents shape of independent draws

        Returns:
        log_alpha_noise: a tensor of shape [sample_shape + (N, N)]
        """
        batch = log_alpha.size(0)
        n = log_alpha.size(-1)
        noise = self.sample_gumbel(sample_shape + log_alpha.shape)
        log_alpha_noise = log_alpha + noise
        return log_alpha_noise




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
