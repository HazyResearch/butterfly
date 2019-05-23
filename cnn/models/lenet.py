'''LeNet in PyTorch.'''
import sys, os, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')
from butterfly import Butterfly
from butterfly.butterfly_multiply import butterfly_mult_untied
# import baselines.toeplitz as toeplitz
import structure.layer as sl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self, method='linear', **kwargs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)

        # print(method, tied_weight, kwargs)
        if method == 'linear':
            self.fc   = nn.Linear(1024, 1024)
        elif method == 'butterfly':
            self.fc   = Butterfly(1024, 1024, bias=True, **kwargs)
            # self.fc   = Butterfly(1024, 1024, tied_weight=False, bias=False, param='regular', nblocks=0)
            # self.fc   = Butterfly(1024, 1024, tied_weight=False, bias=False, param='odo', nblocks=1)
        elif method == 'low-rank':
            self.fc = nn.Sequential(nn.Linear(1024, kwargs['rank'], bias=False), nn.Linear(kwargs['rank'], 1024))
        elif method == 'toeplitz':
            self.fc = sl.ToeplitzLikeC(layer_size=1024, bias=True, **kwargs)
        else: assert False, f"method {method} not supported"

        # self.bias = nn.Parameter(torch.zeros(1024))
        self.logits   = nn.Linear(1024, 10)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc(out))
        # out = out + self.bias
        out = self.logits(out)
        return out


class MLP(nn.Module):
    def __init__(self, method='linear', **kwargs):
        super().__init__()
        if method == 'linear':
            make_layer = lambda name: self.add_module(name, nn.Linear(1024, 1024, bias=True))
        elif method == 'butterfly':
            make_layer = lambda name: self.add_module(name, Butterfly(1024, 1024, bias=True, **kwargs))
            # self.fc   = Butterfly(1024, 1024, tied_weight=False, bias=False, param='regular', nblocks=0)
            # self.fc   = Butterfly(1024, 1024, tied_weight=False, bias=False, param='odo', nblocks=1)
        elif method == 'low-rank':
            make_layer = lambda name: self.add_module(name, nn.Sequential(nn.Linear(1024, kwargs['rank'], bias=False), nn.Linear(kwargs['rank'], 1024, bias=True)))
        elif method == 'toeplitz':
            make_layer = lambda name: self.add_module(name, sl.ToeplitzLikeC(layer_size=1024, bias=True, **kwargs))
        else: assert False, f"method {method} not supported"

        # self.fc10 = make_layer()
        # self.fc11 = make_layer()
        # self.fc12 = make_layer()
        # self.fc2 = make_layer()
        make_layer('fc10')
        make_layer('fc11')
        make_layer('fc12')
        make_layer('fc2')
        make_layer('fc3')
        self.logits = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 3, 1024)
        x = self.fc10(x[:,0,:]) + self.fc11(x[:,1,:]) + self.fc12(x[:,2,:])
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.logits(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, dropout=False, method='linear', tied_weight=False, **kwargs):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.dropout = nn.Dropout() if dropout else nn.Identity()
        self.features_size = 256 * 4 * 4

        self.fc1 = nn.Linear(self.features_size, self.features_size)
        if method == 'linear':
            self.fc   = nn.Linear(self.features_size, self.features_size, bias=False)
        elif method == 'butterfly':
            self.fc   = Butterfly(self.features_size, self.features_size, tied_weight=tied_weight, bias=False, **kwargs)
            # self.fc   = Butterfly(self.features_size, self.features_size, tied_weight=False, bias=False, param='regular', nblocks=0)
            # self.fc   = Butterfly(self.features_size, self.features_size, tied_weight=False, bias=False, param='odo', nblocks=1)
        elif method == 'low-rank':
            self.fc = nn.Sequential(nn.Linear(self.features_size, kwargs['rank'], bias=False), nn.Linear(kwargs['rank'], self.features_size, bias=False))
        else: assert False, f"method {method} not supported"
        self.bias = nn.Parameter(torch.zeros(self.features_size))
        self.fc2 = nn.Linear(4096, 4096)
        # self.fc2 = nn.Identity()
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # self.dropout,
            # self.fc1,
            # nn.ReLU(),
            # nn.Dropout(),
            self.dropout,
            self.fc2,
            nn.ReLU(),
            nn.Linear(self.features_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print("HELLO ", x.size())
        x = x.view(-1, self.features_size)
        x = self.dropout(x)
        x = nn.ReLU(self.fc1(x) + self.bias)
        x = self.classifier(x)
        return x
