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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self, method='linear', **kwargs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)

        if method == 'linear':
            print("LINEAR")
            self.fc   = nn.Linear(1024, 1024)
        elif method == 'butterfly':
            self.fc   = Butterfly(1024, 1024, tied_weight=False, bias=True, param='regular', nblocks=0)
        elif method == 'low-rank':
            print("I AM HEREEEEE")
            self.fc = nn.Sequential(nn.Linear(1024, kwargs['rank'], bias=False), nn.Linear(kwargs['rank'], 1024))
        else: assert False, f"method {method} not supported"

        self.logits   = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc(out))
        out = self.logits(out)
        return out

