import torch.nn as nn
import torch.nn.functional as F

from .lenet import LeNetPadded
from .kops import KOP2d


class ButterfLeNet(LeNetPadded):
    name = 'butterflenet'

    def __init__(self, num_classes=10, pooling_mode='avg', **kwargs):
        nn.Module.__init__(self)
        in_size = 32
        self.conv1 = KOP2d(in_size, 3, 6, 5, **kwargs)
        self.conv2 = KOP2d(in_size // 2, 6, 16, 5, **kwargs)
        self.fc1   = nn.Linear(16*8*8, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        assert pooling_mode in ['avg', 'max']
        self.pool2d = F.avg_pool2d if pooling_mode == 'avg' else F.max_pool2d
