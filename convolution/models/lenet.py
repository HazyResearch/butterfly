'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    name = 'lenet'

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNetPadded(nn.Module):
    name = 'lenetpadded'

    def __init__(self, num_classes=10, padding_mode='circular', pooling_mode='avg'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2, padding_mode=padding_mode)
        self.fc1   = nn.Linear(16*8*8, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        assert pooling_mode in ['avg', 'max']
        self.pool2d = F.avg_pool2d if pooling_mode == 'avg' else F.max_pool2d

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = self.pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
