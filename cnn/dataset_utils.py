import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(config_dataset):
    if config_dataset['name'] == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=False, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=False, download=True, transform=transform_test)
        np_random_state = np.random.get_state()  # To get exactly the same training and validation sets
        np.random.seed(0)
        indices = np.random.permutation(range(len(trainset)))
        np.random.set_state(np_random_state)
        trainset = torch.utils.data.Subset(trainset, indices[:45000])
        # trainset = torch.utils.data.Subset(trainset, indices[:5000])
        validset = torch.utils.data.Subset(validset, indices[-5000:])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        validloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
        return trainloader, validloader, testloader
    else:
        assert False, 'Dataset not implemented'
