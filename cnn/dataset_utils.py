import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(config_dataset):
    if config_dataset['name'] in ['CIFAR10', 'PCIFAR10']:
        if config_dataset['name'] == 'PCIFAR10':
            # fix permutation
            rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            true_perm = torch.randperm(1024)
            torch.set_rng_state(rng_state)
            permutation_transform = [transforms.Lambda(lambda x: x.view(-1,1024)[:,true_perm].view(-1,32,32))]
        else:
            permutation_transform = []

        normalize = transforms.Normalize(
            mean=[0.49139765, 0.48215759, 0.44653141],
            std=[0.24703199, 0.24348481, 0.26158789]
        )

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            normalize
        ] + permutation_transform)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            normalize
        ] + permutation_transform)

        trainset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=True, transform=transform_train)
        validset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=False, transform=transform_test)
        np_random_state = np.random.get_state()  # To get exactly the same training and validation sets
        np.random.seed(0)
        indices = np.random.permutation(range(len(trainset)))
        np.random.set_state(np_random_state)
        trainset = torch.utils.data.Subset(trainset, indices[:40000])
        # trainset = torch.utils.data.Subset(trainset, indices[:5000])
        validset = torch.utils.data.Subset(validset, indices[-10000:])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False, num_workers=4)

        if config_dataset['name'] == 'PCIFAR10':
            trainloader.true_permutation = true_perm
            testloader.true_permutation = true_perm
        return trainloader, testloader
    else:
        assert False, 'Dataset not implemented'
