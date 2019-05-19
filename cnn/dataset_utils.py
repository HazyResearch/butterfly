import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import math
import numpy as np
def bitreversal_permutation(n):
    """Return the bit reversal permutation used in FFT.
    Parameter:
        n: integer, must be a power of 2.
    Return:
        perm: bit reversal permutation, numpy array of size n
    """
    m = int(math.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(m):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    return torch.tensor(perm.squeeze(0))



def get_dataset(config_dataset):
    if config_dataset['name'] in ['CIFAR10', 'PCIFAR10', 'PPCIFAR10']:
        if config_dataset['name'] == 'PCIFAR10':
            # fix permutation
            rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            true_perm = torch.randperm(1024)
            torch.set_rng_state(rng_state)
            permutation_transforms = [transforms.Lambda(lambda x: x.view(-1,1024)[:,true_perm].view(-1,32,32))]
        elif config_dataset['name'] == 'PPCIFAR10':
            # rng_state = torch.get_rng_state()
            # torch.manual_seed(0)
            # true_perm1 = torch.randperm(32)
            # true_perm2 = torch.randperm(32)
            true_perm1 = bitreversal_permutation(32)
            true_perm2 = bitreversal_permutation(32)
            # torch.set_rng_state(rng_state)
            def fn(x):
                # dumb hack because torch doesn't support multiple LongTensor indexing
                return x.transpose(-1,-2)[...,true_perm2].transpose(-1,-2)[...,true_perm1]
            permutation_transforms = [transforms.Lambda(fn)]
        else:
            permutation_transforms = []

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
        ] + permutation_transforms)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            normalize
        ] + permutation_transforms)

            trainset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=True, transform=transforms.ToTensor())
            validset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=False, transform=transforms.ToTensor())
        elif 'transform' in config_dataset and config_dataset['transform'] == 'permute':
            transforms_ = transforms.Compose([transforms.ToTensor()] + permutation_transforms)
            trainset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=True, transform=transforms_)
            validset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=False, transform=transforms_)
        elif 'transform' in config_dataset and config_dataset['transform'] == 'normalize':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                normalize
            ])
            trainset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=True, transform=transform_train)
            validset = torchvision.datasets.CIFAR10(root=project_root+'/data', train=True, download=False, transform=transform_train)
        else:
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

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config_dataset['batch'], shuffle=True, num_workers=4)
        validloader = torch.utils.data.DataLoader(validset, batch_size=config_dataset['batch'], shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config_dataset['batch'], shuffle=False, num_workers=4)

        if config_dataset['name'] == 'PCIFAR10':
            # trainloader.true_permutation = true_perm
            testloader.true_permutation = true_perm
        elif config_dataset['name'] == 'PPCIFAR10':
            # trainloader.true_permutation1 = true_perm1
            # trainloader.true_permutation2 = true_perm2
            testloader.true_permutation = (true_perm1, true_perm2)
        return trainloader, validloader, testloader
    else:
        assert False, 'Dataset not implemented'
