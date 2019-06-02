import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(config_dataset):
    if config_dataset['name'] == 'CIFAR10':
        normalize = transforms.Normalize(
            mean=[0.49139765, 0.48215759, 0.44653141],
            std=[0.24703199, 0.24348481, 0.26158789]
        )
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            normalize,
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

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config_dataset['batch'], shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(validset, batch_size=config_dataset['batch'], shuffle=False, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        return trainloader, validloader, testloader
    elif config_dataset['name'] == 'TinyImageNet':
        # Copied from https://github.com/tjmoon0104/pytorch-tiny-imagenet/blob/851b65f1843d4c9a7d0e737d4743254e0ee6c107/ResNet18_64.ipynb
        normalize = transforms.Normalize(
            mean=[0.4802, 0.4481, 0.3975],
            std=[0.2302, 0.2265, 0.2262]
        )
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            'val': transforms.Compose([transforms.ToTensor(), normalize,]),
            'test': transforms.Compose([transforms.ToTensor(), normalize,])
        }
        data_dir = project_root + '/data/tiny-imagenet-200'
        datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'val','test']}
        num_workers = {'train': 8, 'val': 4, 'test': 4}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=config_dataset['batch'],
                                          shuffle=True, num_workers=num_workers[x])
                       for x in ['train', 'val', 'test']}
        return dataloaders['train'], dataloaders['val', dataloaders['test']]
    else:
        assert False, 'Dataset not implemented'
