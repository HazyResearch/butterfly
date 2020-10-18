from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from pl_bolts.datamodules import CIFAR10DataModule


class CIFAR10(CIFAR10DataModule):

    def __init__(self, data_dir=current_dir, extra_augment=True, **kwargs):
        super().__init__(data_dir, **kwargs)
        if extra_augment:
            augment_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            # By default it only converts to Tensor and normalizes
            self.train_transforms = transforms.Compose(augment_list
                                                    + self.default_transforms().transforms)


class CIFAR100(CIFAR10):

    name = 'cifar100'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DATASET = datasets.CIFAR100

    @property
    def num_classes(self):
        return 100

    def default_transforms(self):
        cf100_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
        return cf100_transforms
