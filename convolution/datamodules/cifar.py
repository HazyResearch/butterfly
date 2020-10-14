from pathlib import Path
current_dir = Path(__file__).parent.absolute()

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from pl_bolts.datamodules import CIFAR10DataModule


class SubsetwIndex(torch.utils.data.Subset):
    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.dataset[index] + (self.random_indices[index], )


class CIFAR10(CIFAR10DataModule):

    def __init__(self, data_dir=current_dir, extra_augment=True, split_size=1, split_seed=42, crossfit_index=0,
                 return_indices=False, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.split_size = split_size
        self.split_seed = split_seed
        self.crossfit_index = crossfit_index
        self.return_indices = return_indices
        if extra_augment:
            augment_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            # By default it only converts to Tensor and normalizes
            self.train_transforms = transforms.Compose(augment_list
                                                    + self.default_transforms().transforms)

    def train_dataloader(self):
        transforms = (self.default_transforms() if self.train_transforms is None
                      else self.train_transforms)
        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms,
                                **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        random_indices = torch.randperm(
            train_length, generator=torch.Generator().manual_seed(self.split_seed)
        )
        if self.split_size > 1:
            dataset_train.indices = [
                idx for idx in dataset_train.indices
                if random_indices[idx] % self.split_size != self.crossfit_index
            ]
        if self.return_indices:
            dataset_train = SubsetwIndex(dataset_train.dataset, dataset_train.indices)
            dataset_train.random_indices = random_indices
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader


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
