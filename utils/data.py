"""
CIFAR-10 data loading utilities.

Provides normalised train/test DataLoaders with standard CIFAR-10 augmentation.
Mean and std are the per-channel statistics of the full CIFAR-10 training set.
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Per-channel mean/std of CIFAR-10 training set
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2023, 0.1994, 0.2010)


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download (if needed) and return CIFAR-10 train/test DataLoaders.

    Training augmentation:
        - Random horizontal flip
        - Random crop with 4-pixel padding (standard for CIFAR-10)
        - Normalise

    Test:
        - Normalise only (no augmentation for reproducible evaluation)

    Args:
        data_dir:    Directory where CIFAR-10 data is stored / will be downloaded.
        batch_size:  Mini-batch size for both loaders.
        num_workers: Number of DataLoader worker processes.

    Returns:
        (train_loader, test_loader)
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader
