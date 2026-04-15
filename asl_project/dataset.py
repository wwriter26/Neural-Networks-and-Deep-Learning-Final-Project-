"""
dataset.py — Dataset loading, 80/10/10 splitting, and augmentation.

Loads the Kaggle ASL Alphabet dataset, applies stratified split,
and returns DataLoaders with appropriate transforms.
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Return image transforms for a given split.

    Train split gets augmentation (flips, brightness, rotation).
    Val/test splits get only resize + normalize.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
            transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
            transforms.ColorJitter(brightness=config.BRIGHTNESS_FACTOR),
            transforms.ToTensor(),                          # Scales to [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],     # Center around 0
                                 std=[0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])


def build_splits():
    """
    Load the full dataset and create stratified 80/10/10 splits.

    Returns:
        train_indices, val_indices, test_indices, class_to_idx mapping
    """
    # Use a dummy transform just to scan the dataset structure
    full_dataset = datasets.ImageFolder(root=str(config.DATA_DIR))
    targets = full_dataset.targets
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_to_idx)

    # Group indices by class for stratified splitting
    class_indices = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    rng = random.Random(config.RANDOM_SEED)
    for c in range(num_classes):
        idxs = class_indices[c][:]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)

        train_indices.extend(idxs[:n_train])
        val_indices.extend(idxs[n_train:n_train + n_val])
        test_indices.extend(idxs[n_train + n_val:])

    return train_indices, val_indices, test_indices, class_to_idx


class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset so we can apply a different transform per split."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataloaders(batch_size: int = None, num_workers: int = None):
    """
    Build and return train, validation, and test DataLoaders.

    Args:
        batch_size: Override config.BATCH_SIZE if provided.
        num_workers: Override config.NUM_WORKERS if provided.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    bs = batch_size or config.BATCH_SIZE
    nw = num_workers if num_workers is not None else config.NUM_WORKERS

    # Load raw dataset (PIL images, no transform yet)
    raw_dataset = datasets.ImageFolder(
        root=str(config.DATA_DIR),
        transform=None,   # We apply transforms in TransformSubset
    )

    train_idx, val_idx, test_idx, class_to_idx = build_splits()

    # Build datasets with appropriate transforms
    train_ds = TransformSubset(raw_dataset, train_idx, get_transforms("train"))
    val_ds = TransformSubset(raw_dataset, val_idx, get_transforms("val"))
    test_ds = TransformSubset(raw_dataset, test_idx, get_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True)

    # Invert class_to_idx → idx_to_class
    class_names = [None] * len(class_to_idx)
    for name, idx in class_to_idx.items():
        class_names[idx] = name

    print(f"Dataset loaded: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    print(f"Classes ({len(class_names)}): {class_names}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}, Labels shape: {labels.shape}")
    print(f"Label sample: {[class_names[l] for l in labels[:8].tolist()]}")
