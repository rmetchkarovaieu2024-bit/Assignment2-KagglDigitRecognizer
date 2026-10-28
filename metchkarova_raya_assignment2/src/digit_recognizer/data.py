"""Data loading for the Kaggle Digit Recognizer competition.

train.csv: 42,000 rows, column ``label`` + 784 pixel columns (28x28, values 0-255).
test.csv : 28,000 rows, 784 pixel columns only.

The pipeline follows the course notebooks:
    pixels / 255  ->  values in [0, 1]          (what ToTensor() does)
    (x - mean) / std                            (what Normalize() does, 1 channel)
    reshape to [B, 1, 28, 28]                   (B, C, H, W format)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE = 28
NUM_CLASSES = 10


def load_train_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read train.csv and return (images, labels).

    images: uint8 array of shape (N, 28, 28)
    labels: int64 array of shape (N,)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md.")
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("train.csv must contain a 'label' column")
    labels = df["label"].to_numpy(dtype=np.int64)
    pixels = df.drop(columns=["label"]).to_numpy(dtype=np.uint8)
    if pixels.shape[1] != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"Expected 784 pixel columns, got {pixels.shape[1]}")
    return pixels.reshape(-1, IMAGE_SIZE, IMAGE_SIZE), labels


def load_test_csv(path: str | Path) -> np.ndarray:
    """Read test.csv and return images as a uint8 array of shape (N, 28, 28)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. See data/README.md.")
    pixels = pd.read_csv(path).to_numpy(dtype=np.uint8)
    if pixels.shape[1] != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"Expected 784 pixel columns, got {pixels.shape[1]}")
    return pixels.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)


def split_train_val(
    images: np.ndarray, labels: np.ndarray, val_size: float = 0.1, seed: int = 42
):
    """Stratified hold-out split: every digit keeps the same proportion in train and validation."""
    return train_test_split(
        images, labels, test_size=val_size, random_state=seed, stratify=labels
    )


def compute_mean_std(images: np.ndarray) -> tuple[float, float]:
    """Mean and std of the pixels (scaled to [0, 1]), computed on the TRAINING split only."""
    x = images.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std())


def get_train_augmentation() -> transforms.Compose:
    """Random, label-preserving transforms used only during training.

    Small rotations, shifts and zooms: a digit stays the same digit.
    No horizontal flip (a flipped 2 or 7 is not a valid digit).
    """
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, fill=0.0
            )
        ]
    )


class DigitDataset(Dataset):
    """PyTorch Dataset that returns (image [1, 28, 28], label) or only the image for test data.

    The optional ``augment`` transform is applied on the [0, 1] image BEFORE normalisation,
    so that the black background (0) used to fill empty borders stays black.
    """

    def __init__(self, images, labels=None, mean=0.0, std=1.0, augment=None):
        x = torch.tensor(np.asarray(images), dtype=torch.float32) / 255.0
        self.images = x.unsqueeze(1)  # (N, 1, 28, 28)
        self.labels = None if labels is None else torch.tensor(np.asarray(labels), dtype=torch.long)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.augment is not None:
            x = self.augment(x)
        x = (x - self.mean) / self.std
        if self.labels is None:
            return x
        return x, self.labels[idx]


def make_loaders(
    x_train,
    y_train,
    x_val,
    y_val,
    mean: float,
    std: float,
    batch_size: int = 128,
    augment: bool = False,
    seed: int = 42,
):
    """Build train and validation DataLoaders. Only the training loader is shuffled/augmented."""
    train_ds = DigitDataset(
        x_train, y_train, mean, std, augment=get_train_augmentation() if augment else None
    )
    val_ds = DigitDataset(x_val, y_val, mean, std)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    return train_loader, val_loader


def make_test_loader(x_test, mean: float, std: float, batch_size: int = 512) -> DataLoader:
    """DataLoader for the unlabelled Kaggle test set (never shuffled, so the order matches ImageId)."""
    return DataLoader(DigitDataset(x_test, None, mean, std), batch_size=batch_size, shuffle=False)
