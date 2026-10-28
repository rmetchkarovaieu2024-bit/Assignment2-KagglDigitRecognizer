"""Small helpers: reproducibility, device selection and parameter counting."""

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Fix the random seeds so that results can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use the GPU if there is one, otherwise the CPU (same idea as in the course notebooks)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
