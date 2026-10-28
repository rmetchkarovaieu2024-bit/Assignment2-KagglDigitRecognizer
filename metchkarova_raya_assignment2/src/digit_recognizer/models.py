"""Models: MLP baseline, baseline CNN (2 conv blocks) and improved CNN.

All models take input of shape [B, 1, 28, 28] and return LOGITS of shape [B, 10].
We never apply softmax inside the model: nn.CrossEntropyLoss expects raw logits.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully connected baseline, as in Session 9 (Linear -> ReLU -> Linear), with one extra layer.

    [B, 1, 28, 28] -> Flatten -> [B, 784] -> 256 -> 128 -> [B, 10]
    """

    def __init__(self, num_classes: int = 10, hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """Baseline CNN with 2 convolutional blocks (Conv -> ReLU -> MaxPool).

    Input              (B, 1, 28, 28)
    Conv2d(1, 32, 3, p=1)  -> (B, 32, 28, 28) -> ReLU -> MaxPool2d(2) -> (B, 32, 14, 14)
    Conv2d(32, 64, 3, p=1) -> (B, 64, 14, 14) -> ReLU -> MaxPool2d(2) -> (B, 64, 7, 7)
    Flatten                -> (B, 3136)
    Linear(3136, 128) -> ReLU -> Linear(128, 10) -> (B, 10) logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv2d -> BatchNorm2d -> ReLU, the building block used in the course CNN notebook."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


class ImprovedCNN(nn.Module):
    """Improved CNN: same 2-block idea, plus BatchNorm, two convs per block and Dropout.

    Input (B, 1, 28, 28)
    Block 1: conv_bn_relu(1, 32), conv_bn_relu(32, 32), MaxPool2d(2), Dropout2d -> (B, 32, 14, 14)
    Block 2: conv_bn_relu(32, 64), conv_bn_relu(64, 64), MaxPool2d(2), Dropout2d -> (B, 64, 7, 7)
    Head   : Flatten -> Linear(3136, 256) -> BatchNorm1d -> ReLU -> Dropout -> Linear(256, 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.4, dropout2d: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(1, 32),
            conv_bn_relu(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout2d),
            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout2d),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODELS = {"mlp": MLP, "cnn": SimpleCNN, "improved_cnn": ImprovedCNN}


def build_model(name: str, **kwargs) -> nn.Module:
    """Create a model by name: 'mlp', 'cnn' or 'improved_cnn'."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODELS)}")
    return MODELS[name](**kwargs)
