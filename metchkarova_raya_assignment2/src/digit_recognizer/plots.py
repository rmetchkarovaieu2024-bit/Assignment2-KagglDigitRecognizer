"""Figures for the analysis: sample digits, learning curves, confusion matrix, misclassified digits."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_samples(images, labels, n: int = 16, path=None):
    """Show the first n images with their labels."""
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for ax, i in zip(axes.flat, range(n)):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(str(labels[i]), fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Training samples (label above each image)")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_history(histories: dict, path=None):
    """Training and validation loss/accuracy curves for one or several models.

    histories: {"model name": history_dict_returned_by_fit}
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (name, h) in enumerate(histories.items()):
        color = f"C{i}"
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], color=color, linestyle="--", label=f"{name} train")
        axes[0].plot(epochs, h["val_loss"], color=color, label=f"{name} validation")
        axes[1].plot(epochs, h["train_acc"], color=color, linestyle="--", label=f"{name} train")
        axes[1].plot(epochs, h["val_acc"], color=color, label=f"{name} validation")
    axes[0].set_title("Loss (cross-entropy)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix (validation)", path=None):
    """Confusion matrix: rows are true digits, columns are predicted digits."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=list(range(10))).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig, cm


def plot_misclassified(images, y_true, y_pred, n: int = 24, path=None):
    """Grid of misclassified validation digits with 'true -> predicted'."""
    wrong = np.where(np.asarray(y_true) != np.asarray(y_pred))[0][:n]
    cols = 8
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(f"true {y_true[i]} / pred {y_pred[i]}", fontsize=8, color="crimson")
    fig.suptitle(f"Misclassified validation digits ({len(wrong)} shown)")
    fig.tight_layout()
    _save(fig, path)
    return fig
