"""digit_recognizer: reusable code for the Kaggle Digit Recognizer assignment."""

from .data import (
    DigitDataset,
    compute_mean_std,
    get_train_augmentation,
    load_test_csv,
    load_train_csv,
    make_loaders,
    make_test_loader,
    split_train_val,
)
from .models import MLP, ImprovedCNN, SimpleCNN, build_model
from .plots import plot_confusion_matrix, plot_history, plot_misclassified, plot_samples
from .predict import check_submission, make_submission, predict_labels, predict_logits
from .train import evaluate, fit, train_one_epoch
from .utils import count_parameters, get_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "DigitDataset",
    "compute_mean_std",
    "get_train_augmentation",
    "load_test_csv",
    "load_train_csv",
    "make_loaders",
    "make_test_loader",
    "split_train_val",
    "MLP",
    "SimpleCNN",
    "ImprovedCNN",
    "build_model",
    "plot_confusion_matrix",
    "plot_history",
    "plot_misclassified",
    "plot_samples",
    "check_submission",
    "make_submission",
    "predict_labels",
    "predict_logits",
    "evaluate",
    "fit",
    "train_one_epoch",
    "count_parameters",
    "get_device",
    "set_seed",
]
