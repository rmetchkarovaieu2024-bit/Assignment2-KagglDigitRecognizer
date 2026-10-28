"""Prediction and Kaggle submission helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def predict_logits(model, loader, device) -> torch.Tensor:
    """Concatenate the logits of every batch. Works with loaders that return x or (x, y)."""
    model.eval()
    outputs = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        outputs.append(model(images.to(device)).cpu())
    return torch.cat(outputs)


@torch.no_grad()
def predict_ensemble(models, loader, device) -> np.ndarray:
    """Ensemble prediction: average the softmax probabilities of several models, then take the argmax.

    Each model makes slightly different mistakes, so averaging their probabilities
    ("soft voting") usually corrects some of the errors of any single model.
    """
    total = None
    for model in models:
        probs = torch.softmax(predict_logits(model, loader, device), dim=1)
        total = probs if total is None else total + probs
    return (total / len(models)).argmax(dim=1).numpy()


def predict_labels(model, loader, device) -> np.ndarray:
    """Predicted class = argmax of the logits (softmax would not change the argmax)."""
    return predict_logits(model, loader, device).argmax(dim=1).numpy()


def make_submission(labels, path: str | Path | None = None) -> pd.DataFrame:
    """Build the Kaggle file: header ImageId,Label with ImageId starting at 1."""
    labels = np.asarray(labels).astype(int)
    df = pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels})
    check_submission(df, expected_rows=len(labels))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def check_submission(df: pd.DataFrame, expected_rows: int = 28000) -> bool:
    """Check the rubric requirements for the results CSV. Raises ValueError if something is wrong."""
    if list(df.columns) != ["ImageId", "Label"]:
        raise ValueError(f"Columns must be ['ImageId', 'Label'], got {list(df.columns)}")
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")
    if df.isna().any().any():
        raise ValueError("Submission contains missing values")
    if not (df["ImageId"].to_numpy() == np.arange(1, expected_rows + 1)).all():
        raise ValueError("ImageId must go from 1 to the number of rows, in order")
    if not pd.api.types.is_integer_dtype(df["Label"]) or not df["Label"].between(0, 9).all():
        raise ValueError("Label must be integers between 0 and 9")
    return True
