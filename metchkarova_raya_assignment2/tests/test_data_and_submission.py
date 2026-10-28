import numpy as np
import pandas as pd
import pytest

from digit_recognizer import (
    DigitDataset,
    check_submission,
    compute_mean_std,
    load_train_csv,
    make_submission,
    split_train_val,
)


def _fake_train_csv(tmp_path, n=50):
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(n, 784))
    df = pd.DataFrame(pixels, columns=[f"pixel{i}" for i in range(784)])
    df.insert(0, "label", np.arange(n) % 10)
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


def test_load_and_reshape(tmp_path):
    images, labels = load_train_csv(_fake_train_csv(tmp_path))
    assert images.shape == (50, 28, 28)
    assert labels.shape == (50,)


def test_dataset_returns_B_1_28_28(tmp_path):
    images, labels = load_train_csv(_fake_train_csv(tmp_path))
    mean, std = compute_mean_std(images)
    x, y = DigitDataset(images, labels, mean, std)[0]
    assert tuple(x.shape) == (1, 28, 28)
    assert int(y) == int(labels[0])


def test_split_is_stratified(tmp_path):
    images, labels = load_train_csv(_fake_train_csv(tmp_path, n=100))
    _, _, _, y_val = split_train_val(images, labels, val_size=0.2)
    assert sorted(np.bincount(y_val)) == [2] * 10


def test_submission_format():
    df = make_submission(np.zeros(28000, dtype=int))
    assert list(df.columns) == ["ImageId", "Label"]
    assert df["ImageId"].iloc[0] == 1 and df["ImageId"].iloc[-1] == 28000
    assert check_submission(df)


def test_bad_submission_is_rejected():
    df = pd.DataFrame({"ImageId": np.arange(1, 28001), "Label": np.full(28000, 11)})
    with pytest.raises(ValueError):
        check_submission(df)
