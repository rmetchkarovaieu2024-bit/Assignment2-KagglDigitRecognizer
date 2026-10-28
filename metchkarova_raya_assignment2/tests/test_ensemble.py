import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from digit_recognizer import build_model, fit, predict_ensemble


def _loader(n=32):
    x = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


def test_predict_ensemble_returns_one_label_per_image():
    models = [build_model("cnn").eval(), build_model("cnn").eval()]
    pred = predict_ensemble(models, _loader(), torch.device("cpu"))
    assert pred.shape == (32,)
    assert ((pred >= 0) & (pred <= 9)).all()


def test_fit_without_validation_keeps_last_weights():
    model = build_model("cnn")
    history = fit(model, _loader(), None, torch.device("cpu"), epochs=1, verbose=False)
    assert history["best_epoch"] == 1
    assert np.isnan(history["best_val_acc"])
