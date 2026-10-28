import pytest
import torch

from digit_recognizer import build_model, count_parameters


@pytest.mark.parametrize("name", ["mlp", "cnn", "improved_cnn"])
def test_model_returns_logits_of_shape_B_10(name):
    model = build_model(name)
    model.eval()
    x = torch.randn(8, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (8, 10)
    # Logits are raw scores, not probabilities: they do not have to sum to 1
    assert torch.isfinite(logits).all()


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("resnet1000")


def test_models_have_trainable_parameters():
    assert count_parameters(build_model("cnn")) > 0
