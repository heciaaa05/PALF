import pytest
import torch
from palf.models import MODEL_REGISTRY, build_model


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_forward_shape(name):
    model = build_model(name, input_len=48, horizon=12)
    y = model(torch.randn(4, 48, 1))
    assert y.shape == (4, 12)