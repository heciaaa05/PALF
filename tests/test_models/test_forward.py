import pytest
import torch
from palf.models import MODEL_REGISTRY, build_model


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_forward_shape_univariate(name):
    model = build_model(name, input_len=48, horizon=12, input_dim=1)
    y = model(torch.randn(4, 48, 1))
    assert y.shape == (4, 12)


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_forward_shape_multivariate(name):
    model = build_model(name, input_len=48, horizon=12, input_dim=3)
    y = model(torch.randn(4, 48, 3))
    assert y.shape == (4, 12)