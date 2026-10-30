import numpy as np
import torch
from torch.utils.data import TensorDataset

from palf.training.trainer import train_model


def _tiny_dataset(n, L=10, H=3, D=1, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, L, D)).astype(np.float32)
    y = rng.normal(size=(n, H)).astype(np.float32)
    pat = rng.integers(0, 4, size=n).astype(np.int64)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y),
                         torch.from_numpy(pat), torch.arange(n))


def _tiny_data():
    return _tiny_dataset(32, seed=1), _tiny_dataset(16, seed=2), _tiny_dataset(16, seed=3), {}


def test_patience_none_runs_full_epochs(tmp_path):
    res = train_model("GRU", epochs=3, batch=8, lr=1e-2, device_name="cpu",
                      seed=42, data=_tiny_data(), patience=None,
                      out_root=str(tmp_path))
    assert res["epochs_run"] == 3
    assert res["stopped_early"] is False
    assert len(res["history"]) == 3


def test_patience_stops_early_when_no_improvement(tmp_path):
    # lr=0.0 -> trong so khong doi -> val_loss giong het nhau moi epoch -> khong bao gio cai thien
    res = train_model("GRU", epochs=10, batch=8, lr=0.0, device_name="cpu",
                      seed=42, data=_tiny_data(), patience=2,
                      out_root=str(tmp_path))
    assert res["stopped_early"] is True
    assert res["epochs_run"] < 10
    assert len(res["history"]) == res["epochs_run"]