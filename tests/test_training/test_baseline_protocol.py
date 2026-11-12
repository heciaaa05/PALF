import importlib.util
import json
import os
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from palf.data.patterns_weather import WEATHER_PATTERNS
from palf.training.trainer import train_model


EXPECTED_8_FILES = [
    "history.csv",
    "epoch_pattern_log.csv",
    "cartography.csv",
    "sample_loss_epochs.npy",
    "sample_index.npy",
    "pattern_labels_train.npy",
    "run_config.json",
    "test_metrics.json",
]


def _make_weather_like_subset(n_train=40, n_val=16, n_test=16, L=24, H=6, D=5, seed=123):
    rng = np.random.default_rng(seed)

    def _mk(n, s_offset):
        r = np.random.default_rng(seed + s_offset)
        x = r.normal(size=(n, L, D)).astype(np.float32)
        y = r.normal(size=(n, H)).astype(np.float32)
        pat = np.arange(n, dtype=np.int64) % len(WEATHER_PATTERNS)
        r.shuffle(pat)
        idx = np.arange(n, dtype=np.int64)
        return TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(pat),
            torch.from_numpy(idx),
        )

    th = {
        "pattern_names": WEATHER_PATTERNS,
        "pattern_scheme": "weather_v1",
        "target_col": "T (degC)",
        "target_idx": 1,
        "target_mean": 9.45,
        "target_std": 8.42,
    }
    meta = dict(th)
    return _mk(n_train, 1), _mk(n_val, 2), _mk(n_test, 3), th, meta


def test_step0_eight_files_and_shapes(tmp_path):
    """(a) Chay nhanh 2 epoch tren mot phan du lieu cho du 8 file;
    (b) kiem tra hinh dang sample_loss_epochs.npy bang (n_train, epochs)."""
    data = _make_weather_like_subset(n_train=40, n_val=16, n_test=16)
    res = train_model(
        "gru",
        epochs=2,
        batch=16,
        lr=1e-3,
        device_name="cpu",
        seed=0,
        data=data,
        out_root=str(tmp_path),
        use_seed_subdir=True,
        patience=None,
    )

    out_dir = res["out_dir"]
    assert out_dir.endswith(os.path.join("gru", "seed0"))

    for fname in EXPECTED_8_FILES:
        fpath = os.path.join(out_dir, fname)
        assert os.path.isfile(fpath), f"Thieu file dau ra: {fname}"

    # (b) Kiem tra hinh dang va kieu du lieu cua 3 file .npy
    loss_ep = np.load(os.path.join(out_dir, "sample_loss_epochs.npy"))
    sample_idx = np.load(os.path.join(out_dir, "sample_index.npy"))
    pat_labels = np.load(os.path.join(out_dir, "pattern_labels_train.npy"))

    assert loss_ep.shape == (40, 2)
    assert loss_ep.dtype == np.float32
    assert not np.isnan(loss_ep).any()

    assert sample_idx.shape == (40,)
    assert np.issubdtype(sample_idx.dtype, np.integer)
    assert np.array_equal(sample_idx, np.arange(40))

    assert pat_labels.shape == (40,)
    assert pat_labels.dtype.kind in ("U", "S")
    assert set(pat_labels.tolist()).issubset(set(WEATHER_PATTERNS))

    # Kiem tra history.csv co epoch_time_sec va n_samples_processed
    hist_df = pd.read_csv(os.path.join(out_dir, "history.csv"))
    for col in ("train_loss", "val_loss", "epoch_time_sec", "n_samples_processed"):
        assert col in hist_df.columns
    assert list(hist_df["n_samples_processed"]) == [40, 40]
    assert (hist_df["epoch_time_sec"] >= 0.0).all()

    # Kiem tra test_metrics.json co MAE, RMSE ca don vi chuan hoa va degC
    with open(os.path.join(out_dir, "test_metrics.json"), "r", encoding="utf-8") as f:
        tm = json.load(f)
    for k in ("mae_norm", "rmse_norm", "mae_degC", "rmse_degC", "target_mean", "target_std"):
        assert k in tm
    assert tm["mae_degC"] == pytest.approx(tm["mae_norm"] * 8.42, rel=1e-5)
    assert tm["rmse_degC"] == pytest.approx(tm["rmse_norm"] * 8.42, rel=1e-5)

    # Kiem tra run_config.json
    with open(os.path.join(out_dir, "run_config.json"), "r", encoding="utf-8") as f:
        rc = json.load(f)
    for k in ("model", "seed", "epochs", "batch", "lr", "device", "pattern_scheme", "git_commit"):
        assert k in rc
    assert rc["model"] == "gru"
    assert rc["seed"] == 0
    assert rc["epochs"] == 2
    assert rc["pattern_scheme"] == "weather_v1"
    assert isinstance(rc["git_commit"], str) and len(rc["git_commit"]) > 0


def test_step0_sample_losses_match_epoch_pattern_log(tmp_path):
    """(c) Trung binh loss cua cac mau cung Pattern trong sample_loss_epochs.npy
    khop epoch_pattern_log.csv (sai so nho)."""
    data = _make_weather_like_subset(n_train=48, n_val=16, n_test=16, seed=77)
    res = train_model(
        "tcn",
        epochs=3,
        batch=16,
        lr=1e-3,
        device_name="cpu",
        seed=1,
        data=data,
        out_root=str(tmp_path),
        use_seed_subdir=True,
    )
    out_dir = res["out_dir"]
    loss_ep = np.load(os.path.join(out_dir, "sample_loss_epochs.npy"))
    pat_labels = np.load(os.path.join(out_dir, "pattern_labels_train.npy"))
    ep_log = pd.read_csv(os.path.join(out_dir, "epoch_pattern_log.csv"))

    for _, row in ep_log.iterrows():
        ep_idx = int(row["epoch"]) - 1
        pat_name = row["pattern"]
        mask = pat_labels == pat_name
        assert int(mask.sum()) == int(row["n"])
        mean_from_npy = float(loss_ep[mask, ep_idx].mean())
        std_from_npy = float(loss_ep[mask, ep_idx].std())
        assert mean_from_npy == pytest.approx(float(row["train_loss_mean"]), abs=1e-5)
        assert std_from_npy == pytest.approx(float(row["train_loss_std"]), abs=1e-5)


def test_step0_same_seed_reproducibility(tmp_path):
    """(d) Chay lai cung seed cho ket qua giong nhau."""
    data1 = _make_weather_like_subset(n_train=32, n_val=16, n_test=16, seed=99)
    data2 = _make_weather_like_subset(n_train=32, n_val=16, n_test=16, seed=99)

    dir1 = os.path.join(str(tmp_path), "run1")
    dir2 = os.path.join(str(tmp_path), "run2")

    r1 = train_model("gru", epochs=2, batch=16, lr=1e-3, device_name="cpu",
                     seed=4, data=data1, out_dir=dir1)
    r2 = train_model("gru", epochs=2, batch=16, lr=1e-3, device_name="cpu",
                     seed=4, data=data2, out_dir=dir2)

    l1 = np.load(os.path.join(dir1, "sample_loss_epochs.npy"))
    l2 = np.load(os.path.join(dir2, "sample_loss_epochs.npy"))
    assert np.allclose(l1, l2, atol=1e-6)
    assert r1["test_metrics"]["mae_degC"] == pytest.approx(r2["test_metrics"]["mae_degC"], abs=1e-6)
    assert r1["test_metrics"]["rmse_degC"] == pytest.approx(r2["test_metrics"]["rmse_degC"], abs=1e-6)


def test_run_observation_cli_and_palf_mode_blocked(tmp_path):
    """Kiem tra script CLI run_observation.py tao du 8 file va chan --mode palf."""
    script_path = os.path.join("experiments", "scripts", "run_observation.py")
    spec = importlib.util.spec_from_file_location("run_observation", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Kiem tra chan --mode palf
    args_palf = mod.parse_args(["--model", "gru", "--mode", "palf"])
    with pytest.raises(RuntimeError, match="mode palf"):
        mod.run_from_args(args_palf)

    # Chay thu 2 epoch tren subset cua Jena
    args_ok = mod.parse_args([
        "--model", "gru",
        "--seed", "0",
        "--epochs", "2",
        "--batch", "16",
        "--subset", "48",
        "--out-root", str(tmp_path),
    ])
    results = mod.run_from_args(args_ok)
    assert len(results) == 1
    out_dir = os.path.join(str(tmp_path), "gru", "seed0")
    for fname in EXPECTED_8_FILES:
        assert os.path.isfile(os.path.join(out_dir, fname))
