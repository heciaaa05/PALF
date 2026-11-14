import os
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from palf.core.config import PalfConfig
from palf.core.policy import PolicyDecision
from palf.data.patterns_weather import WEATHER_PATTERNS
from palf.training.scheduler import AdaptiveScheduler
from palf.training.trainer import train_model


def _make_imbalanced_pat(counts=(80, 120, 100, 700), seed=42):
    rng = np.random.default_rng(seed)
    parts = [np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)]
    pat = np.concatenate(parts)
    rng.shuffle(pat)
    return pat


def _make_subset_data(n_train=64, n_val=16, n_test=16, L=24, H=6, D=5, seed=101):
    def _mk(n, s_off):
        r = np.random.default_rng(seed + s_off)
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


def test_scheduler_a_w1_budget1_contains_all_train_once():
    """(a) w=1, budget=1: epoch chua toan bo train dung 1 lan."""
    pat = _make_imbalanced_pat((80, 120, 100, 700))
    cfg = PalfConfig(budget_ratio=1.0)
    sched = AdaptiveScheduler(pat, WEATHER_PATTERNS, cfg=cfg, seed=7)
    dec = PolicyDecision(weights={p: 1.0 for p in WEATHER_PATTERNS}, info={})

    idx = sched.plan_epoch(epoch=1, decision=dec)
    assert len(idx) == len(pat)
    assert np.array_equal(np.sort(idx), np.arange(len(pat)))


def test_scheduler_b_w1_budget07_proportional_to_pi():
    """(b) w=1, budget=0.7: so mau dung va moi Pattern nhan phan ty le pi_p."""
    counts = (80, 120, 100, 700)
    pat = _make_imbalanced_pat(counts)
    n_train = len(pat)
    cfg = PalfConfig(budget_ratio=0.7, min_floor_samples=32, min_floor_frac=0.05)
    sched = AdaptiveScheduler(pat, WEATHER_PATTERNS, cfg=cfg, seed=11)

    idx = sched.plan_epoch(epoch=2, decision=None)
    n_e = int(round(0.7 * n_train))
    assert len(idx) == n_e
    # Khong lap lai mau trong cung 1 epoch
    assert len(np.unique(idx)) == n_e

    for i, p_name in enumerate(WEATHER_PATTERNS):
        n_p = int((pat[idx] == i).sum())
        expected_p = round(n_e * (counts[i] / n_train))
        assert abs(n_p - expected_p) <= 1


def test_scheduler_c_d_exact_total_and_floor_enforced():
    """(c) Tong so mau moi epoch dung bang n_e; (d) khong Pattern nao duoi san."""
    counts = (80, 120, 100, 700)
    pat = _make_imbalanced_pat(counts)
    n_train = len(pat)
    cfg = PalfConfig(budget_ratio=0.65, min_floor_samples=32, min_floor_frac=0.05)
    sched = AdaptiveScheduler(pat, WEATHER_PATTERNS, cfg=cfg, seed=19)

    # Trong so cuc doan: 1 pattern rat cao (vuot tran N_p), 1 pattern rat thap (cham san)
    weights = {
        "pressure_drop": 3.0,
        "humidity_net_rise": 2.5,
        "temp_anomaly": 0.25,
        "normal": 0.25,
    }
    dec = PolicyDecision(weights=weights, info={})
    idx = sched.plan_epoch(epoch=4, decision=dec)
    n_e = int(round(0.65 * n_train))

    # (c) Tong so mau dung bang n_e va khong lap
    assert len(idx) == n_e
    assert len(np.unique(idx)) == n_e

    # (d) Moi Pattern >= san min_floor_samples (32) va <= N_p
    for i, p_name in enumerate(WEATHER_PATTERNS):
        n_p = int((pat[idx] == i).sum())
        floor_p = min(counts[i], max(32, int(round(0.05 * (counts[i] / n_train) * n_e))))
        assert n_p >= floor_p
        assert n_p <= counts[i]


def test_scheduler_e_same_seed_reproducibility():
    """(e) Cung seed cho cung danh sach mau."""
    pat = _make_imbalanced_pat((80, 120, 100, 700))
    cfg = PalfConfig(budget_ratio=0.7)
    s1 = AdaptiveScheduler(pat, WEATHER_PATTERNS, cfg=cfg, seed=42)
    s2 = AdaptiveScheduler(pat, WEATHER_PATTERNS, cfg=cfg, seed=42)
    w = {"pressure_drop": 1.8, "humidity_net_rise": 1.2, "temp_anomaly": 1.5, "normal": 0.6}

    idx1 = s1.plan_epoch(epoch=5, decision=w)
    idx2 = s2.plan_epoch(epoch=5, decision=w)
    assert np.array_equal(idx1, idx2)

    # Epoch khac nhau cho thu tu khac nhau
    idx_next = s1.plan_epoch(epoch=6, decision=w)
    assert not np.array_equal(idx1, idx_next)


def test_trainer_f_baseline_mode_identical_and_g_palf_mode_files_and_cartography(tmp_path):
    """(f) mode='baseline' giu nguyen ket qua cu tren cung seed;
    (g) chay 2 epoch PALF tren tap con nho ra du file, cartography khong co NaN khi budget=0.7."""
    data1 = _make_subset_data(n_train=64, n_val=16, n_test=16, seed=202)
    data2 = _make_subset_data(n_train=64, n_val=16, n_test=16, seed=202)

    b_dir1 = os.path.join(str(tmp_path), "base1")
    b_dir2 = os.path.join(str(tmp_path), "base2")
    r_b1 = train_model("gru", epochs=2, batch=16, lr=1e-3, seed=3, data=data1, out_dir=b_dir1, mode="baseline")
    r_b2 = train_model("gru", epochs=2, batch=16, lr=1e-3, seed=3, data=data2, out_dir=b_dir2)

    pd.testing.assert_frame_equal(
        r_b1["history"].drop(columns=["epoch_time_sec", "overhead_sec", "elapsed"]),
        r_b2["history"].drop(columns=["epoch_time_sec", "overhead_sec", "elapsed"]),
    )

    # (g) Chay 3 epoch PALF voi budget_ratio=0.7 va warmup=1
    data_palf = _make_subset_data(n_train=64, n_val=16, n_test=16, seed=202)
    palf_dir = os.path.join(str(tmp_path), "palf_run")
    cfg = PalfConfig(policy="state", budget_ratio=0.7, warmup_epochs=1, window=1, min_floor_samples=8)
    r_palf = train_model(
        "gru",
        epochs=3,
        batch=16,
        lr=1e-3,
        seed=3,
        data=data_palf,
        out_dir=palf_dir,
        mode="palf",
        palf_config=cfg,
    )

    expected_files = [
        "history.csv",
        "epoch_pattern_log.csv",
        "cartography.csv",
        "sample_loss_epochs.npy",
        "sample_index.npy",
        "pattern_labels_train.npy",
        "run_config.json",
        "test_metrics.json",
        "policy_log.csv",
    ]
    for fn in expected_files:
        assert os.path.isfile(os.path.join(palf_dir, fn)), f"Thieu file {fn}"

    # Kiem tra sample_loss_epochs.npy co NaN (do budget=0.7 < 1.0) nhung cartography.csv khong co NaN
    loss_ep = np.load(os.path.join(palf_dir, "sample_loss_epochs.npy"))
    assert np.isnan(loss_ep).any()

    carto = pd.read_csv(os.path.join(palf_dir, "cartography.csv"))
    assert not carto["mean_loss"].isna().any()
    assert not carto["std_loss"].isna().any()

    # Kiem tra history.csv co overhead_sec va n_samples_processed == round(0.7 * 64) == 45
    hist = pd.read_csv(os.path.join(palf_dir, "history.csv"))
    assert "overhead_sec" in hist.columns
    assert (hist["n_samples_processed"] == int(round(0.7 * 64))).all()

    # Kiem tra policy_log.csv
    plog = pd.read_csv(os.path.join(palf_dir, "policy_log.csv"))
    assert len(plog) == 3 * len(WEATHER_PATTERNS)
    for col in ("epoch", "pattern", "weight", "need_score", "n_seen", "loss_ratio", "velocity", "instability"):
        assert col in plog.columns
