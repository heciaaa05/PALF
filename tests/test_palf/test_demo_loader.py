import json
import os
import numpy as np
import pandas as pd
import pytest

from palf.evaluation.demo_loader import (
    compute_time_to_target,
    compute_val_auc_samples,
    discover_saved_runs,
    load_saved_run,
    summarize_palf_diagnostics,
    summarize_side_by_side,
)


def _write_dummy_run(run_dir: str, mode: str, config_tag: str, model: str, seed: int, val_losses=(1.0, 0.8, 0.5), n_per_ep=100):
    os.makedirs(run_dir, exist_ok=True)
    epochs = list(range(1, len(val_losses) + 1))
    hist = pd.DataFrame({
        "epoch": epochs,
        "train_loss": [v * 1.1 for v in val_losses],
        "val_loss": list(val_losses),
        "grad_norm": [0.5] * len(epochs),
        "epoch_time_sec": [2.0] * len(epochs),
        "overhead_sec": [0.05 if mode == "palf" else 0.0] * len(epochs),
        "n_samples_processed": [n_per_ep] * len(epochs),
        "elapsed": [2.0 * e for e in epochs],
    })
    hist.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    pats = ["pressure_drop", "humidity_net_rise", "temp_anomaly", "normal"]
    pat_rows = []
    pol_rows = []
    for e in epochs:
        for i, p in enumerate(pats):
            pat_rows.append({
                "epoch": e,
                "pattern": p,
                "n": 25,
                "n_seen": 25 if mode == "baseline" else (15 + 5 * i),
                "train_loss_mean": float(val_losses[e - 1]),
                "train_loss_std": 0.1,
                "val_loss": float(val_losses[e - 1]),
                "grad_norm": 0.5,
            })
            if mode == "palf":
                pol_rows.append({
                    "epoch": e,
                    "pattern": p,
                    "weight": 1.0 + 0.2 * (i - 1),
                    "need_score": 0.5 * (i - 1),
                    "n_seen": 15 + 5 * i,
                    "quota_planned": 15 + 5 * i,
                    "in_warmup": e <= 1,
                    "updated": True,
                    "hit_floor": False,
                    "hit_ceil": False,
                    "loss_ratio": 0.9 - 0.1 * e,
                    "velocity": 0.1,
                    "instability": 0.05,
                })
    pd.DataFrame(pat_rows).to_csv(os.path.join(run_dir, "epoch_pattern_log.csv"), index=False)
    if mode == "palf":
        pd.DataFrame(pol_rows).to_csv(os.path.join(run_dir, "policy_log.csv"), index=False)

    rc = {
        "mode": mode,
        "config_tag": config_tag,
        "model": model,
        "model_name": model.upper(),
        "seed": seed,
        "epochs": len(epochs),
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(rc, f)

    tm = {
        "mae_degC": 1.8 if mode == "palf" else 2.0,
        "rmse_degC": 2.4 if mode == "palf" else 2.6,
        "best_epoch": len(epochs),
        "best_val_loss": float(min(val_losses)),
        "by_pattern": [
            {"pattern": p, "n": 10, "mae_degC": 1.5 + 0.1 * i, "rmse_degC": 2.0 + 0.1 * i}
            for i, p in enumerate(pats)
        ],
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(tm, f)


def test_demo_loader_discover_and_summarize(tmp_path):
    b_root = os.path.join(str(tmp_path), "baseline", "jena")
    p_root = os.path.join(str(tmp_path), "palf", "jena")

    _write_dummy_run(os.path.join(b_root, "gru", "seed0"), "baseline", "baseline", "gru", 0, (1.0, 0.8, 0.5), 100)
    _write_dummy_run(os.path.join(p_root, "state_b0.7_a1.0", "gru", "seed0"), "palf", "state_b0.7_a1.0", "gru", 0, (1.0, 0.50, 0.45), 70)

    runs = discover_saved_runs(b_root, p_root)
    assert len(runs) == 2

    overall_df, pat_df, val_target = summarize_side_by_side(runs)
    assert val_target == pytest.approx(1.02 * 0.5)
    assert len(overall_df) == 2
    assert len(pat_df) == 8

    # Kiem tra so mau va thoi gian toi nguong val_target = 0.51:
    # Baseline dat <= 0.51 o epoch 3 (300 mau, 6.0s); PALF dat <= 0.51 o epoch 2 (140 mau, 4.0s)
    b_row = overall_df[overall_df["Mode"] == "baseline"].iloc[0]
    p_row = overall_df[overall_df["Mode"] == "palf"].iloc[0]
    assert b_row["Epoch tới ngưỡng"] == 3
    assert b_row["Số mẫu tới ngưỡng"] == pytest.approx(300.0)
    assert b_row["Giây tới ngưỡng"] == pytest.approx(6.0)

    assert p_row["Epoch tới ngưỡng"] == 2
    assert p_row["Số mẫu tới ngưỡng"] == pytest.approx(140.0)
    assert p_row["Giây tới ngưỡng"] == pytest.approx(4.0)


def test_demo_loader_palf_diagnostics_and_missing_palf(tmp_path):
    b_dir = os.path.join(str(tmp_path), "base_only")
    _write_dummy_run(b_dir, "baseline", "baseline", "gru", 0)
    b_run = load_saved_run(b_dir)

    # Khi chon baseline hoac chua co PALF, phai bao has_palf=False va message ro rang, khong dung so gia
    diag_empty = summarize_palf_diagnostics(b_run)
    assert diag_empty["has_palf"] is False
    assert "Chưa có kết quả PALF" in diag_empty["message"]
    assert diag_empty["weights_pivot"].empty

    p_dir = os.path.join(str(tmp_path), "palf_only")
    _write_dummy_run(p_dir, "palf", "state_b0.7_a1.0", "gru", 0)
    p_run = load_saved_run(p_dir)
    diag_ok = summarize_palf_diagnostics(p_run)
    assert diag_ok["has_palf"] is True
    assert diag_ok["weights_pivot"].shape == (3, 4)
    assert diag_ok["share_pivot"].shape == (3, 4)
    # Tong ty le mau thuc su moi epoch bang 1.0
    assert np.allclose(diag_ok["share_pivot"].sum(axis=1).to_numpy(), 1.0)
    for k in ("loss_ratio", "velocity", "instability", "need_score"):
        assert k in diag_ok["state_pivots"]
