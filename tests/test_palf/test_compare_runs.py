import json
import os
import numpy as np
import pandas as pd
import pytest

from palf.evaluation.compare_runs import run_comparison_protocol


def _write_known_run(
    run_dir: str,
    mode: str,
    config_tag: str,
    model: str,
    seed: int,
    val_losses=(1.0, 0.8, 0.5),
    n_per_ep=100,
    mae_degC=2.0,
    rmse_degC=2.5,
):
    os.makedirs(run_dir, exist_ok=True)
    epochs = list(range(1, len(val_losses) + 1))
    hist = pd.DataFrame({
        "epoch": epochs,
        "train_loss": [v * 1.05 for v in val_losses],
        "val_loss": list(val_losses),
        "grad_norm": [0.4] * len(epochs),
        "epoch_time_sec": [2.0] * len(epochs),
        "overhead_sec": [0.02 if mode == "palf" else 0.0] * len(epochs),
        "n_samples_processed": [n_per_ep] * len(epochs),
        "elapsed": [2.0 * e for e in epochs],
    })
    hist.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    pats = ["pressure_drop", "humidity_net_rise", "temp_anomaly", "normal"]
    pat_rows = []
    for e in epochs:
        for p in pats:
            pat_rows.append({
                "epoch": e,
                "pattern": p,
                "n": 25,
                "n_seen": n_per_ep // 4,
                "train_loss_mean": float(val_losses[e - 1]),
                "train_loss_std": 0.1,
                "val_loss": float(val_losses[e - 1]),
                "grad_norm": 0.4,
            })
    pd.DataFrame(pat_rows).to_csv(os.path.join(run_dir, "epoch_pattern_log.csv"), index=False)

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
        "mae_degC": float(mae_degC),
        "rmse_degC": float(rmse_degC),
        "mae_norm": float(mae_degC / 8.0),
        "rmse_norm": float(rmse_degC / 8.0),
        "best_epoch": int(np.argmin(val_losses) + 1),
        "best_val_loss": float(min(val_losses)),
        "by_pattern": [
            {"pattern": "pressure_drop", "n": 10, "mae_degC": mae_degC + 0.3, "rmse_degC": rmse_degC + 0.3},
            {"pattern": "humidity_net_rise", "n": 10, "mae_degC": mae_degC + 0.2, "rmse_degC": rmse_degC + 0.2},
            {"pattern": "temp_anomaly", "n": 10, "mae_degC": mae_degC + 0.4, "rmse_degC": rmse_degC + 0.4},
            {"pattern": "normal", "n": 70, "mae_degC": mae_degC - 0.1, "rmse_degC": rmse_degC - 0.1},
        ],
        "by_temp_sign": [
            {"subpattern": "temp_anomaly_heat", "temp_sign": 1, "n": 6, "mae_degC": mae_degC + 0.35, "rmse_degC": rmse_degC + 0.35},
            {"subpattern": "temp_anomaly_cold", "temp_sign": -1, "n": 4, "mae_degC": mae_degC + 0.45, "rmse_degC": rmse_degC + 0.45},
        ],
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(tm, f)


def test_compare_runs_known_values_and_missing_configs(tmp_path):
    """Kiem tra compare_runs tren du lieu nho co ket qua biet truoc:
    - val_target = 1.02 * median(best_val cua A0)
    - so mau va thoi gian toi nguong val_target
    - paired differences giua A2r va A0r (cung nhu A0)
    - tach temp_anomaly_heat / temp_anomaly_cold
    - bao cao ro cac cau hinh con thieu (A1, A2)."""
    b_root = os.path.join(str(tmp_path), "baseline", "jena")
    p_root = os.path.join(str(tmp_path), "palf", "jena")
    out_dir = os.path.join(str(tmp_path), "out_comp")

    # 3 seed cho A0: best_val lan luot la 0.48, 0.50, 0.52 => median = 0.50 => val_target = 1.02 * 0.50 = 0.51
    _write_known_run(os.path.join(b_root, "gru", "seed0"), "baseline", "baseline", "gru", 0, (1.0, 0.70, 0.48), 100, 2.00, 2.50)
    _write_known_run(os.path.join(b_root, "gru", "seed1"), "baseline", "baseline", "gru", 1, (1.0, 0.70, 0.50), 100, 2.10, 2.60)
    _write_known_run(os.path.join(b_root, "gru", "seed2"), "baseline", "baseline", "gru", 2, (1.0, 0.70, 0.52), 100, 2.20, 2.70)

    # 3 seed cho A0r (uniform_b0.7)
    _write_known_run(os.path.join(p_root, "uniform_b0.7", "gru", "seed0"), "palf", "uniform_b0.7", "gru", 0, (1.0, 0.60, 0.50), 70, 2.05, 2.55)
    _write_known_run(os.path.join(p_root, "uniform_b0.7", "gru", "seed1"), "palf", "uniform_b0.7", "gru", 1, (1.0, 0.60, 0.50), 70, 2.15, 2.65)
    _write_known_run(os.path.join(p_root, "uniform_b0.7", "gru", "seed2"), "palf", "uniform_b0.7", "gru", 2, (1.0, 0.60, 0.50), 70, 2.25, 2.75)

    # 3 seed cho A2r (state_b0.7_a1.0): dat val_loss <= 0.51 ngay o epoch 2 (140 mau, 4.0s), MAE giam 0.15 so voi A0r
    _write_known_run(os.path.join(p_root, "state_b0.7_a1.0", "gru", "seed0"), "palf", "state_b0.7_a1.0", "gru", 0, (1.0, 0.50, 0.44), 70, 1.90, 2.40)
    _write_known_run(os.path.join(p_root, "state_b0.7_a1.0", "gru", "seed1"), "palf", "state_b0.7_a1.0", "gru", 1, (1.0, 0.50, 0.45), 70, 2.00, 2.50)
    _write_known_run(os.path.join(p_root, "state_b0.7_a1.0", "gru", "seed2"), "palf", "state_b0.7_a1.0", "gru", 2, (1.0, 0.50, 0.46), 70, 2.10, 2.60)

    res = run_comparison_protocol(
        baseline_root=b_root,
        palf_root=p_root,
        out_dir=out_dir,
        models=("gru",),
        seeds=(0, 1, 2),
    )

    assert res["val_targets"]["gru"] == pytest.approx(0.51)

    sum_df = res["summary_df"].set_index("code")
    assert sum_df.loc["A0", "mae_degC_mean"] == pytest.approx(2.10)
    assert sum_df.loc["A2r", "mae_degC_mean"] == pytest.approx(2.00)
    # A2r dat nguong o epoch 2 => 140 mau, 4.0s tren ca 3 seed
    assert sum_df.loc["A2r", "samples_to_target_mean"] == pytest.approx(140.0)
    assert sum_df.loc["A2r", "time_sec_to_target_mean"] == pytest.approx(4.0)
    # Chi co 3 seed (<5) nen stat_note phai ghi trend_only
    assert "trend_only" in sum_df.loc["A2r", "stat_note"]

    # Kiem tra paired difference giua A2r va A0r: moi seed A2r thap hon A0r dung 0.15 degC
    paired_df = res["paired_df"]
    m_pair = (paired_df["code"] == "A2r") & (paired_df["ref_code"] == "A0r") & (paired_df["metric"] == "mae_degC")
    assert m_pair.sum() == 1
    assert paired_df.loc[m_pair, "diff_mean"].iloc[0] == pytest.approx(-0.15)
    assert paired_df.loc[m_pair, "diff_std"].iloc[0] == pytest.approx(0.0, abs=1e-7)

    # Kiem tra temp_anomaly_heat va temp_anomaly_cold co trong bang pattern
    pat_sum = res["pat_summary_df"]
    assert "temp_anomaly_heat" in set(pat_sum["pattern"])
    assert "temp_anomaly_cold" in set(pat_sum["pattern"])

    # Kiem tra bao cao ro cau hinh con thieu (A1 va A2)
    miss_df = res["missing_df"]
    assert set(miss_df["code"]) == {"A1", "A2"}
    assert (miss_df["status"] == "MISSING_ALL").all()

    # Kiem tra cac file xuat ra
    for fn in (
        "runs_detail.csv",
        "summary_overall.csv",
        "summary_by_pattern.csv",
        "paired_differences.csv",
        "missing_configs.csv",
        "comparison_report.md",
        "val_loss_vs_samples_gru.png",
        "mae_by_pattern_gru.png",
    ):
        assert os.path.isfile(os.path.join(out_dir, fn)), f"Thieu file {fn}"
