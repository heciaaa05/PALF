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
    n_train=100,
    n_val=32,
    n_test=100,
    epochs=None,
    batch=128,
    lr=1e-3,
):
    os.makedirs(run_dir, exist_ok=True)
    ep_list = list(range(1, len(val_losses) + 1))
    hist = pd.DataFrame({
        "epoch": ep_list,
        "train_loss": [v * 1.05 for v in val_losses],
        "val_loss": list(val_losses),
        "grad_norm": [0.4] * len(ep_list),
        "epoch_time_sec": [2.0] * len(ep_list),
        "overhead_sec": [0.02 if mode == "palf" else 0.0] * len(ep_list),
        "n_samples_processed": [n_per_ep] * len(ep_list),
        "elapsed": [2.0 * e for e in ep_list],
    })
    hist.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    pats = ["pressure_drop", "humidity_net_rise", "temp_anomaly", "normal"]
    pat_rows = []
    for e in ep_list:
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

    ep_cfg = int(epochs) if epochs is not None else len(ep_list)
    rc = {
        "mode": mode,
        "config_tag": config_tag,
        "model": model,
        "model_name": model.upper(),
        "seed": seed,
        "epochs": ep_cfg,
        "epochs_run": ep_cfg,
        "batch": int(batch),
        "lr": float(lr),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
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
    """Kiem tra compare_runs tren du lieu nho co ket qua biet truoc (Spec v1 Muc 7)."""
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

    # 3 seed cho A3r (predictive_b0.7_a1.0): dat val_loss <= 0.51 ngay o epoch 2 (140 mau, 4.0s), MAE giam 0.15 so voi A0r
    _write_known_run(os.path.join(p_root, "predictive_b0.7_a1.0", "gru", "seed0"), "palf", "predictive_b0.7_a1.0", "gru", 0, (1.0, 0.50, 0.44), 70, 1.90, 2.40)
    _write_known_run(os.path.join(p_root, "predictive_b0.7_a1.0", "gru", "seed1"), "palf", "predictive_b0.7_a1.0", "gru", 1, (1.0, 0.50, 0.45), 70, 2.00, 2.50)
    _write_known_run(os.path.join(p_root, "predictive_b0.7_a1.0", "gru", "seed2"), "palf", "predictive_b0.7_a1.0", "gru", 2, (1.0, 0.50, 0.46), 70, 2.10, 2.60)

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
    assert sum_df.loc["A3r", "mae_degC_mean"] == pytest.approx(2.00)
    assert sum_df.loc["A3r", "samples_to_target_mean"] == pytest.approx(140.0)
    assert sum_df.loc["A3r", "time_sec_to_target_mean"] == pytest.approx(4.0)
    assert "trend_only" in sum_df.loc["A3r", "stat_note"]

    paired_df = res["paired_df"]
    m_pair = (paired_df["code"] == "A3r") & (paired_df["ref_code"] == "A0r") & (paired_df["metric"] == "mae_degC")
    assert m_pair.sum() == 1
    assert paired_df.loc[m_pair, "diff_mean"].iloc[0] == pytest.approx(-0.15)
    assert paired_df.loc[m_pair, "diff_std"].iloc[0] == pytest.approx(0.0, abs=1e-7)

    pat_sum = res["pat_summary_df"]
    assert "temp_anomaly_heat" in set(pat_sum["pattern"])
    assert "temp_anomaly_cold" in set(pat_sum["pattern"])

    miss_df = res["missing_df"]
    assert set(miss_df["code"]) == {"A0s", "A0w", "A1", "A3"}
    assert (miss_df["status"] == "MISSING_ALL").all()


@pytest.mark.parametrize(
    "bad_field,bad_kwargs,expected_msg",
    [
        ("n_train", {"n_train": 2048}, "n_train"),
        ("n_val", {"n_val": 512}, "n_val"),
        ("n_test", {"n_test": 512}, "n_test"),
        ("epochs", {"epochs": 10}, "epochs"),
        ("batch", {"batch": 64}, "batch"),
        ("lr", {"lr": 3e-4}, "lr"),
    ],
)
def test_compare_runs_rejects_mismatched_fields_and_seeds(tmp_path, bad_field, bad_kwargs, expected_msg):
    """(Prompt 7 Phan 4.4): Kiem tra compare_runs dung voi loi ValueError ro rang
    khi lech tung truong (n_train, n_val, n_test, epochs, batch, lr) hoac lech danh sach seeds."""
    b_root = os.path.join(str(tmp_path), "baseline", "jena")
    p_root = os.path.join(str(tmp_path), "palf", "jena")
    out_dir = os.path.join(str(tmp_path), "out_err")

    for s in (0, 1):
        _write_known_run(os.path.join(b_root, "gru", f"seed{s}"), "baseline", "baseline", "gru", s)
        # Seed 1 cua A0r bi lech truong bad_field
        kw = bad_kwargs if s == 1 else {}
        _write_known_run(os.path.join(p_root, "uniform_b0.7", "gru", f"seed{s}"), "palf", "uniform_b0.7", "gru", s, **kw)

    with pytest.raises(ValueError, match=expected_msg):
        run_comparison_protocol(
            baseline_root=b_root,
            palf_root=p_root,
            out_dir=out_dir,
            models=("gru",),
            seeds=(0, 1),
        )


def test_compare_runs_rejects_mismatched_seed_list(tmp_path):
    """Kiem tra compare_runs dung voi ValueError khi danh sach seed cua A0 va A0r khong khop nhau."""
    b_root = os.path.join(str(tmp_path), "baseline", "jena")
    p_root = os.path.join(str(tmp_path), "palf", "jena")
    out_dir = os.path.join(str(tmp_path), "out_seed_err")

    # A0 co seed 0, 1, 2 nhung A0r chi co seed 0, 1
    for s in (0, 1, 2):
        _write_known_run(os.path.join(b_root, "gru", f"seed{s}"), "baseline", "baseline", "gru", s)
    for s in (0, 1):
        _write_known_run(os.path.join(p_root, "uniform_b0.7", "gru", f"seed{s}"), "palf", "uniform_b0.7", "gru", s)

    with pytest.raises(ValueError, match="seeds"):
        run_comparison_protocol(
            baseline_root=b_root,
            palf_root=p_root,
            out_dir=out_dir,
            models=("gru",),
            seeds=(0, 1, 2),
        )


def test_prompt10_primary_comparisons_ttest_wilcoxon_bonferroni():
    import pandas as pd
    from palf.evaluation.compare_runs import (
        compute_prompt10_primary_comparisons,
        paired_ttest_two_sided,
        paired_wilcoxon_two_sided,
    )

    # Known paired differences
    diffs = np.array([-0.02, -0.03, -0.01, -0.025, -0.015, -0.022, -0.018, -0.024, -0.019, -0.021])
    t_s, t_p = paired_ttest_two_sided(diffs)
    w_s, w_p = paired_wilcoxon_two_sided(diffs)
    assert t_s < -8.0
    assert t_p < 1e-5
    assert w_s == 0.0
    assert w_p == pytest.approx(2.0 / 1024.0)  # Exact 2-sided Wilcoxon for n=10 with all same sign

    rows = []
    for s in range(10):
        rows.append({"model": "gru", "code": "A0s", "seed": s, "mae_degC": 1.80})
        rows.append({"model": "gru", "code": "A0w", "seed": s, "mae_degC": 1.79})
        rows.append({"model": "gru", "code": "A1", "seed": s, "mae_degC": 1.80 + diffs[s]})
        rows.append({"model": "gru", "code": "A3", "seed": s, "mae_degC": 1.80 + diffs[s] + 0.005})
    c_df = compute_prompt10_primary_comparisons(pd.DataFrame(rows)).set_index("comparison_id")
    assert set(c_df.index) == {"C1", "C2", "C3"}
    assert bool(c_df.loc["C1", "bonferroni_significant"]) is True
    assert bool(c_df.loc["C3", "bonferroni_significant"]) is True
    assert c_df.loc["C1", "p_bonferroni"] == pytest.approx(min(1.0, 3.0 * c_df.loc["C1", "t_pvalue"]))
