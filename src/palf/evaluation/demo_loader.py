"""Module doc va tong hop ket qua chay san tren dia cho giao dien Demo (Phan 3) va so sanh."""
import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_time_to_target(hist_df: pd.DataFrame, val_target: float) -> dict:
    """Tinh epoch, so mau tich luy va thoi gian tich luy lan dau dat val_loss <= val_target."""
    if hist_df is None or hist_df.empty or np.isnan(val_target):
        return {
            "reached": False,
            "epoch_to_target": None,
            "samples_to_target": np.nan,
            "time_sec_to_target": np.nan,
        }
    df = hist_df.sort_values("epoch").reset_index(drop=True)
    cum_samples = df["n_samples_processed"].cumsum()
    cum_time = df["epoch_time_sec"].cumsum()
    hits = df.index[df["val_loss"] <= float(val_target)]
    if len(hits) == 0:
        return {
            "reached": False,
            "epoch_to_target": None,
            "samples_to_target": np.nan,
            "time_sec_to_target": np.nan,
        }
    first_idx = int(hits[0])
    return {
        "reached": True,
        "epoch_to_target": int(df.loc[first_idx, "epoch"]),
        "samples_to_target": float(cum_samples.iloc[first_idx]),
        "time_sec_to_target": float(cum_time.iloc[first_idx]),
    }


def compute_val_auc_samples(hist_df: pd.DataFrame) -> float:
    """Dien tich duoi duong val_loss theo so mau tich luy (chuan hoa chia cho tong so mau da duyet,
    tuc trung binh tich phan hinh thang cua val_loss tren truc so mau)."""
    if hist_df is None or hist_df.empty:
        return float("nan")
    df = hist_df.sort_values("epoch").reset_index(drop=True)
    x = df["n_samples_processed"].cumsum().to_numpy(dtype=float)
    y = df["val_loss"].to_numpy(dtype=float)
    if len(x) < 2 or x[-1] <= x[0]:
        return float(y[0]) if len(y) == 1 else float("nan")
    area = float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))
    return float(area / (x[-1] - x[0]))


def load_saved_run(run_dir: str, label: Optional[str] = None) -> dict:
    """Doc 1 thu muc ket qua da chay san tren dia (baseline hoac PALF)."""
    hist_path = os.path.join(run_dir, "history.csv")
    pat_path = os.path.join(run_dir, "epoch_pattern_log.csv")
    cfg_path = os.path.join(run_dir, "run_config.json")
    tm_path = os.path.join(run_dir, "test_metrics.json")
    pol_path = os.path.join(run_dir, "policy_log.csv")

    if not os.path.isfile(hist_path) or not os.path.isfile(tm_path):
        raise FileNotFoundError(f"Thu muc '{run_dir}' thieu history.csv hoac test_metrics.json")

    hist_df = pd.read_csv(hist_path).sort_values("epoch").reset_index(drop=True)
    if "n_samples_processed" not in hist_df.columns:
        hist_df["n_samples_processed"] = 1
    if "epoch_time_sec" not in hist_df.columns:
        hist_df["epoch_time_sec"] = 0.0
    if "overhead_sec" not in hist_df.columns:
        hist_df["overhead_sec"] = 0.0
    hist_df["cum_samples"] = hist_df["n_samples_processed"].cumsum()
    hist_df["cum_time_sec"] = hist_df["epoch_time_sec"].cumsum()

    pat_df = pd.read_csv(pat_path) if os.path.isfile(pat_path) else pd.DataFrame()
    if not pat_df.empty:
        if "n_seen" not in pat_df.columns and "n" in pat_df.columns:
            pat_df["n_seen"] = pat_df["n"]
        if "n_seen" in pat_df.columns:
            ep_tot = pat_df.groupby("epoch")["n_seen"].transform("sum").replace(0, np.nan)
            pat_df["actual_share"] = pat_df["n_seen"] / ep_tot

    with open(tm_path, "r", encoding="utf-8") as f:
        test_metrics = json.load(f)

    run_config = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            run_config = json.load(f)

    has_palf = os.path.isfile(pol_path)
    policy_df = pd.read_csv(pol_path) if has_palf else pd.DataFrame()

    mode = run_config.get("mode", "palf" if has_palf else "baseline")
    config_tag = run_config.get("config_tag", "baseline" if mode == "baseline" else "palf")
    model = str(run_config.get("model", "unknown")).lower()
    seed = int(run_config.get("seed", 0))
    p_cfg = run_config.get("palf_config", {}) or {}
    pol_name = p_cfg.get("policy", "")

    display_tag = config_tag
    if pol_name == "predictive" or "predictive" in str(config_tag) or str(config_tag).startswith("e1_warmup1"):
        display_tag = f"{config_tag} [predictive - không đạt ngưỡng G3]"

    if label is None:
        if mode == "baseline":
            label = f"Baseline (A0) | {model.upper()} | seed{seed}"
        else:
            label = f"PALF ({display_tag}) | {model.upper()} | seed{seed}"

    return {
        "label": label,
        "run_dir": run_dir,
        "mode": mode,
        "config_tag": config_tag,
        "display_tag": display_tag,
        "model": model,
        "seed": seed,
        "has_palf": has_palf,
        "history": hist_df,
        "pattern_log": pat_df,
        "policy_log": policy_df,
        "run_config": run_config,
        "test_metrics": test_metrics,
    }


def discover_saved_runs(
    baseline_root: str = "experiments/results/baseline/jena",
    palf_root: str = "experiments/results/palf/jena",
) -> List[dict]:
    """Quet tat ca cac lan chay baseline va PALF hop le tren dia."""
    runs: List[dict] = []

    if os.path.isdir(baseline_root):
        for model_name in sorted(os.listdir(baseline_root)):
            m_dir = os.path.join(baseline_root, model_name)
            if not os.path.isdir(m_dir):
                continue
            for seed_name in sorted(os.listdir(m_dir)):
                s_dir = os.path.join(m_dir, seed_name)
                if os.path.isfile(os.path.join(s_dir, "history.csv")) and os.path.isfile(
                    os.path.join(s_dir, "test_metrics.json")
                ):
                    try:
                        runs.append(load_saved_run(s_dir))
                    except Exception:
                        pass

    if os.path.isdir(palf_root):
        for tag_name in sorted(os.listdir(palf_root)):
            t_dir = os.path.join(palf_root, tag_name)
            if not os.path.isdir(t_dir):
                continue
            for model_name in sorted(os.listdir(t_dir)):
                m_dir = os.path.join(t_dir, model_name)
                if not os.path.isdir(m_dir):
                    continue
                for seed_name in sorted(os.listdir(m_dir)):
                    s_dir = os.path.join(m_dir, seed_name)
                    if os.path.isfile(os.path.join(s_dir, "history.csv")) and os.path.isfile(
                        os.path.join(s_dir, "test_metrics.json")
                    ):
                        try:
                            r = load_saved_run(s_dir)
                            if r["config_tag"] in ("palf", "baseline"):
                                r["config_tag"] = tag_name
                                r["label"] = f"PALF ({tag_name}) | {r['model'].upper()} | seed{r['seed']}"
                            runs.append(r)
                        except Exception:
                            pass

    return runs


def summarize_side_by_side(
    selected_runs: List[dict], val_target: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Tao bang so sanh tong the va bang MAE/RMSE (°C) theo Pattern cho 2-3 lan chay.
    Neu val_target=None, mac dinh lay 1.02 * best_val_loss cua lan chay dau tien."""
    if not selected_runs:
        return pd.DataFrame(), pd.DataFrame(), float("nan")

    if val_target is None:
        base_candidates = [r for r in selected_runs if r["mode"] == "baseline"]
        ref_run = base_candidates[0] if base_candidates else selected_runs[0]
        best_v = float(ref_run["history"]["val_loss"].min())
        val_target = 1.02 * best_v

    overall_rows = []
    pattern_rows = []

    for r in selected_runs:
        hist = r["history"]
        tm = r["test_metrics"]
        ttt = compute_time_to_target(hist, val_target)
        auc_s = compute_val_auc_samples(hist)
        tot_samples = int(hist["n_samples_processed"].sum())
        tot_time = float(hist["epoch_time_sec"].sum())
        tot_ov = float(hist["overhead_sec"].sum())

        overall_rows.append({
            "Run": r["label"],
            "Mode": r["mode"],
            "Config": r.get("display_tag", r["config_tag"]),
            "Model": r["model"].upper(),
            "Seed": int(r["seed"]),
            "Test MAE (°C)": float(tm.get("mae_degC", np.nan)),
            "Test RMSE (°C)": float(tm.get("rmse_degC", np.nan)),
            "Best Val Loss": float(hist["val_loss"].min()),
            "Best Epoch": int(tm.get("best_epoch", hist.loc[hist["val_loss"].idxmin(), "epoch"])),
            "Epoch tới ngưỡng": ttt["epoch_to_target"],
            "Số mẫu tới ngưỡng": ttt["samples_to_target"],
            "AUC Val (theo mẫu)": auc_s,
            "Tổng mẫu đã duyệt": tot_samples,
        })

        for bp in tm.get("by_pattern", []):
            pattern_rows.append({
                "Run": r["label"],
                "Pattern": bp["pattern"],
                "Số mẫu test": int(bp.get("n", 0)),
                "MAE (°C)": float(bp.get("mae_degC", np.nan)),
                "RMSE (°C)": float(bp.get("rmse_degC", np.nan)),
            })

    return pd.DataFrame(overall_rows), pd.DataFrame(pattern_rows), float(val_target)


def summarize_palf_diagnostics(run_dict: Optional[dict]) -> dict:
    """Trich xuat cac bang pivot cho trang Quan sat PALF:
    - weights_pivot: epoch x pattern (tu policy_log.csv)
    - share_pivot: epoch x pattern (ty le mau thuc su n_seen tu epoch_pattern_log.csv)
    - state_pivots: dict[var_name -> DataFrame(epoch x pattern)] (tu policy_log.csv)
    Neu chua co ket qua PALF, tra ve has_palf=False va message ro rang."""
    if run_dict is None or not run_dict.get("has_palf", False) or run_dict.get("policy_log", pd.DataFrame()).empty:
        return {
            "has_palf": False,
            "message": "Chưa có kết quả PALF trên đĩa (hoặc lần chạy được chọn là baseline).",
            "weights_pivot": pd.DataFrame(),
            "share_pivot": pd.DataFrame(),
            "state_pivots": {},
        }

    pol_df = run_dict["policy_log"]
    pat_df = run_dict["pattern_log"]

    weights_pivot = pol_df.pivot(index="epoch", columns="pattern", values="weight")

    if "quota_planned" in pol_df.columns and (pol_df["quota_planned"] > 0).any():
        q_tot = pol_df.groupby("epoch")["quota_planned"].transform("sum").replace(0, np.nan)
        tmp_q = pol_df.copy()
        tmp_q["quota_share"] = tmp_q["quota_planned"] / q_tot
        quota_share_pivot = tmp_q.pivot(index="epoch", columns="pattern", values="quota_share")
    else:
        quota_share_pivot = pd.DataFrame()

    if not pat_df.empty and "actual_share" in pat_df.columns:
        share_pivot = pat_df.pivot(index="epoch", columns="pattern", values="actual_share")
    elif "n_seen" in pol_df.columns:
        tot = pol_df.groupby("epoch")["n_seen"].transform("sum").replace(0, np.nan)
        tmp = pol_df.copy()
        tmp["actual_share"] = tmp["n_seen"] / tot
        share_pivot = tmp.pivot(index="epoch", columns="pattern", values="actual_share")
    else:
        share_pivot = pd.DataFrame()

    state_pivots = {}
    for var_col in ("loss_ratio", "velocity", "instability", "y_hat", "need_score"):
        if var_col in pol_df.columns and pol_df[var_col].notna().any():
            state_pivots[var_col] = pol_df.pivot(index="epoch", columns="pattern", values=var_col)

    return {
        "has_palf": True,
        "message": "OK",
        "weights_pivot": weights_pivot,
        "quota_share_pivot": quota_share_pivot,
        "share_pivot": share_pivot,
        "state_pivots": state_pivots,
    }


def summarize_pattern_mae_across_seeds(saved_runs: List[dict], model_filter: Optional[str] = None) -> pd.DataFrame:
    """Tong hop bang MAE va RMSE (°C) theo Pattern kem do lech chuan giua cac seed (Mean ± Std)
    cho tung (Model, Config) — phuc vu Man hinh 3 trong Prompt 7 Phan 3.1."""
    rows = []
    for r in saved_runs:
        if model_filter and model_filter != "Tất cả" and r["model"].upper() != model_filter.upper():
            continue
        tm = r.get("test_metrics", {})
        for bp in tm.get("by_pattern", []):
            rows.append({
                "Model": r["model"].upper(),
                "Config": r["config_tag"],
                "Seed": int(r["seed"]),
                "Pattern": bp["pattern"],
                "n_test": int(bp.get("n", 0)),
                "mae_degC": float(bp.get("mae_degC", np.nan)),
                "rmse_degC": float(bp.get("rmse_degC", np.nan)),
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out_rows = []
    for (m, cfg, pat), grp in df.groupby(["Model", "Config", "Pattern"], sort=False):
        n_s = len(grp)
        m_mean = float(grp["mae_degC"].mean())
        m_std = float(grp["mae_degC"].std(ddof=1)) if n_s > 1 else 0.0
        r_mean = float(grp["rmse_degC"].mean())
        r_std = float(grp["rmse_degC"].std(ddof=1)) if n_s > 1 else 0.0
        out_rows.append({
            "Backbone": m,
            "Cấu hình": cfg,
            "Pattern": pat,
            "Số seed": n_s,
            "Số mẫu test": int(grp["n_test"].iloc[0]),
            "MAE (°C) Mean": round(m_mean, 4),
            "MAE (°C) Std": round(m_std, 4),
            "MAE (°C) (Mean ± Std)": f"{m_mean:.4f} ± {m_std:.4f}",
            "RMSE (°C) (Mean ± Std)": f"{r_mean:.4f} ± {r_std:.4f}",
        })
    return pd.DataFrame(out_rows)
