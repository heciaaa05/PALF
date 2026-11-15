"""Giao thuc so sanh cong bang Baseline vs PALF (Spec Phase 5-6 v1 Muc 7, Prompt 7 Phan 4.4-4.5)."""
import math
import os
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .demo_loader import (
    compute_time_to_target,
    compute_val_auc_samples,
    load_saved_run,
)

# Danh sach cau hinh chuan theo Spec v1 Muc 7, Prompt 7 Phan 5.2, Prompt 8 & Prompt 9
# Luu y (Prompt 8 & 9): A3 va A3r phai ghi ro 'predictive (khong dat nguong G3)'; A0s la doi chung chinh cua A1 va A3.
STANDARD_CONFIGS = [
    {"code": "A0", "mode": "baseline", "policy": "uniform", "config_tag": "baseline", "budget_ratio": 1.0, "ref_code": "A0", "desc": "Baseline uniform (DataLoader chuan, budget 1.0)"},
    {"code": "A0s", "mode": "palf", "policy": "uniform", "config_tag": "uniform_b1.0", "budget_ratio": 1.0, "ref_code": "A0", "desc": "Baseline uniform qua Scheduler (budget 1.0)"},
    {"code": "A0w", "mode": "palf", "policy": "loss_shuffled", "config_tag": "loss_shuffled_b1.0_a1.0", "budget_ratio": 1.0, "ref_code": "A0s", "desc": "Doi chung A0w trong so xao tron (budget 1.0)"},
    {"code": "A0r", "mode": "palf", "policy": "uniform", "config_tag": "uniform_b0.7", "budget_ratio": 0.7, "ref_code": "A0s", "desc": "Baseline uniform qua Scheduler (budget 0.7)"},
    {"code": "A1", "mode": "palf", "policy": "loss", "config_tag": "loss_b1.0_a1.0", "budget_ratio": 1.0, "ref_code": "A0s", "desc": "Doi chung B2 chi dung loss_ratio (budget 1.0)"},
    {"code": "A3", "mode": "palf", "policy": "predictive", "config_tag": "predictive_b1.0_a0.5_w2", "budget_ratio": 1.0, "ref_code": "A0s", "desc": "predictive (khong dat nguong G3, budget 1.0)"},
    {"code": "A3r", "mode": "palf", "policy": "predictive", "config_tag": "predictive_b0.7_a0.5_w2", "budget_ratio": 0.7, "ref_code": "A0r", "desc": "predictive (khong dat nguong G3, budget 0.7)"},
]

EXPLORATORY_CONFIGS = [
    {"code": "E1", "mode": "palf", "policy": "predictive", "config_tag": "e1_warmup1_b1.0_a0.5_w2", "budget_ratio": 1.0, "ref_code": "A0s", "desc": "[Kham pha E1] predictive (khong dat nguong G3, warmup=1)"},
    {"code": "E2", "mode": "palf", "policy": "loss_abs", "config_tag": "loss_abs_b1.0_a0.5_w2", "budget_ratio": 1.0, "ref_code": "A0s", "desc": "[Kham pha E2] loss_abs (loss tuyet doi theo Pattern, warmup=3)"},
]


def resolve_config_run_dir(
    baseline_root: str,
    palf_root: str,
    cfg_spec: dict,
    model: str,
    seed: int,
) -> Tuple[str, str]:
    """Tra ve (run_dir, matched_tag) cho (cfg_spec, model, seed).
    Uu tien tim dung cfg_spec['config_tag']; neu khong co tren dia va cfg_spec co 'policy' + 'budget_ratio',
    tim thu muc bat dau bang '<policy>_b<budget_ratio:.1f>' (ho tro tham so alpha, w_max da tinh chinh o Buoc 5.1)."""
    m_low = model.lower()
    if cfg_spec["mode"] == "baseline":
        return os.path.join(baseline_root, m_low, f"seed{seed}"), "baseline"

    exact_tag = cfg_spec["config_tag"]
    exact_dir = os.path.join(palf_root, exact_tag, m_low, f"seed{seed}")
    if os.path.isfile(os.path.join(exact_dir, "history.csv")):
        return exact_dir, exact_tag

    pol = cfg_spec.get("policy")
    br = cfg_spec.get("budget_ratio")
    if pol and br is not None and os.path.isdir(palf_root):
        prefix = f"{pol}_b{float(br):.1f}"
        for cand_tag in sorted(os.listdir(palf_root)):
            if cand_tag == exact_tag or cand_tag.startswith(prefix):
                cand_dir = os.path.join(palf_root, cand_tag, m_low, f"seed{seed}")
                if os.path.isfile(os.path.join(cand_dir, "history.csv")):
                    return cand_dir, cand_tag

    return exact_dir, exact_tag


def validate_protocol_consistency(found_runs: List[dict], check_seeds: bool = True) -> None:
    """Kiem tra nghiem ngat tinh nhat quan cua run_config.json giua moi lan chay dua vao bang (Prompt 7 Phan 4.4).
    Neu n_train, n_val, n_test, epochs, batch (batch_size), lr hoac danh sach seed lech nhau
    -> raise ValueError neu ro cap nao va truong nao lech (khong chi canh bao)."""
    if not found_runs:
        return

    fields_to_check = ("n_train", "n_val", "n_test", "epochs", "batch", "lr")

    # Nhom theo backbone de so sanh cac cau hinh tren cung backbone (va giua cac backbone cho cung tap du lieu)
    ref_run = found_runs[0]
    ref_rc = ref_run.get("run_config", {})
    ref_id = f"{ref_run['model'].upper()}/{ref_run['code']}/seed{ref_run['seed']}"

    for r in found_runs[1:]:
        rc = r.get("run_config", {})
        cur_id = f"{r['model'].upper()}/{r['code']}/seed{r['seed']}"

        for fld in fields_to_check:
            v_ref = ref_rc.get(fld)
            v_cur = rc.get(fld)
            # Ho tro alias batch_size neu file cu ghi batch_size
            if fld == "batch":
                if v_ref is None:
                    v_ref = ref_rc.get("batch_size")
                if v_cur is None:
                    v_cur = rc.get("batch_size")
            if v_ref is None or v_cur is None:
                raise ValueError(
                    f"Thieu truong '{fld}' trong run_config.json khi so sanh ({ref_id} vs {cur_id})."
                )
            if isinstance(v_ref, float) or isinstance(v_cur, float):
                mismatch = not np.isclose(float(v_ref), float(v_cur), rtol=1e-6, atol=1e-9)
            else:
                mismatch = int(v_ref) != int(v_cur)
            if mismatch:
                raise ValueError(
                    f"Loi lech cau hinh '{fld}': {ref_id} ({fld}={v_ref}) != {cur_id} ({fld}={v_cur}). "
                    f"Khong duoc tron lan chay thu (smoke) voi lan chay day du trong bang so sanh!"
                )

    if check_seeds:
        for m in sorted({r["model"] for r in found_runs}):
            m_runs = [r for r in found_runs if r["model"] == m]
            seeds_by_code: Dict[str, List[int]] = {}
            for r in m_runs:
                seeds_by_code.setdefault(r["code"], []).append(int(r["seed"]))
            codes = sorted(seeds_by_code.keys())
            if len(codes) >= 2:
                base_c = "A0" if "A0" in seeds_by_code else codes[0]
                base_seeds = sorted(seeds_by_code[base_c])
                for c in codes:
                    c_seeds = sorted(seeds_by_code[c])
                    if c_seeds != base_seeds:
                        raise ValueError(
                            f"Loi lech danh sach 'seeds' tren backbone '{m.upper()}': "
                            f"{base_c} (seeds={base_seeds}) != {c} (seeds={c_seeds})."
                        )


def collect_protocol_runs(
    baseline_root: str = "experiments/results/baseline/jena",
    palf_root: str = "experiments/results/palf/jena",
    models: Tuple[str, ...] = ("gru", "tcn", "transformer", "patchtst"),
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    config_specs: Optional[List[dict]] = None,
    strict_validation: bool = True,
) -> Tuple[List[dict], pd.DataFrame]:
    """Doc tat ca cac lan chay theo danh sach cau hinh.
    Tra ve (found_runs, missing_df) — khong im lang bo qua cau hinh thieu,
    va kiem tra nghiem ngat tinh dong nhat cua run_config.json."""
    specs = config_specs if config_specs is not None else STANDARD_CONFIGS
    found_runs: List[dict] = []
    missing_rows: List[dict] = []

    for m in models:
        m_low = m.lower()
        for cfg in specs:
            found_seeds = []
            missing_seeds = []
            matched_tag_used = cfg["config_tag"]
            for s in seeds:
                r_dir, matched_tag = resolve_config_run_dir(baseline_root, palf_root, cfg, m_low, s)
                if os.path.isfile(os.path.join(r_dir, "history.csv")) and os.path.isfile(
                    os.path.join(r_dir, "test_metrics.json")
                ):
                    matched_tag_used = matched_tag
                    r = load_saved_run(
                        r_dir, label=f"{cfg['code']} ({matched_tag}) | {m_low.upper()} | seed{s}"
                    )
                    r["code"] = cfg["code"]
                    r["config_tag"] = matched_tag
                    r["ref_code"] = cfg["ref_code"]
                    r["budget_ratio"] = cfg["budget_ratio"]
                    r["config_desc"] = cfg["desc"]
                    r["model"] = m_low
                    r["seed"] = int(s)
                    found_runs.append(r)
                    found_seeds.append(int(s))
                else:
                    missing_seeds.append(int(s))

            if missing_seeds:
                missing_rows.append({
                    "model": m_low,
                    "code": cfg["code"],
                    "config_tag": matched_tag_used,
                    "description": cfg["desc"],
                    "n_found": len(found_seeds),
                    "found_seeds": ",".join(str(x) for x in found_seeds) if found_seeds else "none",
                    "n_missing": len(missing_seeds),
                    "missing_seeds": ",".join(str(x) for x in missing_seeds),
                    "status": "PARTIAL" if found_seeds else "MISSING_ALL",
                })

    if strict_validation and found_runs:
        validate_protocol_consistency(found_runs, check_seeds=True)

    return found_runs, pd.DataFrame(missing_rows)


def compute_val_targets_from_a0(found_runs: List[dict]) -> Dict[str, float]:
    """Tinh val_target = 1.02 * median(best_val_loss cua A0 theo seed) cho tung backbone."""
    targets: Dict[str, float] = {}
    models = sorted({r["model"] for r in found_runs})
    for m in models:
        a0_runs = [r for r in found_runs if r["model"] == m and r["code"] == "A0"]
        if not a0_runs:
            a0_runs = [r for r in found_runs if r["model"] == m]
        best_vals = [float(r["history"]["val_loss"].min()) for r in a0_runs]
        targets[m] = float(1.02 * np.median(best_vals)) if best_vals else float("nan")
    return targets


def build_detail_and_pattern_tables(
    found_runs: List[dict], val_targets: Dict[str, float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tao bang chi tiet tung (model, code, seed) va bang theo Pattern (bao gom temp_anomaly_heat/cold)."""
    detail_rows = []
    pattern_rows = []

    for r in found_runs:
        m = r["model"]
        v_target = val_targets.get(m, np.nan)
        hist = r["history"]
        tm = r["test_metrics"]

        ttt = compute_time_to_target(hist, v_target)
        auc_s = compute_val_auc_samples(hist)
        best_val = float(hist["val_loss"].min())
        best_ep = int(tm.get("best_epoch", hist.loc[hist["val_loss"].idxmin(), "epoch"]))
        tot_samples = int(hist["n_samples_processed"].sum())
        tot_time = float(hist["epoch_time_sec"].sum())
        mean_ep_time = float(hist["epoch_time_sec"].mean())
        tot_ov = float(hist["overhead_sec"].sum())
        mean_ov = float(hist["overhead_sec"].mean())

        detail_rows.append({
            "model": m,
            "code": r["code"],
            "config_tag": r["config_tag"],
            "ref_code": r["ref_code"],
            "seed": int(r["seed"]),
            "epochs_run": int(len(hist)),
            "mae_degC": float(tm.get("mae_degC", np.nan)),
            "rmse_degC": float(tm.get("rmse_degC", np.nan)),
            "mae_norm": float(tm.get("mae_norm", np.nan)),
            "rmse_norm": float(tm.get("rmse_norm", np.nan)),
            "best_val_loss": best_val,
            "best_epoch": best_ep,
            "val_target": float(v_target),
            "reached_target": bool(ttt["reached"]),
            "epoch_to_target": ttt["epoch_to_target"],
            "samples_to_target": float(ttt["samples_to_target"]),
            "time_sec_to_target": float(ttt["time_sec_to_target"]),
            "val_auc_samples": float(auc_s),
            "total_samples": tot_samples,
            "total_time_sec": tot_time,
            "mean_epoch_time_sec": mean_ep_time,
            "total_overhead_sec": tot_ov,
            "mean_overhead_sec": mean_ov,
        })

        by_pat = tm.get("by_pattern", [])
        norm_mae = np.nan
        norm_rmse = np.nan
        for bp in by_pat:
            if bp["pattern"] == "normal":
                norm_mae = float(bp.get("mae_degC", np.nan))
                norm_rmse = float(bp.get("rmse_degC", np.nan))

        for bp in by_pat:
            p_mae = float(bp.get("mae_degC", np.nan))
            p_rmse = float(bp.get("rmse_degC", np.nan))
            pattern_rows.append({
                "model": m,
                "code": r["code"],
                "config_tag": r["config_tag"],
                "seed": int(r["seed"]),
                "pattern": bp["pattern"],
                "group_type": "main_pattern",
                "n_test": int(bp.get("n", 0)),
                "mae_degC": p_mae,
                "rmse_degC": p_rmse,
                "gap_mae_vs_normal_degC": float(p_mae - norm_mae) if np.isfinite(norm_mae) else np.nan,
                "gap_rmse_vs_normal_degC": float(p_rmse - norm_rmse) if np.isfinite(norm_rmse) else np.nan,
            })

        for sp in tm.get("by_temp_sign", []):
            s_mae = float(sp.get("mae_degC", np.nan))
            s_rmse = float(sp.get("rmse_degC", np.nan))
            pattern_rows.append({
                "model": m,
                "code": r["code"],
                "config_tag": r["config_tag"],
                "seed": int(r["seed"]),
                "pattern": sp["subpattern"],
                "group_type": "temp_sign_split",
                "n_test": int(sp.get("n", 0)),
                "mae_degC": s_mae,
                "rmse_degC": s_rmse,
                "gap_mae_vs_normal_degC": float(s_mae - norm_mae) if np.isfinite(norm_mae) else np.nan,
                "gap_rmse_vs_normal_degC": float(s_rmse - norm_rmse) if np.isfinite(norm_rmse) else np.nan,
            })

    return pd.DataFrame(detail_rows), pd.DataFrame(pattern_rows)


def summarize_overall_and_paired(detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tong hop mean ± std theo (model, code) va hieu so ghep theo seed (paired difference) so voi ref_code (A0 hoac A0r)."""
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics = [
        "mae_degC",
        "rmse_degC",
        "best_val_loss",
        "best_epoch",
        "samples_to_target",
        "time_sec_to_target",
        "val_auc_samples",
        "mean_epoch_time_sec",
        "mean_overhead_sec",
    ]
    summary_rows = []
    for (m, code), grp in detail_df.groupby(["model", "code"], sort=False):
        n_seeds = len(grp)
        row = {
            "model": m,
            "code": code,
            "config_tag": grp["config_tag"].iloc[0],
            "ref_code": grp["ref_code"].iloc[0],
            "n_seeds": int(n_seeds),
            "seeds": ",".join(str(s) for s in sorted(grp["seed"].tolist())),
            "n_reached_target": int(grp["reached_target"].sum()),
            "stat_note": "trend_only (<5 seeds)" if n_seeds < 5 else f"{n_seeds}_seeds_complete",
        }
        for col in metrics:
            vals = grp[col].dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    paired_rows = []
    for m, m_grp in detail_df.groupby("model", sort=False):
        by_code = {c: g.set_index("seed") for c, g in m_grp.groupby("code", sort=False)}
        for code, g_cur in by_code.items():
            if code == "A0":
                continue
            ref_candidates = [g_cur["ref_code"].iloc[0]]
            if code in ("A1", "A3", "A2", "A0r", "E1", "E2") and "A0s" in by_code and "A0s" not in ref_candidates:
                ref_candidates.insert(0, "A0s")
            if "A0" not in ref_candidates:
                ref_candidates.append("A0")
            if code in ("A3", "A2", "E1", "E2") and "A1" in by_code and "A1" not in ref_candidates:
                ref_candidates.append("A1")
            if code == "A1" and "A0w" in by_code and "A0w" not in ref_candidates:
                ref_candidates.append("A0w")
            if code in ("E1", "E2") and "A3" in by_code and "A3" not in ref_candidates:
                ref_candidates.insert(0, "A3")
            for ref_c in ref_candidates:
                if ref_c not in by_code:
                    continue
                g_ref = by_code[ref_c]
                common_seeds = sorted(set(g_cur.index).intersection(set(g_ref.index)))
                if not common_seeds:
                    continue
                for col in ("mae_degC", "rmse_degC", "best_val_loss", "samples_to_target", "time_sec_to_target", "val_auc_samples"):
                    diffs = (g_cur.loc[common_seeds, col] - g_ref.loc[common_seeds, col]).dropna()
                    if len(diffs) == 0:
                        continue
                    paired_rows.append({
                        "model": m,
                        "code": code,
                        "ref_code": ref_c,
                        "metric": col,
                        "n_paired_seeds": int(len(diffs)),
                        "paired_seeds": ",".join(str(s) for s in common_seeds),
                        "diff_mean": float(diffs.mean()),
                        "diff_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                        "per_seed_diffs": "; ".join(f"s{s}:{float(g_cur.loc[s, col] - g_ref.loc[s, col]):+.4f}" for s in common_seeds if pd.notna(g_cur.loc[s, col]) and pd.notna(g_ref.loc[s, col])),
                        "stat_note": "Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed)" if len(diffs) < 5 else f"Du {len(diffs)} seed ghep cap",
                    })

    return summary_df, pd.DataFrame(paired_rows)


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3.0e-14
    fpmin = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest_two_sided(diffs: np.ndarray) -> Tuple[float, float]:
    """Kiem dinh t ghep cap hai phia tren mang hieu so d = x - y."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))
    if std_d < 1e-15:
        if abs(mean_d) < 1e-15:
            return 0.0, 1.0
        return float("inf") if mean_d > 0 else float("-inf"), 0.0
    t_stat = mean_d / (std_d / math.sqrt(n))
    df = n - 1
    x = df / (df + t_stat * t_stat)
    p_val = float(_betainc(0.5 * df, 0.5, x))
    return float(t_stat), min(1.0, max(0.0, p_val))


def paired_wilcoxon_two_sided(diffs: np.ndarray) -> Tuple[float, float]:
    """Kiem dinh Wilcoxon signed-rank hai phia (exact voi n_nz <= 16, xap xi chuan khi lon hon)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    nz = d[np.abs(d) > 1e-15]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    abs_nz = np.abs(nz)
    ranks = pd.Series(abs_nz).rank(method="average").to_numpy(dtype=float)
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    w_stat = min(w_plus, w_minus)

    if n <= 16:
        # Exact distribution over all 2^n sign assignments (sum of subset ranks)
        sums = np.array([0.0], dtype=float)
        for r in ranks:
            sums = np.concatenate([sums, sums + r])
        total_r = float(ranks.sum())
        w_obs_min = min(w_plus, total_r - w_plus)
        # Two-sided p-value: P(min(W+, W-) <= w_obs_min)
        mins = np.minimum(sums, total_r - sums)
        p_val = float(np.mean(mins <= w_obs_min + 1e-12))
        return float(w_stat), min(1.0, max(0.0, p_val))

    mean_w = n * (n + 1) / 4.0
    _, counts = np.unique(abs_nz, return_counts=True)
    tie_corr = float(np.sum(counts ** 3 - counts))
    var_w = (n * (n + 1) * (2 * n + 1) - 0.5 * tie_corr) / 24.0
    if var_w <= 0:
        return float(w_stat), 1.0
    z = (abs(w_plus - mean_w) - 0.5) / math.sqrt(var_w)
    p_val = math.erfc(max(0.0, z) / math.sqrt(2.0))
    return float(w_stat), min(1.0, max(0.0, p_val))


def compute_prompt10_primary_comparisons(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Tinh 3 doi chieu chinh C1 (A1-A0s), C2 (A3-A1), C3 (A1-A0w) voi Bonferroni (alpha=0.05/3=0.0167)."""
    if detail_df.empty:
        return pd.DataFrame()
    specs = [
        ("C1", "A1", "A0s", "Co cau theo loss (A1) vs lay deu qua Scheduler (A0s)"),
        ("C2", "A3", "A1", "Learning State (A3) vs loss don thuan (A1)"),
        ("C3", "A1", "A0w", "Thong tin Pattern (A1) vs trong so xao tron cung phan bo (A0w)"),
    ]
    rows = []
    alpha_bonf = 0.05 / 3.0
    for m, m_grp in detail_df.groupby("model", sort=False):
        by_code = {c: g.set_index("seed") for c, g in m_grp.groupby("code", sort=False)}
        for cid, cur_c, ref_c, desc in specs:
            if cur_c not in by_code or ref_c not in by_code:
                continue
            g_cur = by_code[cur_c]
            g_ref = by_code[ref_c]
            common = sorted(set(g_cur.index).intersection(set(g_ref.index)))
            if not common:
                continue
            diffs = (g_cur.loc[common, "mae_degC"] - g_ref.loc[common, "mae_degC"]).to_numpy(dtype=float)
            valid_mask = np.isfinite(diffs)
            diffs = diffs[valid_mask]
            common_v = [s for s, ok in zip(common, valid_mask) if ok]
            n = len(diffs)
            if n == 0:
                continue
            t_stat, t_p = paired_ttest_two_sided(diffs)
            w_stat, w_p = paired_wilcoxon_two_sided(diffs)
            p_bonf = min(1.0, t_p * 3.0) if np.isfinite(t_p) else float("nan")
            n_neg = int(np.sum(diffs < -1e-12))
            n_pos = int(np.sum(diffs > 1e-12))
            rows.append({
                "model": m,
                "comparison_id": cid,
                "code": cur_c,
                "ref_code": ref_c,
                "description": desc,
                "n_seeds": n,
                "seeds": ",".join(str(s) for s in common_v),
                "diff_mean": float(np.mean(diffs)),
                "diff_std": float(np.std(diffs, ddof=1)) if n > 1 else 0.0,
                "n_neg": n_neg,
                "n_pos": n_pos,
                "t_stat": float(t_stat),
                "t_pvalue": float(t_p),
                "p_bonferroni": float(p_bonf),
                "bonferroni_significant": bool(np.isfinite(t_p) and t_p < alpha_bonf),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_pvalue": float(w_p),
                "per_seed_diffs": "; ".join(f"s{s}:{d:+.4f}" for s, d in zip(common_v, diffs)),
            })
    return pd.DataFrame(rows)


def summarize_by_pattern(pattern_df: pd.DataFrame) -> pd.DataFrame:
    if pattern_df.empty:
        return pd.DataFrame()
    rows = []
    for (m, code, pat, gtype), grp in pattern_df.groupby(["model", "code", "pattern", "group_type"], sort=False):
        rows.append({
            "model": m,
            "code": code,
            "config_tag": grp["config_tag"].iloc[0],
            "pattern": pat,
            "group_type": gtype,
            "n_seeds": int(len(grp)),
            "n_test": int(grp["n_test"].iloc[0]),
            "mae_degC_mean": float(grp["mae_degC"].mean()),
            "mae_degC_std": float(grp["mae_degC"].std(ddof=1)) if len(grp) > 1 else 0.0,
            "rmse_degC_mean": float(grp["rmse_degC"].mean()),
            "rmse_degC_std": float(grp["rmse_degC"].std(ddof=1)) if len(grp) > 1 else 0.0,
            "gap_mae_vs_normal_mean": float(grp["gap_mae_vs_normal_degC"].mean()),
        })
    return pd.DataFrame(rows)


def plot_comparison_figures(
    found_runs: List[dict],
    pat_summary_df: pd.DataFrame,
    val_targets: Dict[str, float],
    out_dir: str,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved_figs: List[str] = []
    models = sorted({r["model"] for r in found_runs})

    for m in models:
        m_runs = [r for r in found_runs if r["model"] == m]
        if not m_runs:
            continue

        fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
        by_code: Dict[str, List[dict]] = {}
        for r in m_runs:
            by_code.setdefault(r["code"], []).append(r)

        for code, r_list in by_code.items():
            tag = r_list[0]["config_tag"]
            all_hist = pd.concat([r["history"][["epoch", "cum_samples", "val_loss"]] for r in r_list], ignore_index=True)
            grp = all_hist.groupby("epoch", as_index=False).agg(
                cum_samples=("cum_samples", "mean"),
                val_mean=("val_loss", "mean"),
                val_std=("val_loss", "std"),
            )
            ax1.plot(grp["cum_samples"], grp["val_mean"], marker="o", markersize=3.5, label=f"{code} ({tag}, n={len(r_list)})")
            if len(r_list) > 1 and grp["val_std"].notna().any():
                std_v = grp["val_std"].fillna(0.0)
                ax1.fill_between(grp["cum_samples"], grp["val_mean"] - std_v, grp["val_mean"] + std_v, alpha=0.15)

        vt = val_targets.get(m, np.nan)
        if np.isfinite(vt):
            ax1.axhline(vt, color="red", linestyle="--", linewidth=1.0, label=f"val_target = {vt:.4f}")
        ax1.set_title(f"{m.upper()} — Val Loss theo Số mẫu train tích lũy")
        ax1.set_xlabel("Số mẫu train tích lũy (cumulative samples)")
        ax1.set_ylabel("Validation MSE (chuẩn hóa)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"val_loss_vs_samples_{m}.png")
        fig1.savefig(p1, dpi=150)
        plt.close(fig1)
        saved_figs.append(p1)

        sub_pat = pat_summary_df[(pat_summary_df["model"] == m) & (pat_summary_df["group_type"] == "main_pattern")]
        if not sub_pat.empty:
            fig2, ax2 = plt.subplots(figsize=(8.0, 4.5))
            piv_m = sub_pat.pivot(index="pattern", columns="code", values="mae_degC_mean")
            piv_s = sub_pat.pivot(index="pattern", columns="code", values="mae_degC_std").fillna(0.0)
            piv_m.plot(kind="bar", yerr=piv_s, ax=ax2, capsize=3, rot=15)
            ax2.set_title(f"{m.upper()} — Test MAE (°C) theo Pattern")
            ax2.set_xlabel("Pattern")
            ax2.set_ylabel("Test MAE (°C)")
            ax2.grid(True, axis="y", alpha=0.3)
            ax2.legend(title="Cấu hình", fontsize=8)
            plt.tight_layout()
            p2 = os.path.join(out_dir, f"mae_by_pattern_{m}.png")
            fig2.savefig(p2, dpi=150)
            plt.close(fig2)
            saved_figs.append(p2)

    return saved_figs


def _format_overall_table(summary_df: pd.DataFrame) -> List[str]:
    if summary_df.empty:
        return ["*Không tìm thấy lần chạy nào.*"]
    out = [
        "| Backbone | Mã | Config Tag | Mô tả | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Best Epoch | Mẫu tới `val_target` | Ghi chú |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    desc_map = {c["code"]: c["desc"] for c in STANDARD_CONFIGS + EXPLORATORY_CONFIGS}
    for _, r in summary_df.iterrows():
        samp_str = f"{r['samples_to_target_mean']:.0f}" if pd.notna(r["samples_to_target_mean"]) else "Chưa đạt"
        desc_str = desc_map.get(r["code"], str(r["config_tag"]))
        out.append(
            f"| `{r['model'].upper()}` | **{r['code']}** | `{r['config_tag']}` | {desc_str} | {int(r['n_seeds'])} ({r['seeds']}) | "
            f"{r['mae_degC_mean']:.4f} ± {r['mae_degC_std']:.4f} | "
            f"{r['rmse_degC_mean']:.4f} ± {r['rmse_degC_std']:.4f} | "
            f"{r['best_val_loss_mean']:.4f} ± {r['best_val_loss_std']:.4f} | "
            f"{r.get('best_epoch_mean', float('nan')):.1f} | "
            f"{samp_str} ({int(r['n_reached_target'])}/{int(r['n_seeds'])}) | "
            f"{r['stat_note']} |"
        )
    return out


def _format_pattern_table(pat_summary_df: pd.DataFrame) -> List[str]:
    if pat_summary_df.empty:
        return ["*Không có dữ liệu theo Pattern.*"]
    out = [
        "| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in pat_summary_df.iterrows():
        out.append(
            f"| `{r['model'].upper()}` | **{r['code']}** | `{r['group_type']}` | `{r['pattern']}` | {int(r['n_test'])} | "
            f"{r['mae_degC_mean']:.4f} ± {r['mae_degC_std']:.4f} | "
            f"{r['rmse_degC_mean']:.4f} ± {r['rmse_degC_std']:.4f} | "
            f"{r['gap_mae_vs_normal_mean']:+.4f} |"
        )
    return out


def _format_paired_table(paired_df: pd.DataFrame) -> List[str]:
    if paired_df.empty:
        return ["*Chưa có cặp cấu hình PALF vs đối chứng cùng seed để tính hiệu số ghép cặp.*"]
    out = [
        "| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for _, r in paired_df[paired_df["metric"] != "time_sec_to_target"].iterrows():
        out.append(
            f"| `{r['model'].upper()}` | **{r['code']}** | **{r['ref_code']}** | `{r['metric']}` | {int(r['n_paired_seeds'])} | "
            f"{r['diff_mean']:+.4f} ± {r['diff_std']:.4f} | `{r['per_seed_diffs']}` | {r['stat_note']} |"
        )
    return out


def write_comparison_markdown(
    summary_df: pd.DataFrame,
    pat_summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    out_path: str,
    extra_sections: Optional[dict] = None,
) -> str:
    lines = [
        "# Báo cáo So sánh Giao thức Công bằng: Baseline vs PALF (Spec v1 — Mục 7 & Prompt 8)",
        "",
        "> **Quy tắc thống kê, đồng nhất & định danh G3 (Spec v1 Mục 7 & Prompt 8):**",
        "> 1. Mọi lần chạy trong bảng bắt buộc có cùng `n_train`, `n_val`, `n_test`, `epochs`, `batch`, `lr` và cùng danh sách `seeds` (các lần chạy thử nhỏ lưu riêng ở `experiments/results/smoke/`).",
        "> 2. Tiêu chí G3 đặt trước cho kết quả **\"lui về phương án dự phòng\"** (chỉ 2/4 backbone đạt $\\Delta R^2 \\ge 0.02$ sau kiểm soát epoch và dấu $\\beta$ không cùng chiều ở PatchTST). Do đó, cấu hình **A3** và **A3r** được gọi đúng tên là **`predictive (không đạt ngưỡng G3)`**.",
        "> 3. **Ghi chú về thời gian chạy:** Thời gian chạy (`epoch_time_sec`, `time_sec_to_target`, `overhead_sec`) không so sánh được giữa các máy/lần chạy khác nhau và đường mã lấy mẫu khác nhau, không dùng để kết luận và được chuyển xuống **Phụ lục** ở cuối báo cáo.",
        "",
        "## 1. Bảng tổng hợp chính theo Cấu hình & Backbone",
        "",
    ]

    lines.extend(_format_overall_table(summary_df))

    if extra_sections:
        if "summary_3seeds_df" in extra_sections and not extra_sections["summary_3seeds_df"].empty:
            lines.extend([
                "",
                "### 1b. Bảng đối chiếu nguyên vẹn trên 3 Seed gốc (`seeds 0, 1, 2` — theo Mục 8.3)",
                "",
            ])
            lines.extend(_format_overall_table(extra_sections["summary_3seeds_df"]))

        if "summary_4seeds_df" in extra_sections and not extra_sections["summary_4seeds_df"].empty:
            lines.extend([
                "",
                "### 1c. Bảng trung bình 4 Seed độc lập (`seeds 1, 2, 3, 4` — tách khỏi `seed 0` đã dùng chọn siêu tham số ở Bước 5.1)",
                "",
            ])
            lines.extend(_format_overall_table(extra_sections["summary_4seeds_df"]))

    lines.extend(["", "## 2. Bảng MAE và RMSE (°C) theo từng Pattern (kể cả tách `temp_anomaly` nóng/lạnh)", ""])
    lines.extend(_format_pattern_table(pat_summary_df))

    lines.extend([
        "",
        "## 3. So sánh ghép cặp theo Seed (Paired Differences vs A0s / A0 / A0r / A1 — A0s là đối chứng chính của A1, A3 theo Mục 9.1)",
        "",
        "> **Cách đọc đăng ký trước đối với `A0s` (Prompt 9 — Mục 9.1):**",
        "> - **Kết quả thực nghiệm trên 5 seed GRU:** `A0s - A0 = -0.0007 ± 0.0164 °C` (`|hiệu| = 0.0007 °C` nhỏ hơn nhiều so với độ lệch chuẩn giữa các seed của `A0` là `0.0255 °C` và của `A0s` là `0.0161 °C` $\\Rightarrow$ **`A0s ≈ A0`**).",
        "> - **Theo quy tắc đăng ký trước (Nhánh 1 — `A0s ≈ A0`):** Khi giữ nguyên lấy mẫu đều 100% không hoàn lại mỗi epoch (`uniform, budget=1.0`), đường mã `AdaptiveScheduler` (`A0s: 1.7829 ± 0.0161 °C`) cho kết quả trùng khít với `DataLoader` chuẩn (`A0: 1.7836 ± 0.0255 °C`). Do đó, phần cải thiện của `A1` (`-0.0211 ± 0.0142 °C` so với `A0s`), `A3` (`-0.0148 ± 0.0118 °C` so với `A0s`) và `A0r` (`-0.0151 ± 0.0079 °C` so với `A0s`) **không phải do bản thân đường mã `AdaptiveScheduler` khi lấy đủ 100% mẫu**, mà xuất hiện khi thay đổi cơ chế lấy mẫu sau warmup (giảm mẫu ngẫu nhiên 30% ở `A0r` hoặc lấy mẫu có hoàn lại theo trọng số ở `A1/A3`).",
        "> - **Theo quy tắc đăng ký trước (Nhánh 3 & tổng hợp):** Độ giảm của `A1` (`-0.0211 °C`) và `A3` (`-0.0148 °C`) so với `A0s` nhỉnh hơn/xấp xỉ 1 độ lệch chuẩn liên-seed (`0.0161 °C`), nên được ghi nhận là **xu hướng kèm cảnh báo $n=5$ nhỏ và không hiệu chỉnh đa so sánh**. Tuy nhiên, vì `A3` (`1.7682 °C`) và `A1` (`1.7618 °C`) **vẫn chưa phân biệt được với đối chứng giảm mẫu đồng đều không dùng Pattern (`A0r: 1.7678 ± 0.0118 °C`)** và `A3 - A1 = +0.0063 ± 0.0148 °C`, `A3r - A0r = +0.0021 ± 0.0043 °C`, nên **không có bằng chứng cho thấy trọng số theo Pattern hay công thức `predictive` mang lại lợi ích vượt trên việc giảm mẫu / lấy mẫu có hoàn lại**.",
        "",
    ])
    lines.extend(_format_paired_table(paired_df))

    if extra_sections and "primary_c123_df" in extra_sections and not extra_sections["primary_c123_df"].empty:
        c_df = extra_sections["primary_c123_df"]
        lines.extend([
            "",
            "### 3c. Ba đối chiếu chính Prompt 10 (Hiệu chỉnh Bonferroni: ngưỡng $\\alpha_{\\text{Bonf}} = 0.05 / 3 = 0.0167$, kiểm định t ghép cặp hai phía & Wilcoxon)",
            "",
            "| Backbone | Mã | Đối chiếu | Số seed | Hiệu MAE (Mean ± Std) (°C) | Số seed âm/dương | t-stat | p-value (t) | p-Bonferroni (×3) | Đạt ngưỡng 0.0167? | Wilcoxon (W, p) | Hiệu số từng seed |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|---|",
        ])
        for _, r in c_df.iterrows():
            sig_str = "**Đạt**" if r["bonferroni_significant"] else "Không đạt"
            lines.append(
                f"| `{r['model'].upper()}` | **{r['comparison_id']}** | **{r['code']} − {r['ref_code']}** | {int(r['n_seeds'])} | "
                f"{r['diff_mean']:+.4f} ± {r['diff_std']:.4f} | {int(r['n_neg'])}/{int(r['n_seeds'])} âm ({int(r['n_pos'])} dương) | "
                f"{r['t_stat']:+.3f} | {r['t_pvalue']:.4f} | {r['p_bonferroni']:.4f} | {sig_str} | "
                f"W={r['wilcoxon_stat']:.1f}, p={r['wilcoxon_pvalue']:.4f} | `{r['per_seed_diffs']}` |"
            )
        lines.append("")
        # Ap dung dung bang cach doc dang ky truoc (Prompt 10)
        by_cid = {r["comparison_id"]: r for _, r in c_df.iterrows()}
        c1 = by_cid.get("C1")
        c2 = by_cid.get("C2")
        c3 = by_cid.get("C3")
        lines.append("> **Kết luận theo đúng bảng Cách đọc đăng ký trước (Prompt 10):**")
        if c1 is not None and c3 is not None:
            if c3["bonferroni_significant"] and c3["diff_mean"] < 0:
                lines.append(
                    f"> - **C3 đạt (`p_t = {c3['t_pvalue']:.4f} < 0.0167`):** Với GRU trên Jena, ưu tiên Pattern theo loss (`A1`) giảm MAE nhỏ (`{-c3['diff_mean']:.4f} °C`) so với trọng số ngẫu nhiên cùng phân bố (`A0w`)."
                )
            elif c1["bonferroni_significant"] and c1["diff_mean"] < 0:
                lines.append(
                    f"> - **C3 không đạt (`p_t = {c3['t_pvalue']:.4f}`), C1 đạt (`p_t = {c1['t_pvalue']:.4f} < 0.0167`):** Cải thiện so với lấy đều (`A0s`) là do **đổi cơ cấu mẫu nói chung** (lặp mẫu / phân bổ lại tần suất batch), chưa chứng minh được do thông tin Pattern."
                )
            else:
                lines.append(
                    f"> - **C1 không đạt (`p_t = {c1['t_pvalue']:.4f} ≥ 0.0167`):** Không có hiệu ứng ổn định của trọng số theo loss (`A1`) so với lấy đều (`A0s`) ở 10 seed; hiệu ứng ở 5 seed là biến động ngẫu nhiên."
                )
        if c2 is not None:
            c2_dir = c2["n_neg"] if c2["diff_mean"] < 0 else c2["n_pos"]
            if c2["bonferroni_significant"] and c2_dir >= 8:
                lines.append(
                    f"> - **C2 đạt (`p_t = {c2['t_pvalue']:.4f} < 0.0167`, cùng dấu {c2_dir}/{int(c2['n_seeds'])} seed):** Learning State có khác biệt so với loss đơn thuần (cảnh báo: một bộ dữ liệu Jena, một backbone GRU)."
                )
            else:
                lines.append(
                    f"> - **C2 không đạt (`A3 − A1 = {c2['diff_mean']:+.4f} °C`, `p_t = {c2['t_pvalue']:.4f}`):** **Learning State đa biến không hơn loss đơn thuần.** (Kết quả chính của đề tài)."
                )

    if extra_sections and "paired_3seeds_df" in extra_sections and not extra_sections["paired_3seeds_df"].empty:
        lines.extend([
            "",
            "### 3b. Hiệu số ghép cặp trên 3 Seed gốc (`seeds 0, 1, 2`)",
            "",
        ])
        lines.extend(_format_paired_table(extra_sections["paired_3seeds_df"]))

    if extra_sections and "exploratory_summary_df" in extra_sections and not extra_sections["exploratory_summary_df"].empty:
        lines.extend([
            "",
            "## 4. Hai thí nghiệm khám phá đặt trước trên GRU (`E1: warmup_epochs=1` & `E2: policy=loss_abs`, `seeds 0, 1, 2`)",
            "",
            "> **Ghi chú tiền đăng ký (Prompt 8 — Mục 8.4):** Cả hai thí nghiệm **E1** (`policy=predictive`, `warmup_epochs=1`, `alpha=0.5`, `w_max=2.0`) và **E2** (`policy=loss_abs`, `warmup_epochs=3`, `alpha=0.5`, `w_max=2.0`) được báo cáo đầy đủ bất kể kết quả cải thiện hay không, đặt cạnh `A0`, `A0r`, `A1`, `A3` trên cùng `seeds 0, 1, 2`.",
            "",
            "### 4a. Bảng tổng hợp toàn cục (`E1`, `E2` đặt cạnh `A0`, `A0r`, `A1`, `A3` — `seeds 0, 1, 2`)",
            "",
        ])
        lines.extend(_format_overall_table(extra_sections["exploratory_summary_df"]))

        if "exploratory_pat_df" in extra_sections and not extra_sections["exploratory_pat_df"].empty:
            lines.extend([
                "",
                "### 4b. Bảng Test MAE & RMSE (°C) theo từng Pattern (`E1`, `E2` — `seeds 0, 1, 2`)",
                "",
            ])
            lines.extend(_format_pattern_table(extra_sections["exploratory_pat_df"]))

        if "exploratory_paired_df" in extra_sections and not extra_sections["exploratory_paired_df"].empty:
            lines.extend([
                "",
                "### 4c. Hiệu số ghép cặp theo Seed của `E1` và `E2` so với `A0`, `A1`, `A3` (`seeds 0, 1, 2`)",
                "",
            ])
            lines.extend(_format_paired_table(extra_sections["exploratory_paired_df"]))

        if "tcn_summary_df" in extra_sections and not extra_sections["tcn_summary_df"].empty:
            lines.extend([
                "",
                "## 4b. Nhân bản trên Backbone TCN (`A0, A0s, A1, A3` — `seeds 0, 1, 2`, Mục 9.2)",
                "",
                "> **Ghi chú nhân bản trên TCN (Prompt 9 — Mục 9.2):**",
                "> - Chạy với cùng siêu tham số từ GRU (`alpha=0.5, w_max=2.0` cho `A3`, `alpha=1.0, w_max=3.0` cho `A1`, **không tinh chỉnh lại trên TCN**) và hệ số ngoài mẫu `experiments/analysis/phase3_real/predictive_coef_tcn_seeds34.json` (`fit_scope: \"tcn\"`, `fit_seeds: [3, 4]`) để kiểm tra tính lặp lại trên kiến trúc tích chập nhân quả giãn nở (TCN). Kết luận chỉ mô tả xu hướng (`3 seeds`).",
                "> - **Quan sát mô tả trên TCN:** TCN hội tụ sớm hơn cả GRU (`best_epoch = 3.0` ở `A0`/`A0s` và `3.7` ở `A1`/`A3`). Ở `2/3` seed (`seed 0, 2`), điểm `val_loss` tốt nhất đạt ngay tại `epoch 3` (cuối giai đoạn `warmup_epochs = 3`), các epoch 4–8 sau khi Policy can thiệp không vượt qua mốc `val_loss` của epoch 3 nên checkpoint tốt nhất của `A0s, A1, A3` trùng nhau (`hiệu số = 0.0000 °C` ở `s0, s2`); ở `seed 1` (`best_epoch = 5`), `A1` (`1.7799 °C`) và `A3` (`1.8061 °C`) có MAE cao hơn nhẹ so với `A0s` (`1.7633 °C`). Nhóm `temp_anomaly` tiếp tục có sai số cao nhất ở mọi cấu hình (`1.9502–1.9798 °C`). Như vậy, hiện tượng hội tụ sớm và không tách biệt được hiệu quả của Policy lặp lại nhất quán trên backbone thứ hai (TCN).",
                "",
                "### Bảng tổng hợp TCN (`seeds 0, 1, 2`)",
                "",
            ])
            lines.extend(_format_overall_table(extra_sections["tcn_summary_df"]))

            if "tcn_pat_df" in extra_sections and not extra_sections["tcn_pat_df"].empty:
                lines.extend([
                    "",
                    "### Bảng Test MAE & RMSE (°C) theo từng Pattern trên TCN (`seeds 0, 1, 2`)",
                    "",
                ])
                lines.extend(_format_pattern_table(extra_sections["tcn_pat_df"]))

            if "tcn_paired_df" in extra_sections and not extra_sections["tcn_paired_df"].empty:
                lines.extend([
                    "",
                    "### Hiệu số ghép cặp theo Seed trên TCN (`A0s - A0`, `A1 - A0s`, `A3 - A0s`, `A3 - A1`)",
                    "",
                ])
                lines.extend(_format_paired_table(extra_sections["tcn_paired_df"]))

    lines.extend(["", "## 5. Báo cáo các cấu hình còn thiếu (Missing Configurations)", ""])
    if missing_df.empty:
        lines.append("Tất cả các cấu hình và seed trong danh sách kiểm tra đều đã có đủ trên đĩa.")
    else:
        lines.append("| Backbone | Mã | Config Tag | Mô tả | Trạng thái | Seed đã có | Seed còn thiếu |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in missing_df.iterrows():
            lines.append(
                f"| `{r['model'].upper()}` | **{r['code']}** | `{r['config_tag']}` | {r['description']} | "
                f"`{r['status']}` | `{r['found_seeds']}` ({int(r['n_found'])}) | `{r['missing_seeds']}` ({int(r['n_missing'])}) |"
            )

    lines.extend([
        "",
        "## Phụ lục: Thời gian chạy (Chỉ tham khảo kỹ thuật — Không dùng để so sánh/kết luận)",
        "",
        "> **Lưu ý:** Thời gian chạy mỗi epoch (`Epoch time`), thời gian đạt ngưỡng (`Giây tới val_target`) và `Overhead PALF` phụ thuộc vào tải CPU tại thời điểm chạy cũng như sự khác biệt đường mã giữa `DataLoader(train)` chuẩn và `DataLoader(Subset(train, indices))`. Các số liệu này **không dùng để kết luận hiệu năng thuật toán**.",
        "",
    ])
    if not summary_df.empty:
        lines.append("| Backbone | Mã | Config Tag | Số seed | Giây tới `val_target` | Epoch time (s) | Overhead PALF (s) |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for _, r in summary_df.iterrows():
            t_str = f"{r['time_sec_to_target_mean']:.1f}s" if pd.notna(r["time_sec_to_target_mean"]) else "Chưa đạt"
            lines.append(
                f"| `{r['model'].upper()}` | **{r['code']}** | `{r['config_tag']}` | {int(r['n_seeds'])} ({r['seeds']}) | "
                f"{t_str} | {r['mean_epoch_time_sec_mean']:.2f}s | {r['mean_overhead_sec_mean']:.4f}s |"
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    content = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def run_comparison_protocol(
    baseline_root: str = "experiments/results/baseline/jena",
    palf_root: str = "experiments/results/palf/jena",
    out_dir: str = "experiments/analysis/comparison",
    models: Tuple[str, ...] = ("gru", "tcn", "transformer", "patchtst"),
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    config_specs: Optional[List[dict]] = None,
    strict_validation: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    effective_specs = config_specs
    if effective_specs is None and len(seeds) > 5:
        # Prompt 10: kiem dinh 10 seed tap trung vao A0, A0s, A0w, A0r, A1, A3 (A3r chi chay 5 seed 0..4)
        effective_specs = [c for c in STANDARD_CONFIGS if c["code"] in ("A0", "A0s", "A0w", "A0r", "A1", "A3")]

    found_runs, missing_df = collect_protocol_runs(
        baseline_root=baseline_root,
        palf_root=palf_root,
        models=models,
        seeds=seeds,
        config_specs=effective_specs,
        strict_validation=strict_validation,
    )
    val_targets = compute_val_targets_from_a0(found_runs)
    detail_df, pattern_df = build_detail_and_pattern_tables(found_runs, val_targets)
    summary_df, paired_df = summarize_overall_and_paired(detail_df)
    pat_summary_df = summarize_by_pattern(pattern_df)

    detail_df.to_csv(os.path.join(out_dir, "runs_detail.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "summary_overall.csv"), index=False)
    pat_summary_df.to_csv(os.path.join(out_dir, "summary_by_pattern.csv"), index=False)
    paired_df.to_csv(os.path.join(out_dir, "paired_differences.csv"), index=False)
    missing_df.to_csv(os.path.join(out_dir, "missing_configs.csv"), index=False)

    saved_figs = plot_comparison_figures(found_runs, pat_summary_df, val_targets, out_dir)

    extra_sections: dict = {}
    primary_c123_df = compute_prompt10_primary_comparisons(detail_df)
    if not primary_c123_df.empty:
        extra_sections["primary_c123_df"] = primary_c123_df
        primary_c123_df.to_csv(os.path.join(out_dir, "primary_comparisons_prompt10.csv"), index=False)

    if len(seeds) >= 5 and set((0, 1, 2)).issubset(set(seeds)):
        det_3 = detail_df[detail_df["seed"].isin([0, 1, 2])].copy()
        sum_3, pair_3 = summarize_overall_and_paired(det_3)
        extra_sections["summary_3seeds_df"] = sum_3
        extra_sections["paired_3seeds_df"] = pair_3
        sum_3.to_csv(os.path.join(out_dir, "summary_overall_3seeds.csv"), index=False)

        det_4 = detail_df[detail_df["seed"].isin([1, 2, 3, 4])].copy()
        sum_4, _ = summarize_overall_and_paired(det_4)
        extra_sections["summary_4seeds_df"] = sum_4
        sum_4.to_csv(os.path.join(out_dir, "summary_overall_seeds1234.csv"), index=False)

    if config_specs is None:
        exp_specs = [c for c in STANDARD_CONFIGS if c["code"] in ("A0", "A0s", "A0r", "A1", "A3")] + EXPLORATORY_CONFIGS
        exp_runs, _ = collect_protocol_runs(
            baseline_root=baseline_root,
            palf_root=palf_root,
            models=models,
            seeds=(0, 1, 2),
            config_specs=exp_specs,
            strict_validation=False,
        )
        has_e1_e2 = any(r["code"] in ("E1", "E2") for r in exp_runs)
        if has_e1_e2:
            exp_vt = compute_val_targets_from_a0(exp_runs)
            exp_det, exp_pat = build_detail_and_pattern_tables(exp_runs, exp_vt)
            exp_sum, exp_pair = summarize_overall_and_paired(exp_det)
            exp_pat_sum = summarize_by_pattern(exp_pat)
            extra_sections["exploratory_summary_df"] = exp_sum
            extra_sections["exploratory_pat_df"] = exp_pat_sum[exp_pat_sum["code"].isin(["A0", "A0s", "A1", "A3", "E1", "E2"])]
            extra_sections["exploratory_paired_df"] = exp_pair[exp_pair["code"].isin(["E1", "E2"])]
            exp_sum.to_csv(os.path.join(out_dir, "summary_exploratory_e1_e2.csv"), index=False)

        if "tcn" not in [m.lower() for m in models]:
            tcn_specs = [c for c in STANDARD_CONFIGS if c["code"] in ("A0", "A0s", "A1", "A3")]
            tcn_runs, _ = collect_protocol_runs(
                baseline_root=baseline_root,
                palf_root=palf_root,
                models=("tcn",),
                seeds=(0, 1, 2),
                config_specs=tcn_specs,
                strict_validation=False,
            )
            if any(r["code"] != "A0" for r in tcn_runs):
                tcn_vt = compute_val_targets_from_a0(tcn_runs)
                tcn_det, tcn_pat = build_detail_and_pattern_tables(tcn_runs, tcn_vt)
                tcn_sum, tcn_pair = summarize_overall_and_paired(tcn_det)
                tcn_pat_sum = summarize_by_pattern(tcn_pat)
                extra_sections["tcn_summary_df"] = tcn_sum
                extra_sections["tcn_pat_df"] = tcn_pat_sum
                extra_sections["tcn_paired_df"] = tcn_pair
                tcn_sum.to_csv(os.path.join(out_dir, "summary_tcn_3seeds.csv"), index=False)
                tcn_pat_sum.to_csv(os.path.join(out_dir, "summary_by_pattern_tcn.csv"), index=False)
                tcn_pair.to_csv(os.path.join(out_dir, "paired_differences_tcn.csv"), index=False)
                plot_comparison_figures(tcn_runs, tcn_pat_sum, tcn_vt, out_dir)

    report_path = os.path.join(out_dir, "comparison_report.md")
    write_comparison_markdown(summary_df, pat_summary_df, paired_df, missing_df, report_path, extra_sections=extra_sections)

    return {
        "found_runs": found_runs,
        "val_targets": val_targets,
        "detail_df": detail_df,
        "summary_df": summary_df,
        "pat_summary_df": pat_summary_df,
        "paired_df": paired_df,
        "primary_c123_df": primary_c123_df,
        "missing_df": missing_df,
        "saved_figs": saved_figs,
        "report_path": report_path,
    }
