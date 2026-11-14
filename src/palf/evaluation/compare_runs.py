"""Giao thuc so sanh cong bang Baseline vs PALF (Spec Phase 5-6 Muc 7, Phan 4.2)."""
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

# Quy uoc 5 cau hinh chuan theo Spec Muc 7.1
STANDARD_CONFIGS = [
    {"code": "A0", "mode": "baseline", "config_tag": "baseline", "budget_ratio": 1.0, "ref_code": "A0", "desc": "Baseline uniform (budget 1.0)"},
    {"code": "A0r", "mode": "palf", "config_tag": "uniform_b0.7", "budget_ratio": 0.7, "ref_code": "A0", "desc": "Baseline uniform qua Scheduler (budget 0.7)"},
    {"code": "A1", "mode": "palf", "config_tag": "loss_b1.0_a1.0", "budget_ratio": 1.0, "ref_code": "A0", "desc": "Doi chung B2 chi dung loss_ratio (budget 1.0)"},
    {"code": "A2", "mode": "palf", "config_tag": "state_b1.0_a1.0", "budget_ratio": 1.0, "ref_code": "A0", "desc": "PALF Learning State (budget 1.0)"},
    {"code": "A2r", "mode": "palf", "config_tag": "state_b0.7_a1.0", "budget_ratio": 0.7, "ref_code": "A0r", "desc": "PALF Learning State giam ngan sach (budget 0.7)"},
]


def resolve_config_run_dir(
    baseline_root: str,
    palf_root: str,
    cfg_spec: dict,
    model: str,
    seed: int,
) -> str:
    if cfg_spec["mode"] == "baseline":
        return os.path.join(baseline_root, model.lower(), f"seed{seed}")
    return os.path.join(palf_root, cfg_spec["config_tag"], model.lower(), f"seed{seed}")


def collect_protocol_runs(
    baseline_root: str = "experiments/results/baseline/jena",
    palf_root: str = "experiments/results/palf/jena",
    models: Tuple[str, ...] = ("gru", "tcn", "transformer", "patchtst"),
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    config_specs: Optional[List[dict]] = None,
) -> Tuple[List[dict], pd.DataFrame]:
    """Doc tat ca cac lan chay theo danh sach cau hinh A0, A0r, A1, A2, A2r.
    Tra ve (found_runs, missing_df) — khong im lang bo qua cau hinh thieu."""
    specs = config_specs if config_specs is not None else STANDARD_CONFIGS
    found_runs: List[dict] = []
    missing_rows: List[dict] = []

    for m in models:
        m_low = m.lower()
        for cfg in specs:
            found_seeds = []
            missing_seeds = []
            for s in seeds:
                r_dir = resolve_config_run_dir(baseline_root, palf_root, cfg, m_low, s)
                if os.path.isfile(os.path.join(r_dir, "history.csv")) and os.path.isfile(
                    os.path.join(r_dir, "test_metrics.json")
                ):
                    r = load_saved_run(r_dir, label=f"{cfg['code']} ({cfg['config_tag']}) | {m_low.upper()} | seed{s}")
                    r["code"] = cfg["code"]
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
                    "config_tag": cfg["config_tag"],
                    "description": cfg["desc"],
                    "n_found": len(found_seeds),
                    "found_seeds": ",".join(str(x) for x in found_seeds) if found_seeds else "none",
                    "n_missing": len(missing_seeds),
                    "missing_seeds": ",".join(str(x) for x in missing_seeds),
                    "status": "PARTIAL" if found_seeds else "MISSING_ALL",
                })

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
            "stat_note": "trend_only (<5 seeds)" if n_seeds < 5 else "5_seeds_complete",
        }
        for col in metrics:
            vals = grp[col].dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) > 0 else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Paired seed differences: moi cau hinh so voi ref_code (va luon kem so voi A0 neu ref_code != A0)
    paired_rows = []
    for m, m_grp in detail_df.groupby("model", sort=False):
        by_code = {c: g.set_index("seed") for c, g in m_grp.groupby("code", sort=False)}
        for code, g_cur in by_code.items():
            if code == "A0":
                continue
            ref_candidates = [g_cur["ref_code"].iloc[0]]
            if "A0" not in ref_candidates:
                ref_candidates.append("A0")
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
                        "stat_note": "Chi mo ta xu huong (khong tuyen bo y nghia thong ke voi <5 seed)" if len(diffs) < 5 else "Du 5 seed ghep cap",
                    })

    return summary_df, pd.DataFrame(paired_rows)


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

        # 1. Hinh val_loss theo so mau da xu ly (trung binh theo seed cho tung code)
        fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
        by_code: Dict[str, List[dict]] = {}
        for r in m_runs:
            by_code.setdefault(r["code"], []).append(r)

        for code, r_list in by_code.items():
            tag = r_list[0]["config_tag"]
            # Gop theo epoch
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

        # 2. Hinh cot MAE (°C) theo Pattern
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


def write_comparison_markdown(
    summary_df: pd.DataFrame,
    pat_summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    out_path: str,
) -> str:
    lines = [
        "# Báo cáo So sánh Giao thức Công bằng: Baseline vs PALF (Mục 7)",
        "",
        "> **Quy tắc thống kê (Spec Mục 7):** Với cấu hình có ít hơn 5 seed (`n_seeds < 5`), báo cáo **chỉ mô tả xu hướng** và hiệu số ghép theo từng seed, tuyệt đối **không tuyên bố ý nghĩa thống kê**.",
        "",
        "## 1. Bảng tổng hợp toàn cục theo Cấu hình & Backbone",
        "",
    ]

    if summary_df.empty:
        lines.append("*Không tìm thấy lần chạy nào.*")
    else:
        lines.append(
            "| Backbone | Mã | Config Tag | Số seed | Test MAE (°C) | Test RMSE (°C) | Best Val Loss | Mẫu tới `val_target` | Giây tới `val_target` | AUC Val (mẫu) | Epoch time (s) | Overhead PALF (s) | Ghi chú |"
        )
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in summary_df.iterrows():
            samp_str = f"{r['samples_to_target_mean']:.0f}" if pd.notna(r["samples_to_target_mean"]) else "Chưa đạt"
            t_str = f"{r['time_sec_to_target_mean']:.1f}s" if pd.notna(r["time_sec_to_target_mean"]) else "Chưa đạt"
            lines.append(
                f"| `{r['model'].upper()}` | **{r['code']}** | `{r['config_tag']}` | {int(r['n_seeds'])} ({r['seeds']}) | "
                f"{r['mae_degC_mean']:.4f} ± {r['mae_degC_std']:.4f} | "
                f"{r['rmse_degC_mean']:.4f} ± {r['rmse_degC_std']:.4f} | "
                f"{r['best_val_loss_mean']:.4f} ± {r['best_val_loss_std']:.4f} | "
                f"{samp_str} ({int(r['n_reached_target'])}/{int(r['n_seeds'])}) | {t_str} | "
                f"{r['val_auc_samples_mean']:.4f} | {r['mean_epoch_time_sec_mean']:.2f}s | "
                f"{r['mean_overhead_sec_mean']:.4f}s | {r['stat_note']} |"
            )

    lines.extend(["", "## 2. Bảng MAE và RMSE (°C) theo từng Pattern (kể cả tách `temp_anomaly` nóng/lạnh)", ""])
    if pat_summary_df.empty:
        lines.append("*Không có dữ liệu theo Pattern.*")
    else:
        lines.append("| Backbone | Mã | Nhóm | Pattern | Số mẫu test | MAE (°C) (Mean ± Std) | RMSE (°C) (Mean ± Std) | Chênh lệch MAE so với `normal` (°C) |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        for _, r in pat_summary_df.iterrows():
            lines.append(
                f"| `{r['model'].upper()}` | **{r['code']}** | `{r['group_type']}` | `{r['pattern']}` | {int(r['n_test'])} | "
                f"{r['mae_degC_mean']:.4f} ± {r['mae_degC_std']:.4f} | "
                f"{r['rmse_degC_mean']:.4f} ± {r['rmse_degC_std']:.4f} | "
                f"{r['gap_mae_vs_normal_mean']:+.4f} |"
            )

    lines.extend(["", "## 3. So sánh ghép cặp theo Seed (Paired Differences vs A0 / A0r)", ""])
    if paired_df.empty:
        lines.append("*Chưa có cặp cấu hình PALF vs đối chứng cùng seed để tính hiệu số ghép cặp.*")
    else:
        lines.append("| Backbone | Cấu hình | Đối chứng | Chỉ số | Số seed ghép | Hiệu số (Mean ± Std) | Hiệu số từng seed | Ghi chú |")
        lines.append("|---|---|---|---|---:|---:|---|---|")
        for _, r in paired_df.iterrows():
            lines.append(
                f"| `{r['model'].upper()}` | **{r['code']}** | **{r['ref_code']}** | `{r['metric']}` | {int(r['n_paired_seeds'])} | "
                f"{r['diff_mean']:+.4f} ± {r['diff_std']:.4f} | `{r['per_seed_diffs']}` | {r['stat_note']} |"
            )

    lines.extend(["", "## 4. Báo cáo các cấu hình còn thiếu (Missing Configurations)", ""])
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
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    found_runs, missing_df = collect_protocol_runs(
        baseline_root=baseline_root,
        palf_root=palf_root,
        models=models,
        seeds=seeds,
        config_specs=config_specs,
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
    report_path = os.path.join(out_dir, "comparison_report.md")
    write_comparison_markdown(summary_df, pat_summary_df, paired_df, missing_df, report_path)

    return {
        "found_runs": found_runs,
        "val_targets": val_targets,
        "detail_df": detail_df,
        "summary_df": summary_df,
        "pat_summary_df": pat_summary_df,
        "paired_df": paired_df,
        "missing_df": missing_df,
        "saved_figs": saved_figs,
        "report_path": report_path,
    }
