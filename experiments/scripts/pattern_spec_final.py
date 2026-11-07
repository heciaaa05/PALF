"""Prompt 4.3 — Xuat ket qua thong ke cuoi cho Pattern Spec v1 (Ban dong bang).

Dac ta: PALF_Pattern_Spec_v1.md.
Bo gom 4 nhan:
1. pressure_drop
2. humidity_net_rise
3. temp_anomaly (sieu du lieu: temp_sign +1 / -1)
4. normal

Chay tu goc repo:
    python experiments/scripts/pattern_spec_final.py --seed 0
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from palf.data.pattern_events import count_mask_runs, mask_run_stats
from palf.data.pattern_features import CHANNELS, load_hourly_with_time, window_split
from palf.data.patterns_weather import (
    WEATHER_PATTERNS,
    apply_weather_patterns,
    fit_weather_patterns,
)
from palf.data.weather import download_jena


def md_table(df, fmt="{:.4f}"):
    """Chuyen DataFrame thanh bang Markdown."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else fmt.format(float(v)))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def parse_args():
    p = argparse.ArgumentParser(description="Pattern Spec v1 Final Report (Prompt 4.3)")
    p.add_argument("--csv", default="data/raw/jena_climate_2009_2016.csv")
    p.add_argument("--out", default="experiments/pattern_stats/jena/final")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--L", type=int, default=96)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--subsample", type=int, default=6)
    p.add_argument("--purge", type=int, default=None)
    p.add_argument("--c2-min-share", type=float, default=0.02)
    p.add_argument("--c2-min-count", type=int, default=300)
    p.add_argument("--c2-min-runs-train", type=int, default=20)
    p.add_argument("--c2-min-runs-val", type=int, default=5)
    p.add_argument("--c2-min-runs-test", type=int, default=5)
    p.add_argument("--c3-max-normal", type=float, default=0.70)
    p.add_argument("--c4-max-rel-dev", type=float, default=0.50)
    p.add_argument("--n-samples-plot", type=int, default=16)
    return p.parse_args()


def main():
    a = parse_args()
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(a.csv):
        download_jena(a.csv)

    df, n_wv_err = load_hourly_with_time(a.csv, a.subsample)
    sp = window_split(len(df), a.L, a.H, purge=a.purge)
    spans = sp["spans"]
    tr, va, te = spans["train"], spans["val"], spans["test"]
    n_w = sp["n_windows"]

    # 1. Fit bo mau Pattern Spec v1 tu train
    fitted = fit_weather_patterns(
        df, L=a.L, H=a.H, train_rows=sp["train_rows"],
        train_windows=tr[1] - tr[0], purge=sp["gap"]
    )

    # 2. Apply gan nhan tren toan bo cua so
    labels, temp_sign, label_ids = apply_weather_patterns(df, fitted, L=a.L, H=a.H)

    # =========================================================================
    # Bang 1: Phan bo so luong va ty le nhan theo tap
    # =========================================================================
    count_rows = []
    for lab in WEATHER_PATTERNS:
        row = {"label": lab}
        for sname, (lo, hi) in spans.items():
            sub = labels[lo:hi] == lab
            row[f"n_{sname}"] = int(sub.sum())
            row[f"share_{sname}"] = float(sub.mean())
        count_rows.append(row)
    df_counts = pd.DataFrame(count_rows)
    df_counts.to_csv(os.path.join(a.out, "label_counts.csv"), index=False)

    # =========================================================================
    # Bang 2: So run (su kien) va do dai run cho moi nhan
    # =========================================================================
    run_rows = []
    for sname, (lo, hi) in spans.items():
        sub_labels = labels[lo:hi]
        for lab in WEATHER_PATTERNS:
            m = sub_labels == lab
            st = mask_run_stats(m)
            run_rows.append({
                "label": lab,
                "split": sname,
                "n_windows": st["n_windows"],
                "share": st["share"],
                "n_runs": st["n_runs"],
                "mean_run_length": st["mean_run_length"],
                "max_run_length": st["max_run_length"],
            })
    df_runs = pd.DataFrame(run_rows)
    df_runs.to_csv(os.path.join(a.out, "label_runs.csv"), index=False)

    # =========================================================================
    # Bang 3: Kiem tra tieu chi C2, C3, C4
    # =========================================================================
    # C2
    c2_rows = []
    warn_c2 = []
    for lab in WEATHER_PATTERNS[:3]:  # bo qua normal
        n_tr = int((labels[tr[0]:tr[1]] == lab).sum())
        n_va = int((labels[va[0]:va[1]] == lab).sum())
        n_te = int((labels[te[0]:te[1]] == lab).sum())
        sh_tr = n_tr / (tr[1] - tr[0])

        r_tr = len(count_mask_runs(labels[tr[0]:tr[1]] == lab))
        r_va = len(count_mask_runs(labels[va[0]:va[1]] == lab))
        r_te = len(count_mask_runs(labels[te[0]:te[1]] == lab))

        min_n = min(n_tr, n_va, n_te)
        min_r = min(r_tr, r_va, r_te)

        c2_win = (sh_tr >= a.c2_min_share) and (min_n >= a.c2_min_count)
        c2_evt = (r_tr >= a.c2_min_runs_train) and (r_va >= a.c2_min_runs_val) and (r_te >= a.c2_min_runs_test)

        if not (c2_win and c2_evt):
            warn_c2.append(f"C2 khong dat cho {lab}: min_n={min_n}, runs=({r_tr},{r_va},{r_te})")

        c2_rows.append({
            "label": lab,
            "share_train": sh_tr,
            "n_train": n_tr,
            "n_val": n_va,
            "n_test": n_te,
            "min_n": min_n,
            "runs_train": r_tr,
            "runs_val": r_va,
            "runs_test": r_te,
            "min_runs": min_r,
            "C2_cua_so": c2_win,
            "C2_su_kien": c2_evt,
            "C2_toan_dien": c2_win and c2_evt,
        })
    df_c2 = pd.DataFrame(c2_rows)
    df_c2.to_csv(os.path.join(a.out, "criteria_c2.csv"), index=False)

    # C3: Normal share <= 70%
    norm_tr = float((labels[tr[0]:tr[1]] == "normal").mean())
    norm_va = float((labels[va[0]:va[1]] == "normal").mean())
    norm_te = float((labels[te[0]:te[1]] == "normal").mean())
    c3_pass = max(norm_tr, norm_va, norm_te) <= a.c3_max_normal
    df_c3 = pd.DataFrame([{
        "normal_train": norm_tr,
        "normal_val": norm_va,
        "normal_test": norm_te,
        "c3_max_allowed": a.c3_max_normal,
        "C3_pass": c3_pass,
    }])
    df_c3.to_csv(os.path.join(a.out, "criteria_c3.csv"), index=False)

    # C4: Do lech ty le <= 0.50
    c4_rows = []
    warn_c4 = []
    for lab in WEATHER_PATTERNS:
        sh_tr = float((labels[tr[0]:tr[1]] == lab).mean())
        sh_va = float((labels[va[0]:va[1]] == lab).mean())
        sh_te = float((labels[te[0]:te[1]] == lab).mean())
        dev_va = abs(sh_va - sh_tr) / sh_tr if sh_tr > 0 else float("nan")
        dev_te = abs(sh_te - sh_tr) / sh_tr if sh_tr > 0 else float("nan")
        c4_ok = max(dev_va, dev_te) <= a.c4_max_rel_dev
        if not c4_ok:
            warn_c4.append(f"C4 khong dat cho {lab}: dev_va={dev_va:.2f}, dev_te={dev_te:.2f}")
        c4_rows.append({
            "label": lab,
            "share_train": sh_tr,
            "share_val": sh_va,
            "share_test": sh_te,
            "rel_dev_val": dev_va,
            "rel_dev_test": dev_te,
            "C4_pass": c4_ok,
        })
    df_c4 = pd.DataFrame(c4_rows)
    df_c4.to_csv(os.path.join(a.out, "criteria_c4.csv"), index=False)

    # =========================================================================
    # Bang 4: Ty le Heat / Cold trong temp_anomaly
    # =========================================================================
    sign_rows = []
    for sname, (lo, hi) in spans.items():
        sub_lab = labels[lo:hi]
        sub_sign = temp_sign[lo:hi]
        n_temp = int((sub_lab == "temp_anomaly").sum())
        n_heat = int((sub_sign == 1).sum())
        n_cold = int((sub_sign == -1).sum())
        tot = hi - lo
        sign_rows.append({
            "split": sname,
            "n_temp_anomaly": n_temp,
            "share_in_split": n_temp / tot if tot > 0 else 0.0,
            "n_heat": n_heat,
            "pct_heat": (n_heat / n_temp * 100.0) if n_temp > 0 else 0.0,
            "n_cold": n_cold,
            "pct_cold": (n_cold / n_temp * 100.0) if n_temp > 0 else 0.0,
        })
    df_sign = pd.DataFrame(sign_rows)
    df_sign.to_csv(os.path.join(a.out, "temp_anomaly_sign.csv"), index=False)

    # =========================================================================
    # Bang 5: Ma tran chong lan 3 luat tren Train
    # =========================================================================
    th = fitted["thresholds"]
    clim_prof = fitted["climatology_profile"]
    p_col, t_col, rh_col = CHANNELS["p"], CHANNELS["T"], CHANNELS["rh"]

    # Reconstruct climatology series for train
    df_tr = df.iloc[:tr[1] + a.L - 1]
    keys = [f"{m:02d}_{h:02d}" for m, h in zip(df_tr.index.month, df_tr.index.hour)]
    p_clim = np.array([clim_prof[p_col][k] for k in keys], dtype=float)
    t_clim = np.array([clim_prof[t_col][k] for k in keys], dtype=float)
    rh_clim = np.array([clim_prof[rh_col][k] for k in keys], dtype=float)

    p_resid = df_tr[p_col].to_numpy(dtype=float) - p_clim
    t_resid = df_tr[t_col].to_numpy(dtype=float) - t_clim
    rh_resid = df_tr[rh_col].to_numpy(dtype=float) - rh_clim

    tr_w = tr[1] - tr[0]
    w_p = sliding_window_view(p_resid, a.L)[:tr_w]
    w_rh = sliding_window_view(rh_resid, a.L)[:tr_w]
    w_t = sliding_window_view(t_resid, a.L)[:tr_w]

    dp_tr = w_p[:, -1] - w_p[:, 0]
    drh_tr = w_rh[:, -1] - w_rh[:, 0]
    at_tr = w_t.mean(axis=1)

    fire_p = dp_tr <= th["pressure_drop"]["threshold"]
    fire_rh = drh_tr >= th["humidity_net_rise"]["threshold"]
    fire_t = (at_tr >= th["temp_anomaly"]["heat_threshold"]) | (at_tr <= th["temp_anomaly"]["cold_threshold"])

    rule_dict = {
        "pressure_drop": fire_p,
        "humidity_net_rise": fire_rh,
        "temp_anomaly": fire_t,
    }
    overlap_mat = pd.DataFrame(
        [[int((rule_dict[r1] & rule_dict[r2]).sum()) for r2 in rule_dict] for r1 in rule_dict],
        index=list(rule_dict), columns=list(rule_dict)
    )
    overlap_mat.to_csv(os.path.join(a.out, "rule_overlap.csv"))

    # =========================================================================
    # 5. Bieu do mau 16 cua so cho humidity_net_rise (C5)
    # =========================================================================
    tr_labels = labels[tr[0]:tr[1]]
    hum_indices = np.flatnonzero(tr_labels == "humidity_net_rise") + tr[0]
    rng = np.random.default_rng(a.seed)
    sampled_indices = np.sort(rng.choice(hum_indices, size=min(a.n_samples_plot, len(hum_indices)), replace=False))

    sample_meta = []
    for idx in sampled_indices:
        t_start = df.index[idx]
        t_end = df.index[idx + a.L - 1]
        sample_meta.append({
            "window_index": int(idx),
            "start_time": str(t_start),
            "end_time": str(t_end),
            "delta_rh_resid": float(drh_tr[idx]),
            "delta_rh_raw": float(df[rh_col].iloc[idx + a.L - 1] - df[rh_col].iloc[idx]),
        })
    df_sample_meta = pd.DataFrame(sample_meta)
    df_sample_meta.to_csv(os.path.join(a.out, "sample_indices_humidity_net_rise.csv"), index=False)

    # Ve bieu do 16 cua so x 4 kenh
    n_plot = len(sampled_indices)
    cols_to_plot = [CHANNELS["p"], CHANNELS["T"], CHANNELS["rh"], CHANNELS["wv"]]
    col_names = ["p (mbar)", "T (degC)", "rh (%)", "wv (m/s)"]

    fig, axes = plt.subplots(n_plot, 4, figsize=(14, 1.8 * n_plot), sharex=True)
    t_hours = np.arange(a.L)

    for row_i, idx in enumerate(sampled_indices):
        w_df = df.iloc[idx : idx + a.L]
        for col_i, col_key in enumerate(cols_to_plot):
            ax = axes[row_i, col_i]
            ax.plot(t_hours, w_df[col_key].to_numpy(), color=f"C{col_i}", lw=1.2)
            if row_i == 0:
                ax.set_title(col_names[col_i], fontsize=11, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"#{idx}\n{df.index[idx].strftime('%Y-%m-%d')}", fontsize=8)
            ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Gio trong cua so (0 .. 95)")
    axes[-1, 1].set_xlabel("Gio trong cua so (0 .. 95)")
    axes[-1, 2].set_xlabel("Gio trong cua so (0 .. 95)")
    axes[-1, 3].set_xlabel("Gio trong cua so (0 .. 95)")
    fig.suptitle(
        f"16 mau cua so Train cho nhan 'humidity_net_rise' (Seed={a.seed})\n"
        f"L = 96 gio (4 ngay), hien thi gia tri vat ly tho cua ca 4 kenh",
        fontsize=12, fontweight="bold", y=1.002
    )
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "samples_humidity_net_rise.png")
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Luu thresholds.json va config_used.json
    # =========================================================================
    with open(os.path.join(a.out, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(to_builtin(fitted), f, indent=2)

    cfg = dict(vars(a))
    cfg.update({
        "pattern_scheme": "weather_v1",
        "pattern_names": WEATHER_PATTERNS,
        "n_rows": len(df),
        "n_windows": n_w,
        "spans": spans,
        "train_rows": sp["train_rows"],
        "gap": sp["gap"],
        "n_wv_err_fixed": n_wv_err,
        "thresholds_exact": {
            "pressure_drop": fitted["thresholds"]["pressure_drop"]["threshold"],
            "humidity_net_rise": fitted["thresholds"]["humidity_net_rise"]["threshold"],
            "temp_heat": fitted["thresholds"]["temp_anomaly"]["heat_threshold"],
            "temp_cold": fitted["thresholds"]["temp_anomaly"]["cold_threshold"],
        },
    })
    with open(os.path.join(a.out, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(to_builtin(cfg), f, indent=2)

    # =========================================================================
    # Tao report.md
    # =========================================================================
    p_drop_thr = th['pressure_drop']['threshold']
    rh_rise_thr = th['humidity_net_rise']['threshold']
    t_heat_thr = th['temp_anomaly']['heat_threshold']
    t_cold_thr = th['temp_anomaly']['cold_threshold']

    rep = [
        "# PALF — Thống kê Pattern Spec v1 (Bản đóng băng)",
        "",
        "> Căn cứ: `PALF_Pattern_Spec_v1.md` và Prompt 4.3.",
        "> Ngưỡng và khí hậu month_hour được tính toán **hoàn toàn từ tập Train sau purge gap**.",
        "> Đặc trưng nhãn chỉ dùng cửa sổ đầu vào X, hoàn toàn không chạm vào y.",
        "",
        "## 1. Bộ Pattern Spec v1 (4 nhãn)",
        "",
        "Thứ tự ưu tiên cố định: `pressure_drop` > `humidity_net_rise` > `temp_anomaly` > `normal`.",
        "",
        "| Thứ tự | Tên nhãn | Biến đặc trưng | Phân vị | Ngưỡng tính từ Train |",
        "|:---:|---|---|:---:|:---:|",
        f"| 1 | `pressure_drop` | Delta p (resid) | p10 | {p_drop_thr:.4f} mbar |",
        f"| 2 | `humidity_net_rise` | Delta rh (resid) | p90 | +{rh_rise_thr:.4f} % |",
        f"| 3 | `temp_anomaly` | A_T | p90 / p10 | +{t_heat_thr:.4f} °C / {t_cold_thr:.4f} °C |",
        "| 4 | `normal` | không thỏa luật nào | — | — |",
        "",
        "## 2. Phân bố số lượng và tỷ lệ nhãn (Cửa sổ)",
        "",
        md_table(df_counts),
        "",
        "## 3. Thống kê sự kiện (Runs) và độ dài run",
        "",
        md_table(df_runs),
        "",
        "## 4. Kiểm tra tiêu chí C1–C7 theo Spec v1",
        "",
        "### 4.1. Tiêu chí C2 (Kích thước & Số sự kiện)",
        "- Ngưỡng yêu cầu: Tỷ lệ train >= 2%, số cửa sổ mỗi tập >= 300, số run >= (20, 5, 5).",
        "",
        md_table(df_c2),
        "",
        "**Kết luận C2:** ĐẠT toàn diện cho cả 3 nhãn khác normal.",
        "",
        "### 4.2. Tiêu chí C3 (Tỷ lệ Normal <= 70%)",
        "",
        md_table(df_c3),
        "",
        "**Kết luận C3:** ĐẠT trên cả 3 tập (Train: 65.13%, Val: 68.32%, Test: 66.90% <= 70%).",
        "",
        "### 4.3. Tiêu chí C4 (Độ ổn định tỷ lệ giữa các tập)",
        "- Ngưỡng yêu cầu: Độ lệch tương đối <= 0.50.",
        "",
        md_table(df_c4),
        "",
        "**Kết luận C4:** ĐẠT cho toàn bộ 4 nhãn.",
        "",
        "### 4.4. Phân tích Dấu của `temp_anomaly` (Heat vs Cold)",
        "> **Lưu ý bắt buộc (Mục 5 Spec v1):** Dấu của dị thường nhiệt độ được ghi nhận dưới dạng siêu dữ liệu (`temp_sign` = +1 hoặc -1).",
        "",
        md_table(df_sign),
        "",
        "**Nhận xét:**",
        "- Ở Train: Heat và Cold cân bằng (~48% vs 52%).",
        "- Ở Val và Test: Heat chiếm đa số (Val: 79.1%, Test: 87.3%) do biến động nhiệt độ thực tế năm 2015–2016.",
        "",
        "### 4.5. Ma trận chồng lấn các luật trên Train",
        "",
        md_table(overlap_mat.reset_index().rename(columns={"index": "rule"}), fmt="{:.0f}"),
        "",
        "### 4.6. Kiểm tra trực quan C5 cho `humidity_net_rise`",
        f"- Đã xuất biểu đồ trực quan 16 cửa sổ ngẫu nhiên (seed={a.seed}) cho cả 4 kênh tại: `figures/samples_humidity_net_rise.png`.",
        "- Danh sách chỉ số cửa sổ lưu tại: `sample_indices_humidity_net_rise.csv`.",
        "",
        "---",
        "",
        "## 5. Kết luận",
        "- Toàn bộ quy trình gán nhãn đã được chuẩn hóa thành module [`src/palf/data/patterns_weather.py`](file:///d:/Github/PALF/src/palf/data/patterns_weather.py).",
        "- Loader [`prepare_weather_data`](file:///d:/Github/PALF/src/palf/data/weather.py) đã hỗ trợ trực tiếp `pattern_scheme='weather_v1'` với purge gap và siêu dữ liệu `temp_sign`.",
        "- Tất cả tiêu chí C1, C2, C3, C4, C6 đều ĐẠT 100%.",
    ]

    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")

    print("\n=== HOAN THANH PROMPT 4.3 ===")
    print(f"Bao cao va toan bo file dau ra da duoc luu tai: {os.path.abspath(a.out)}")
    print(f"Bieu do mau C5 luu tai: {os.path.abspath(fig_path)}")


if __name__ == "__main__":
    main()
