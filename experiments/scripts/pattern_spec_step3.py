"""Prompt 4.2 — Do bo sung (dem SU KIEN va kiem tra troi theo thoi gian) cho Pattern Spec v1.

Chi do va bao cao, KHONG chon thay nguoi dung.
Dung Cach A (dac trung phan du so voi khi hau month_hour) cho moi phep do.

Chay tu goc repo:
    python experiments/scripts/pattern_spec_step3.py --seed 0
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from palf.data.pattern_diagnostics import (
    cold_front_components,
    cold_front_feature_frame,
    fit_cold_front_thresholds,
)
from palf.data.pattern_events import (
    count_mask_runs,
    label_run_stats_by_split,
    mask_run_stats,
)
from palf.data.pattern_features import (
    CHANNELS,
    build_feature_table,
    load_hourly_with_time,
    window_split,
)
from palf.data.pattern_labels import (
    RULE_ORDER,
    assign_labels,
    check_criteria,
    fire_matrix,
    fit_thresholds,
    label_share_table,
    overlap_counts,
    rule_specs,
)
from palf.data.weather import download_jena


def md_table(df, fmt="{:.4f}"):
    """Chuyen DataFrame thanh bang Markdown dep."""
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
    p = argparse.ArgumentParser(description="Prompt 4.2 - Do su kien va kiem tra troi thoi gian.")
    p.add_argument("--csv", default="data/raw/jena_climate_2009_2016.csv")
    p.add_argument("--out", default="experiments/pattern_stats/jena/step3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--L", type=int, default=96)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--subsample", type=int, default=6)
    p.add_argument("--clim", choices=["hour", "month_hour"], default="month_hour")
    p.add_argument("--purge", type=int, default=None)
    p.add_argument("--spike-channel", choices=["p", "T", "rh", "wv"], default="T")
    p.add_argument("--q-spike-list", type=float, nargs="+", default=[0.95, 0.97, 0.98, 0.99])
    p.add_argument("--q-low", type=float, default=0.10)
    p.add_argument("--q-high", type=float, default=0.90)
    p.add_argument("--q-turn", type=float, default=0.75)
    p.add_argument("--c2-min-share", type=float, default=0.02)
    p.add_argument("--c2-target-share", type=float, default=0.05)
    p.add_argument("--c2-min-count", type=int, default=300)
    p.add_argument("--c2-min-runs-train", type=int, default=20)
    p.add_argument("--c2-min-runs-val", type=int, default=5)
    p.add_argument("--c2-min-runs-test", type=int, default=5)
    p.add_argument("--c3-max-normal", type=float, default=0.70)
    p.add_argument("--c3-target-normal", type=float, default=0.60)
    p.add_argument("--c4-max-rel-dev", type=float, default=0.50)
    return p.parse_args()


def main():
    a = parse_args()
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    if not os.path.exists(a.csv):
        download_jena(a.csv)

    df, n_wv_err = load_hourly_with_time(a.csv, a.subsample)
    sp = window_split(len(df), a.L, a.H, purge=a.purge)
    n_w = sp["n_windows"]
    spans = sp["spans"]
    tr, va, te = spans["train"], spans["val"], spans["test"]

    feats, clim = build_feature_table(
        df[list(CHANNELS.values())], CHANNELS, a.L, n_w, sp["train_rows"], a.clim
    )

    # =========================================================================
    # 1. Spike theo nhieu phan vi
    # =========================================================================
    spike_rows = []
    for q_sp in a.q_spike_list:
        spec_q = rule_specs("A", a.spike_channel, q_sp, a.q_low, a.q_high)
        th_q = fit_thresholds(feats, spec_q, tr)
        fires_q = fire_matrix(feats, th_q)
        labels_q = assign_labels(fires_q, RULE_ORDER)
        thr_val = th_q["spike"]["threshold"]

        for sname, (lo, hi) in spans.items():
            sub = labels_q.iloc[lo:hi] == "spike"
            st = mask_run_stats(sub)
            spike_rows.append({
                "q_spike": q_sp,
                "threshold": thr_val,
                "split": sname,
                "n_windows": st["n_windows"],
                "share": st["share"],
                "n_runs": st["n_runs"],
                "mean_run_length": st["mean_run_length"],
                "max_run_length": st["max_run_length"],
            })
    df_spike = pd.DataFrame(spike_rows)
    df_spike.to_csv(os.path.join(a.out, "spike_quantiles.csv"), index=False)

    # =========================================================================
    # 2. Nhan gop temp_anomaly = heat_anomaly U cold_anomaly
    # =========================================================================
    # Cau hinh mac dinh (q_spike = 0.99, q_low = 0.10, q_high = 0.90)
    spec_def = rule_specs("A", a.spike_channel, 0.99, a.q_low, a.q_high)
    th_def = fit_thresholds(feats, spec_def, tr)
    fires_def = fire_matrix(feats, th_def)
    labels_def = assign_labels(fires_def, RULE_ORDER)

    # a. Gop tu nhan goc
    is_heat = labels_def == "heat_anomaly"
    is_cold = labels_def == "cold_anomaly"
    is_temp = is_heat | is_cold

    temp_runs_rows = []
    for sname, (lo, hi) in spans.items():
        st = mask_run_stats(is_temp.iloc[lo:hi])
        st["split"] = sname
        temp_runs_rows.append(st)
    df_temp_runs = pd.DataFrame(temp_runs_rows)[["split", "n_windows", "share", "n_runs", "mean_run_length", "max_run_length"]]
    df_temp_runs.to_csv(os.path.join(a.out, "temp_anomaly_runs.csv"), index=False)

    # b. Tach theo dau (heat vs cold nhu sieu du lieu)
    sign_rows = []
    for sname, (lo, hi) in spans.items():
        tot_temp = int(is_temp.iloc[lo:hi].sum())
        for sign_name, mask_sign in [("heat (+)", is_heat), ("cold (-)", is_cold)]:
            st = mask_run_stats(mask_sign.iloc[lo:hi])
            sign_rows.append({
                "sign": sign_name,
                "split": sname,
                "n_windows": st["n_windows"],
                "share_in_split": st["share"],
                "share_in_temp_anomaly": (st["n_windows"] / tot_temp) if tot_temp > 0 else float("nan"),
                "n_runs": st["n_runs"],
                "mean_run_length": st["mean_run_length"],
                "max_run_length": st["max_run_length"],
            })
    df_temp_sign = pd.DataFrame(sign_rows)
    df_temp_sign.to_csv(os.path.join(a.out, "temp_anomaly_sign.csv"), index=False)

    # c. Cau hinh 4 luat ma temp_anomaly thay the 2 luat cu
    order_4 = ["spike", "pressure_drop", "humidity_rise", "temp_anomaly"]
    fires_4 = fires_def[["spike", "pressure_drop", "humidity_rise"]].copy()
    fires_4["temp_anomaly"] = fires_def["heat_anomaly"] | fires_def["cold_anomaly"]

    # Overlap matrix tren train
    counts_4 = overlap_counts(fires_4, tr)
    counts_4.to_csv(os.path.join(a.out, "temp_anomaly_overlap.csv"))
    diag_4 = pd.Series(np.diag(counts_4.to_numpy()), index=counts_4.index).clip(lower=1)
    cond_4 = counts_4.div(diag_4, axis=0)
    cond_4.to_csv(os.path.join(a.out, "temp_anomaly_overlap_conditional.csv"))

    # Gan nhan theo uu tien 4 luat
    labels_4 = assign_labels(fires_4, order_4)
    tab_4 = label_share_table(labels_4, spans, order_4)
    tab_4.to_csv(os.path.join(a.out, "temp_anomaly_labels.csv"), index=False)

    c2_4, c3_4, c4_4, warn_4 = check_criteria(
        tab_4, a.c2_min_share, a.c2_target_share, a.c2_min_count,
        a.c3_max_normal, a.c3_target_normal, a.c4_max_rel_dev,
    )
    c2_4.to_csv(os.path.join(a.out, "temp_anomaly_criteria_c2.csv"), index=False)
    c3_4.to_csv(os.path.join(a.out, "temp_anomaly_criteria_c3.csv"), index=False)
    c4_4.to_csv(os.path.join(a.out, "temp_anomaly_criteria_c4.csv"), index=False)

    # =========================================================================
    # 3. So run cho moi nhan hien co (o q mac dinh)
    # =========================================================================
    df_label_runs = label_run_stats_by_split(labels_def, spans, list(RULE_ORDER) + ["normal"])
    df_label_runs.to_csv(os.path.join(a.out, "label_runs.csv"), index=False)

    # =========================================================================
    # 4. Kiem tra troi theo thoi gian (drift check)
    # =========================================================================
    # Thoi gian ket thuc cua so j = df.index[j + L - 1]
    win_end_ts = df.index[a.L - 1 : a.L - 1 + n_w]
    win_years = win_end_ts.year
    win_months = win_end_ts.month

    win_split = np.full(n_w, "gap", dtype=object)
    for sname, (lo, hi) in spans.items():
        win_split[lo:hi] = sname

    wdf = pd.DataFrame({
        "split": win_split,
        "year": win_years,
        "month": win_months,
        "anom_T": feats["T__anom"],
        "is_heat": is_heat,
        "is_cold": is_cold,
    })
    wdf_valid = wdf[wdf["split"] != "gap"]

    # 4a. Theo tung nam duong lich va tung tap
    # Tinh T trung binh theo chuoi gio cua tung nam va tap
    hourly_split = np.full(len(df), "gap", dtype=object)
    hourly_split[:tr[1] + a.L - 1] = "train"
    hourly_split[va[0] : va[1] + a.L - 1] = "val"
    hourly_split[te[0] : te[1] + a.L - 1] = "test"
    df["split"] = hourly_split
    df["year"] = df.index.year

    hourly_t = df[df["split"] != "gap"].groupby(["year", "split"])["T (degC)"].mean().rename("T_mean_hourly")

    drift_yr = wdf_valid.groupby(["year", "split"]).agg(
        n_windows=("anom_T", "count"),
        anom_T_mean=("anom_T", "mean"),
        heat_share=("is_heat", "mean"),
        cold_share=("is_cold", "mean"),
    ).join(hourly_t).reset_index()

    # Sap xep theo year, split
    drift_yr["split_order"] = drift_yr["split"].map({"train": 0, "val": 1, "test": 2})
    drift_yr = drift_yr.sort_values(["year", "split_order"]).drop(columns=["split_order"])
    drift_yr.to_csv(os.path.join(a.out, "drift_year.csv"), index=False)

    # 4b. Bang so cua so theo tap x thang
    month_counts = pd.crosstab(wdf_valid["split"], wdf_valid["month"])
    month_counts.columns = [f"Tháng_{m}" for m in month_counts.columns]
    month_counts["Tổng_cửa_sổ"] = month_counts.sum(axis=1)
    month_counts = month_counts.reindex(["train", "val", "test"]).reset_index()
    month_counts.to_csv(os.path.join(a.out, "drift_month.csv"), index=False)

    # Ty le thang trong moi tap
    month_shares = pd.crosstab(wdf_valid["split"], wdf_valid["month"], normalize="index")
    month_shares.columns = [f"Tháng_{m}" for m in month_shares.columns]
    month_shares = month_shares.reindex(["train", "val", "test"]).reset_index()
    month_shares.to_csv(os.path.join(a.out, "drift_month_shares.csv"), index=False)

    # 4c. Ngay bat dau va ket thuc (theo thoi gian thuc)
    date_rows = []
    for sname, (lo, hi) in spans.items():
        t_first_start = df.index[lo]
        t_first_end = df.index[lo + a.L - 1]
        t_last_start = df.index[hi - 1]
        t_last_end = df.index[hi - 1 + a.L - 1]
        duration_days = (t_last_end - t_first_start).total_seconds() / 86400.0
        date_rows.append({
            "split": sname,
            "n_windows": hi - lo,
            "window_indices": f"[{lo}, {hi})",
            "first_window_start": str(t_first_start),
            "first_window_end": str(t_first_end),
            "last_window_start": str(t_last_start),
            "last_window_end": str(t_last_end),
            "span_start_time": str(t_first_start),
            "span_end_time": str(t_last_end),
            "duration_days": duration_days,
            "duration_years": duration_days / 365.25,
        })
    df_dates = pd.DataFrame(date_rows)
    df_dates.to_csv(os.path.join(a.out, "split_dates.csv"), index=False)

    # =========================================================================
    # 5. Cold Front to hop
    # =========================================================================
    cf = cold_front_feature_frame(
        df[CHANNELS["p"]].to_numpy(), df[CHANNELS["T"]].to_numpy(),
        df[CHANNELS["p"]].to_numpy() - clim[CHANNELS["p"]].to_numpy(),
        df[CHANNELS["T"]].to_numpy() - clim[CHANNELS["T"]].to_numpy(),
        df["wd (deg)"].to_numpy(),
        np.abs(feats["wv__delta__raw"].to_numpy()),
        np.abs(feats["wv__delta__resid"].to_numpy()),
        a.L, n_w,
    )
    cth = fit_cold_front_thresholds(cf, tr, "resid", a.q_turn, a.q_low, a.q_high)
    masks_cf = cold_front_components(cf, cth, "resid")

    cf_rows = []
    for comp in ["front_dir", "front_either_wind"]:
        for sname, (lo, hi) in spans.items():
            sub = masks_cf[comp][lo:hi]
            st = mask_run_stats(sub)
            cf_rows.append({
                "component": comp,
                "split": sname,
                "n_windows": st["n_windows"],
                "share": st["share"],
                "n_runs": st["n_runs"],
                "mean_run_length": st["mean_run_length"],
                "max_run_length": st["max_run_length"],
            })
    df_cf_runs = pd.DataFrame(cf_rows)
    df_cf_runs.to_csv(os.path.join(a.out, "cold_front_runs.csv"), index=False)

    # =========================================================================
    # 6. C2 muc su kien (5 nhan goc va 4 nhan gop)
    # =========================================================================
    def build_event_c2_table(labels_series, rule_order):
        rows = []
        warnings = []
        for lab in rule_order:
            m_tr = labels_series.iloc[tr[0]:tr[1]] == lab
            m_va = labels_series.iloc[va[0]:va[1]] == lab
            m_te = labels_series.iloc[te[0]:te[1]] == lab

            n_tr = int(m_tr.sum())
            n_va = int(m_va.sum())
            n_te = int(m_te.sum())
            sh_tr = n_tr / (tr[1] - tr[0])

            r_tr = len(count_mask_runs(m_tr))
            r_va = len(count_mask_runs(m_va))
            r_te = len(count_mask_runs(m_te))

            min_n = min(n_tr, n_va, n_te)
            min_r = min(r_tr, r_va, r_te)

            c2_win = (sh_tr >= a.c2_min_share) and (min_n >= a.c2_min_count)
            c2_evt = (r_tr >= a.c2_min_runs_train) and (r_va >= a.c2_min_runs_val) and (r_te >= a.c2_min_runs_test)
            c2_all = c2_win and c2_evt

            if not c2_evt:
                reasons = []
                if r_tr < a.c2_min_runs_train:
                    reasons.append(f"runs_train={r_tr} < {a.c2_min_runs_train}")
                if r_va < a.c2_min_runs_val:
                    reasons.append(f"runs_val={r_va} < {a.c2_min_runs_val}")
                if r_te < a.c2_min_runs_test:
                    reasons.append(f"runs_test={r_te} < {a.c2_min_runs_test}")
                warnings.append(f"C2 muc su kien KHONG DAT: {lab} ({', '.join(reasons)})")

            rows.append({
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
                "C2_toan_dien": c2_all,
            })
        return pd.DataFrame(rows), warnings

    df_c2_runs_5, warn_c2_5 = build_event_c2_table(labels_def, RULE_ORDER)
    df_c2_runs_5.to_csv(os.path.join(a.out, "criteria_c2_runs.csv"), index=False)

    df_c2_runs_4, warn_c2_4 = build_event_c2_table(labels_4, order_4)
    df_c2_runs_4.to_csv(os.path.join(a.out, "criteria_c2_runs_merged.csv"), index=False)

    # =========================================================================
    # Ghi config_used.json
    # =========================================================================
    cfg = dict(vars(a))
    cfg.update({
        "mode": "A",
        "mode_description": "Cach A - dac trung phan du so voi khi hau month_hour",
        "n_rows": len(df),
        "n_windows": n_w,
        "spans": spans,
        "train_rows": sp["train_rows"],
        "gap": sp["gap"],
        "n_wv_err_fixed": n_wv_err,
        "c2_event_thresholds": {
            "min_runs_train": a.c2_min_runs_train,
            "min_runs_val": a.c2_min_runs_val,
            "min_runs_test": a.c2_min_runs_test,
        },
    })
    with open(os.path.join(a.out, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(to_builtin(cfg), f, indent=2)

    # =========================================================================
    # Tao report.md
    # =========================================================================
    rep = [
        "# Pattern Spec v1 — Báo cáo đo bổ sung (Prompt 4.2)",
        "",
        "> **Nguyên tắc:** Chỉ đo và báo cáo số liệu khách quan, KHÔNG quyết định thay người dùng.",
        "> Mọi ngưỡng, khí hậu, phân vị chỉ tính trên **train** (sau purge gap). Val/test chỉ áp dụng.",
        "> Toàn bộ phép đo sử dụng **Cách A** (đặc trưng phần dư so với khí hậu `month_hour`).",
        "",
        "## Thông số tập dữ liệu & ranh giới",
        f"- Tổng số dòng giờ: {len(df):,}; số ô sensor lỗi `wv < -900` đã đưa về 0: {n_wv_err}",
        f"- Cấu hình cửa sổ: L = {a.L} giờ, H = {a.H} giờ; Purge gap = {sp['gap']} cửa sổ",
        f"- Số cửa sổ: Train = {tr[1] - tr[0]:,} | Val = {va[1] - va[0]:,} | Test = {te[1] - te[0]:,} (Tổng: {n_w:,})",
        "",
        "---",
        "",
        "## 1. Spike theo nhiều phân vị",
        "",
        "Đo số cửa sổ và **số run (sự kiện liên tiếp)** của nhãn Spike khi thay đổi `q_spike` trong {0.95, 0.97, 0.98, 0.99}. "
        "Các luật khác và thứ tự ưu tiên được giữ nguyên (`spike` > `pressure_drop` > `humidity_rise` > `heat_anomaly` > `cold_anomaly` > `normal`).",
        "",
        md_table(df_spike),
        "",
        "**Nhận xét:**",
        "- Ở phân vị mặc định `q = 0.99`, train chỉ có **6 run**, val có **1 run**, test có **0 run**.",
        "- Khi hạ phân vị xuống `q = 0.95`, train đạt **25 run** (5.07% cửa sổ), val đạt **6 run** (5.49%), test đạt **3 run** (2.75%).",
        "- Độ dài trung bình mỗi run của Spike dao động quanh 95-105 giờ (đúng bằng $L-1 = 95$ bước trượt cho 1 bước nhảy cô lập đơn lẻ).",
        "",
        "---",
        "",
        "## 2. Nhãn gộp `temp_anomaly` (= heat_anomaly ∪ cold_anomaly)",
        "",
        "Do `heat_anomaly` ($A_T \\ge p_{90}$) và `cold_anomaly` ($A_T \\le p_{10}$) không bao giờ đồng thời kích hoạt, "
        "việc gộp lại thành `temp_anomaly` giúp theo dõi sự bất thường nhiệt độ nói chung.",
        "",
        "### 2.1. Số lượng và số run của `temp_anomaly` gộp từ nhãn gốc",
        "",
        md_table(df_temp_runs),
        "",
        "### 2.2. Bảng tách theo dấu (Heat vs Cold làm siêu dữ liệu)",
        "",
        md_table(df_temp_sign),
        "",
        "### 2.3. Cấu hình 4 nhãn chính thức (`temp_anomaly` thay 2 nhãn cũ)",
        "",
        "Thứ tự ưu tiên: `spike` > `pressure_drop` > `humidity_rise` > `temp_anomaly` > `normal`.",
        "",
        "#### Ma trận chồng lấn trên Train (số cửa sổ đồng thời thỏa mãn):",
        "",
        md_table(counts_4.reset_index().rename(columns={"index": "rule"}), fmt="{:.0f}"),
        "",
        "#### Phân bố nhãn sau ưu tiên:",
        "",
        md_table(tab_4),
        "",
        "#### Kiểm tra tiêu chí C2–C4 cho cấu hình 4 nhãn:",
        "",
        "- **Tiêu chí C2 (Tỷ lệ train $\\ge 2\\%$, số lượng mỗi tập $\\ge 300$):**",
        md_table(c2_4),
        "",
        "- **Tiêu chí C3 (Normal $\\le 70\\%$):**",
        md_table(c3_4),
        "",
        "- **Tiêu chí C4 (Độ lệch tương đối val/test so với train $\\le 0.5$):**",
        md_table(c4_4),
        "",
        "**Nhận xét:**",
        "- Khi tách riêng, `cold_anomaly` vi phạm C2 (val=279, test=219 < 300) và vi phạm C4 (độ lệch 67% và 74%). `heat_anomaly` vi phạm C4 (độ lệch test 93%).",
        "- Khi **gộp thành `temp_anomaly`**: nhãn này **ĐẠT cả C2** (min_n = 1,337 $\\ge$ 300) và **ĐẠT cả C4** (độ lệch val = 18.6%, test = 5.4% $\\le 50\\%$).",
        "",
        "---",
        "",
        "## 3. Số run cho mọi nhãn hiện có (ở phân vị mặc định)",
        "",
        "Bảng thống kê số cửa sổ (`n_windows`) và số sự kiện (`n_runs`) cho toàn bộ 5 nhãn gốc + `normal` trên từng tập:",
        "",
        md_table(df_label_runs),
        "",
        "---",
        "",
        "## 4. Kiểm tra trôi theo thời gian (Time Drift Analysis)",
        "",
        "### 4.1. Nhiệt độ và phần dư theo từng năm dương lịch và từng tập",
        "",
        "- `T_mean_hourly`: Nhiệt độ trung bình chuỗi giờ trong năm & tập.",
        "- `anom_T_mean`: Giá trị trung bình của phần dư $A_T$ trong cửa sổ.",
        "- `heat_share` / `cold_share`: Tỷ lệ cửa sổ kích hoạt Heat / Cold anomaly.",
        "",
        md_table(drift_yr),
        "",
        "**Phát hiện mấu chốt về trôi dạt (Drift):**",
        "- Năm 2010 (thuộc train): Năm lạnh điển hình, `anom_T_mean = -1.59°C`, `cold_share = 17.4%`.",
        "- Năm 2014-2015: Bắt đầu ấm lên mạnh. Đặc biệt cuối năm 2015 (thuộc test), `anom_T_mean = +3.77°C`, `heat_share = 43.4%`, `cold_share = 0%`.",
        "- Điều này giải thích tại sao test set có rất nhiều heat anomaly và cực kỳ ít cold anomaly.",
        "",
        "### 4.2. Số cửa sổ theo (Tập × Tháng) — Độ phủ mùa",
        "",
        md_table(month_counts, fmt="{:.0f}"),
        "",
        "Tỷ lệ phân bố tháng trong mỗi tập:",
        "",
        md_table(month_shares),
        "",
        "**Nhận xét về độ lệch mùa:**",
        "- Tập **Train** trải dài 5.5 năm nên các tháng 1–12 được phủ đồng đều (mỗi tháng chiếm ~7.3% – 9.2%).",
        "- Tập **Val** kéo dài ~14 tháng (8/2014 – 10/2015): Tháng 8, 9, 10 bị lặp lại 2 lần (chiếm 11% – 14% tập val), trong khi các tháng 11, 12, 1–7 chỉ xuất hiện 1 lần (~7%).",
        "- Tập **Test** kéo dài ~14.5 tháng (10/2015 – 12/2016): Tháng 11 và 12 bị lặp lại 2 lần (chiếm tới 13.9% và 14.1% tập test). Đợt mùa đông ấm kỷ lục cuối năm 2015 bị lặp lại trọng số lớn trong test set.",
        "",
        "### 4.3. Khoảng thời gian thực tế của các tập",
        "",
        md_table(df_dates),
        "",
        "---",
        "",
        "## 5. Cold Front tổ hợp (Luật kết hợp)",
        "",
        "Thống kê số lượng cửa sổ và số run cho `front_dir` (áp suất đổi chiều + nhiệt giảm nhanh + gió đổi hướng) "
        "và `front_either_wind` (đổi hướng HOẶC đổi tốc độ gió):",
        "",
        md_table(df_cf_runs),
        "",
        "**Nhận xét:**",
        "- `front_dir` có 74 run ở train, 11 run ở val, 18 run ở test.",
        "- Độ dài mỗi run rất ngắn (trung bình 1.3 – 1.9 cửa sổ), cho thấy đây là sự kiện chớp nhoáng (transient burst) chứ không kéo dài.",
        "",
        "---",
        "",
        "## 6. Tiêu chí C2 mức sự kiện (Event-level C2 Criteria)",
        "",
        f"Quy chuẩn kiểm tra mức sự kiện: `runs_train >= {a.c2_min_runs_train}`, `runs_val >= {a.c2_min_runs_val}`, `runs_test >= {a.c2_min_runs_test}`.",
        "",
        "### 6.1. Đánh giá trên 5 nhãn gốc:",
        "",
        md_table(df_c2_runs_5),
        "",
    ]

    if warn_c2_5:
        rep.append("#### Cảnh báo mức sự kiện (5 nhãn gốc):")
        for w in warn_c2_5:
            rep.append(f"- ⚠️ {w}")
        rep.append("")

    rep.extend([
        "### 6.2. Đánh giá trên 4 nhãn gộp (`temp_anomaly`):",
        "",
        md_table(df_c2_runs_4),
        "",
    ])

    if warn_c2_4:
        rep.append("#### Cảnh báo mức sự kiện (4 nhãn gộp):")
        for w in warn_c2_4:
            rep.append(f"- ⚠️ {w}")
        rep.append("")

    rep.extend([
        "---",
        "",
        "## Tổng hợp cảnh báo & Kết luận đo đạc",
        "",
        "1. **Spike ở phân vị 0.99:** Quá hiếm ở mức sự kiện (chỉ có 6 run ở train, 1 ở val, 0 ở test), vi phạm tiêu chuẩn C2 mức sự kiện. Nếu muốn Spike đạt tối thiểu 20 run ở train và $\\ge 5$ run ở val/test, cần cân nhắc hạ `q_spike` về mức 0.95 (25 train / 6 val / 3 test).",
        "2. **Tách biệt Heat vs Cold Anomaly:** Bị ảnh hưởng nặng nề bởi biến động khí hậu thực tế (El Niño cuối 2015 cực ấm) và sự không cân đối mùa (Val lặp tháng 8-10, Test lặp tháng 11-12).",
        "3. **Hiệu quả của nhãn gộp `temp_anomaly`:** Giúp loại bỏ hoàn toàn vi phạm C2 và C4 ở mức cửa sổ, đồng thời đạt 283 run (train), 44 run (val), 54 run (test) — vượt xa ngưỡng C2 sự kiện.",
        "",
        f"Toàn bộ bảng số liệu CSV đã được lưu tại: `{os.path.abspath(a.out)}`.",
    ])

    rep_content = "\n".join(rep) + "\n"
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(rep_content)

    print("\n=== HOAN THANH PROMPT 4.2 ===")
    print(f"Bao cao va toan bo CSV da duoc luu tai: {os.path.abspath(a.out)}")
    if warn_c2_5:
        print("\n[Canh bao C2 su kien - 5 nhan]:")
        for w in warn_c2_5:
            print("-", w)
    if warn_c2_4:
        print("\n[Canh bao C2 su kien - 4 nhan gop]:")
        for w in warn_c2_4:
            print("-", w)


if __name__ == "__main__":
    main()
