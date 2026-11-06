"""Pattern Spec v1: do va bao cao Jena. Khong chon thay nguoi dung.

Chay tu goc repo, sau `conda activate palf`:

    python experiments/scripts/pattern_spec_report.py --seed 0
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from palf.data.pattern_diagnostics import (
    assign_max_z, assignment_diff, cold_front_components, cold_front_feature_frame,
    component_counts, fit_cold_front_thresholds, iter_sensitivity_specs, pearson,
    run_length_summary, sample_window_indices,
)
from palf.data.pattern_features import (
    CHANNELS, FEATURES, QUANTILES, build_feature_table, load_hourly_with_time,
    variance_explained, window_split,
)
from palf.data.pattern_labels import (
    RULE_ORDER, add_raw_mean, assign_labels, check_criteria, fire_matrix,
    fit_thresholds, label_share_table, overlap_counts, rule_specs,
)
from palf.data.weather import download_jena

MODES = (
    ("A", "resid", "Cach A - dac trung phan du so voi khi hau"),
    ("B", "raw", "Cach B - dac trung tho"),
)
LABELS = list(RULE_ORDER) + ["normal"]


def md_table(df, fmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (bool, np.bool_)):
                cells.append(str(bool(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else fmt.format(float(v)))
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


def plot_feature_hists(ftr, short, col, path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    for ax, f in zip(axes[:3], FEATURES):
        for mode in ("raw", "resid"):
            ax.hist(ftr[f"{short}__{f}__{mode}"].dropna(), bins=80, alpha=0.55,
                    density=True, label=mode)
        ax.set_title(f"{col} | {f}")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    axes[3].hist(ftr[f"{short}__anom"].dropna(), bins=80, density=True, color="C2")
    axes[3].set_title(f"{col} | A (anom)")
    axes[3].set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_samples(df, columns, indices, L, path, title):
    indices = [int(j) for j in indices]
    if not indices:
        return False
    fig, axes = plt.subplots(len(indices), len(columns),
                             figsize=(3.15 * len(columns), 1.45 * len(indices)),
                             sharex=True, squeeze=False)
    t = np.arange(L)
    for r, j in enumerate(indices):
        block = df.iloc[j:j + L]
        end = df.index[j + L - 1]
        for c, col in enumerate(columns):
            ax = axes[r, c]
            ax.plot(t, block[col].to_numpy(), lw=0.9)
            if r == 0:
                ax.set_title(col, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{j}\n{end:%Y-%m-%d %H}", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def sensitivity_frame(feats, spans, priority, mode, spike_channel, q_spike, q_low, q_high, steps):
    base_share = None
    rows = []
    for kind, rule, dpp, spec, clipped, q_used in iter_sensitivity_specs(
            mode, spike_channel, q_spike, q_low, q_high, priority, steps):
        th = fit_thresholds(feats, spec, spans["train"])
        labels = assign_labels(fire_matrix(feats, th), priority)
        tab = label_share_table(labels, spans, priority).set_index("label")
        if base_share is None:
            base_share = tab
        for label in LABELS:
            rec = {
                "mode": mode, "kind": kind, "rule": rule, "delta_pp": int(dpp),
                "q_used": float(q_used) if q_used is not None else float("nan"),
                "clipped": bool(clipped), "label": label,
                "q_spike": float(spec["spike"][2]),
                "q_pressure": float(spec["pressure_drop"][2]),
                "q_humidity": float(spec["humidity_rise"][2]),
                "q_heat": float(spec["heat_anomaly"][2]),
                "q_cold": float(spec["cold_anomaly"][2]),
            }
            for split in spans:
                share = float(tab.loc[label, f"share_{split}"])
                rec[f"share_{split}"] = share
                rec[f"d_share_{split}"] = share - float(base_share.loc[label, f"share_{split}"])
            rows.append(rec)
    return pd.DataFrame(rows)


def compact_sensitivity(sens):
    sub = sens[sens["kind"] != "baseline"].copy()
    wide = sub.pivot_table(
        index=["mode", "kind", "rule", "delta_pp", "clipped"],
        columns="label", values="d_share_train", aggfunc="first",
    )
    wide = wide.reindex(columns=LABELS)
    wide = wide.reset_index()
    return wide.sort_values(["mode", "kind", "rule", "delta_pp"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Do Pattern Spec v1 tren Jena. Khong chon thay nguoi dung.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--subsample", type=int, default=6)
    ap.add_argument("--clim", choices=["hour", "month_hour"], default="month_hour")
    ap.add_argument("--purge", type=int, default=None)
    ap.add_argument("--spike-channel", default="T", choices=list(CHANNELS))
    ap.add_argument("--q-spike", type=float, default=0.99)
    ap.add_argument("--q-low", type=float, default=0.10)
    ap.add_argument("--q-high", type=float, default=0.90)
    ap.add_argument("--priority", nargs="+", default=list(RULE_ORDER), choices=RULE_ORDER)
    ap.add_argument("--q-turn", type=float, default=0.75,
                    help="Phan vi train cua min(|d1|,|d2|) ap suat cho Cold Front.")
    ap.add_argument("--q-cool", type=float, default=None, help="Mac dinh = --q-low.")
    ap.add_argument("--q-wind", type=float, default=None, help="Mac dinh = --q-high.")
    ap.add_argument("--c2-min-share", type=float, default=0.02)
    ap.add_argument("--c2-target-share", type=float, default=0.05)
    ap.add_argument("--c2-min-count", type=int, default=300)
    ap.add_argument("--c3-max-normal", type=float, default=0.70)
    ap.add_argument("--c3-target-normal", type=float, default=0.60)
    ap.add_argument("--c4-max-rel-dev", type=float, default=0.5)
    ap.add_argument("--sens-pp", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/pattern_stats/jena")
    a = ap.parse_args()
    q_cool = a.q_low if a.q_cool is None else a.q_cool
    q_wind = a.q_high if a.q_wind is None else a.q_wind
    priority = list(a.priority)

    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    raw_df, n_wv_err = load_hourly_with_time(a.csv or download_jena(), a.subsample)
    wd_col = next(c for c in raw_df.columns if c.lower().startswith("wd"))
    missing = [c for c in CHANNELS.values() if c not in raw_df.columns]
    if missing:
        raise SystemExit(f"Thieu cot: {missing}")
    df = raw_df[list(CHANNELS.values())].copy()
    wd = raw_df[wd_col]
    if (df.filter(like="wv") < 0).any().any():
        raise SystemExit("Con gia tri gio am sau khi sua loi cam bien.")

    sp = window_split(len(df), a.L, a.H, purge=a.purge)
    n_w, spans = sp["n_windows"], sp["spans"]
    tr, va, te = spans["train"], spans["val"], spans["test"]
    feats, clim = build_feature_table(df, CHANNELS, a.L, n_w, sp["train_rows"], a.clim)
    feats = add_raw_mean(feats, df, CHANNELS, a.L, n_w)
    ftr = feats.iloc[tr[0]:tr[1]]
    train_df = df.iloc[:sp["train_rows"]]

    # --- 1. Chu ky ---
    cyc_rows = []
    for s, col in CHANNELS.items():
        x = train_df[col]
        cyc_rows.append({
            "channel": col,
            "r2_hour": variance_explained(x.to_numpy(), x.index.hour),
            "r2_month_hour": variance_explained(x.to_numpy(), x.index.month, x.index.hour),
            "acf_lag24": float(x.autocorr(24)),
            "acf_lag96": float(x.autocorr(96)),
            "corr_delta_raw_end_hour": pearson(ftr[f"{s}__delta__raw"], ftr["end_hour"]),
            "eta2_delta_raw": variance_explained(ftr[f"{s}__delta__raw"].to_numpy(),
                                                 ftr["end_hour"].to_numpy()),
            "eta2_delta_resid": variance_explained(ftr[f"{s}__delta__resid"].to_numpy(),
                                                   ftr["end_hour"].to_numpy()),
        })
    cyc = pd.DataFrame(cyc_rows)

    cmp_rows = []
    for s, col in CHANNELS.items():
        r, e = ftr[f"{s}__delta__raw"], ftr[f"{s}__delta__resid"]
        gap = r - e
        cmp_rows.append({
            "channel": col,
            "std_raw": float(r.std()), "std_resid": float(e.std()),
            "std_ratio": float(e.std() / r.std()),
            "std_raw_minus_resid": float(gap.std()),
            "clim_std_ratio": float(gap.std() / r.std()),
            "corr_raw_resid": pearson(r, e),
            "iqr_raw": float(r.quantile(0.75) - r.quantile(0.25)),
            "iqr_resid": float(e.quantile(0.75) - e.quantile(0.25)),
            "p01_raw": float(r.quantile(0.01)), "p99_raw": float(r.quantile(0.99)),
            "p01_resid": float(e.quantile(0.01)), "p99_resid": float(e.quantile(0.99)),
        })
    cmp_df = pd.DataFrame(cmp_rows)

    # --- 2. Phan vi ---
    q_rows = []
    for s, col in CHANNELS.items():
        combos = [(f, m) for f in FEATURES for m in ("raw", "resid")] + [("anom", "resid")]
        for f, m in combos:
            key = f"{s}__anom" if f == "anom" else f"{s}__{f}__{m}"
            v = ftr[key]
            row = {"channel": col, "feature": f, "mode": m}
            row.update({f"p{int(round(q * 100)):02d}": float(v.quantile(q)) for q in QUANTILES})
            q_rows.append(row)
        plot_feature_hists(ftr, s, col, os.path.join(fig_dir, f"feat_hist_{s}.png"))
    qdf = pd.DataFrame(q_rows)

    # --- 3-5, 7-10 theo tung cach ---
    cf = cold_front_feature_frame(
        df[CHANNELS["p"]].to_numpy(), df[CHANNELS["T"]].to_numpy(),
        df[CHANNELS["p"]].to_numpy() - clim[CHANNELS["p"]].to_numpy(),
        df[CHANNELS["T"]].to_numpy() - clim[CHANNELS["T"]].to_numpy(),
        wd.to_numpy(),
        np.abs(feats["wv__delta__raw"].to_numpy()),
        np.abs(feats["wv__delta__resid"].to_numpy()),
        a.L, n_w,
    )
    if not np.allclose(cf["t_delta__raw"], feats["T__delta__raw"], equal_nan=True):
        raise RuntimeError("Delta nhiet tho lech khoi dac trung cua so")
    if not np.allclose(cf["t_delta__resid"], feats["T__delta__resid"], equal_nan=True):
        raise RuntimeError("Delta nhiet phan du lech khoi dac trung cua so")

    all_warn = []
    rule_thresholds = {}
    z_stats = {}
    n_ties = {}
    sens_parts = []
    run_parts = []
    run_all_parts = []
    front_parts = []
    front_thr_rows = []
    sample_rows = []
    mode_blocks = []

    for mode, tag, title in MODES:
        spec = rule_specs(mode, a.spike_channel, a.q_spike, a.q_low, a.q_high)
        th = fit_thresholds(feats, spec, spans["train"])
        fires = fire_matrix(feats, th)
        labels = assign_labels(fires, priority)
        z_labels, _, stats, ties = assign_max_z(feats, th, spans["train"], priority)
        tab = label_share_table(labels, spans, priority)
        c2, c3, c4, warn = check_criteria(
            tab, a.c2_min_share, a.c2_target_share, a.c2_min_count,
            a.c3_max_normal, a.c3_target_normal, a.c4_max_rel_dev,
        )
        counts = overlap_counts(fires, spans["train"])
        diag = pd.Series(np.diag(counts.to_numpy()), index=counts.index).clip(lower=1)
        cond = counts.div(diag, axis=0)
        firing_rows = []
        for name in th:
            row = {"rule": name}
            for sname, (lo, hi) in spans.items():
                row[f"share_{sname}"] = float(fires[name].iloc[lo:hi].mean())
            firing_rows.append(row)
        firing = pd.DataFrame(firing_rows)
        th_df = pd.DataFrame(th).T.reset_index().rename(columns={"index": "rule"})
        diff = assignment_diff(labels, z_labels, spans, fires, priority)
        diff.insert(0, "mode", mode)
        diff.insert(1, "n_z_ties_all_windows", ties)
        per_run, all_run = run_length_summary(labels, spans, LABELS)
        per_run.insert(0, "mode", mode)
        all_run.insert(0, "mode", mode)
        sens = sensitivity_frame(
            feats, spans, priority, mode, a.spike_channel,
            a.q_spike, a.q_low, a.q_high, a.sens_pp,
        )
        cth = fit_cold_front_thresholds(cf, spans["train"], tag, a.q_turn, q_cool, q_wind)
        masks = cold_front_components(cf, cth, tag)
        front = component_counts(masks, spans)
        front.insert(0, "mode", mode)

        rule_thresholds[mode] = th
        z_stats[mode] = stats
        n_ties[mode] = ties
        sens_parts.append(sens)
        run_parts.append(per_run)
        run_all_parts.append(all_run)
        front_parts.append(front)
        front_thr_rows.append({"mode": mode, **cth})
        all_warn.extend([f"[{mode}] {w}" for w in warn])

        picked = sample_window_indices(labels, spans["train"], LABELS, a.n_samples, a.seed)
        for name, idxs in picked.items():
            path = os.path.join(fig_dir, f"samples_{mode}_{name}.png")
            plot_samples(df, list(CHANNELS.values()), idxs, a.L, path,
                         f"Cach {mode} | {name} | train | seed {a.seed}")
            for j in idxs:
                sample_rows.append({
                    "mode": mode, "label": name, "window": int(j),
                    "end_time": str(df.index[int(j) + a.L - 1]),
                    "figure": os.path.relpath(path, a.out).replace("\\", "/"),
                })

        th_df.to_csv(os.path.join(a.out, f"thresholds_{mode}.csv"), index=False)
        firing.to_csv(os.path.join(a.out, f"rule_firing_{mode}.csv"), index=False)
        tab.to_csv(os.path.join(a.out, f"label_counts_{mode}.csv"), index=False)
        counts.to_csv(os.path.join(a.out, f"overlap_counts_{mode}.csv"))
        cond.to_csv(os.path.join(a.out, f"overlap_conditional_{mode}.csv"))
        c2.to_csv(os.path.join(a.out, f"criteria_c2_{mode}.csv"), index=False)
        c3.to_csv(os.path.join(a.out, f"criteria_c3_{mode}.csv"), index=False)
        c4.to_csv(os.path.join(a.out, f"criteria_c4_{mode}.csv"), index=False)
        diff.to_csv(os.path.join(a.out, f"priority_vs_maxz_{mode}.csv"), index=False)

        block = [
            f"## {title}", "",
            "Thu tu uu tien: " + " > ".join(priority) + " > normal", "",
            "### 3. Nguong va ty le kich hoat (truoc uu tien)", "",
            md_table(th_df), "",
            md_table(firing), "",
            "### 4. Ma tran chong lan tren train (so cua so; duong cheo = so cua so cua luat)", "",
            md_table(counts.reset_index().rename(columns={"index": "rule"}), "{:.0f}"), "",
            "P(cot | hang) tren train:", "",
            md_table(cond.reset_index().rename(columns={"index": "rule"}), "{:.3f}"), "",
            "### 5. Nhan sau uu tien, va tieu chi C2-C4", "",
            md_table(tab), "",
            md_table(c2), "",
            md_table(c3), "",
            md_table(c4), "",
        ]
        if warn:
            block += ["Canh bao:"] + [f"- {w}" for w in warn] + [""]
        else:
            block += ["Khong co tieu chi C2-C4 nao khong dat o cach nay.", ""]
        block += [
            "### 7. Uu tien co dinh va diem z lon nhat", "",
            f"So cua so hoa z (ca chuoi, hoa thi giu luat dung truoc trong priority): {ties}", "",
            md_table(diff.drop(columns=["mode"])), "",
        ]
        mode_blocks.append("\n".join(block))
        print(f"\n===== {title} =====")
        print(tab.round(4).to_string(index=False))
        for w in warn:
            print("[CANH BAO]", w)

    sens_df = pd.concat(sens_parts, ignore_index=True)
    sens_wide = compact_sensitivity(sens_df)
    runs = pd.concat(run_parts, ignore_index=True)
    runs_all = pd.concat(run_all_parts, ignore_index=True)
    fronts = pd.concat(front_parts, ignore_index=True)
    front_thr = pd.DataFrame(front_thr_rows)
    samples = pd.DataFrame(sample_rows)

    # --- Ghi bang ---
    cyc.to_csv(os.path.join(a.out, "cycle_strength.csv"), index=False)
    cmp_df.to_csv(os.path.join(a.out, "delta_raw_vs_resid.csv"), index=False)
    qdf.to_csv(os.path.join(a.out, "feature_quantiles.csv"), index=False)
    train_df.groupby([train_df.index.month, train_df.index.hour]).mean().to_csv(
        os.path.join(a.out, "climatology_month_hour.csv"))
    fronts.to_csv(os.path.join(a.out, "cold_front_counts.csv"), index=False)
    front_thr.to_csv(os.path.join(a.out, "cold_front_thresholds.csv"), index=False)
    sens_df.to_csv(os.path.join(a.out, "threshold_sensitivity.csv"), index=False)
    sens_wide.to_csv(os.path.join(a.out, "threshold_sensitivity_train.csv"), index=False)
    runs.to_csv(os.path.join(a.out, "run_lengths.csv"), index=False)
    runs_all.to_csv(os.path.join(a.out, "run_lengths_overall.csv"), index=False)
    samples.to_csv(os.path.join(a.out, "sample_indices.csv"), index=False)
    pd.concat([pd.read_csv(os.path.join(a.out, f"priority_vs_maxz_{m}.csv")) for m, _, _ in MODES],
              ignore_index=True).to_csv(os.path.join(a.out, "priority_vs_maxz.csv"), index=False)

    hour_shift = (a.L - 1) % 24
    thr_doc = {
        "spans": spans,
        "train_rows": sp["train_rows"],
        "gap": sp["gap"],
        "clim": a.clim,
        "feature_quantiles_train": {
            f"{r['channel']}|{r['feature']}|{r['mode']}":
                {k: v for k, v in r.items() if k.startswith("p") and k[1:].isdigit()}
            for r in qdf.to_dict("records")
        },
        "rules": rule_thresholds,
        "cold_front": front_thr.to_dict("records"),
        "zscore_train": z_stats,
        "n_z_ties": n_ties,
    }
    with open(os.path.join(a.out, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(to_builtin(thr_doc), f, indent=2)
    cfg = dict(vars(a))
    cfg.update({
        "q_cool": q_cool, "q_wind": q_wind,
        "n_rows": len(df), "n_windows": n_w, "gap": sp["gap"], "spans": spans,
        "train_rows": sp["train_rows"],
        "n_wv_cells_below_-900_set_to_zero": n_wv_err,
        "wd_column": wd_col,
        "statistics": "physical units after wv<-900 set to 0; quantiles, climatology, z from train only",
        "features_use_y": False,
        "z_tie_break": "earlier rule in --priority",
        "cold_front_declared": (
            "front_dir = sign(d1)!=sign(d2) AND min(|d1|,|d2|)>=train quantile q_turn "
            "AND delta_T<=train quantile q_cool AND circular |wd_end-wd_start|>=train quantile q_wind. "
            "d1=p_mid-p_start, d2=p_end-p_mid, mid=L//2. "
            "Cach A dung chuoi phan du cho p va T; Cach B dung chuoi tho. "
            "Huong gio luon la hieu vong qua atan2(sin, cos), khong phai hieu tho."
        ),
        "other_cold_front_counts_are_not_a_ranking": True,
        "sample_population": "train windows",
        "run_length_population": "within each split; purge gap is not inside a split",
    })
    with open(os.path.join(a.out, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(to_builtin(cfg), f, indent=2)

    lines = [
        "# Pattern stats - Jena",
        "",
        "Chi do, khong chon thay nguoi dung. Khong co ket luan nen dung cach nao.",
        "Khi hau, phan vi, diem z va ACF chi tinh tren train sau purge gap. Val/test chi ap dung nguong do.",
        "Dac trung va nhan chi dung X. Don vi vat ly, sau khi gan o `wv` / `max. wv` < -900 ve 0.",
        "",
        f"- So dong gio: {len(df)}; so o wv/max.wv < -900 da gan 0: {n_wv_err}",
        f"- L = {a.L}, H = {a.H}, purge gap = {sp['gap']} cua so moi ranh gioi",
        f"- Cua so: train {tr[1] - tr[0]}, val {va[1] - va[0]}, test {te[1] - te[0]} (tong {n_w})",
        f"- Khi hau: {sp['train_rows']} dong dau, khoa {a.clim}",
        f"- A_k = mean(x - khi hau) trong cua so. Cach B, heat/cold dung mean tho cua T.",
        f"- Kenh spike (tham so, khong phai ket luan): {a.spike_channel}. "
        f"Phan vi: spike {a.q_spike}, thap {a.q_low}, cao {a.q_high}.",
        f"- Seed mau cua so: {a.seed}. Khong co buoc ngau nhien nao khac.",
        f"- C2 dat khi share_train >= {a.c2_min_share:.0%} va min(n) qua 3 tap >= {a.c2_min_count}. "
        f"Cot {a.c2_target_share:.0%} chi la moc trong, khong doi ket qua dat/khong dat.",
        f"- C3 canh bao khi Normal > {a.c3_max_normal:.0%} o bat ky tap. Cot {a.c3_target_normal:.0%} la moc trong.",
        f"- C4: |share - share_train| / share_train <= {a.c4_max_rel_dev} (tham so cua script; prompt khong chot so nay).",
        "",
        "## 1. Chu ky ngay/mua",
        "",
        "r2 = ty le phuong sai chuoi gio giai thich boi trung binh nhom. acf = tu tuong quan.",
        "corr_delta_raw_end_hour = Pearson giua Delta tho va gio ket thuc (gio la so 0-23).",
        "eta2_delta = ty le phuong sai cua Delta giai thich boi gio ket thuc, gio duoc xem la nhom.",
        "",
        md_table(cyc),
        "",
        "Delta tho va Delta phan du:",
        "",
        md_table(cmp_df),
        "",
        f"Cua so co {a.L} diem nen moc ket thuc lech moc bat dau {a.L - 1} gio; "
        f"({a.L - 1}) mod 24 = {hour_shift}. Gio ket thuc = gio bat dau + {hour_shift} (mod 24).",
        f"eta2 cua Delta tho theo gio ket thuc nam trong "
        f"[{cyc['eta2_delta_raw'].min():.4f}, {cyc['eta2_delta_raw'].max():.4f}].",
        f"std(Delta phan du) / std(Delta tho) nam trong "
        f"[{cmp_df['std_ratio'].min():.4f}, {cmp_df['std_ratio'].max():.4f}].",
        f"std(Delta tho - Delta phan du) / std(Delta tho) nam trong "
        f"[{cmp_df['clim_std_ratio'].min():.4f}, {cmp_df['clim_std_ratio'].max():.4f}]. "
        "Hieu nay bang khi hau tai diem cuoi tru khi hau tai diem dau.",
        "",
        "## 2. Phan vi dac trung tren train",
        "",
        "A (anom) la mean(x - khi hau) trong cua so, chi co o cot mode = resid.",
        "",
        md_table(qdf),
        "",
        "Histogram (train, truc tan suat log): `figures/feat_hist_<p|T|rh|wv>.png`.",
        "",
        "\n".join(mode_blocks),
        "## 6. Cold Front - chi dem, khong gan nhan",
        "",
        "Luat khai bao `front_dir`: ap suat doi chieu (hai nua cua so trai dau va "
        f"min(|d1|,|d2|) >= phan vi {a.q_turn:.2f} cua chinh dai luong do tren moi cua so train) "
        f"VA delta T <= phan vi {q_cool:.2f} VA |delta wd| vong dau-cuoi >= phan vi {q_wind:.2f}.",
        "d1 = p(mid) - p(dau), d2 = p(cuoi) - p(mid), mid = L//2.",
        "Cach A dung chuoi p/T da tru khi hau. Cach B dung chuoi tho. Huong gio khong tru khi hau.",
        "`front_speed`, `front_either_wind`, `front_dir_fall_rise` la so dem cua dinh nghia gio / chieu ap khac, cung nguong. Bang khong xep hang chung.",
        "",
        "Nguong train:",
        "",
        md_table(front_thr),
        "",
        "So cua so:",
        "",
        md_table(fronts),
        "",
        "## 8. Do nhay nguong",
        "",
        "Moi o la d_share_train: ty le nhan tren train sau khi dich, tru ty le goc.",
        "`kind=rule`: chi dich phan vi cua mot luat. `kind=all`: dich q_spike, q_low va q_high cung mot luc "
        "(q_low gan vao ca pressure_drop va cold_anomaly; q_high gan vao ca humidity_rise va heat_anomaly).",
        "Phan vi bi cat ve [0, 1] thi clipped = True.",
        "",
        md_table(sens_wide),
        "",
        "Day du train/val/test: `threshold_sensitivity.csv`.",
        "",
        "## 9. Mau cua so tren train",
        "",
        f"Moi nhan lay toi da {a.n_samples} cua so train, seed {a.seed}, khong lap, ve theo thu tu thoi gian.",
        "Don vi vat ly, 4 kenh. Anh: `figures/samples_<A|B>_<nhan>.png`. Chi so: `sample_indices.csv`.",
        "",
        md_table(samples.groupby(["mode", "label"]).size().reset_index(name="n_plotted"), "{:.0f}"),
        "",
        "## 10. Do dai cum cua so lien tiep cung nhan",
        "",
        "Tinh trong tung tap sau khi gan theo uu tien co dinh. Buoc cua so = 1 gio. "
        "Purge gap nam giua cac tap nen khong tao them mot dut cum ben trong tap.",
        "mean_run_length = so cua so trung binh cua mot cum. n_windows / n_runs la do dai trung binh khi gop moi nhan.",
        "",
        "Gop moi nhan:",
        "",
        md_table(runs_all),
        "",
        "Theo nhan:",
        "",
        md_table(runs),
        "",
        "## Canh bao C2-C4",
        "",
    ]
    if all_warn:
        lines += [f"- {w}" for w in all_warn]
    else:
        lines.append("Khong co tieu chi C2-C4 nao khong dat.")
    lines.append("")
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[Chu ky]")
    print(cyc.round(3).to_string(index=False))
    print("\n[Cold Front]")
    print(fronts.round(4).to_string(index=False))
    print("\n[Run length tong]")
    print(runs_all.round(3).to_string(index=False))
    if all_warn:
        print("\n[CANH BAO CHUNG]")
        for w in all_warn:
            print("-", w)
    print("\nDa luu:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
