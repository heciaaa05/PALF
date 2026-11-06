import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palf.data.weather import download_jena
from palf.data.pattern_features import (
    CHANNELS, FEATURES, QUANTILES, load_hourly_with_time, window_split,
    build_feature_table, variance_explained,
)


def md_table(df, fmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_features(ftr, short, col, path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    for ax, f in zip(axes[:3], FEATURES):
        for mode in ("raw", "resid"):
            ax.hist(ftr[f"{short}__{f}__{mode}"].dropna(), bins=100, alpha=0.55,
                    density=True, label=mode)
        ax.set_title(f"{col} | {f}")
        ax.set_yscale("log")
        ax.legend()
    axes[3].hist(ftr[f"{short}__anom"].dropna(), bins=100, density=True)
    axes[3].set_title(f"{col} | anom (resid)")
    axes[3].set_yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--subsample", type=int, default=6)
    ap.add_argument("--clim", choices=["hour", "month_hour"], default="month_hour")
    ap.add_argument("--purge", type=int, default=None,
                    help="So cua so bo o moi ranh gioi tap. Mac dinh L+H-1.")
    ap.add_argument("--out", default="experiments/pattern_stats/jena")
    a = ap.parse_args()

    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df, n_wv_err = load_hourly_with_time(a.csv or download_jena(), a.subsample)
    df = df[list(CHANNELS.values())]
    sp = window_split(len(df), a.L, a.H, purge=a.purge)
    n_w, spans = sp["n_windows"], sp["spans"]
    tr, va, te = spans["train"], spans["val"], spans["test"]

    feats, clim = build_feature_table(df, CHANNELS, a.L, n_w, sp["train_rows"], a.clim)
    ftr = feats.iloc[tr[0]:tr[1]]
    train_df = df.iloc[:sp["train_rows"]]

    # --- Muc 1: chu ky ngay/mua ---
    rows = []
    for s, col in CHANNELS.items():
        x = train_df[col]
        rows.append({
            "channel": col,
            "r2_hour": variance_explained(x.to_numpy(), x.index.hour),
            "r2_month_hour": variance_explained(x.to_numpy(), x.index.month, x.index.hour),
            "acf_lag24": float(x.autocorr(24)),
            "acf_lag96": float(x.autocorr(96)),
            "eta2_delta_raw": variance_explained(ftr[f"{s}__delta__raw"].to_numpy(),
                                                 ftr["end_hour"].to_numpy()),
            "eta2_delta_resid": variance_explained(ftr[f"{s}__delta__resid"].to_numpy(),
                                                   ftr["end_hour"].to_numpy()),
        })
    cyc = pd.DataFrame(rows)

    rows = []
    for s, col in CHANNELS.items():
        r, e = ftr[f"{s}__delta__raw"], ftr[f"{s}__delta__resid"]
        rows.append({
            "channel": col,
            "std_raw": r.std(), "std_resid": e.std(), "std_ratio": e.std() / r.std(),
            "iqr_raw": r.quantile(0.75) - r.quantile(0.25),
            "iqr_resid": e.quantile(0.75) - e.quantile(0.25),
            "p01_raw": r.quantile(0.01), "p99_raw": r.quantile(0.99),
            "p01_resid": e.quantile(0.01), "p99_resid": e.quantile(0.99),
        })
    cmp_df = pd.DataFrame(rows)

    # --- Muc 2: phan vi dac trung tren train ---
    rows = []
    for s, col in CHANNELS.items():
        combos = [(f, m) for f in FEATURES for m in ("raw", "resid")] + [("anom", "resid")]
        for f, m in combos:
            key = f"{s}__anom" if f == "anom" else f"{s}__{f}__{m}"
            v = ftr[key]
            row = {"channel": col, "feature": f, "mode": m}
            row.update({f"p{int(round(q * 100)):02d}": float(v.quantile(q)) for q in QUANTILES})
            rows.append(row)
        plot_features(ftr, s, col, os.path.join(fig_dir, f"feat_hist_{s}.png"))
    qdf = pd.DataFrame(rows)

    # --- Luu ket qua ---
    cyc.to_csv(os.path.join(a.out, "cycle_strength.csv"), index=False)
    cmp_df.to_csv(os.path.join(a.out, "delta_raw_vs_resid.csv"), index=False)
    qdf.to_csv(os.path.join(a.out, "feature_quantiles.csv"), index=False)
    train_df.groupby([train_df.index.month, train_df.index.hour]).mean().to_csv(
        os.path.join(a.out, "climatology_month_hour.csv"))

    thr = {
        "spans": spans, "train_rows": sp["train_rows"], "gap": sp["gap"], "clim": a.clim,
        "quantiles": {
            f"{r['channel']}|{r['feature']}|{r['mode']}":
                {k: v for k, v in r.items() if k.startswith("p") and k[1:].isdigit()}
            for r in qdf.to_dict("records")
        },
    }
    with open(os.path.join(a.out, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thr, f, indent=2, default=float)
    cfg = dict(vars(a))
    cfg.update({"n_rows": len(df), "n_windows": n_w, "gap": sp["gap"], "spans": spans,
                "n_wv_sensor_errors_set_to_zero": n_wv_err})
    with open(os.path.join(a.out, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=float)

    lines = [
        "# Pattern stats - Jena (buoc 1: chu ky va phan phoi dac trung)",
        "",
        "Chi do, khong chon thay nguoi dung. Khi hau, ACF va phan vi chi tinh tren train (sau purge gap).",
        "",
        f"- So dong (gio): {len(df)}; hang loi cam bien gio da gan ve 0: {n_wv_err}",
        f"- L = {a.L}, H = {a.H}, purge gap = {sp['gap']} cua so moi ranh gioi",
        f"- Cua so: train {tr[1] - tr[0]}, val {va[1] - va[0]}, test {te[1] - te[0]} (tong {n_w})",
        f"- Khi hau tinh tu {sp['train_rows']} dong dau, khoa nhom: {a.clim}",
        "- Dac trung chi dung X. anom = mean(x - khi hau) trong cua so.",
        "",
        "## 1. Chu ky ngay/mua",
        "",
        "r2_* = ty le phuong sai chuoi gio giai thich boi trung binh nhom. "
        "acf_* = tu tuong quan. eta2_* = ty le phuong sai cua Delta giai thich boi gio ket thuc cua so.",
        "",
        md_table(cyc),
        "",
        "Delta tho vs phan du so voi khi hau:",
        "",
        md_table(cmp_df),
        "",
        f"Ghi chu: L = {a.L} la boi cua 24 ({a.L % 24 == 0}); gio bat dau va gio ket thuc cua so lech nhau 1 gio.",
        "",
        "## 2. Phan vi dac trung tren train",
        "",
        md_table(qdf),
        "",
        "Histogram: figures/feat_hist_<p|T|rh|wv>.png",
    ]
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("Cua so: train", tr[1] - tr[0], "| val", va[1] - va[0], "| test", te[1] - te[0],
          "| purge", sp["gap"], "| loi gio:", n_wv_err)
    print("\n[Chu ky]")
    print(cyc.round(3).to_string(index=False))
    print("\n[Delta tho vs phan du]")
    print(cmp_df.round(3).to_string(index=False))
    print("\nDa luu vao:", a.out)


if __name__ == "__main__":
    main()