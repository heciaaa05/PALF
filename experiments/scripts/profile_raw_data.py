import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from palf.data.weather import download_jena

BOUNDS = {
    "p (mbar)": (900, 1050), "T (degC)": (-30, 45), "Tpot (K)": (250, 320),
    "Tdew (degC)": (-40, 30), "rh (%)": (0, 100), "VPmax (mbar)": (0, 80),
    "VPact (mbar)": (0, 80), "VPdef (mbar)": (0, 80), "sh (g/kg)": (0, 20),
    "H2OC (mmol/mol)": (0, 30), "rho (g/m**3)": (1100, 1400),
    "wv (m/s)": (0, 30), "max. wv (m/s)": (0, 40), "wd (deg)": (0, 360),
}
QS = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]

# (ten nhan, nhom kenh, loai dac trung, muc phan vi)
RULES = [
    ("pressure_drop", "pressure", "delta", "low"),
    ("rapid_cooling", "temp", "delta", "low"),
    ("rapid_warming", "temp", "delta", "high"),
    ("heat_wave", "temp", "roll_mean", "high"),
    ("humidity_surge", "humidity", "delta", "high"),
    ("wind_gust", "wind", "level", "extreme"),
]


def load_hourly(csv_path, subsample=6):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c.lower())
    ts = pd.to_datetime(df[date_col], format="%d.%m.%Y %H:%M:%S")
    df = df.drop(columns=[date_col])
    df.index = ts
    df = df.iloc[::subsample]
    n_wv_err = 0
    for col in df.columns:
        if "wv" in col.lower():
            bad = df[col] < -900
            n_wv_err += int(bad.sum())
            df.loc[bad, col] = 0.0  # giong weather.py
    return df, n_wv_err


def channel_summary(df):
    rows = []
    for col in df.columns:
        x = df[col]
        lo, hi = BOUNDS.get(col, (-np.inf, np.inf))
        bad = int(((x < lo) | (x > hi)).sum())
        rows.append({
            "channel": col, "mean": x.mean(), "std": x.std(), "min": x.min(), "max": x.max(),
            "n_missing": int(x.isna().sum()), "bound_lo": lo, "bound_hi": hi,
            "n_out_of_bounds": bad, "pct_out_of_bounds": 100.0 * bad / len(x),
        })
    return pd.DataFrame(rows)


def cycle_strength(df, train_rows):
    tr = df.iloc[:train_rows]
    rows = []
    for col in tr.columns:
        x = tr[col].dropna()
        tot = float(((x - x.mean()) ** 2).sum())
        if tot == 0:
            continue

        def r2(keys):
            m = x.groupby(keys).transform("mean")
            return float(1.0 - ((x - m) ** 2).sum() / tot)

        rows.append({
            "channel": col,
            "acf_lag24": float(x.autocorr(24)),
            "r2_hour": r2([x.index.hour]),
            "r2_month": r2([x.index.month]),
            "r2_month_hour": r2([x.index.month, x.index.hour]),
        })
    return pd.DataFrame(rows)


def deseason(df, train_rows):
    """Tru trung binh theo (thang, gio) tinh tren train. Bo qua huong gio (bien tuan hoan)."""
    out = df.copy()
    train = df.iloc[:train_rows]
    idx = pd.MultiIndex.from_arrays([df.index.month, df.index.hour])
    for col in df.columns:
        if col.startswith("wd"):
            continue
        prof = train[col].groupby([train.index.month, train.index.hour]).mean()
        out[col] = df[col].to_numpy() - prof.reindex(idx).to_numpy()
    return out


def change_percentiles(w, train_rows, ks):
    rows = []
    for col in w.columns:
        for k in ks:
            d = w[col].diff(k).iloc[:train_rows].dropna()
            if col.startswith("wd"):
                d = (d + 180.0) % 360.0 - 180.0
            q = d.quantile(QS)
            row = {"channel": col, "k": k, "std": float(d.std())}
            row.update({f"p{int(round(v * 100)):02d}": float(q[v]) for v in QS})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_hist(w, train_rows, k, channels, path):
    chans = [c for c in channels if c in w.columns]
    if not chans:
        return
    fig, axes = plt.subplots(1, len(chans), figsize=(4.2 * len(chans), 3.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, chans):
        d = w[c].diff(k).iloc[:train_rows].dropna()
        ax.hist(d, bins=120)
        for q in (0.10, 0.90, 0.99):
            ax.axvline(d.quantile(q), linestyle="--", linewidth=1)
        ax.set_title(f"{c} | delta {k}h")
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def candidate_flags(w, train_rows, a):
    cols = {"pressure": a.pressure_col, "temp": a.temp_col,
            "humidity": a.humidity_col, "wind": a.wind_col}
    qmap = {"low": a.q_low, "high": a.q_high, "extreme": a.q_extreme}
    flags, rules = {}, []
    for name, ch, kind, level in RULES:
        col = cols[ch]
        if col not in w.columns:
            continue
        if kind == "delta":
            s = w[col].diff(a.k_short)
        elif kind == "roll_mean":
            s = w[col].rolling(a.sustain).mean()
        else:
            s = w[col]
        thr = float(s.iloc[:train_rows].quantile(qmap[level]))
        flags[name] = (s <= thr) if level == "low" else (s >= thr)
        rules.append({"pattern": name, "channel": col, "feature": kind,
                      "op": "<=" if level == "low" else ">=",
                      "quantile": qmap[level], "threshold": thr})
    return pd.DataFrame(flags), pd.DataFrame(rules)


def md_table(df, fmt="{:.4f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--subsample", type=int, default=6)
    ap.add_argument("--target-col", default="T (degC)")
    ap.add_argument("--temp-col", default="T (degC)")
    ap.add_argument("--pressure-col", default="p (mbar)")
    ap.add_argument("--humidity-col", default="rh (%)")
    ap.add_argument("--wind-col", default="wv (m/s)")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 3, 6, 12, 24])
    ap.add_argument("--k-short", type=int, default=3)
    ap.add_argument("--sustain", type=int, default=24)
    ap.add_argument("--q-low", type=float, default=0.10)
    ap.add_argument("--q-high", type=float, default=0.90)
    ap.add_argument("--q-extreme", type=float, default=0.99)
    ap.add_argument("--deseason", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or ("experiments/results/data_profile/jena" + ("_deseason" if a.deseason else ""))
    os.makedirs(out, exist_ok=True)

    df, n_wv_err = load_hourly(a.csv or download_jena(), a.subsample)
    n = len(df)
    n_w = n - a.L - a.H + 1
    i1, i2 = int(n_w * 0.7), int(n_w * 0.85)
    train_rows = i1 + a.L - 1  # so dong thuoc phan input cua cac cua so train
    gaps = df.index.to_series().diff().dt.total_seconds() / 3600.0
    n_gap_1h, n_gap_6h = int((gaps > 1).sum()), int((gaps > 6).sum())

    work = deseason(df, train_rows) if a.deseason else df

    summary = channel_summary(df)
    cyc = cycle_strength(df, train_rows)
    chg = change_percentiles(work, train_rows, a.ks)
    flags, rules = candidate_flags(work, train_rows, a)

    end = flags.iloc[a.L - 1: a.L - 1 + n_w].reset_index(drop=True)
    end["stable"] = ~end.any(axis=1)
    splits = {"train": slice(0, i1), "val": slice(i1, i2), "test": slice(i2, n_w)}
    freq = []
    for pat in end.columns:
        row = {"pattern": pat}
        for s, sl in splits.items():
            row[f"{s}_share"] = float(end.iloc[sl][pat].mean())
            row[f"{s}_n"] = int(end.iloc[sl][pat].sum())
        freq.append(row)
    freq = pd.DataFrame(freq)

    names = [c for c in end.columns if c != "stable"]
    tr = end.iloc[:i1][names].astype(int)
    counts = tr.T.dot(tr)
    diag = pd.Series(np.diag(counts), index=counts.index).clip(lower=1)
    cond = counts.div(diag, axis=0)

    summary.to_csv(os.path.join(out, "channel_summary.csv"), index=False)
    cyc.to_csv(os.path.join(out, "cycle_strength.csv"), index=False)
    chg.to_csv(os.path.join(out, "change_rate_percentiles.csv"), index=False)
    rules.to_csv(os.path.join(out, "pattern_rules.csv"), index=False)
    freq.to_csv(os.path.join(out, "pattern_frequency.csv"), index=False)
    counts.to_csv(os.path.join(out, "pattern_overlap_counts.csv"))
    cond.to_csv(os.path.join(out, "pattern_overlap_conditional.csv"))
    plot_hist(work, train_rows, a.k_short,
              [a.temp_col, a.pressure_col, a.humidity_col, a.wind_col],
              os.path.join(out, "change_rate_hist.png"))

    tgt = cyc[cyc["channel"] == a.target_col]
    present = {k: (v in df.columns) for k, v in
               {"pressure": a.pressure_col, "temp": a.temp_col,
                "humidity": a.humidity_col, "wind": a.wind_col}.items()}
    lines = [
        "# Muc 8 - Ban dien nhap tu du lieu thuc (Jena)",
        "",
        f"- Che do tinh Delta/nguong: {'phan du khi hau (deseason theo thang x gio)' if a.deseason else 'gia tri tho'}",
        f"- Tan suat: 1 gio (lay 1 trong moi {a.subsample} dong). So dong: {n}. Khoang trong >1h: {n_gap_1h}, >6h: {n_gap_6h}",
        f"- Loi cam bien gio (< -900) da gan ve 0: {n_wv_err} dong",
        "",
        "## 1. Cach xu ly chu ky ngay/mua",
        "",
        md_table(tgt) if len(tgt) else "(khong co kenh muc tieu trong bang chu ky)",
        "",
        f"- L = {a.L} = {a.L / 24:.1f} x 24 gio (L la boi cua chu ky ngay: {a.L % 24 == 0}).",
        "- Quyet dinh: ___ (A: phan du khi hau / B: L boi chu ky ngay)",
        "",
        "## 2. L va H",
        "",
        f"- L = {a.L}, H = {a.H}. So cua so: {n_w} (train {i1}, val {i2 - i1}, test {n_w - i2}).",
        "",
        "## 3. Kenh tham gia luat",
        "",
        "- Co trong dataset: " + ", ".join(f"{k}={'co' if v else 'KHONG'}" for k, v in present.items()),
        "",
        f"## 4. Kenh muc tieu: {a.target_col}",
        "",
        "## 5. Nguong cu the (chi tinh tren train)",
        "",
        md_table(rules),
        "",
        "## 6. Tan suat nhan ung vien (so nhan cuoi: 4-7, chot sau khi co ket qua C2, C3)",
        "",
        md_table(freq),
        "",
        "## 7. Chong lan: P(cot | hang) tren train",
        "",
        md_table(cond.reset_index().rename(columns={"index": "given"})),
        "",
        "- Thu tu uu tien khi chong lan: ___",
    ]
    with open(os.path.join(out, "section8_draft.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("So dong:", n, "| khoang trong >1h:", n_gap_1h, ">6h:", n_gap_6h, "| loi gio:", n_wv_err)
    print("Cua so: train", i1, "val", i2 - i1, "test", n_w - i2)
    print("\n[Chu ky]")
    print(cyc.round(3).to_string(index=False))
    print("\n[Tan suat nhan]")
    print(freq.round(4).to_string(index=False))
    print("\n[Chong lan P(cot|hang)]")
    print(cond.round(3).to_string())
    print("\nDa luu vao:", out)


if __name__ == "__main__":
    main()