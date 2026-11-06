"""Dac trung cua so cho Pattern Spec v1 (chi do, khong chon thay nguoi dung).

Quy uoc:
- Cua so j: dau vao X = hang [j, j+L-1], muc tieu y = hang [j+L, j+L+H-1].
- Dac trung chi dung X. Khi hau va nguong chi tinh tren cac hang cua train.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = {"p": "p (mbar)", "T": "T (degC)", "rh": "rh (%)", "wv": "wv (m/s)"}
FEATURES = ("delta", "jump", "sigma")
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def load_hourly_with_time(csv_path, subsample=6):
    """Doc Jena giu lai moc thoi gian. Cung quy tac sua loi gio nhu weather.load_jena."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c.lower())
    ts = pd.to_datetime(df[date_col], format="%d.%m.%Y %H:%M:%S")
    df = df.drop(columns=[date_col])
    df.index = pd.DatetimeIndex(ts)
    df = df.iloc[::subsample].copy()
    n_wv_err = 0
    for col in df.columns:
        if "wv" in col.lower():
            bad = df[col] < -900
            n_wv_err += int(bad.sum())
            df.loc[bad, col] = 0.0
    return df, n_wv_err


def window_split(n_rows, L, H, train_frac=0.70, val_end_frac=0.85, purge=None):
    """Chia cua so theo thoi gian co purge gap (mac dinh L+H-1: khong chia se hang nao)."""
    n_w = n_rows - L - H + 1
    if n_w <= 0:
        raise ValueError("Chuoi qua ngan so voi L + H")
    gap = L + H - 1 if purge is None else int(purge)
    i1 = int(n_w * train_frac)
    i2 = int(n_w * val_end_frac)
    spans = {"train": (0, i1), "val": (i1 + gap, i2), "test": (i2 + gap, n_w)}
    for name, (a, b) in spans.items():
        if b <= a:
            raise ValueError(f"Tap {name} rong sau khi purge")
    return {"n_windows": n_w, "gap": gap, "spans": spans, "train_rows": i1 + L - 1}


def climatology(df, train_rows, keys="month_hour"):
    """Trung binh theo gio (hoac thang x gio) chi tinh tren train_rows hang dau."""
    train = df.iloc[:train_rows]
    if keys == "hour":
        tk = [train.index.hour]
        idx = pd.Index(df.index.hour)
    else:
        tk = [train.index.month, train.index.hour]
        idx = pd.MultiIndex.from_arrays([df.index.month, df.index.hour])
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for col in df.columns:
        prof = train[col].groupby(tk).mean()
        out[col] = prof.reindex(idx).to_numpy()
    return out


def variance_explained(values, *groups):
    """Ty le phuong sai cua values giai thich boi trung binh theo nhom (eta^2)."""
    v = np.asarray(values, dtype=float)
    ok = ~np.isnan(v)
    s = pd.Series(v[ok])
    keys = [np.asarray(g)[ok] for g in groups]
    tot = float(((s - s.mean()) ** 2).sum())
    if tot == 0.0:
        return float("nan")
    m = s.groupby(keys).transform("mean")
    return float(1.0 - ((s - m) ** 2).sum() / tot)


def window_features(x, L, n_w):
    """Delta = x_L - x_1, jump = max|x_{t+1}-x_t|, sigma = std, mean. Chi dung X."""
    W = sliding_window_view(np.asarray(x, dtype=float), L)[:n_w]
    d = np.abs(np.diff(W, axis=1))
    return {
        "delta": W[:, -1] - W[:, 0],
        "jump": d.max(axis=1),
        "sigma": W.std(axis=1),
        "mean": W.mean(axis=1),
    }


def build_feature_table(df, channels, L, n_w, train_rows, clim_keys="month_hour"):
    """Tra ve (bang dac trung (n_w, ...), khi hau).
    Cot: '{k}__{delta|jump|sigma}__{raw|resid}', '{k}__anom' va end_hour, end_month.
    anom = mean(x - khi hau) trong cua so = mean(x) - mean(khi hau theo moc thoi gian cua cua so)."""
    clim = climatology(df[list(channels.values())], train_rows, clim_keys)
    cols = {}
    for short, col in channels.items():
        x = df[col].to_numpy(dtype=float)
        r = x - clim[col].to_numpy(dtype=float)
        raw = window_features(x, L, n_w)
        res = window_features(r, L, n_w)
        for f in FEATURES:
            cols[f"{short}__{f}__raw"] = raw[f]
            cols[f"{short}__{f}__resid"] = res[f]
        cols[f"{short}__anom"] = res["mean"]
    out = pd.DataFrame(cols)
    end_idx = df.index[L - 1: L - 1 + n_w]
    out["end_hour"] = end_idx.hour.to_numpy()
    out["end_month"] = end_idx.month.to_numpy()
    return out, clim


def add_wind_direction_components(df, col="wd (deg)"):
    """Huong gio la bien vong: thay bang sin/cos, khong tinh Delta tho tren do."""
    out = df.drop(columns=[col]).copy()
    rad = np.deg2rad(df[col].to_numpy(dtype=float))
    out["wd_sin"] = np.sin(rad)
    out["wd_cos"] = np.cos(rad)
    return out