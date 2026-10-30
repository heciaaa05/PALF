import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from .synthetic import compute_thresholds, label_patterns

JENA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"


def download_jena(raw_dir="data/raw"):
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, "jena_climate_2009_2016.csv")
    if os.path.exists(csv_path):
        return csv_path
    req = urllib.request.Request(JENA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as f, open(csv_path, "wb") as out:
            out.write(f.read())
    return csv_path


def load_jena(csv_path, subsample=6):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    date_cols = [c for c in df.columns if "date" in c.lower()]
    df = df.drop(columns=date_cols)
    df = df.iloc[::subsample].reset_index(drop=True)  # 10 phut -> moi `subsample` buoc (mac dinh = 1 gio)
    for col in df.columns:
        if "wv" in col.lower():
            df.loc[df[col] < -900, col] = 0.0  # loi du lieu da biet: gio am bat thuong
    return df


def build_real_windows(values, L, H, target_idx):
    n = len(values) - L - H + 1
    idx = np.arange(L + H)[None, :] + np.arange(n)[:, None]
    w = values[idx]
    x = w[:, :L, :].astype(np.float32)
    y = w[:, L:, target_idx].astype(np.float32)
    return x, y


def prepare_weather_data(target_col="T (degC)", L=96, H=24, subsample=6, csv_path=None):
    csv_path = csv_path or download_jena()
    df = load_jena(csv_path, subsample=subsample)
    if target_col not in df.columns:
        raise ValueError(f"Khong tim thay cot '{target_col}'. Cac cot: {list(df.columns)}")
    cols = list(df.columns)
    target_idx = cols.index(target_col)
    values = df.to_numpy(dtype=np.float32)

    cut = int(len(values) * 0.7)
    mu, sd = values[:cut].mean(0), values[:cut].std(0)
    sd[sd == 0] = 1.0
    values_n = (values - mu) / sd

    x, y = build_real_windows(values_n, L, H, target_idx)
    n_w = len(x)
    i1, i2 = int(n_w * 0.7), int(n_w * 0.85)

    th = compute_thresholds(x[:i1, :, target_idx])
    pat = label_patterns(x[:, :, target_idx], th)

    def ds(a, b):
        return TensorDataset(
            torch.from_numpy(x[a:b]), torch.from_numpy(y[a:b]),
            torch.from_numpy(pat[a:b]), torch.arange(b - a),
        )

    meta = {"columns": cols, "target_col": target_col, "target_idx": target_idx}
    return ds(0, i1), ds(i1, i2), ds(i2, n_w), th, meta
