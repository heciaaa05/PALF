import numpy as np
import torch
from torch.utils.data import TensorDataset

PATTERNS = ["stable", "rising", "falling", "spike"]


def make_series(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    s = 2.0 * np.sin(2 * np.pi * t / 24) + 0.8 * np.sin(2 * np.pi * t / (24 * 30))
    s = s + rng.normal(0, 0.3, n)
    # spike ngau nhien mo phong su kien cuc doan
    for p in np.where(rng.random(n) < 0.004)[0]:
        amp = rng.uniform(4, 8)
        k = np.arange(min(8, n - p))
        s[p:p + len(k)] += amp * np.exp(-k / 3)
    return s.astype(np.float32)


def build_windows(series, L, H):
    n = len(series) - L - H + 1
    idx = np.arange(L + H)[None, :] + np.arange(n)[:, None]
    w = series[idx]
    return w[:, :L], w[:, L:]


def compute_thresholds(x_train, tail=12):
    t = x_train[:, -tail:]
    slope = np.abs(t[:, -1] - t[:, 0]) / (tail - 1)
    diff_std = np.diff(t, axis=1).std()
    return {"slope": float(np.median(slope)), "spike": float(4 * diff_std)}


def label_patterns(x, th, tail=12):
    """Rule-based pattern tren 12 buoc cuoi cua cua so input."""
    t = x[:, -tail:]
    slope = (t[:, -1] - t[:, 0]) / (tail - 1)
    labels = np.zeros(len(x), dtype=np.int64)
    labels[slope > th["slope"]] = 1
    labels[slope < -th["slope"]] = 2
    labels[np.abs(np.diff(t, axis=1)).max(1) > th["spike"]] = 3
    return labels


def prepare_data(n=20000, L=48, H=12, seed=42):
    s = make_series(n, seed)
    cut = int(len(s) * 0.7)
    mu, sd = s[:cut].mean(), s[:cut].std()
    s = ((s - mu) / sd).astype(np.float32)

    x, y = build_windows(s, L, H)
    n_w = len(x)
    i1, i2 = int(n_w * 0.7), int(n_w * 0.85)
    th = compute_thresholds(x[:i1])
    pat = label_patterns(x, th)

    def ds(a, b):
        return TensorDataset(
            torch.from_numpy(x[a:b]).unsqueeze(-1),
            torch.from_numpy(y[a:b]),
            torch.from_numpy(pat[a:b]),
            torch.arange(b - a),
        )

    return ds(0, i1), ds(i1, i2), ds(i2, n_w), th