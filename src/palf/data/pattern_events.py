"""Do bo sung muc su kien (consecutive runs) va kiem tra troi cho Pattern Spec v1.

Chi do va bao cao. Ranh gioi chi tinh tren train sau purge gap. Khong dung y.
"""
import numpy as np
import pandas as pd


def count_mask_runs(mask):
    """Dem cac doan True lien tiep (events/runs) trong mot mang hoac series boolean.

    Tra ve: np.ndarray (dtype=int) chua do dai cua tung doan True.
    Tra ve mang rong neu mang rong hoac khong chua bat ky gia tri True nao.
    """
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0 or not np.any(arr):
        return np.array([], dtype=int)
    padded = np.pad(arr, (1, 1), mode="constant", constant_values=False)
    diff = np.diff(padded.astype(int))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return (ends - starts).astype(int)


def mask_run_stats(mask):
    """Thong ke cac doan True: so cua so, ty le, so runs, do dai trung binh va max."""
    arr = np.asarray(mask, dtype=bool)
    n_w = int(arr.sum())
    n_tot = len(arr)
    share = float(n_w / n_tot) if n_tot > 0 else float("nan")
    lens = count_mask_runs(arr)
    n_r = int(len(lens))
    mean_l = float(lens.mean()) if n_r > 0 else float("nan")
    max_l = int(lens.max()) if n_r > 0 else 0
    return {
        "n_windows": n_w,
        "share": share,
        "n_runs": n_r,
        "mean_run_length": mean_l,
        "max_run_length": max_l,
    }


def split_mask_runs(mask, spans):
    """Tinh thong ke doan True cho tung tap (khong noi qua purge gap)."""
    rows = []
    m = np.asarray(mask, dtype=bool)
    for split, (lo, hi) in spans.items():
        sub = m[lo:hi]
        st = mask_run_stats(sub)
        st["split"] = split
        rows.append(st)
    return pd.DataFrame(rows)


def label_run_stats_by_split(labels, spans, names):
    """Thong ke so runs va do dai cho moi nhan trong tung tap."""
    rows = []
    for split, (lo, hi) in spans.items():
        sub = labels.iloc[lo:hi].to_numpy()
        for name in names:
            m = sub == name
            st = mask_run_stats(m)
            st["label"] = name
            st["split"] = split
            rows.append(st)
    df = pd.DataFrame(rows)
    # Dat lai thu tu cot cho dep
    return df[["label", "split", "n_windows", "share", "n_runs", "mean_run_length", "max_run_length"]]
