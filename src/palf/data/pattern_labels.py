"""Gan nhan thu theo luat phan vi (chi do, khong chon thay nguoi dung).

- Nguong chi tinh tren cac cua so train.
- Nhan chi dung dac trung cua X (xem pattern_features), khong dung y.
- Cach A = dac trung phan du so voi khi hau; Cach B = dac trung tho.
"""
import numpy as np
import pandas as pd

from .pattern_features import window_features

RULE_ORDER = ["spike", "pressure_drop", "humidity_rise", "heat_anomaly", "cold_anomaly"]


def rule_specs(mode, spike_channel="T", q_spike=0.99, q_low=0.10, q_high=0.90):
    """Tra ve {ten luat: (cot dac trung, toan tu, muc phan vi)}.
    mode 'A': dac trung phan du. mode 'B': dac trung tho, heat/cold dung trung binh tho cua T."""
    m = "resid" if mode == "A" else "raw"
    anom_t = "T__anom" if mode == "A" else "T__mean__raw"
    return {
        "spike": (f"{spike_channel}__jump__{m}", ">=", q_spike),
        "pressure_drop": (f"p__delta__{m}", "<=", q_low),
        "humidity_rise": (f"rh__delta__{m}", ">=", q_high),
        "heat_anomaly": (anom_t, ">=", q_high),
        "cold_anomaly": (anom_t, "<=", q_low),
    }


def add_raw_mean(feats, df, channels, L, n_w):
    """Them trung binh tho trong cua so cho moi kenh: cot '{k}__mean__raw'."""
    out = feats.copy()
    for short, col in channels.items():
        out[f"{short}__mean__raw"] = window_features(df[col].to_numpy(dtype=float), L, n_w)["mean"]
    return out


def fit_thresholds(feats, spec, train_span):
    a, b = train_span
    th = {}
    for name, (col, op, q) in spec.items():
        v = feats[col].iloc[a:b].dropna()
        if len(v) == 0:
            raise ValueError(f"Khong co du lieu train cho cot {col}")
        th[name] = {"feature": col, "op": op, "quantile": q, "threshold": float(v.quantile(q))}
    return th


def fire_matrix(feats, th):
    """DataFrame bool (n_w, so luat): luat nao kich hoat tren cua so nao. NaN -> False."""
    cols = {}
    for name, t in th.items():
        v = feats[t["feature"]]
        cols[name] = (v <= t["threshold"]) if t["op"] == "<=" else (v >= t["threshold"])
    return pd.DataFrame(cols)


def assign_labels(fires, priority):
    """Luat dung truoc trong `priority` thang. Khong luat nao kich hoat -> 'normal'."""
    for name in priority:
        if name not in fires.columns:
            raise ValueError(f"Luat '{name}' khong co trong bang kich hoat")
    labels = pd.Series("normal", index=fires.index, dtype=object)
    for name in reversed(list(priority)):
        labels[fires[name].to_numpy()] = name
    return labels


def overlap_counts(fires, span):
    a, b = span
    m = fires.iloc[a:b].astype(int)
    return m.T.dot(m)


def label_share_table(labels, spans, priority):
    rows = []
    for name in list(priority) + ["normal"]:
        row = {"label": name}
        for s, (a, b) in spans.items():
            part = labels.iloc[a:b]
            row[f"n_{s}"] = int((part == name).sum())
            row[f"share_{s}"] = float((part == name).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def check_criteria(tab, c2_min_share=0.02, c2_target_share=0.05, c2_min_count=300,
                   c3_max_normal=0.70, c3_target_normal=0.60, c4_max_rel_dev=0.5):
    """Tra ve (c2, c3, c4, canh_bao). Chi bao dat/khong dat, khong chon thay nguoi dung."""
    warnings = []

    nn = tab[tab["label"] != "normal"].copy()
    nn["min_n"] = nn[["n_train", "n_val", "n_test"]].min(axis=1)
    k_min, k_tar = f"share>={c2_min_share:.0%}", f"share>={c2_target_share:.0%}"
    k_cnt = f"min_n>={c2_min_count}"
    nn[k_min] = nn["share_train"] >= c2_min_share
    nn[k_tar] = nn["share_train"] >= c2_target_share
    nn[k_cnt] = nn["min_n"] >= c2_min_count
    nn["C2"] = nn[k_min] & nn[k_cnt]
    c2 = nn[["label", "share_train", "n_train", "n_val", "n_test", "min_n", k_min, k_tar, k_cnt, "C2"]]
    for _, r in c2.iterrows():
        if not r["C2"]:
            warnings.append(f"C2 khong dat: {r['label']} (share_train={r['share_train']:.2%}, "
                            f"min_n={int(r['min_n'])})")

    nr = tab[tab["label"] == "normal"].iloc[0]
    shares = [float(nr["share_train"]), float(nr["share_val"]), float(nr["share_test"])]
    k3a, k3b = f"all<={c3_max_normal:.0%}", f"all<={c3_target_normal:.0%}"
    c3 = pd.DataFrame([{
        "normal_train": shares[0], "normal_val": shares[1], "normal_test": shares[2],
        k3a: bool(max(shares) <= c3_max_normal), k3b: bool(max(shares) <= c3_target_normal),
    }])
    if not c3.iloc[0][k3a]:
        warnings.append(f"C3 khong dat: Normal = {shares[0]:.1%} / {shares[1]:.1%} / {shares[2]:.1%} "
                        f"(train/val/test), nguong {c3_max_normal:.0%}")

    rows = []
    for _, r in tab.iterrows():
        base = float(r["share_train"])
        dv = abs(float(r["share_val"]) - base) / base if base > 0 else float("nan")
        dt = abs(float(r["share_test"]) - base) / base if base > 0 else float("nan")
        ok = bool(dv <= c4_max_rel_dev and dt <= c4_max_rel_dev)  # NaN -> False
        rows.append({"label": r["label"], "share_train": base, "share_val": float(r["share_val"]),
                     "share_test": float(r["share_test"]), "rel_dev_val": dv,
                     "rel_dev_test": dt, "C4": ok})
        if not ok:
            warnings.append(f"C4 khong dat: {r['label']} (lech tuong doi val={dv:.2f}, test={dt:.2f}, "
                            f"nguong {c4_max_rel_dev})")
    c4 = pd.DataFrame(rows)
    return c2, c3, c4, warnings