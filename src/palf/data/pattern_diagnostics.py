"""Do them cho Pattern Spec v1: mat tran lanh, gan theo z, do nhay, do dai chuoi.

Chi do. Nguong va diem z chi tinh tren cua so train. Moi dac trung chi dung X, khong dung y.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .pattern_features import window_features
from .pattern_labels import fire_matrix, rule_specs


def pearson(a, b):
    """He so tuong quan Pearson. Tra NaN neu mot ben khong bien thien."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def half_deltas(x, L, n_w):
    """Chia cua so tai diem giua: d1 = x_mid - x_1, d2 = x_L - x_mid. Chi dung X."""
    if L < 2:
        raise ValueError("L phai >= 2")
    W = sliding_window_view(np.asarray(x, dtype=float), L)[:n_w]
    mid = L // 2
    return W[:, mid] - W[:, 0], W[:, -1] - W[:, mid]


def turn_magnitude(d1, d2):
    """Do doi chieu: min(|d1|, |d2|) khi d1 va d2 trai dau, nguoc lai 0."""
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    rev = (d1 * d2) < 0.0
    mag = np.minimum(np.abs(d1), np.abs(d2))
    out = np.zeros(len(mag), dtype=float)
    out[rev] = mag[rev]
    return out, rev


def circular_abs_delta_deg(wd_deg, L, n_w):
    """|goc| giua huong gio dau va cuoi cua so, qua sin/cos. Don vi do, trong [0, 180].

    Khong dung hieu tho (wd_L - wd_1), vi wd la bien vong.
    """
    if L < 2:
        raise ValueError("L phai >= 2")
    W = sliding_window_view(np.asarray(wd_deg, dtype=float), L)[:n_w]
    rad = np.deg2rad(W[:, -1] - W[:, 0])
    return np.abs(np.rad2deg(np.arctan2(np.sin(rad), np.cos(rad))))


def cold_front_feature_frame(pressure, temperature, pressure_resid, temperature_resid,
                             wind_dir_deg, wv_abs_delta_raw, wv_abs_delta_resid, L, n_w):
    """Bang dac trung Cold Front. Chuoi p/T/wd dai bang so dong; |delta wv| dai n_w."""
    cols = {}
    series = {
        "raw": (pressure, temperature),
        "resid": (pressure_resid, temperature_resid),
    }
    for tag, (p, t) in series.items():
        d1, d2 = half_deltas(p, L, n_w)
        cols[f"p_d1__{tag}"] = d1
        cols[f"p_d2__{tag}"] = d2
        cols[f"t_delta__{tag}"] = window_features(t, L, n_w)["delta"]
    wd = circular_abs_delta_deg(wind_dir_deg, L, n_w)
    if len(wd) != n_w:
        raise ValueError("So cua so huong gio khong khop n_w")
    cols["wd_abs_turn"] = wd
    for name, arr in (("raw", wv_abs_delta_raw), ("resid", wv_abs_delta_resid)):
        v = np.asarray(arr, dtype=float)
        if len(v) != n_w:
            raise ValueError(f"wv_abs_delta__{name} phai dai {n_w}")
        cols[f"wv_abs_delta__{name}"] = v
    return pd.DataFrame(cols)


def fit_cold_front_thresholds(feat, train_span, tag, q_turn, q_cool, q_wind):
    """Phan vi train. tag 'raw' hoac 'resid' cho ap suat va nhiet; gio tinh rieng.

    - Chan ap: min(|d1|, |d2|) tren MOI cua so train (ke ca khong doi chieu).
    - Nhiet: delta T.
    - Gio: |goc| dau-cuoi, va |delta| toc do gio cua cung tag.
    """
    a, b = train_span
    d1 = feat[f"p_d1__{tag}"].iloc[a:b].to_numpy(dtype=float)
    d2 = feat[f"p_d2__{tag}"].iloc[a:b].to_numpy(dtype=float)
    leg = np.minimum(np.abs(d1), np.abs(d2))
    return {
        "tag": tag,
        "q_turn": float(q_turn),
        "thr_leg": float(pd.Series(leg).quantile(q_turn)),
        "q_cool": float(q_cool),
        "thr_t_delta": float(feat[f"t_delta__{tag}"].iloc[a:b].quantile(q_cool)),
        "q_wind": float(q_wind),
        "thr_wd": float(feat["wd_abs_turn"].iloc[a:b].quantile(q_wind)),
        "thr_wv": float(feat[f"wv_abs_delta__{tag}"].iloc[a:b].quantile(q_wind)),
    }


def cold_front_components(feat, thr, tag):
    """Mask tung ve va cac giao. Khong gan nhan; chi dem o tang script.

    Luat khai bao (front_dir): ap suat doi chieu manh VA nhiet giam nhanh VA |delta wd| vong lon.
    Cac mask khac la so dem phu, cung nguong train, de so sanh dinh nghia gio / chieu ap.
    """
    d1 = feat[f"p_d1__{tag}"].to_numpy(dtype=float)
    d2 = feat[f"p_d2__{tag}"].to_numpy(dtype=float)
    leg = np.minimum(np.abs(d1), np.abs(d2))
    reversal = (d1 * d2) < 0.0
    strong = reversal & (leg >= thr["thr_leg"])
    fall_then_rise = strong & (d1 < 0.0) & (d2 > 0.0)
    cool = feat[f"t_delta__{tag}"].to_numpy(dtype=float) <= thr["thr_t_delta"]
    wind_dir = feat["wd_abs_turn"].to_numpy(dtype=float) >= thr["thr_wd"]
    wind_spd = feat[f"wv_abs_delta__{tag}"].to_numpy(dtype=float) >= thr["thr_wv"]
    return {
        "pressure_reversal": strong,
        "pressure_fall_then_rise": fall_then_rise,
        "rapid_cooling": cool,
        "wind_dir": wind_dir,
        "wind_speed": wind_spd,
        "front_dir": strong & cool & wind_dir,
        "front_dir_fall_rise": fall_then_rise & cool & wind_dir,
        "front_speed": strong & cool & wind_spd,
        "front_either_wind": strong & cool & (wind_dir | wind_spd),
    }


def component_counts(masks, spans):
    """Dem mask theo dung khoang cua tung tap. Cua so nam trong purge gap khong bi dem."""
    rows = []
    for name, mask in masks.items():
        m = np.asarray(mask, dtype=bool)
        row = {"component": name}
        for split, (a, b) in spans.items():
            part = m[a:b]
            row[f"n_{split}"] = int(part.sum())
            row[f"share_{split}"] = float(part.mean()) if len(part) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def extremity_z(values, train_span, op):
    """Diem z theo huong cua luat, mean/std (ddof=0) chi tu train. >= : (x-mu)/sd; <= : (mu-x)/sd."""
    if op not in (">=", "<="):
        raise ValueError(f"Toan tu khong ho tro: {op}")
    v = np.asarray(values, dtype=float)
    a, b = train_span
    tr = v[a:b]
    tr = tr[np.isfinite(tr)]
    if len(tr) == 0:
        raise ValueError("Khong co gia tri train huu han de tinh z")
    mu = float(tr.mean())
    sd = float(tr.std())
    if sd == 0.0:
        sd = 1.0
    z = (v - mu) / sd if op == ">=" else (mu - v) / sd
    return z, mu, sd


def assign_max_z(feats, thresholds, train_span, priority):
    """Trong cac luat dang kich hoat, lay z lon nhat. Hoa: luat dung truoc trong priority.

    Khong luat nao kich hoat -> 'normal'. Tra (nhan, bang z, thong ke train, so cua so hoa).
    """
    fires = fire_matrix(feats, thresholds)
    z_cols, stats = {}, {}
    for name in priority:
        t = thresholds[name]
        z, mu, sd = extremity_z(feats[t["feature"]].to_numpy(dtype=float), train_span, t["op"])
        z_cols[name] = z
        stats[name] = {"mean_train": mu, "std_train": sd, "op": t["op"], "feature": t["feature"]}
    Z = pd.DataFrame(z_cols, index=feats.index)
    score = Z.copy()
    for name in priority:
        score.loc[~fires[name].to_numpy(), name] = -np.inf
    arr = score.to_numpy(dtype=float)
    mx = np.max(arr, axis=1)
    any_fire = np.isfinite(mx)
    n_at = np.sum(arr == mx[:, None], axis=1)
    n_ties = int(np.sum(any_fire & (n_at > 1)))
    labels = np.full(len(feats), "normal", dtype=object)
    assigned = np.zeros(len(feats), dtype=bool)
    for i, name in enumerate(priority):
        take = any_fire & ~assigned & (arr[:, i] == mx)
        labels[take] = name
        assigned |= take
    return pd.Series(labels, index=feats.index), Z, stats, n_ties


def assignment_diff(priority_labels, z_labels, spans, fires, priority):
    """Phan tram cua so doi nhan giua uu tien co dinh va z lon nhat, tren moi tap."""
    multi = fires[list(priority)].sum(axis=1) >= 2
    rows = []
    for split, (a, b) in spans.items():
        p = priority_labels.iloc[a:b].to_numpy()
        z = z_labels.iloc[a:b].to_numpy()
        m = multi.iloc[a:b].to_numpy()
        changed = p != z
        n_multi = int(m.sum())
        rows.append({
            "split": split,
            "n": int(len(p)),
            "n_changed": int(changed.sum()),
            "pct_changed": float(changed.mean()) if len(p) else float("nan"),
            "n_multi_fire": n_multi,
            "n_changed_multi": int((changed & m).sum()),
            "pct_changed_multi": float((changed & m).sum() / n_multi) if n_multi else float("nan"),
        })
    return pd.DataFrame(rows)


def shift_quantiles(q_spike, q_low, q_high, which, delta_pp):
    """Dich phan vi them delta_pp diem phan tram. which: spike|low|high|all. Cat ve [0, 1]."""
    if which not in ("spike", "low", "high", "all"):
        raise ValueError(which)
    base = {"spike": float(q_spike), "low": float(q_low), "high": float(q_high)}
    keys = list(base) if which == "all" else [which]
    out, clipped = dict(base), {k: False for k in base}
    for k in keys:
        raw = base[k] + float(delta_pp) / 100.0
        clipped_q = min(1.0, max(0.0, raw))
        out[k] = clipped_q
        clipped[k] = abs(clipped_q - raw) > 1e-12
    return out, clipped


def iter_sensitivity_specs(mode, spike_channel, q_spike, q_low, q_high, priority, steps_pp):
    """Sinh spec: tung luat mot, roi tat ca phan vi cung luc. Moi buoc la ± diem phan tram."""
    base = rule_specs(mode, spike_channel, q_spike, q_low, q_high)
    yield "baseline", "baseline", 0, dict(base), False, None
    for rule in priority:
        col, op, q0 = base[rule]
        q0 = float(q0)
        for step in steps_pp:
            for sign in (-1, 1):
                dpp = sign * int(step)
                raw = q0 + dpp / 100.0
                new_q = min(1.0, max(0.0, raw))
                spec = dict(base)
                spec[rule] = (col, op, new_q)
                yield "rule", rule, dpp, spec, abs(new_q - raw) > 1e-12, new_q
    for step in steps_pp:
        for sign in (-1, 1):
            dpp = sign * int(step)
            qs, flags = shift_quantiles(q_spike, q_low, q_high, "all", dpp)
            spec = rule_specs(mode, spike_channel, qs["spike"], qs["low"], qs["high"])
            yield "all", "all", dpp, spec, any(flags.values()), None


def consecutive_runs(lab):
    """Do dai va nhan cua tung cum lien tiep cung nhan. Tra (do dai, nhan cum)."""
    lab = np.asarray(lab)
    n = len(lab)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=object)
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = lab[1:] != lab[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)
    return (ends - starts).astype(int), lab[starts]


def run_length_summary(labels, spans, names):
    """Do dai cum trong tung tap (khong noi qua purge gap). Tra (theo nhan, tong)."""
    per_label, overall = [], []
    for split, (lo, hi) in spans.items():
        lab = np.asarray(labels.iloc[lo:hi])
        lengths, run_lab = consecutive_runs(lab)
        overall.append({
            "split": split,
            "n_windows": int(len(lab)),
            "n_runs": int(len(lengths)),
            "mean_run_length": float(lengths.mean()) if len(lengths) else float("nan"),
            "median_run_length": float(np.median(lengths)) if len(lengths) else float("nan"),
        })
        for name in names:
            lens = lengths[run_lab == name]
            per_label.append({
                "split": split,
                "label": name,
                "n_windows": int((lab == name).sum()) if len(lab) else 0,
                "n_runs": int(len(lens)),
                "mean_run_length": float(lens.mean()) if len(lens) else float("nan"),
                "median_run_length": float(np.median(lens)) if len(lens) else float("nan"),
                "max_run_length": int(lens.max()) if len(lens) else 0,
            })
    return pd.DataFrame(per_label), pd.DataFrame(overall)


def sample_window_indices(labels, span, names, n, seed):
    """Toi da n chi so cua so moi nhan, trong span, khong lap. Seed co dinh. Ket qua sap theo thoi gian."""
    rng = np.random.default_rng(seed)
    lo, hi = span
    part = labels.iloc[lo:hi].to_numpy()
    out = {}
    for name in names:
        idx = np.flatnonzero(part == name) + lo
        if len(idx) == 0:
            out[name] = np.array([], dtype=int)
            continue
        take = min(int(n), len(idx))
        choice = rng.choice(idx, size=take, replace=False)
        out[name] = np.sort(choice.astype(int))
    return out
