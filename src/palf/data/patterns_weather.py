"""Pattern Analyzer v1 cho du lieu thoi tiet (Jena Climate) - Ban dong bang.

Dac ta: PALF_Pattern_Spec_v1.md.
Bo gom dung 4 nhan:
1. pressure_drop (uu tien 1): Delta p (phan du) <= p10 (-12.219 mbar)
2. humidity_net_rise (uu tien 2): Delta rh (phan du) >= p90 (+20.428 %)
3. temp_anomaly (uu tien 3): A_T >= p90 (+4.7405 degC) hoac <= p10 (-4.6102 degC)
4. normal (uu tien 4): khong thoa luat nao

Sieu du lieu: temp_sign (+1 heat, -1 cold, 0 cho cac nhan khac/normal).
Nguyet tac:
- Nguong va khi hau chi tinh tren train (sau purge gap).
- Nhan va dac trung chi dung X, khong dung y.
"""
import json
import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from .pattern_features import CHANNELS, climatology, window_features, window_split

WEATHER_PATTERNS = ["pressure_drop", "humidity_net_rise", "temp_anomaly", "normal"]
PATTERN_TO_ID = {name: i for i, name in enumerate(WEATHER_PATTERNS)}
ID_TO_PATTERN = {i: name for i, name in enumerate(WEATHER_PATTERNS)}

DEFAULT_QUANTILES = {
    "pressure_drop": 0.10,
    "humidity_net_rise": 0.90,
    "temp_heat": 0.90,
    "temp_cold": 0.10,
}


def fit_weather_patterns(df, L=96, H=24, train_rows=None, train_windows=None,
                         purge=None, clim_key="month_hour"):
    """Tinh khi hau month_hour va cac nguong phan vi chi tren tap train.

    df: DataFrame voi DatetimeIndex va cac cot kenh CHANNELS.
    """
    n_rows = len(df)
    sp = window_split(n_rows, L, H, purge=purge)
    n_w = sp["n_windows"]
    tr_span = sp["spans"]["train"]
    tr_w = tr_span[1] - tr_span[0] if train_windows is None else int(train_windows)
    tr_rows = sp["train_rows"] if train_rows is None else int(train_rows)

    # 1. Tinh khi hau tren train_rows hang dau
    clim_df = climatology(df, tr_rows, keys=clim_key)

    # 2. Tinh phan du cho cac kenh p, T, rh
    p_col = CHANNELS["p"]
    t_col = CHANNELS["T"]
    rh_col = CHANNELS["rh"]

    p_resid = df[p_col].to_numpy(dtype=float) - clim_df[p_col].to_numpy(dtype=float)
    t_resid = df[t_col].to_numpy(dtype=float) - clim_df[t_col].to_numpy(dtype=float)
    rh_resid = df[rh_col].to_numpy(dtype=float) - clim_df[rh_col].to_numpy(dtype=float)

    # 3. Tinh dac trung tren cac cua so train
    # Delta = x_L - x_1 (cuoi tru dau)
    w_p = sliding_window_view(p_resid, L)[:tr_w]
    w_rh = sliding_window_view(rh_resid, L)[:tr_w]
    w_t = sliding_window_view(t_resid, L)[:tr_w]

    delta_p_train = w_p[:, -1] - w_p[:, 0]
    delta_rh_train = w_rh[:, -1] - w_rh[:, 0]
    anom_t_train = w_t.mean(axis=1)

    # 4. Tinh cac nguong phan vi
    thr_pressure_drop = float(pd.Series(delta_p_train).quantile(DEFAULT_QUANTILES["pressure_drop"]))
    thr_humidity_rise = float(pd.Series(delta_rh_train).quantile(DEFAULT_QUANTILES["humidity_net_rise"]))
    thr_temp_heat = float(pd.Series(anom_t_train).quantile(DEFAULT_QUANTILES["temp_heat"]))
    thr_temp_cold = float(pd.Series(anom_t_train).quantile(DEFAULT_QUANTILES["temp_cold"]))

    # Luu thong tin profile khi hau (month, hour) -> mean
    clim_profile = {}
    train_slice = df.iloc[:tr_rows]
    for c in [p_col, t_col, rh_col]:
        grouped = train_slice[c].groupby([train_slice.index.month, train_slice.index.hour]).mean()
        clim_profile[c] = {f"{m:02d}_{h:02d}": float(v) for (m, h), v in grouped.items()}

    fitted = {
        "pattern_scheme": "weather_v1",
        "pattern_names": WEATHER_PATTERNS,
        "L": int(L),
        "H": int(H),
        "purge_gap": int(sp["gap"]),
        "train_rows": int(tr_rows),
        "train_windows": int(tr_w),
        "clim_key": clim_key,
        "quantiles": DEFAULT_QUANTILES,
        "thresholds": {
            "pressure_drop": {
                "feature": "p__delta__resid",
                "op": "<=",
                "quantile": DEFAULT_QUANTILES["pressure_drop"],
                "threshold": thr_pressure_drop,
                "unit": "mbar",
            },
            "humidity_net_rise": {
                "feature": "rh__delta__resid",
                "op": ">=",
                "quantile": DEFAULT_QUANTILES["humidity_net_rise"],
                "threshold": thr_humidity_rise,
                "unit": "%",
            },
            "temp_anomaly": {
                "feature": "T__anom",
                "heat_op": ">=",
                "heat_quantile": DEFAULT_QUANTILES["temp_heat"],
                "heat_threshold": thr_temp_heat,
                "cold_op": "<=",
                "cold_quantile": DEFAULT_QUANTILES["temp_cold"],
                "cold_threshold": thr_temp_cold,
                "unit": "degC",
            },
        },
        "climatology_profile": clim_profile,
    }
    return fitted


def _extract_window_features(df_or_feats, fitted, L=96, H=None):
    """Trich xuat 3 dac trung can thiet (delta_p_resid, delta_rh_resid, anom_t) tu feats hoac df."""
    if isinstance(df_or_feats, pd.DataFrame) and "p__delta__resid" in df_or_feats.columns:
        dp = df_or_feats["p__delta__resid"].to_numpy(dtype=float)
        drh = df_or_feats["rh__delta__resid"].to_numpy(dtype=float)
        at = df_or_feats["T__anom"].to_numpy(dtype=float)
        return dp, drh, at

    # Truong hop nhan vao df goc (co DatetimeIndex)
    df = df_or_feats
    clim_prof = fitted["climatology_profile"]
    p_col, t_col, rh_col = CHANNELS["p"], CHANNELS["T"], CHANNELS["rh"]

    # Reconstruct climatology series from profile
    months = df.index.month
    hours = df.index.hour
    keys = [f"{m:02d}_{h:02d}" for m, h in zip(months, hours)]

    p_clim = np.array([clim_prof[p_col][k] for k in keys], dtype=float)
    t_clim = np.array([clim_prof[t_col][k] for k in keys], dtype=float)
    rh_clim = np.array([clim_prof[rh_col][k] for k in keys], dtype=float)

    p_resid = df[p_col].to_numpy(dtype=float) - p_clim
    t_resid = df[t_col].to_numpy(dtype=float) - t_clim
    rh_resid = df[rh_col].to_numpy(dtype=float) - rh_clim

    h_val = fitted.get("H", 0) if H is None else H
    n_w = len(df) - L - h_val + 1
    w_p = sliding_window_view(p_resid, L)[:n_w]
    w_rh = sliding_window_view(rh_resid, L)[:n_w]
    w_t = sliding_window_view(t_resid, L)[:n_w]

    dp = w_p[:, -1] - w_p[:, 0]
    drh = w_rh[:, -1] - w_rh[:, 0]
    at = w_t.mean(axis=1)
    return dp, drh, at


def apply_weather_patterns(df_or_feats, fitted, L=96, H=None):
    """Gan nhan theo 4 luat va thu tu uu tien cua Spec v1.

    Tra ve:
        labels: np.ndarray (dtype=object) chua ten 4 nhan.
        temp_sign: np.ndarray (dtype=int64) chua +1 (heat), -1 (cold), 0 (khong ap dung).
        label_ids: np.ndarray (dtype=int64) chua 0, 1, 2, 3 tuong ung WEATHER_PATTERNS.
    """
    dp, drh, at = _extract_window_features(
        df_or_feats, fitted, L=fitted.get("L", L), H=fitted.get("H", H)
    )
    n = len(dp)

    th = fitted["thresholds"]
    thr_p = th["pressure_drop"]["threshold"]
    thr_rh = th["humidity_net_rise"]["threshold"]
    thr_th = th["temp_anomaly"]["heat_threshold"]
    thr_tc = th["temp_anomaly"]["cold_threshold"]

    fire_p = dp <= thr_p
    fire_rh = drh >= thr_rh
    fire_heat = at >= thr_th
    fire_cold = at <= thr_tc
    fire_temp = fire_heat | fire_cold

    labels = np.full(n, "normal", dtype=object)
    label_ids = np.full(n, PATTERN_TO_ID["normal"], dtype=np.int64)
    temp_sign = np.zeros(n, dtype=np.int64)

    # Priority 1: pressure_drop
    m_p = fire_p
    labels[m_p] = "pressure_drop"
    label_ids[m_p] = PATTERN_TO_ID["pressure_drop"]

    # Priority 2: humidity_net_rise
    m_rh = (~m_p) & fire_rh
    labels[m_rh] = "humidity_net_rise"
    label_ids[m_rh] = PATTERN_TO_ID["humidity_net_rise"]

    # Priority 3: temp_anomaly
    m_temp = (~m_p) & (~m_rh) & fire_temp
    labels[m_temp] = "temp_anomaly"
    label_ids[m_temp] = PATTERN_TO_ID["temp_anomaly"]

    # Ghi dau cho temp_anomaly (+1 heat, -1 cold, cac nhan khac luon = 0)
    temp_sign[m_temp & fire_heat] = 1
    temp_sign[m_temp & fire_cold] = -1

    return labels, temp_sign, label_ids


class WeatherPatternAnalyzer:
    """Class quan ly quy trinh fit, apply va serialization cho Pattern Spec v1."""

    def __init__(self, fitted=None):
        self.fitted = fitted

    def fit(self, df, L=96, H=24, train_rows=None, train_windows=None, purge=None):
        self.fitted = fit_weather_patterns(
            df, L=L, H=H, train_rows=train_rows, train_windows=train_windows, purge=purge
        )
        return self

    def apply(self, df_or_feats):
        if self.fitted is None:
            raise RuntimeError("Analyzer chua duoc fit. Goi fit() truoc.")
        return apply_weather_patterns(df_or_feats, self.fitted)

    def save_json(self, path):
        if self.fitted is None:
            raise RuntimeError("Khong co du lieu fitted de luu.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.fitted, f, indent=2)

    @classmethod
    def load_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            fitted = json.load(f)
        return cls(fitted=fitted)
