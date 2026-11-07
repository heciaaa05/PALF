import numpy as np
import pandas as pd
import pytest
import torch

from palf.data.patterns_weather import (
    WEATHER_PATTERNS,
    WeatherPatternAnalyzer,
    apply_weather_patterns,
    fit_weather_patterns,
)
from palf.data.pattern_features import CHANNELS, load_hourly_with_time, window_split
from palf.data.weather import prepare_weather_data

CH = {"p": "p (mbar)", "T": "T (degC)", "rh": "rh (%)", "wv": "wv (m/s)"}


def _dummy_weather_df(n=500, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="h")
    df = pd.DataFrame({c: rng.normal(size=n).cumsum() for c in CH.values()}, index=idx)
    df["wd (deg)"] = rng.uniform(0, 360, size=n)
    return df


def test_priority_pressure_drop_beats_temp_anomaly():
    """Thu tu uu tien: cua so thoa ca pressure_drop va temp_anomaly phai nhan pressure_drop."""
    fitted = {
        "pattern_scheme": "weather_v1",
        "thresholds": {
            "pressure_drop": {"threshold": -10.0},
            "humidity_net_rise": {"threshold": 20.0},
            "temp_anomaly": {"heat_threshold": 4.0, "cold_threshold": -4.0},
        },
    }
    feats = pd.DataFrame({
        "p__delta__resid": [-15.0, -15.0, 0.0],
        "rh__delta__resid": [0.0, 0.0, 25.0],
        "T__anom": [6.0, -6.0, 5.0],
    })
    labels, temp_sign, label_ids = apply_weather_patterns(feats, fitted)

    assert labels[0] == "pressure_drop"
    assert temp_sign[0] == 0

    assert labels[1] == "pressure_drop"
    assert temp_sign[1] == 0

    assert labels[2] == "humidity_net_rise"
    assert temp_sign[2] == 0


def test_normal_when_no_rule_and_total_count():
    """Cua so khong thoa luat nao nhan normal; tong nhan bang so cua so."""
    fitted = {
        "pattern_scheme": "weather_v1",
        "thresholds": {
            "pressure_drop": {"threshold": -10.0},
            "humidity_net_rise": {"threshold": 20.0},
            "temp_anomaly": {"heat_threshold": 4.0, "cold_threshold": -4.0},
        },
    }
    feats = pd.DataFrame({
        "p__delta__resid": [0.0, 5.0, -5.0],
        "rh__delta__resid": [0.0, 10.0, -10.0],
        "T__anom": [0.0, 1.0, -1.0],
    })
    labels, temp_sign, label_ids = apply_weather_patterns(feats, fitted)

    assert len(labels) == 3
    assert list(labels) == ["normal", "normal", "normal"]
    assert np.all(temp_sign == 0)
    assert np.all(label_ids == 3)


def test_temp_sign_values_and_isolation():
    """temp_sign dung dau (+1 heat, -1 cold) va bang 0 voi cac nhan khac."""
    fitted = {
        "pattern_scheme": "weather_v1",
        "thresholds": {
            "pressure_drop": {"threshold": -10.0},
            "humidity_net_rise": {"threshold": 20.0},
            "temp_anomaly": {"heat_threshold": 4.0, "cold_threshold": -4.0},
        },
    }
    feats = pd.DataFrame({
        "p__delta__resid": [0.0, 0.0, -12.0, 0.0],
        "rh__delta__resid": [0.0, 0.0, 0.0, 22.0],
        "T__anom": [5.0, -5.0, 5.0, -5.0],
    })
    labels, temp_sign, label_ids = apply_weather_patterns(feats, fitted)

    assert labels[0] == "temp_anomaly"
    assert temp_sign[0] == 1

    assert labels[1] == "temp_anomaly"
    assert temp_sign[1] == -1

    assert labels[2] == "pressure_drop"
    assert temp_sign[2] == 0

    assert labels[3] == "humidity_net_rise"
    assert temp_sign[3] == 0


def test_changing_only_forecast_targets_does_not_change_labels_or_thresholds():
    """Doi y ngau nhien khong lam doi nhan hoac nguong (chong ro ri y)."""
    L, H, n = 24, 6, 400
    df = _dummy_weather_df(n=n, seed=10)
    sp = window_split(n, L, H)

    df_y = df.copy()
    df_y.iloc[-H:] += 10000.0

    fit_a = fit_weather_patterns(df, L=L, H=H, train_rows=sp["train_rows"],
                                 train_windows=sp["spans"]["train"][1], purge=sp["gap"])
    fit_b = fit_weather_patterns(df_y, L=L, H=H, train_rows=sp["train_rows"],
                                 train_windows=sp["spans"]["train"][1], purge=sp["gap"])

    for k in ("pressure_drop", "humidity_net_rise"):
        assert fit_a["thresholds"][k]["threshold"] == pytest.approx(fit_b["thresholds"][k]["threshold"])
    assert fit_a["thresholds"]["temp_anomaly"]["heat_threshold"] == pytest.approx(
        fit_b["thresholds"]["temp_anomaly"]["heat_threshold"]
    )
    assert fit_a["thresholds"]["temp_anomaly"]["cold_threshold"] == pytest.approx(
        fit_b["thresholds"]["temp_anomaly"]["cold_threshold"]
    )

    lab_a, sign_a, ids_a = apply_weather_patterns(df, fit_a, L=L, H=H)
    lab_b, sign_b, ids_b = apply_weather_patterns(df_y, fit_b, L=L, H=H)

    assert np.array_equal(lab_a, lab_b)
    assert np.array_equal(sign_a, sign_b)
    assert np.array_equal(ids_a, ids_b)


def test_changing_val_test_does_not_change_fitted_thresholds():
    """Doi val/test khong lam doi nguong da fit tu train."""
    L, H, n = 24, 6, 400
    df = _dummy_weather_df(n=n, seed=20)
    sp = window_split(n, L, H)

    fit_base = fit_weather_patterns(df, L=L, H=H, train_rows=sp["train_rows"],
                                    train_windows=sp["spans"]["train"][1], purge=sp["gap"])

    val0 = sp["train_rows"] + sp["gap"]
    df_tweaked = df.copy()
    df_tweaked.iloc[val0:] += 500.0

    fit_tweaked = fit_weather_patterns(df_tweaked, L=L, H=H, train_rows=sp["train_rows"],
                                       train_windows=sp["spans"]["train"][1], purge=sp["gap"])

    for k in ("pressure_drop", "humidity_net_rise"):
        assert fit_base["thresholds"][k]["threshold"] == pytest.approx(fit_tweaked["thresholds"][k]["threshold"])
    assert fit_base["thresholds"]["temp_anomaly"]["heat_threshold"] == pytest.approx(
        fit_tweaked["thresholds"]["temp_anomaly"]["heat_threshold"]
    )
    assert fit_base["thresholds"]["temp_anomaly"]["cold_threshold"] == pytest.approx(
        fit_tweaked["thresholds"]["temp_anomaly"]["cold_threshold"]
    )


def test_exact_threshold_values_on_jena():
    """Kiem tra cac nguong fit tu Jena train khop dung Spec v1 da dong bang."""
    df, _ = load_hourly_with_time("data/raw/jena_climate_2009_2016.csv")
    sp = window_split(len(df), 96, 24)
    fitted = fit_weather_patterns(
        df, L=96, H=24, train_rows=sp["train_rows"],
        train_windows=sp["spans"]["train"][1], purge=sp["gap"]
    )
    th = fitted["thresholds"]
    assert th["pressure_drop"]["threshold"] == pytest.approx(-12.219, abs=1e-3)
    assert th["humidity_net_rise"]["threshold"] == pytest.approx(20.428, abs=1e-3)
    assert th["temp_anomaly"]["heat_threshold"] == pytest.approx(4.7405, abs=1e-3)
    assert th["temp_anomaly"]["cold_threshold"] == pytest.approx(-4.6102, abs=1e-3)


def test_prepare_weather_data_contracts():
    """Kiem tra hop dong loader prepare_weather_data voi weather_v1 va generic."""
    # weather_v1
    tr, va, te, th, meta = prepare_weather_data(pattern_scheme="weather_v1")
    assert len(tr) == 48981
    assert len(va) == 10377
    assert len(te) == 10377
    assert th["pattern_names"] == WEATHER_PATTERNS
    assert meta["pattern_names"] == WEATHER_PATTERNS
    assert "temp_sign" in meta
    assert len(meta["temp_sign"]) == 69973
    assert set(np.unique(meta["temp_sign"])).issubset({-1, 0, 1})

    # generic
    tr_g, va_g, te_g, th_g, meta_g = prepare_weather_data(pattern_scheme="generic")
    assert th_g["pattern_names"] == ["stable", "rising", "falling", "spike"]
    assert meta_g["pattern_names"] == ["stable", "rising", "falling", "spike"]
