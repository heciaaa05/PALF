import numpy as np
import pandas as pd

from palf.data.pattern_features import (
    add_wind_direction_components, build_feature_table, load_hourly_with_time,
    variance_explained, window_features, window_split,
)
from palf.data.weather import load_jena

CH = {"p": "p (mbar)", "T": "T (degC)", "rh": "rh (%)", "wv": "wv (m/s)"}


def _make_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="h")
    return pd.DataFrame({c: rng.normal(size=n).cumsum() for c in CH.values()}, index=idx)


def test_window_features_known_values():
    f = window_features(np.arange(10, dtype=float), L=4, n_w=5)
    assert np.allclose(f["delta"], 3.0)
    assert np.allclose(f["jump"], 1.0)
    assert np.allclose(f["sigma"], np.std([0, 1, 2, 3]))
    assert np.allclose(f["mean"], [1.5, 2.5, 3.5, 4.5, 5.5])


def test_window_split_purge_shares_no_row():
    L, H, n = 6, 3, 500
    sp = window_split(n_rows=n, L=L, H=H)
    s = sp["spans"]

    def rows(span):
        a, b = span
        return a, (b - 1) + L + H - 1  # hang dau, hang cuoi (gom ca y)

    tr, va, te = rows(s["train"]), rows(s["val"]), rows(s["test"])
    assert sp["gap"] == L + H - 1
    assert tr[1] < va[0]
    assert va[1] < te[0]
    assert te[1] <= n - 1


def test_train_features_ignore_everything_after_train_inputs():
    L, H = 24, 6
    df = _make_df(3000)
    sp = window_split(len(df), L, H)
    n_w = sp["n_windows"]
    a, b = sp["spans"]["train"]
    f1, c1 = build_feature_table(df, CH, L, n_w, sp["train_rows"], "hour")
    df2 = df.copy()
    df2.iloc[sp["train_rows"]:] += 1000.0  # doi y cua cua so train cuoi va toan bo val/test
    f2, c2 = build_feature_table(df2, CH, L, n_w, sp["train_rows"], "hour")
    assert np.allclose(c1.to_numpy(), c2.to_numpy(), equal_nan=True)
    assert np.allclose(f1.iloc[a:b].to_numpy(), f2.iloc[a:b].to_numpy(), equal_nan=True)
    for col in f1.columns:
        assert f1[col].iloc[a:b].quantile(0.9) == f2[col].iloc[a:b].quantile(0.9)


def test_loader_matches_weather_load_jena(tmp_path):
    n = 60
    idx = pd.date_range("2009-01-01 00:10:00", periods=n, freq="10min")
    err = np.arange(n) % 6 == 0
    raw = pd.DataFrame({
        "Date Time": idx.strftime("%d.%m.%Y %H:%M:%S"),
        "p (mbar)": np.linspace(990, 1000, n),
        "T (degC)": np.linspace(0, 5, n),
        "wv (m/s)": np.where(err, -9999.0, 1.5),
        "max. wv (m/s)": np.where(err, -9999.0, 2.5),
    })
    path = tmp_path / "mini.csv"
    raw.to_csv(path, index=False)
    mine, n_err = load_hourly_with_time(str(path), subsample=6)
    ref = load_jena(str(path), subsample=6)
    assert np.allclose(mine.to_numpy(), ref.to_numpy())
    assert mine.index.is_monotonic_increasing
    assert n_err > 0
    assert (mine.filter(like="wv") >= 0).all().all()


def test_wind_direction_components_are_unit_circle():
    df = pd.DataFrame({"wd (deg)": [0.0, 90.0, 180.0, 270.0, 359.0], "x": 1.0})
    out = add_wind_direction_components(df)
    assert "wd (deg)" not in out.columns
    assert np.allclose(out["wd_sin"] ** 2 + out["wd_cos"] ** 2, 1.0)
    assert np.isclose(out["wd_sin"].iloc[1], 1.0)


def test_delta_raw_minus_resid_is_climate_at_endpoints():
    n, L, H = 24 * 40, 48, 6
    idx = pd.date_range("2012-01-01", periods=n, freq="h")
    hour = np.asarray(idx.hour)
    df = pd.DataFrame({
        "p (mbar)": 1000 + 2 * np.sin(2 * np.pi * hour / 24),
        "T (degC)": 10 + 8 * np.sin(2 * np.pi * hour / 24),
        "rh (%)": 60 + 10 * np.cos(2 * np.pi * hour / 24),
        "wv (m/s)": 2 + 0.05 * hour,
    }, index=idx)
    sp = window_split(n, L, H)
    feats, clim = build_feature_table(df, CH, L, sp["n_windows"], sp["train_rows"], "hour")
    j = np.arange(sp["n_windows"])
    for short, col in CH.items():
        c = clim[col].to_numpy(dtype=float)
        got = feats[f"{short}__delta__raw"].to_numpy() - feats[f"{short}__delta__resid"].to_numpy()
        assert np.allclose(got, c[j + L - 1] - c[j])


def test_variance_explained_extremes():
    g = np.array([0, 1] * 50)
    assert np.isclose(variance_explained(g.astype(float), g), 1.0)
    rng = np.random.default_rng(0)
    assert np.isclose(variance_explained(rng.normal(size=100), np.zeros(100)), 0.0)