import math

import numpy as np
import pandas as pd
import pytest

from palf.data.pattern_diagnostics import (
    assign_max_z, assignment_diff, circular_abs_delta_deg, cold_front_components,
    cold_front_feature_frame, component_counts, consecutive_runs, extremity_z,
    fit_cold_front_thresholds, half_deltas, iter_sensitivity_specs, pearson,
    run_length_summary, sample_window_indices, shift_quantiles, turn_magnitude,
)
from palf.data.pattern_features import build_feature_table, window_features, window_split
from palf.data.pattern_labels import (
    RULE_ORDER, add_raw_mean, assign_labels, fire_matrix, fit_thresholds, rule_specs,
)

CH = {"p": "p (mbar)", "T": "T (degC)", "rh": "rh (%)", "wv": "wv (m/s)"}


def test_pearson_extremes():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert math.isnan(pearson([1, 1, 1], [1, 2, 3]))


def test_half_deltas_and_turn_magnitude():
    x = np.array([10, 1, 1, 1, 0, 2, 2, 8], dtype=float)
    d1, d2 = half_deltas(x, L=8, n_w=1)
    assert d1[0] == pytest.approx(-10.0)
    assert d2[0] == pytest.approx(8.0)
    mag, rev = turn_magnitude(d1, d2)
    assert bool(rev[0]) and mag[0] == pytest.approx(8.0)
    mag2, rev2 = turn_magnitude(np.array([2.0, 0.0]), np.array([3.0, -1.0]))
    assert list(rev2) == [False, False]
    assert np.allclose(mag2, 0.0)


def test_circular_delta_crosses_zero_not_raw_difference():
    wd = np.array([350, 0, 0, 10, 10, 180], dtype=float)
    out = circular_abs_delta_deg(wd, L=4, n_w=3)
    assert np.allclose(out, [20.0, 10.0, 180.0])
    assert abs(350 - 10) != pytest.approx(out[0])


def test_cold_front_known_mask_and_span_counts():
    feat = pd.DataFrame({
        "p_d1__raw": [-8, -8, 5, -1, -8, -8],
        "p_d2__raw": [6, 6, 5, -1, 6, 6],
        "t_delta__raw": [-5, 1, -5, -5, -5, -5],
        "wd_abs_turn": [90, 90, 90, 90, 5, 90],
        "wv_abs_delta__raw": [0, 0, 0, 0, 0, 4],
    })
    thr = {"thr_leg": 6.0, "thr_t_delta": -5.0, "thr_wd": 90.0, "thr_wv": 3.0}
    masks = cold_front_components(feat, thr, "raw")
    assert list(masks["front_dir"]) == [True, False, False, False, False, True]
    assert list(masks["front_speed"]) == [False, False, False, False, False, True]
    assert list(masks["pressure_reversal"]) == [True, True, False, False, True, True]
    counts = component_counts(
        {"front_dir": np.array([True, True, True, False, False, True])},
        {"train": (0, 2), "val": (3, 6)},
    )
    assert int(counts.loc[0, "n_train"]) == 2
    assert int(counts.loc[0, "n_val"]) == 1  # index 2 nam trong purge gap, khong dem


def test_cold_front_frame_matches_window_delta_and_train_quantile():
    rng = np.random.default_rng(0)
    n, L = 80, 8
    p = rng.normal(size=n).cumsum()
    t = rng.normal(size=n).cumsum()
    frame = cold_front_feature_frame(p, t, p - 1.0, t - 2.0, rng.uniform(0, 360, n),
                                    np.abs(window_features(p, L, n - L + 1)["delta"]),
                                    np.abs(window_features(t, L, n - L + 1)["delta"]),
                                    L, n - L + 1)
    assert np.allclose(frame["t_delta__raw"], window_features(t, L, n - L + 1)["delta"])
    thr = fit_cold_front_thresholds(frame, (0, 40), "raw", 0.75, 0.10, 0.90)
    leg = np.minimum(np.abs(frame["p_d1__raw"].iloc[:40]), np.abs(frame["p_d2__raw"].iloc[:40]))
    assert thr["thr_leg"] == pytest.approx(float(pd.Series(leg).quantile(0.75)))
    frame2 = frame.copy()
    frame2.iloc[40:] += 1000.0
    thr2 = fit_cold_front_thresholds(frame2, (0, 40), "raw", 0.75, 0.10, 0.90)
    assert thr2["thr_leg"] == pytest.approx(thr["thr_leg"])
    assert thr2["thr_t_delta"] == pytest.approx(thr["thr_t_delta"])
    assert thr2["thr_wd"] == pytest.approx(thr["thr_wd"])


def test_max_z_beats_priority_only_when_more_extreme_and_ties_break_by_order():
    a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    feats = pd.DataFrame({"a": a, "b": b})
    th = {
        "spike": {"feature": "a", "op": ">=", "quantile": 0.9, "threshold": 3.5},
        "pressure_drop": {"feature": "b", "op": ">=", "quantile": 0.9, "threshold": 3.5},
    }
    priority = ["spike", "pressure_drop"]
    pri = assign_labels(fire_matrix(feats, th), priority)
    zlab, Z, _, n_ties = assign_max_z(feats, th, (0, 5), priority)
    assert list(pri) == ["normal", "normal", "normal", "normal", "spike"]
    assert list(zlab) == ["normal", "normal", "normal", "normal", "pressure_drop"]
    assert Z["pressure_drop"].iloc[-1] > Z["spike"].iloc[-1]
    assert n_ties == 0

    tied = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 1.0, 2.0, 3.0]})
    th_tie = {
        "spike": {"feature": "a", "op": ">=", "quantile": 0.5, "threshold": 2.5},
        "pressure_drop": {"feature": "b", "op": ">=", "quantile": 0.5, "threshold": 2.5},
    }
    ztie, _, _, n_tie = assign_max_z(tied, th_tie, (0, 4), priority)
    assert list(ztie) == ["normal", "normal", "normal", "spike"]
    assert n_tie == 1
    diff = assignment_diff(pri, zlab, {"all": (0, 5)}, fire_matrix(feats, th), priority)
    assert int(diff.loc[0, "n_changed"]) == 1
    assert diff.loc[0, "pct_changed"] == pytest.approx(0.2)


def test_extremity_z_flips_with_operator():
    z_hi, mu, sd = extremity_z([0.0, 1.0, 2.0], (0, 3), ">=")
    z_lo, _, _ = extremity_z([0.0, 1.0, 2.0], (0, 3), "<=")
    assert mu == pytest.approx(1.0)
    assert np.allclose(z_hi, -z_lo)
    assert sd > 0


def test_shift_quantiles_clips_and_sensitivity_spec_isolates_one_rule():
    qs, clipped = shift_quantiles(0.99, 0.10, 0.90, "spike", 2)
    assert qs["spike"] == pytest.approx(1.0) and clipped["spike"]
    assert qs["low"] == pytest.approx(0.10) and not clipped["low"]
    qs, clipped = shift_quantiles(0.99, 0.10, 0.90, "all", -2)
    assert (qs["spike"], qs["low"], qs["high"]) == pytest.approx((0.97, 0.08, 0.88))
    assert not any(clipped.values())

    jobs = list(iter_sensitivity_specs("A", "T", 0.99, 0.10, 0.90, ["spike", "pressure_drop"], [1]))
    kinds = [(j[0], j[1], j[2]) for j in jobs]
    assert ("baseline", "baseline", 0) in kinds
    assert ("rule", "spike", 1) in kinds and ("all", "all", -1) in kinds
    one = next(j for j in jobs if j[0] == "rule" and j[1] == "spike" and j[2] == 1)
    assert one[3]["spike"][2] == pytest.approx(1.0)
    assert one[3]["pressure_drop"][2] == pytest.approx(0.10)


def test_higher_spike_quantile_does_not_add_fires():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    feats = pd.DataFrame({
        "T__jump__resid": x, "p__delta__resid": x, "rh__delta__resid": x, "T__anom": x,
    })
    lo = fit_thresholds(feats, rule_specs("A", q_spike=0.90), (0, 400))
    hi = fit_thresholds(feats, rule_specs("A", q_spike=0.99), (0, 400))
    assert hi["spike"]["threshold"] >= lo["spike"]["threshold"]
    f_lo = fire_matrix(feats, lo)["spike"].to_numpy()
    f_hi = fire_matrix(feats, hi)["spike"].to_numpy()
    assert f_hi.sum() <= f_lo.sum()
    assert not np.any(~f_lo & f_hi)


def test_run_length_and_sample_indices_are_seeded():
    lengths, labs = consecutive_runs(np.array(["a", "a", "b", "b", "b", "a"]))
    assert list(lengths) == [2, 3, 1]
    assert list(labs) == ["a", "b", "a"]
    labels = pd.Series(["a"] * 10 + ["b"] * 10 + ["a"] * 5)
    per, overall = run_length_summary(labels, {"train": (0, 25)}, ["a", "b"])
    row_a = per[(per["label"] == "a") & (per["split"] == "train")].iloc[0]
    assert int(row_a["n_runs"]) == 2
    assert row_a["mean_run_length"] == pytest.approx(7.5)
    assert int(overall.iloc[0]["n_runs"]) == 3
    assert overall.iloc[0]["mean_run_length"] == pytest.approx(25 / 3)

    s1 = sample_window_indices(labels, (0, 25), ["a", "b", "missing"], n=4, seed=7)
    s2 = sample_window_indices(labels, (0, 25), ["a", "b", "missing"], n=4, seed=7)
    assert np.array_equal(s1["a"], s2["a"])
    assert len(s1["a"]) == 4 and set(s1["a"]).issubset(set(np.flatnonzero(labels.to_numpy() == "a")))
    assert len(s1["missing"]) == 0
    s3 = sample_window_indices(labels, (0, 25), ["a"], n=4, seed=8)
    assert not np.array_equal(s1["a"], s3["a"])


def _weather_frame(n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="h")
    df = pd.DataFrame({c: rng.normal(size=n).cumsum() for c in CH.values()}, index=idx)
    df["wd (deg)"] = rng.uniform(0, 360, size=n)
    return df


def _label_bundle(df, L=24, H=6):
    sp = window_split(len(df), L, H)
    n_w = sp["n_windows"]
    feats, clim = build_feature_table(df[list(CH.values())], CH, L, n_w, sp["train_rows"], "hour")
    feats = add_raw_mean(feats, df[list(CH.values())], CH, L, n_w)
    pcol, tcol, wvcol = CH["p"], CH["T"], CH["wv"]
    cf = cold_front_feature_frame(
        df[pcol].to_numpy(), df[tcol].to_numpy(),
        df[pcol].to_numpy() - clim[pcol].to_numpy(),
        df[tcol].to_numpy() - clim[tcol].to_numpy(),
        df["wd (deg)"].to_numpy(),
        np.abs(feats["wv__delta__raw"].to_numpy()),
        np.abs(feats["wv__delta__resid"].to_numpy()),
        L, n_w,
    )
    out = {"sp": sp, "feats": feats, "cf": cf}
    for mode, tag in (("A", "resid"), ("B", "raw")):
        th = fit_thresholds(feats, rule_specs(mode), sp["spans"]["train"])
        labels = assign_labels(fire_matrix(feats, th), RULE_ORDER)
        zlab, _, _, _ = assign_max_z(feats, th, sp["spans"]["train"], RULE_ORDER)
        cth = fit_cold_front_thresholds(cf, sp["spans"]["train"], tag, 0.75, 0.10, 0.90)
        out[mode] = {"th": th, "labels": labels, "z": zlab, "cth": cth}
    return out


def test_changing_only_forecast_targets_does_not_change_labels_or_thresholds():
    L, H, n = 24, 6, 900
    df = _weather_frame(n, seed=2)
    # Hang [n-H, n) chi nam trong y cua cac cua so cuoi, khong nam trong X nao.
    df_y = df.copy()
    df_y.iloc[-H:] += 5000.0
    a, b = _label_bundle(df, L, H), _label_bundle(df_y, L, H)
    assert np.allclose(a["feats"].to_numpy(), b["feats"].to_numpy(), equal_nan=True)
    assert np.allclose(a["cf"].to_numpy(), b["cf"].to_numpy(), equal_nan=True)
    for mode in ("A", "B"):
        for name in a[mode]["th"]:
            assert a[mode]["th"][name]["threshold"] == pytest.approx(b[mode]["th"][name]["threshold"])
        assert np.array_equal(a[mode]["labels"].to_numpy(), b[mode]["labels"].to_numpy())
        assert np.array_equal(a[mode]["z"].to_numpy(), b[mode]["z"].to_numpy())
        for k in ("thr_leg", "thr_t_delta", "thr_wd", "thr_wv"):
            assert a[mode]["cth"][k] == pytest.approx(b[mode]["cth"][k])


def test_thresholds_unchanged_when_only_val_and_test_change():
    L, H, n = 24, 6, 900
    df = _weather_frame(n, seed=3)
    base = _label_bundle(df, L, H)
    val0 = base["sp"]["spans"]["val"][0]
    tweaked = df.copy()
    tweaked.iloc[val0:] += 1000.0
    alt = _label_bundle(tweaked, L, H)
    tr = base["sp"]["spans"]["train"]
    va = base["sp"]["spans"]["val"]
    assert np.allclose(base["feats"].iloc[tr[0]:tr[1]].to_numpy(),
                       alt["feats"].iloc[tr[0]:tr[1]].to_numpy(), equal_nan=True)
    assert not np.allclose(base["feats"].iloc[va[0]:va[1]].to_numpy(),
                           alt["feats"].iloc[va[0]:va[1]].to_numpy(), equal_nan=True)
    for mode in ("A", "B"):
        for name in base[mode]["th"]:
            assert base[mode]["th"][name]["threshold"] == pytest.approx(alt[mode]["th"][name]["threshold"])
        assert np.array_equal(base[mode]["labels"].iloc[tr[0]:tr[1]].to_numpy(),
                              alt[mode]["labels"].iloc[tr[0]:tr[1]].to_numpy())
        for k in ("thr_leg", "thr_t_delta", "thr_wd", "thr_wv"):
            assert base[mode]["cth"][k] == pytest.approx(alt[mode]["cth"][k])
