import numpy as np
import pandas as pd
import pytest

from palf.data.pattern_features import build_feature_table, window_split
from palf.data.pattern_labels import (
    RULE_ORDER, add_raw_mean, assign_labels, check_criteria, fire_matrix,
    fit_thresholds, overlap_counts, rule_specs,
)

CH = {"p": "p (mbar)", "T": "T (degC)", "rh": "rh (%)", "wv": "wv (m/s)"}


def _make_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="h")
    return pd.DataFrame({c: rng.normal(size=n).cumsum() for c in CH.values()}, index=idx)


def test_priority_first_rule_wins_and_normal_when_none():
    fires = pd.DataFrame({
        "spike": [True, False, False, False],
        "pressure_drop": [True, True, False, False],
        "heat_anomaly": [False, True, True, False],
    })
    labels = assign_labels(fires, ["spike", "pressure_drop", "heat_anomaly"])
    assert list(labels) == ["spike", "pressure_drop", "heat_anomaly", "normal"]
    labels2 = assign_labels(fires, ["heat_anomaly", "pressure_drop", "spike"])
    assert list(labels2) == ["pressure_drop", "heat_anomaly", "heat_anomaly", "normal"]


def test_rule_specs_modes():
    a, b = rule_specs("A"), rule_specs("B")
    assert a["pressure_drop"][0] == "p__delta__resid"
    assert b["pressure_drop"][0] == "p__delta__raw"
    assert a["heat_anomaly"][0] == "T__anom"
    assert b["heat_anomaly"][0] == "T__mean__raw"
    assert a["spike"][1] == ">=" and a["cold_anomaly"][1] == "<="


def test_thresholds_and_train_labels_ignore_everything_after_train_inputs():
    L, H = 24, 6
    df = _make_df(3000)
    sp = window_split(len(df), L, H)
    n_w, spans = sp["n_windows"], sp["spans"]
    a, b = spans["train"]

    def run(d):
        f, _ = build_feature_table(d, CH, L, n_w, sp["train_rows"], "hour")
        f = add_raw_mean(f, d, CH, L, n_w)
        out = {}
        for mode in ("A", "B"):
            th = fit_thresholds(f, rule_specs(mode), spans["train"])
            labels = assign_labels(fire_matrix(f, th), RULE_ORDER)
            out[mode] = (th, labels.iloc[a:b].to_numpy())
        return out

    d2 = df.copy()
    d2.iloc[sp["train_rows"]:] += 1000.0  # doi y cua cua so train cuoi va toan bo val/test
    r1, r2 = run(df), run(d2)
    for mode in ("A", "B"):
        for name in r1[mode][0]:
            assert r1[mode][0][name]["threshold"] == pytest.approx(r2[mode][0][name]["threshold"])
        assert (r1[mode][1] == r2[mode][1]).all()


def test_overlap_diagonal_matches_rule_counts():
    rng = np.random.default_rng(1)
    fires = pd.DataFrame(rng.random((200, 3)) < 0.3, columns=["a", "b", "c"])
    m = overlap_counts(fires, (0, 150))
    assert np.array_equal(np.diag(m.to_numpy()), fires.iloc[:150].sum().to_numpy())
    assert np.array_equal(m.to_numpy(), m.to_numpy().T)


def test_check_criteria_flags_failures():
    tab = pd.DataFrame({
        "label": ["spike", "pressure_drop", "normal"],
        "n_train": [100, 3000, 6900], "n_val": [10, 900, 1090], "n_test": [100, 3000, 6900],
        "share_train": [0.01, 0.30, 0.69],
        "share_val": [0.01, 0.10, 0.89],
        "share_test": [0.01, 0.30, 0.69],
    })
    c2, c3, c4, warn = check_criteria(tab)
    assert not bool(c2.loc[c2["label"] == "spike", "C2"].iloc[0])
    assert bool(c2.loc[c2["label"] == "pressure_drop", "C2"].iloc[0])
    assert not bool(c3.iloc[0]["all<=70%"])
    assert not bool(c4.loc[c4["label"] == "pressure_drop", "C4"].iloc[0])
    assert any(w.startswith("C2") for w in warn)
    assert any(w.startswith("C3") for w in warn)
    assert any(w.startswith("C4") for w in warn)