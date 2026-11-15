"""Prompt 10 — Kiem thu doi chung A0w (`--shuffle-weights`): hoan vi trong so giua cac Pattern."""
import numpy as np
import pytest

from palf.core import PalfConfig, PolicyEngine


PATTERNS = ["pressure_drop", "humidity_net_rise", "temp_anomaly", "normal"]
FREQS = {"pressure_drop": 0.1, "humidity_net_rise": 0.1, "temp_anomaly": 0.15, "normal": 0.65}


def _states(n_epochs=12, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_epochs):
        out.append({
            p: {
                "loss_ratio": float(rng.uniform(0.3, 1.2)),
                "velocity": float(rng.normal()),
                "instability": float(abs(rng.normal())),
                "loss_mean": float(rng.uniform(0.05, 0.3)),
            }
            for p in PATTERNS
        })
    return out


def _cfg(shuffle, s_seed=None):
    return PalfConfig(policy="loss", alpha=1.0, w_max=3.0, budget_ratio=1.0,
                      shuffle_weights=shuffle, shuffle_seed=s_seed)


def _run(engine, states):
    return [engine.decide(e, st) for e, st in enumerate(states, start=1)]


def test_shuffle_same_seed_is_reproducible_and_different_seed_differs():
    states = _states()
    d1 = _run(PolicyEngine(PATTERNS, FREQS, _cfg(True, 100000)), states)
    d2 = _run(PolicyEngine(PATTERNS, FREQS, _cfg(True, 100000)), states)
    d3 = _run(PolicyEngine(PATTERNS, FREQS, _cfg(True, 100001)), states)
    for a, b in zip(d1, d2):
        assert a.weights == b.weights
        assert a.info.get("shuffle_source") == b.info.get("shuffle_source")
    srcs1 = [tuple(d.info.get("shuffle_source", {}).values()) for d in d1]
    srcs3 = [tuple(d.info.get("shuffle_source", {}).values()) for d in d3]
    assert srcs1 != srcs3


def test_shuffled_weight_value_set_equals_a1_each_epoch():
    """Cung dau vao: tap gia tri w_raw / w_new va so lan cham san/tran cua A0w bang A1 o moi epoch."""
    states = _states()
    a1 = _run(PolicyEngine(PATTERNS, FREQS, _cfg(False)), states)
    a0w = _run(PolicyEngine(PATTERNS, FREQS, _cfg(True, 100003)), states)
    n_permuted = 0
    for e, (x, y) in enumerate(zip(a1, a0w), start=1):
        if e <= 3:
            assert x.weights == y.weights == {p: 1.0 for p in PATTERNS}
            continue
        assert sorted(x.info["w_raw"].values()) == pytest.approx(sorted(y.info["w_raw"].values()))
        assert sorted(x.info["w_new"].values()) == pytest.approx(sorted(y.info["w_new"].values()))
        assert sum(x.info["hit_floor"].values()) == sum(y.info["hit_floor"].values())
        assert sum(x.info["hit_ceil"].values()) == sum(y.info["hit_ceil"].values())
        # w_new cua Pattern dich phai bang w_new cua Pattern nguon trong A1
        for dst, src in y.info["shuffle_source"].items():
            assert y.info["w_new"][dst] == pytest.approx(x.info["w_new"][src])
        if any(dst != src for dst, src in y.info["shuffle_source"].items()):
            n_permuted += 1
        # Trong so cuoi van nam trong rao chan
        for w in y.weights.values():
            assert 0.25 <= w <= 3.0
    assert n_permuted >= 1


def test_shuffle_does_not_touch_global_numpy_rng():
    states = _states()
    np.random.seed(123)
    expected = np.random.rand(5)
    np.random.seed(123)
    _run(PolicyEngine(PATTERNS, FREQS, _cfg(True, 100000)), states)
    got = np.random.rand(5)
    assert np.allclose(expected, got)


def test_shuffle_policy_ignores_val_test_keys():
    states = _states()
    poisoned = []
    for st in states:
        new_st = {}
        for p, d in st.items():
            nd = dict(d)
            nd.update({"val_loss": 999.0, "test_loss": -999.0, "val_mae": 1e6})
            new_st[p] = nd
        poisoned.append(new_st)
    clean = _run(PolicyEngine(PATTERNS, FREQS, _cfg(True, 100000)), states)
    dirty = _run(PolicyEngine(PATTERNS, FREQS, _cfg(True, 100000)), poisoned)
    for a, b in zip(clean, dirty):
        assert a.weights == b.weights
        for p in PATTERNS:
            assert not any(k.startswith(("val", "test")) for k in b.info["state"][p])


def test_shuffle_config_tag_roundtrip_and_uniform_rejected():
    cfg = _cfg(True, 100005)
    assert cfg.config_tag() == "loss_shuffled_b1.0_a1.0"
    assert PalfConfig.from_dict(cfg.to_dict()) == cfg
    assert _cfg(False).config_tag() == "loss_b1.0_a1.0"
    with pytest.raises(ValueError):
        PalfConfig(policy="uniform", shuffle_weights=True)
