import numpy as np
import pytest

from palf.core import (
    LearningStateTracker,
    PalfConfig,
    PolicyDecision,
    PolicyEngine,
)


PATTERNS = ["pressure_drop", "humidity_net_rise", "temp_anomaly", "normal"]
FREQS = {"pressure_drop": 0.1, "humidity_net_rise": 0.1, "temp_anomaly": 0.15, "normal": 0.65}


def test_palf_config_roundtrip_and_defaults():
    cfg = PalfConfig()
    assert cfg.policy == "state"
    assert cfg.state_vars == ("loss_ratio", "velocity", "instability")
    assert cfg.state_signs["loss_ratio"] == 1.0
    assert cfg.state_signs["velocity"] == 1.0
    assert cfg.state_signs["instability"] == 0.5

    d = cfg.to_dict()
    cfg2 = PalfConfig.from_dict(d)
    assert cfg2 == cfg
    assert cfg.config_tag() == "state_b1.0_a1.0"


def test_tracker_formulas_nan_and_custom_registration():
    cfg = PalfConfig(window=2, state_vars=("loss_ratio", "velocity", "instability"))
    tracker = LearningStateTracker(PATTERNS, cfg)

    # Epoch 1: chua du lich su cho velocity va instability -> NaN
    stats_e1 = {
        "pressure_drop": {"loss_mean": 2.0, "loss_std": 0.4, "n_seen": 50, "grad_norm": 0.5},
        "humidity_net_rise": {"loss_mean": 4.0, "loss_std": 0.8, "n_seen": 50, "grad_norm": 0.6},
        "temp_anomaly": {"loss_mean": 1.0, "loss_std": 0.2, "n_seen": 50, "grad_norm": 0.3},
        "normal": {"loss_mean": 2.0, "loss_std": 0.2, "n_seen": 200, "grad_norm": 0.4},
    }
    st1 = tracker.update(1, stats_e1)
    for p in PATTERNS:
        assert st1[p]["loss_ratio"] == pytest.approx(1.0)
        assert np.isnan(st1[p]["velocity"])
        assert np.isnan(st1[p]["instability"])

    # Epoch 2: loss_ratio cua pressure_drop = 1.6 / 2.0 = 0.8 -> velocity = (1.0 - 0.8) / 1 = 0.2
    stats_e2 = {
        "pressure_drop": {"loss_mean": 1.6, "loss_std": 0.3, "n_seen": 50, "grad_norm": 0.4},
        "humidity_net_rise": {"loss_mean": 2.0, "loss_std": 0.5, "n_seen": 50, "grad_norm": 0.5},
        "temp_anomaly": {"loss_mean": 1.0, "loss_std": 0.2, "n_seen": 50, "grad_norm": 0.3},
        "normal": {"loss_mean": 1.8, "loss_std": 0.2, "n_seen": 200, "grad_norm": 0.3},
    }
    st2 = tracker.update(2, stats_e2)
    assert st2["pressure_drop"]["loss_ratio"] == pytest.approx(0.8)
    assert st2["pressure_drop"]["velocity"] == pytest.approx(0.2)
    assert st2["pressure_drop"]["instability"] == pytest.approx(0.1)

    # Dang ky them bien moi khong can sua loi
    cfg_custom = PalfConfig(
        state_vars=("loss_ratio", "custom_double_lr"),
        state_signs={"loss_ratio": 1.0, "custom_double_lr": 0.5},
    )
    tracker_c = LearningStateTracker(PATTERNS, cfg_custom)
    tracker_c.register_variable("custom_double_lr", lambda hist, w: hist[-1]["loss_ratio"] * 2.0)
    st_c = tracker_c.update(1, stats_e1)
    assert st_c["pressure_drop"]["custom_double_lr"] == pytest.approx(2.0)


def test_policy_warmup_and_uniform_give_weight_one():
    cfg_state = PalfConfig(policy="state", warmup_epochs=3)
    eng_state = PolicyEngine(PATTERNS, FREQS, cfg_state)
    state = {
        "pressure_drop": {"loss_ratio": 0.9, "velocity": 0.2, "instability": 0.1},
        "humidity_net_rise": {"loss_ratio": 0.5, "velocity": 0.05, "instability": 0.02},
        "temp_anomaly": {"loss_ratio": 0.7, "velocity": 0.1, "instability": 0.05},
        "normal": {"loss_ratio": 0.3, "velocity": 0.01, "instability": 0.01},
    }
    for ep in (1, 2, 3):
        dec = eng_state.decide(ep, state)
        assert isinstance(dec, PolicyDecision)
        assert all(w == pytest.approx(1.0) for w in dec.weights.values())
        assert dec.info["in_warmup"] is True

    # Che do uniform luon cho trong so 1 ke ca sau warmup
    cfg_uni = PalfConfig(policy="uniform", warmup_epochs=1)
    eng_uni = PolicyEngine(PATTERNS, FREQS, cfg_uni)
    dec_uni = eng_uni.decide(10, state)
    assert all(w == pytest.approx(1.0) for w in dec_uni.weights.values())


def test_policy_bounds_identical_patterns_and_nan_safety():
    cfg = PalfConfig(
        policy="state",
        warmup_epochs=1,
        alpha=5.0,
        w_min=0.25,
        w_max=3.0,
        ema_beta=0.0,
    )
    eng = PolicyEngine(PATTERNS, FREQS, cfg)

    # Hai pattern pressure_drop va humidity_net_rise giong het nhau; temp_anomaly co NaN
    state = {
        "pressure_drop": {"loss_ratio": 0.8, "velocity": 0.15, "instability": 0.05},
        "humidity_net_rise": {"loss_ratio": 0.8, "velocity": 0.15, "instability": 0.05},
        "temp_anomaly": {"loss_ratio": np.nan, "velocity": np.nan, "instability": np.nan},
        "normal": {"loss_ratio": 0.2, "velocity": 0.01, "instability": 0.01},
    }
    dec = eng.decide(2, state)

    # Trong so luon nam trong [w_min, w_max]
    for p, w in dec.weights.items():
        assert cfg.w_min <= w <= cfg.w_max
        assert not np.isnan(w)

    # Hai Pattern giong het nhau cho trong so bang nhau
    assert dec.weights["pressure_drop"] == pytest.approx(dec.weights["humidity_net_rise"])

    # Pattern co NaN thi z=0 va trong so bang 1.0
    assert dec.info["need_score"]["temp_anomaly"] == pytest.approx(0.0)
    assert dec.weights["temp_anomaly"] == pytest.approx(1.0)


def test_policy_rejects_val_test_signals_and_is_deterministic():
    cfg = PalfConfig(policy="state", warmup_epochs=1)
    tracker = LearningStateTracker(PATTERNS, cfg)
    eng = PolicyEngine(PATTERNS, FREQS, cfg)

    # Truyen val_loss vao tracker hoac policy phai bi chan ngay (Nguyen tac bat bien 1)
    bad_stats = {p: {"loss_mean": 1.0, "loss_std": 0.1, "n_seen": 10, "val_loss": 0.9} for p in PATTERNS}
    with pytest.raises(ValueError, match="chi nhan tin hieu train"):
        tracker.update(1, bad_stats)

    bad_state = {p: {"loss_ratio": 0.8, "velocity": 0.1, "instability": 0.05, "val_loss": 0.5} for p in PATTERNS}
    with pytest.raises(ValueError, match="khoa bi cam"):
        eng.decide(2, bad_state)

    # Kiem tra tinh xac dinh
    good_state = {
        "pressure_drop": {"loss_ratio": 0.9, "velocity": 0.2, "instability": 0.08},
        "humidity_net_rise": {"loss_ratio": 0.6, "velocity": 0.1, "instability": 0.04},
        "temp_anomaly": {"loss_ratio": 0.8, "velocity": 0.15, "instability": 0.06},
        "normal": {"loss_ratio": 0.4, "velocity": 0.02, "instability": 0.01},
    }
    e1 = PolicyEngine(PATTERNS, FREQS, cfg)
    e2 = PolicyEngine(PATTERNS, FREQS, cfg)
    d1 = e1.decide(4, good_state)
    d2 = e2.decide(4, good_state)
    assert d1.weights == d2.weights
