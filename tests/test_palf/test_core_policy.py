import json
import os
import numpy as np
import pytest

from palf.core import (
    LearningStateTracker,
    PalfConfig,
    PolicyDecision,
    PolicyEngine,
)
from palf.training.scheduler import AdaptiveScheduler


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

    state = {
        "pressure_drop": {"loss_ratio": 0.8, "velocity": 0.15, "instability": 0.05},
        "humidity_net_rise": {"loss_ratio": 0.8, "velocity": 0.15, "instability": 0.05},
        "temp_anomaly": {"loss_ratio": np.nan, "velocity": np.nan, "instability": np.nan},
        "normal": {"loss_ratio": 0.2, "velocity": 0.01, "instability": 0.01},
    }
    dec = eng.decide(2, state)

    for p, w in dec.weights.items():
        assert cfg.w_min <= w <= cfg.w_max
        assert not np.isnan(w)

    assert dec.weights["pressure_drop"] == pytest.approx(dec.weights["humidity_net_rise"])
    assert dec.info["need_score"]["temp_anomaly"] == pytest.approx(0.0)
    assert dec.weights["temp_anomaly"] == pytest.approx(1.0)


def test_tracker_rejects_val_and_policy_ignores_val_test_keys(tmp_path):
    """(Prompt 7 Phan 1 Buoc 4b): Tracker chan val/test, va PolicyEngine nhan them khoa val/test
    van cho quyet dinh Y HET tren moi che do (bao gom 'predictive')."""
    coef_path = os.path.join(str(tmp_path), "coef.json")
    coef_data = {
        "target": "delta_loss_k3",
        "features": ["loss_ratio", "instability"],
        "standardize": {
            "loss_ratio": {"mean": 0.5, "std": 0.2},
            "instability": {"mean": 0.05, "std": 0.02},
        },
        "intercept": -0.05,
        "coef": {"loss_ratio": -0.1, "instability": 0.2},
        "fit_scope": "gru",
        "fit_seeds": [3, 4],
        "controls_epoch": True,
    }
    with open(coef_path, "w", encoding="utf-8") as f:
        json.dump(coef_data, f)

    cfg = PalfConfig(policy="state", warmup_epochs=1)
    tracker = LearningStateTracker(PATTERNS, cfg)
    bad_stats = {p: {"loss_mean": 1.0, "loss_std": 0.1, "n_seen": 10, "val_loss": 0.9} for p in PATTERNS}
    with pytest.raises(ValueError, match="chi nhan tin hieu train"):
        tracker.update(1, bad_stats)

    clean_state = {
        "pressure_drop": {"loss_ratio": 0.9, "velocity": 0.2, "instability": 0.08},
        "humidity_net_rise": {"loss_ratio": 0.6, "velocity": 0.1, "instability": 0.04},
        "temp_anomaly": {"loss_ratio": 0.8, "velocity": 0.15, "instability": 0.06},
        "normal": {"loss_ratio": 0.4, "velocity": 0.02, "instability": 0.01},
    }
    state_with_val = {
        p: dict(clean_state[p], val_loss=999.0 * (i + 1), test_loss=888.0)
        for i, p in enumerate(PATTERNS)
    }
    state_with_val["val_loss"] = 123.45

    for pol_mode in ("uniform", "loss", "state", "predictive"):
        c_pol = PalfConfig(policy=pol_mode, warmup_epochs=1, predictive_coef_path=coef_path)
        e_clean = PolicyEngine(PATTERNS, FREQS, c_pol)
        e_val = PolicyEngine(PATTERNS, FREQS, c_pol)
        d_clean = e_clean.decide(4, clean_state)
        d_val = e_val.decide(4, state_with_val)
        assert d_clean.weights == d_val.weights
        assert d_clean.info["need_score"] == d_val.info["need_score"]


def test_predictive_policy_a_sign_flip_c_validation_d_differs_from_loss(tmp_path):
    """(Prompt 7 Phan 1 Buoc 4a, 4c, 4d):
    (a) Cung dau vao, predictive voi coef dao dau cho trong so dao chieu;
    (c) File thieu dac trung hoac thieu standardize -> loi ro rang;
    (d) Cung dau vao, predictive khac loss (khi he so instability != 0)."""
    state = {
        "pressure_drop": {"loss_ratio": 0.80, "velocity": 0.10, "instability": 0.02},
        "humidity_net_rise": {"loss_ratio": 0.70, "velocity": 0.10, "instability": 0.12},
        "temp_anomaly": {"loss_ratio": 0.50, "velocity": 0.10, "instability": 0.09},
        "normal": {"loss_ratio": 0.30, "velocity": 0.10, "instability": 0.01},
    }

    # (a) Coef dao dau -> trong so dao chieu
    coef_pos = {
        "target": "delta_loss_k3",
        "features": ["loss_ratio", "instability"],
        "standardize": {
            "loss_ratio": {"mean": 0.5, "std": 0.2},
            "instability": {"mean": 0.05, "std": 0.03},
        },
        "intercept": 0.0,
        "coef": {"loss_ratio": -0.1, "instability": 0.2},
    }
    coef_neg = {
        "target": "delta_loss_k3",
        "features": ["loss_ratio", "instability"],
        "standardize": coef_pos["standardize"],
        "intercept": 0.0,
        "coef": {"loss_ratio": 0.1, "instability": -0.2},
    }
    p_pos = os.path.join(str(tmp_path), "pos.json")
    p_neg = os.path.join(str(tmp_path), "neg.json")
    with open(p_pos, "w", encoding="utf-8") as f:
        json.dump(coef_pos, f)
    with open(p_neg, "w", encoding="utf-8") as f:
        json.dump(coef_neg, f)

    cfg_pos = PalfConfig(policy="predictive", warmup_epochs=1, alpha=0.5, w_min=0.1, w_max=5.0, ema_beta=0.0, predictive_coef_path=p_pos)
    cfg_neg = PalfConfig(policy="predictive", warmup_epochs=1, alpha=0.5, w_min=0.1, w_max=5.0, ema_beta=0.0, predictive_coef_path=p_neg)

    d_pos = PolicyEngine(PATTERNS, FREQS, cfg_pos).decide(2, state)
    d_neg = PolicyEngine(PATTERNS, FREQS, cfg_neg).decide(2, state)

    for p in PATTERNS:
        assert d_pos.info["need_score"][p] == pytest.approx(-d_neg.info["need_score"][p], abs=1e-6)
        assert (d_pos.weights[p] - 1.0) == pytest.approx(-(d_neg.weights[p] - 1.0), abs=1e-6)

    # (d) Cung dau vao, predictive khac loss (vi instability != 0)
    cfg_loss = PalfConfig(policy="loss", warmup_epochs=1, alpha=0.5, w_min=0.1, w_max=5.0, ema_beta=0.0)
    d_loss = PolicyEngine(PATTERNS, FREQS, cfg_loss).decide(2, state)
    assert any(abs(d_pos.weights[p] - d_loss.weights[p]) > 1e-3 for p in PATTERNS)

    # (c) File thieu standardize hoac thieu coef -> raise ValueError ro rang
    bad_no_std = {"features": ["loss_ratio"], "intercept": 0.0, "coef": {"loss_ratio": -0.1}}
    p_bad1 = os.path.join(str(tmp_path), "bad1.json")
    with open(p_bad1, "w", encoding="utf-8") as f:
        json.dump(bad_no_std, f)
    with pytest.raises(ValueError, match="standardize"):
        PolicyEngine(PATTERNS, FREQS, PalfConfig(policy="predictive", predictive_coef_path=p_bad1))

    bad_missing_feat = {
        "features": ["loss_ratio", "instability"],
        "standardize": {"loss_ratio": {"mean": 0.0, "std": 1.0}},
        "intercept": 0.0,
        "coef": {"loss_ratio": -0.1, "instability": 0.2},
    }
    p_bad2 = os.path.join(str(tmp_path), "bad2.json")
    with open(p_bad2, "w", encoding="utf-8") as f:
        json.dump(bad_missing_feat, f)
    with pytest.raises(ValueError, match="instability"):
        PolicyEngine(PATTERNS, FREQS, PalfConfig(policy="predictive", predictive_coef_path=p_bad2))


def test_loss_abs_policy_and_warmup1_tracker():
    """(Prompt 8.4 E1 & E2):
    - E1: Tracker tinh duoc instability tu epoch 2 tro di khi warmup_epochs=1;
    - E2: policy='loss_abs' trao trong so cao hon cho Pattern co loss tuyet doi cao hon (du loss_ratio thap hon),
      va van khong doc val/test."""
    cfg_e2 = PalfConfig(policy="loss_abs", warmup_epochs=1, alpha=0.5, w_min=0.25, w_max=2.0, ema_beta=0.0)
    tracker = LearningStateTracker(PATTERNS, cfg_e2)

    # Epoch 1: temp_anomaly co loss khoi dau rat cao (2.0), normal thap (0.5)
    st1 = tracker.update(1, {
        "pressure_drop": {"loss_mean": 1.0, "loss_std": 0.1, "n_seen": 100, "grad_norm": 0.5},
        "humidity_net_rise": {"loss_mean": 0.8, "loss_std": 0.1, "n_seen": 100, "grad_norm": 0.5},
        "temp_anomaly": {"loss_mean": 2.0, "loss_std": 0.2, "n_seen": 100, "grad_norm": 0.5},
        "normal": {"loss_mean": 0.5, "loss_std": 0.05, "n_seen": 700, "grad_norm": 0.5},
    })
    # Epoch 2: temp_anomaly giam con 0.8 (loss_ratio = 0.40 thap nhat, nhung loss tuyet doi 0.80 cao nhat!)
    # normal giam con 0.45 (loss_ratio = 0.90 cao nhat, nhung loss tuyet doi 0.45 thap nhat!)
    st2 = tracker.update(2, {
        "pressure_drop": {"loss_mean": 0.60, "loss_std": 0.1, "n_seen": 100, "grad_norm": 0.4},
        "humidity_net_rise": {"loss_mean": 0.55, "loss_std": 0.1, "n_seen": 100, "grad_norm": 0.4},
        "temp_anomaly": {"loss_mean": 0.80, "loss_std": 0.2, "n_seen": 100, "grad_norm": 0.4},
        "normal": {"loss_mean": 0.45, "loss_std": 0.05, "n_seen": 700, "grad_norm": 0.4},
    })

    # E1 check: instability o epoch 2 phai la so huu han (khong phai NaN) cho moi pattern
    for p in PATTERNS:
        assert np.isfinite(st2[p]["instability"]), f"instability cua {p} o epoch 2 phai huu han"
        assert np.isfinite(st2[p]["loss_abs"]), f"loss_abs cua {p} o epoch 2 phai huu han"

    # E2 check: temp_anomaly co loss tuyet doi cao nhat (0.80) -> nhan trong so cao nhat
    eng_clean = PolicyEngine(PATTERNS, FREQS, cfg_e2)
    d2_clean = eng_clean.decide(2, st2)
    assert d2_clean.weights["temp_anomaly"] > d2_clean.weights["pressure_drop"] > d2_clean.weights["humidity_net_rise"] > d2_clean.weights["normal"]

    # Kiem tra Policy loss_abs khong doc val/test
    st2_with_val = {p: dict(st2[p], val_loss=999.0 if p == "normal" else 0.001, test_mae=888.0) for p in PATTERNS}
    eng_val = PolicyEngine(PATTERNS, FREQS, cfg_e2)
    d2_val = eng_val.decide(2, st2_with_val)
    assert d2_clean.weights == d2_val.weights
    assert d2_clean.info["need_score"] == d2_val.info["need_score"]


def test_a0s_uniform_budget_1_0_exact_one_pass_per_epoch():
    """(Prompt 9 Muc 9.1): Test bat buoc truoc khi chay A0s:
    Khi policy='uniform' va budget_ratio=1.0, moi epoch gom dung 1 lan moi mau train
    (len(indices) == n_train va set(indices) == {0, 1, ..., n_train - 1})."""
    rng = np.random.default_rng(42)
    n_train = 1000
    pat_train = rng.choice(len(PATTERNS), size=n_train, p=[0.1, 0.1, 0.15, 0.65])
    cfg_a0s = PalfConfig(policy="uniform", budget_ratio=1.0)
    eng_a0s = PolicyEngine(PATTERNS, FREQS, cfg_a0s)
    sched_a0s = AdaptiveScheduler(pat_train, PATTERNS, cfg=cfg_a0s, seed=0)

    state = {p: {"loss_ratio": 0.5, "velocity": 0.1, "instability": 0.05} for p in PATTERNS}
    for ep in range(1, 6):
        dec = eng_a0s.decide(ep, state)
        indices = sched_a0s.plan_epoch(ep, dec)
        assert len(indices) == n_train
        assert np.array_equal(np.sort(indices), np.arange(n_train, dtype=np.int64))
        for p in PATTERNS:
            assert sched_a0s.last_quotas[p] == sched_a0s.pattern_counts[p]
