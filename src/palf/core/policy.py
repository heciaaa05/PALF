"""Policy Engine: chuyen Learning State thanh trong so lay mau theo Pattern."""
from dataclasses import dataclass
from typing import Dict
import numpy as np

from .config import PalfConfig
from .state import FORBIDDEN_VAL_TEST_KEYS


@dataclass
class PolicyDecision:
    weights: dict      # weights[p] > 0
    info: dict         # diem can hoc, z-score, co cham san/tran... (de ghi log va hien thi)


class PolicyEngine:
    """Tinh diem can hoc N_p va trong so lay mau w_p cho tung Pattern."""

    def __init__(self, pattern_names, freqs: dict, cfg: PalfConfig = None):
        self.pattern_names = [str(p) for p in pattern_names]
        self.freqs = {str(p): float(freqs.get(p, 1.0 / max(1, len(self.pattern_names)))) for p in self.pattern_names}
        self.cfg = cfg if cfg is not None else PalfConfig()
        self._prev_weights: Dict[str, float] = {p: 1.0 for p in self.pattern_names}

    def _active_vars_and_signs(self):
        if self.cfg.policy == "uniform":
            return (), {}
        if self.cfg.policy == "loss":
            return ("loss_ratio",), {"loss_ratio": 1.0}
        # policy == "state"
        vars_tuple = tuple(self.cfg.state_vars)
        signs = {v: float(self.cfg.state_signs.get(v, 1.0)) for v in vars_tuple}
        return vars_tuple, signs

    def decide(self, epoch: int, state: dict) -> PolicyDecision:
        if not isinstance(state, dict):
            raise TypeError("state phai la dict anh xa pattern -> dict cac bien Learning State.")

        for k_top in state.keys():
            if str(k_top).lower() in FORBIDDEN_VAL_TEST_KEYS:
                raise ValueError(f"PolicyEngine khong duoc nhan du lieu '{k_top}' (chi dung tin hieu train).")

        active_vars, active_signs = self._active_vars_and_signs()
        in_warmup = int(epoch) <= int(self.cfg.warmup_epochs)

        # Ghi nhan gia tri state dau vao
        clean_state = {}
        for p in self.pattern_names:
            p_st = state.get(p, {}) if isinstance(state.get(p, {}), dict) else {}
            for k_inner in p_st.keys():
                if str(k_inner).lower() in FORBIDDEN_VAL_TEST_KEYS:
                    raise ValueError(f"PolicyEngine phat hien khoa bi cam '{k_inner}' trong state['{p}'].")
            clean_state[p] = {
                v: float(p_st.get(v, np.nan)) if p_st.get(v) is not None else float("nan")
                for v in (active_vars if active_vars else self.cfg.state_vars)
            }

        # 1. Che do uniform hoac dang trong giai doan warm-up -> moi w = 1.0
        if self.cfg.policy == "uniform" or in_warmup:
            weights = {p: 1.0 for p in self.pattern_names}
            self._prev_weights = dict(weights)
            info = {
                "epoch": int(epoch),
                "policy": self.cfg.policy,
                "in_warmup": bool(in_warmup),
                "updated": False,
                "need_score": {p: 0.0 for p in self.pattern_names},
                "z_scores": {p: {v: 0.0 for v in active_vars} for p in self.pattern_names},
                "state": clean_state,
                "w_raw": {p: 1.0 for p in self.pattern_names},
                "w_new": {p: 1.0 for p in self.pattern_names},
                "hit_floor": {p: False for p in self.pattern_names},
                "hit_ceil": {p: False for p in self.pattern_names},
            }
            return PolicyDecision(weights=weights, info=info)

        # 2. Kiem tra chu ky cap nhat update_every sau warm-up
        steps_since_warmup = int(epoch) - int(self.cfg.warmup_epochs) - 1
        should_update = (steps_since_warmup % int(self.cfg.update_every)) == 0

        # Tinh z-score giua cac Pattern cho tung bien active
        z_by_var: Dict[str, Dict[str, float]] = {v: {} for v in active_vars}
        for v in active_vars:
            vals = np.array([clean_state[p].get(v, np.nan) for p in self.pattern_names], dtype=float)
            finite_mask = np.isfinite(vals)
            if finite_mask.sum() < 2:
                for p in self.pattern_names:
                    z_by_var[v][p] = 0.0
            else:
                mu = float(np.mean(vals[finite_mask]))
                sd = float(np.std(vals[finite_mask], ddof=0))
                if sd < 1e-12:
                    for p in self.pattern_names:
                        z_by_var[v][p] = 0.0
                else:
                    for p, val in zip(self.pattern_names, vals):
                        if np.isfinite(val):
                            z_val = (float(val) - mu) / sd
                            z_by_var[v][p] = float(np.clip(z_val, -self.cfg.z_clip, self.cfg.z_clip))
                        else:
                            z_by_var[v][p] = 0.0

        z_scores_by_pat = {p: {v: z_by_var[v][p] for v in active_vars} for p in self.pattern_names}

        need_scores = {}
        w_raw = {}
        w_new = {}
        hit_floor = {}
        hit_ceil = {}

        for p in self.pattern_names:
            # Neu bat ky bien active nao cua Pattern p la NaN -> z=0, N_p=0, w_new=1.0 (Muc 5.3)
            has_nan = any(np.isnan(clean_state[p].get(v, np.nan)) for v in active_vars)
            if has_nan:
                for v in active_vars:
                    z_scores_by_pat[p][v] = 0.0
                n_p = 0.0
                raw_p = 1.0
                new_p = 1.0
                h_fl = False
                h_cl = False
            else:
                n_p = float(sum(active_signs[v] * z_scores_by_pat[p][v] for v in active_vars))
                raw_p = float(1.0 + self.cfg.alpha * n_p)
                h_fl = bool(raw_p < self.cfg.w_min)
                h_cl = bool(raw_p > self.cfg.w_max)
                new_p = float(np.clip(raw_p, self.cfg.w_min, self.cfg.w_max))

            need_scores[p] = n_p
            w_raw[p] = raw_p
            w_new[p] = new_p
            hit_floor[p] = h_fl
            hit_ceil[p] = h_cl

        if should_update:
            weights = {}
            beta = float(self.cfg.ema_beta)
            for p in self.pattern_names:
                has_nan = any(np.isnan(clean_state[p].get(v, np.nan)) for v in active_vars)
                if has_nan:
                    w_ema = 1.0
                else:
                    w_prev = self._prev_weights.get(p, 1.0)
                    w_ema = beta * w_prev + (1.0 - beta) * w_new[p]
                weights[p] = float(np.clip(w_ema, self.cfg.w_min, self.cfg.w_max))
            self._prev_weights = dict(weights)
        else:
            weights = dict(self._prev_weights)

        info = {
            "epoch": int(epoch),
            "policy": self.cfg.policy,
            "in_warmup": False,
            "updated": bool(should_update),
            "need_score": need_scores,
            "z_scores": z_scores_by_pat,
            "state": clean_state,
            "w_raw": w_raw,
            "w_new": w_new,
            "hit_floor": hit_floor,
            "hit_ceil": hit_ceil,
        }
        return PolicyDecision(weights=weights, info=info)
