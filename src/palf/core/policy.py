"""Policy Engine: chuyen Learning State thanh trong so lay mau theo Pattern (Spec Phase 5-6 v1)."""
from dataclasses import dataclass
import json
import os
from typing import Dict, Optional
import numpy as np

from .config import PalfConfig


@dataclass
class PolicyDecision:
    weights: dict      # weights[p] > 0
    info: dict         # diem can hoc, y_hat, z-score, co cham san/tran... (de ghi log va hien thi)


def load_and_validate_predictive_coef(coef_source) -> dict:
    """Doc va kiem tra tinh hop le cua predictive_coef.json (hoac dict).
    Neu thieu features, standardize hoac coef -> raise ValueError ro rang (Prompt 7 Phan 1 Buoc 4c)."""
    if isinstance(coef_source, dict):
        data = dict(coef_source)
    elif isinstance(coef_source, str):
        if not os.path.isfile(coef_source):
            raise FileNotFoundError(
                f"Khong tim thay file he so predictive_coef tai '{coef_source}'."
            )
        with open(coef_source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise TypeError("predictive_coef phai la duong dan file JSON hoac dict.")

    features = data.get("features")
    if not isinstance(features, (list, tuple)) or len(features) == 0:
        raise ValueError("predictive_coef thieu danh sach 'features' hop le.")

    if "standardize" not in data or not isinstance(data["standardize"], dict):
        raise ValueError("predictive_coef thieu muc 'standardize' (mean/std cho tung dac trung).")

    if "coef" not in data or not isinstance(data["coef"], dict):
        raise ValueError("predictive_coef thieu muc 'coef' (he so hoi quy cho tung dac trung).")

    if "intercept" not in data or data["intercept"] is None:
        raise ValueError("predictive_coef thieu 'intercept'.")

    clean_std = {}
    clean_coef = {}
    for feat in features:
        if feat not in data["standardize"] or not isinstance(data["standardize"][feat], dict):
            raise ValueError(f"predictive_coef thieu 'standardize' cho dac trung '{feat}'.")
        st_f = data["standardize"][feat]
        if "mean" not in st_f or "std" not in st_f:
            raise ValueError(f"standardize['{feat}'] phai co du 'mean' va 'std'.")
        sd_val = float(st_f["std"])
        if not np.isfinite(sd_val) or sd_val <= 0.0:
            raise ValueError(f"standardize['{feat}']['std']={sd_val} phai > 0.")
        clean_std[str(feat)] = {"mean": float(st_f["mean"]), "std": sd_val}

        if feat not in data["coef"] or data["coef"][feat] is None:
            raise ValueError(f"predictive_coef thieu he so 'coef' cho dac trung '{feat}'.")
        clean_coef[str(feat)] = float(data["coef"][feat])

    return {
        "target": str(data.get("target", "delta_loss_k3")),
        "features": [str(f) for f in features],
        "standardize": clean_std,
        "intercept": float(data["intercept"]),
        "coef": clean_coef,
        "fit_scope": str(data.get("fit_scope", "unknown")),
        "fit_seeds": list(data.get("fit_seeds", [])),
        "controls_epoch": bool(data.get("controls_epoch", True)),
    }


class PolicyEngine:
    """Tinh diem can hoc N_p va trong so lay mau w_p cho tung Pattern."""

    def __init__(
        self,
        pattern_names,
        freqs: dict,
        cfg: PalfConfig = None,
        predictive_coef: Optional[dict] = None,
    ):
        self.pattern_names = [str(p) for p in pattern_names]
        self.freqs = {
            str(p): float(freqs.get(p, 1.0 / max(1, len(self.pattern_names))))
            for p in self.pattern_names
        }
        self.cfg = cfg if cfg is not None else PalfConfig()
        self._prev_weights: Dict[str, float] = {p: 1.0 for p in self.pattern_names}

        self.predictive_coef: Optional[dict] = None
        if self.cfg.policy == "predictive":
            src = predictive_coef if predictive_coef is not None else self.cfg.predictive_coef_path
            self.predictive_coef = load_and_validate_predictive_coef(src)

    def _active_vars_and_signs(self):
        if self.cfg.policy == "uniform":
            return (), {}
        if self.cfg.policy == "loss":
            return ("loss_ratio",), {"loss_ratio": 1.0}
        if self.cfg.policy == "loss_abs":
            return ("loss_abs",), {"loss_abs": 1.0}
        if self.cfg.policy == "predictive":
            feats = tuple(self.predictive_coef["features"])
            return feats, {f: float(self.predictive_coef["coef"][f]) for f in feats}
        # policy == "state"
        vars_tuple = tuple(self.cfg.state_vars)
        signs = {v: float(self.cfg.state_signs.get(v, 1.0)) for v in vars_tuple}
        return vars_tuple, signs

    def decide(self, epoch: int, state: dict) -> PolicyDecision:
        if not isinstance(state, dict):
            raise TypeError("state phai la dict anh xa pattern -> dict cac bien Learning State.")

        active_vars, active_signs = self._active_vars_and_signs()
        in_warmup = int(epoch) <= int(self.cfg.warmup_epochs)

        # Chi trich xuat cac bien Learning State hop le cua tung Pattern (bo qua moi khoa val/test neu co)
        tracked_vars = tuple(dict.fromkeys(list(active_vars) + list(self.cfg.state_vars)))
        clean_state = {}
        raw_lm = {}
        for p in self.pattern_names:
            p_st = state.get(p, {}) if isinstance(state.get(p, {}), dict) else {}
            clean_state[p] = {
                v: float(p_st.get(v, np.nan)) if p_st.get(v) is not None else float("nan")
                for v in tracked_vars
            }
            if "loss_mean" in p_st and p_st["loss_mean"] is not None:
                raw_lm[p] = float(p_st["loss_mean"])

        # Ho tro tinh loss_abs tu loss_mean neu state truyen truc tiep loss_mean ma chua chia trung binh toan bo
        if "loss_abs" in tracked_vars:
            need_compute_abs = any(
                np.isnan(clean_state[p].get("loss_abs", np.nan)) and p in raw_lm and np.isfinite(raw_lm[p])
                for p in self.pattern_names
            )
            if need_compute_abs:
                valid_lms = [v for v in raw_lm.values() if np.isfinite(v) and v >= 0.0]
                g_mean = float(np.mean(valid_lms)) if valid_lms else float("nan")
                if np.isfinite(g_mean) and g_mean > 0.0:
                    for p in self.pattern_names:
                        if np.isnan(clean_state[p].get("loss_abs", np.nan)) and p in raw_lm and np.isfinite(raw_lm[p]):
                            clean_state[p]["loss_abs"] = float(raw_lm[p] / g_mean)

        # 1. Che do uniform hoac dang trong giai doan warm-up -> moi w = 1.0
        if self.cfg.policy == "uniform" or in_warmup:
            weights = {p: 1.0 for p in self.pattern_names}
            self._prev_weights = dict(weights)
            info = {
                "epoch": int(epoch),
                "policy": self.cfg.policy,
                "in_warmup": bool(in_warmup),
                "updated": False,
                "y_hat": {p: float("nan") for p in self.pattern_names},
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

        y_hat_by_pat = {p: float("nan") for p in self.pattern_names}
        z_scores_by_pat = {p: {} for p in self.pattern_names}
        need_scores = {}

        if self.cfg.policy == "predictive":
            # Chuan hoa tung dac trung bang mean/std TRONG FILE predictive_coef.json (khong tinh lai)
            p_coef = self.predictive_coef
            raw_needs = np.full(len(self.pattern_names), np.nan, dtype=float)
            for idx_p, p in enumerate(self.pattern_names):
                has_nan = any(np.isnan(clean_state[p].get(v, np.nan)) for v in active_vars)
                if has_nan:
                    for v in active_vars:
                        z_scores_by_pat[p][v] = 0.0
                    y_hat_by_pat[p] = float("nan")
                    raw_needs[idx_p] = np.nan
                else:
                    yh = float(p_coef["intercept"])
                    for v in active_vars:
                        mu_v = p_coef["standardize"][v]["mean"]
                        sd_v = p_coef["standardize"][v]["std"]
                        x_std = (float(clean_state[p][v]) - mu_v) / sd_v
                        z_scores_by_pat[p][v] = float(x_std)
                        yh += float(p_coef["coef"][v]) * x_std
                    y_hat_by_pat[p] = float(yh)
                    raw_needs[idx_p] = float(-yh)

            # Dua raw_need = -y_hat qua cung duong chuan hoa z-score giua cac Pattern -> z_clip
            finite_mask = np.isfinite(raw_needs)
            if finite_mask.sum() < 2:
                for p in self.pattern_names:
                    need_scores[p] = 0.0
            else:
                mu_n = float(np.mean(raw_needs[finite_mask]))
                sd_n = float(np.std(raw_needs[finite_mask], ddof=0))
                for idx_p, p in enumerate(self.pattern_names):
                    if not np.isfinite(raw_needs[idx_p]) or sd_n < 1e-12:
                        need_scores[p] = 0.0
                    else:
                        z_n = (float(raw_needs[idx_p]) - mu_n) / sd_n
                        need_scores[p] = float(np.clip(z_n, -self.cfg.z_clip, self.cfg.z_clip))
        else:
            # Che do 'loss' hoac 'state': tinh z-score giua cac Pattern cho tung bien active
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

            for p in self.pattern_names:
                has_nan = any(np.isnan(clean_state[p].get(v, np.nan)) for v in active_vars)
                if has_nan:
                    for v in active_vars:
                        z_scores_by_pat[p][v] = 0.0
                    need_scores[p] = 0.0
                else:
                    for v in active_vars:
                        z_scores_by_pat[p][v] = z_by_var[v][p]
                    need_scores[p] = float(sum(active_signs[v] * z_scores_by_pat[p][v] for v in active_vars))

        w_raw = {}
        w_new = {}
        hit_floor = {}
        hit_ceil = {}

        for p in self.pattern_names:
            has_nan = any(np.isnan(clean_state[p].get(v, np.nan)) for v in active_vars)
            if has_nan:
                raw_p = 1.0
                new_p = 1.0
                h_fl = False
                h_cl = False
            else:
                n_p = need_scores[p]
                raw_p = float(1.0 + self.cfg.alpha * n_p)
                h_fl = bool(raw_p < self.cfg.w_min)
                h_cl = bool(raw_p > self.cfg.w_max)
                new_p = float(np.clip(raw_p, self.cfg.w_min, self.cfg.w_max))

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
            "y_hat": y_hat_by_pat,
            "need_score": need_scores,
            "z_scores": z_scores_by_pat,
            "state": clean_state,
            "w_raw": w_raw,
            "w_new": w_new,
            "hit_floor": hit_floor,
            "hit_ceil": hit_ceil,
        }
        return PolicyDecision(weights=weights, info=info)
