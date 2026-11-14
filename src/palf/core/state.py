"""Learning State Tracker: theo doi trang thai hoc theo tung Pattern chi tu tin hieu train."""
from typing import Callable, Dict, List
import numpy as np

from .config import PalfConfig


def _calc_loss_ratio(history: List[dict], window: int) -> float:
    if not history:
        return float("nan")
    cur = history[-1].get("loss_ratio", np.nan)
    return float(cur) if not np.isnan(cur) else float("nan")


def _calc_velocity(history: List[dict], window: int) -> float:
    """Toc do giam loss chuan hoa trong cua so W: (l_tilde(e-W) - l_tilde(e)) / W.
    Neu chua du lich su (< 2 diem hop le hoac diem hien tai la NaN) thi tra ve NaN."""
    if len(history) < 2:
        return float("nan")
    cur_lr = history[-1].get("loss_ratio", np.nan)
    if np.isnan(cur_lr):
        return float("nan")

    # Tim diem tham chieu cach toi da W buoc ve truoc
    start_idx = max(0, len(history) - 1 - window)
    for idx in range(start_idx, len(history) - 1):
        ref_lr = history[idx].get("loss_ratio", np.nan)
        span = (len(history) - 1) - idx
        if not np.isnan(ref_lr) and span > 0:
            return float((ref_lr - cur_lr) / span)
    return float("nan")


def _calc_instability(history: List[dict], window: int) -> float:
    """Do lech chuan cua loss_ratio trong cua so W epoch gan nhat."""
    if len(history) < 2:
        return float("nan")
    cur_lr = history[-1].get("loss_ratio", np.nan)
    if np.isnan(cur_lr):
        return float("nan")

    recent = [
        h.get("loss_ratio", np.nan)
        for h in history[-max(2, window) :]
        if not np.isnan(h.get("loss_ratio", np.nan))
    ]
    if len(recent) < 2:
        return float("nan")
    return float(np.std(recent, ddof=0))


def _calc_within_dispersion(history: List[dict], window: int) -> float:
    if not history:
        return float("nan")
    h = history[-1]
    lm = h.get("loss_mean", np.nan)
    ls = h.get("loss_std", np.nan)
    if np.isnan(lm) or np.isnan(ls) or abs(lm) < 1e-12:
        return float("nan")
    return float(ls / abs(lm))


def _calc_grad_norm(history: List[dict], window: int) -> float:
    if not history:
        return float("nan")
    gn = history[-1].get("grad_norm", np.nan)
    return float(gn) if not np.isnan(gn) else float("nan")


BUILTIN_STATE_FUNCTIONS: Dict[str, Callable[[List[dict], int], float]] = {
    "loss_ratio": _calc_loss_ratio,
    "velocity": _calc_velocity,
    "instability": _calc_instability,
    "within_dispersion": _calc_within_dispersion,
    "grad_norm": _calc_grad_norm,
}

ALLOWED_TRAIN_STAT_KEYS = {"loss_mean", "loss_std", "n_seen", "grad_norm"}
FORBIDDEN_VAL_TEST_KEYS = {"val_loss", "test_loss", "val_mae", "test_mae", "val_mse", "test_mse", "val", "test"}


class LearningStateTracker:
    """Theo doi va tinh cac bien Learning State cho tung Pattern tu thong ke train moi epoch."""

    _global_registry: Dict[str, Callable[[List[dict], int], float]] = dict(BUILTIN_STATE_FUNCTIONS)

    def __init__(self, pattern_names, cfg: PalfConfig = None):
        self.pattern_names = [str(p) for p in pattern_names]
        self.cfg = cfg if cfg is not None else PalfConfig()
        self._registry: Dict[str, Callable[[List[dict], int], float]] = dict(self._global_registry)
        self.initial_loss: Dict[str, float] = {p: np.nan for p in self.pattern_names}
        self.history: Dict[str, List[dict]] = {p: [] for p in self.pattern_names}

    @classmethod
    def register_variable_global(cls, name: str, fn: Callable[[List[dict], int], float]):
        """Dang ky ham tinh bien state moi o muc lop ma khong can sua loi."""
        cls._global_registry[str(name)] = fn

    def register_variable(self, name: str, fn: Callable[[List[dict], int], float]):
        """Dang ky ham tinh bien state moi cho instance hien tai."""
        self._registry[str(name)] = fn

    def update(self, epoch: int, stats: dict) -> dict:
        """stats[p] = {'loss_mean','loss_std','n_seen','grad_norm'} (chi tin hieu train).
        Tra ve state[p] = {ten_bien: float} (NaN neu chua du lich su)."""
        if not isinstance(stats, dict):
            raise TypeError("stats phai la dict anh xa pattern -> thong ke train.")

        # Kiem tra nguyen tac bat bien 1: khong nhan bat ky khoa val/test nao
        for top_k in stats.keys():
            if str(top_k).lower() in FORBIDDEN_VAL_TEST_KEYS:
                raise ValueError(f"LearningStateTracker khong duoc nhan du lieu '{top_k}' (chi dung tin hieu train).")

        state_out = {}
        window = int(self.cfg.window)

        for p in self.pattern_names:
            p_stat = stats.get(p, {})
            if not isinstance(p_stat, dict):
                raise TypeError(f"stats['{p}'] phai la dict thong ke train.")
            for k_stat in p_stat.keys():
                if str(k_stat).lower() in FORBIDDEN_VAL_TEST_KEYS:
                    raise ValueError(
                        f"stats['{p}'] chua khoa bi cam '{k_stat}'. LearningStateTracker chi nhan tin hieu train."
                    )

            n_seen = int(p_stat.get("n_seen", 0)) if p_stat.get("n_seen") is not None else 0
            loss_mean = float(p_stat.get("loss_mean", np.nan)) if n_seen > 0 else float("nan")
            loss_std = float(p_stat.get("loss_std", np.nan)) if n_seen > 0 else float("nan")
            grad_norm = float(p_stat.get("grad_norm", np.nan))

            if np.isnan(self.initial_loss[p]) and not np.isnan(loss_mean) and loss_mean > 0:
                self.initial_loss[p] = loss_mean

            base_l1 = self.initial_loss[p]
            if not np.isnan(loss_mean) and not np.isnan(base_l1) and base_l1 > 0:
                loss_ratio = float(loss_mean / base_l1)
            else:
                loss_ratio = float("nan")

            rec = {
                "epoch": int(epoch),
                "n_seen": n_seen,
                "loss_mean": loss_mean,
                "loss_std": loss_std,
                "grad_norm": grad_norm,
                "loss_ratio": loss_ratio,
            }
            self.history[p].append(rec)

            p_state = {}
            for var_name in self.cfg.state_vars:
                if var_name not in self._registry:
                    raise KeyError(
                        f"Bien Learning State '{var_name}' chua duoc dang ky. "
                        f"Cac bien hien co: {list(self._registry.keys())}"
                    )
                val = self._registry[var_name](self.history[p], window)
                p_state[var_name] = float(val) if val is not None else float("nan")

            state_out[p] = p_state

        return state_out
