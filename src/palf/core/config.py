"""Cau hinh trung tam PalfConfig cho Policy Engine va Adaptive Scheduler (Spec Phase 5-6 v1).

CANH BAO (Prompt 7 Buoc 3b):
- `DEFAULT_STATE_SIGNS` la gia thuyet ban dau tu v0 va CHUA DUOC Phase 3 xac nhan tren ca 4 backbone.
- Che do `policy="state"` chi giu lai de doi chieu lich su, KHONG dung trong bang so sanh chinh.
- Bang so sanh chinh su dung `policy="predictive"` (doc he so hoi quy tu `predictive_coef_path`).
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


DEFAULT_STATE_VARS: Tuple[str, ...] = ("loss_ratio", "velocity", "instability")
DEFAULT_STATE_SIGNS: Dict[str, float] = {
    "loss_ratio": 1.0,
    "velocity": 1.0,
    "instability": 0.5,
}
DEFAULT_PREDICTIVE_COEF_PATH: str = "experiments/analysis/phase3_real/predictive_coef.json"


@dataclass
class PalfConfig:
    """Cau hinh cho Policy Engine va Adaptive Scheduler.
    Luu y: Che do 'state' dung DEFAULT_STATE_SIGNS chua duoc Phase 3 xac nhan (chi giu de doi chieu);
    khuyen nghi dung 'predictive' (PALF v1) hoac 'loss' (B2) / 'uniform' (A0r)."""

    policy: str = "state"        # "predictive" (PALF v1) | "state" (v0) | "loss" (B2) | "uniform" (A0r)
    warmup_epochs: int = 3
    update_every: int = 1        # cap nhat trong so moi bao nhieu epoch
    alpha: float = 1.0           # do manh phan ung
    w_min: float = 0.25          # trong so san (skip mem)
    w_max: float = 3.0
    ema_beta: float = 0.5        # lam muot trong so theo thoi gian
    budget_ratio: float = 1.0    # rho: ty le so mau moi epoch so voi n_train
    window: int = 3              # W: cua so epoch tinh xu huong/dao dong
    state_vars: Tuple[str, ...] = DEFAULT_STATE_VARS
    state_signs: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STATE_SIGNS))
    predictive_coef_path: str = DEFAULT_PREDICTIVE_COEF_PATH
    z_clip: float = 3.0          # gioi han tuyet doi cua z-score
    min_floor_samples: int = 32  # san so mau toi thieu moi Pattern (Muc 6)
    min_floor_frac: float = 0.05 # ty le san 5% * pi_p * n_e (Muc 6)
    # Prompt 10 — Doi chung A0w "trong so xao tron": hoan vi ngau nhien trong so tinh duoc giua cac Pattern
    # moi epoch (truoc EMA / rao chan / Scheduler). RNG rieng, doc lap voi seed huan luyen.
    shuffle_weights: bool = False
    shuffle_seed: Optional[int] = None

    def __post_init__(self):
        if self.policy not in ("predictive", "state", "loss", "loss_abs", "uniform"):
            raise ValueError(
                f"policy='{self.policy}' khong hop le. Chon: 'predictive', 'state', 'loss', 'loss_abs', 'uniform'."
            )
        if self.shuffle_weights and self.policy == "uniform":
            raise ValueError("shuffle_weights=True khong co y nghia voi policy='uniform'.")
        self.shuffle_weights = bool(self.shuffle_weights)
        if self.shuffle_seed is not None:
            self.shuffle_seed = int(self.shuffle_seed)
        if not (0.0 < self.budget_ratio <= 1.0):
            raise ValueError(f"budget_ratio={self.budget_ratio} phai nam trong (0, 1].")
        if self.w_min <= 0.0 or self.w_max < self.w_min:
            raise ValueError(f"Khoang trong so [{self.w_min}, {self.w_max}] khong hop le.")
        if not (0.0 <= self.ema_beta < 1.0):
            raise ValueError(f"ema_beta={self.ema_beta} phai nam trong [0, 1).")
        if self.window < 1:
            raise ValueError(f"window={self.window} phai >= 1.")
        if self.update_every < 1:
            raise ValueError(f"update_every={self.update_every} phai >= 1.")
        self.state_vars = tuple(self.state_vars)
        if not self.state_signs:
            self.state_signs = {v: DEFAULT_STATE_SIGNS.get(v, 1.0) for v in self.state_vars}
        else:
            self.state_signs = {str(k): float(v) for k, v in self.state_signs.items()}
        self.predictive_coef_path = str(self.predictive_coef_path)

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "warmup_epochs": int(self.warmup_epochs),
            "update_every": int(self.update_every),
            "alpha": float(self.alpha),
            "w_min": float(self.w_min),
            "w_max": float(self.w_max),
            "ema_beta": float(self.ema_beta),
            "budget_ratio": float(self.budget_ratio),
            "window": int(self.window),
            "state_vars": list(self.state_vars),
            "state_signs": dict(self.state_signs),
            "predictive_coef_path": str(self.predictive_coef_path),
            "z_clip": float(self.z_clip),
            "min_floor_samples": int(self.min_floor_samples),
            "min_floor_frac": float(self.min_floor_frac),
            "shuffle_weights": bool(self.shuffle_weights),
            "shuffle_seed": None if self.shuffle_seed is None else int(self.shuffle_seed),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PalfConfig":
        if d is None:
            return cls()
        kwargs = dict(d)
        if "state_vars" in kwargs and kwargs["state_vars"] is not None:
            kwargs["state_vars"] = tuple(kwargs["state_vars"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return cls(**filtered)

    def config_tag(self) -> str:
        if self.policy == "uniform":
            return f"uniform_b{self.budget_ratio:.1f}"
        pol = f"{self.policy}_shuffled" if self.shuffle_weights else self.policy
        base = f"{pol}_b{self.budget_ratio:.1f}_a{self.alpha:.1f}"
        if abs(self.w_max - 3.0) > 1e-6:
            return f"{base}_w{self.w_max:g}"
        return base
