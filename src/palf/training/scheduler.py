"""Adaptive Scheduler: phan bo han ngach va lay mau khong lap theo Pattern cho moi epoch."""
from typing import Dict
import numpy as np

from palf.core.config import PalfConfig


class AdaptiveScheduler:
    """Lap ke hoach danh sach chi so mau train cho tung epoch dua tren trong so Policy."""

    def __init__(self, pat_train, pattern_names, cfg: PalfConfig = None, seed: int = 0):
        self.pat = np.asarray(pat_train, dtype=np.int64)
        self.pattern_names = [str(p) for p in pattern_names]
        self.cfg = cfg if cfg is not None else PalfConfig()
        self.seed = int(seed)
        self.n_train = len(self.pat)

        self.pattern_indices: Dict[str, np.ndarray] = {}
        self.pattern_counts: Dict[str, int] = {}
        self.freqs: Dict[str, float] = {}

        for idx_p, name in enumerate(self.pattern_names):
            ids = np.where(self.pat == idx_p)[0].astype(np.int64)
            self.pattern_indices[name] = ids
            self.pattern_counts[name] = int(len(ids))
            self.freqs[name] = float(len(ids) / self.n_train) if self.n_train > 0 else 0.0

        self.last_quotas: Dict[str, int] = {name: 0 for name in self.pattern_names}

    def _extract_weights(self, decision) -> Dict[str, float]:
        if decision is None:
            return {p: 1.0 for p in self.pattern_names}
        if hasattr(decision, "weights") and isinstance(decision.weights, dict):
            return {p: float(decision.weights.get(p, 1.0)) for p in self.pattern_names}
        if isinstance(decision, dict):
            return {p: float(decision.get(p, 1.0)) for p in self.pattern_names}
        return {p: 1.0 for p in self.pattern_names}

    def compute_quotas(self, weights: Dict[str, float]) -> Dict[str, int]:
        """Tinh han ngach n_p cho tung Pattern thoa:
        - Tong sum(n_p) == n_e = round(budget_ratio * n_train)
        - San: n_p >= floor_p
        - Tran: n_p <= N_p
        - Phan du sau cat san/tran duoc chia lai theo ty le q_p."""
        if self.n_train == 0:
            return {p: 0 for p in self.pattern_names}

        n_e = int(round(self.cfg.budget_ratio * self.n_train))
        n_e = max(1, min(self.n_train, n_e))

        active_pats = [p for p in self.pattern_names if self.pattern_counts[p] > 0]
        if not active_pats:
            return {p: 0 for p in self.pattern_names}

        # Neu n_e == n_train, moi Pattern bat buoc lay du N_p (khong the vuot tran N_p)
        if n_e == self.n_train:
            return {p: self.pattern_counts[p] for p in self.pattern_names}

        # 1. Tinh ty le muc tieu q_p ~ w_p * pi_p
        raw_q = np.array(
            [max(0.0, float(weights.get(p, 1.0))) * self.freqs[p] for p in active_pats],
            dtype=float,
        )
        if raw_q.sum() <= 0:
            raw_q = np.array([self.freqs[p] for p in active_pats], dtype=float)
        q = raw_q / raw_q.sum()

        # 2. Tinh san floor_p va tran cap_p = N_p
        caps = np.array([self.pattern_counts[p] for p in active_pats], dtype=int)
        floors = np.array(
            [
                min(
                    self.pattern_counts[p],
                    max(
                        int(self.cfg.min_floor_samples),
                        int(round(self.cfg.min_floor_frac * self.freqs[p] * n_e)),
                    ),
                )
                for p in active_pats
            ],
            dtype=int,
        )
        # Neu tong san vuot n_e (chi xay ra khi tap du lieu thu nghiem rat nho), co dan san theo ty le
        if floors.sum() > n_e:
            scaled = np.floor(floors * (n_e / floors.sum())).astype(int)
            floors = np.minimum(caps, np.maximum(0, scaled))

        # 3. Khoi tao han ngach va cat theo [floors, caps]
        quotas = np.round(n_e * q).astype(int)
        quotas = np.clip(quotas, floors, caps)

        # 4. Chia lai phan chenh lech cho cac Pattern con cho theo ty le q_p
        for _ in range(len(active_pats) + 4):
            diff = int(n_e - quotas.sum())
            if diff == 0:
                break
            if diff > 0:
                avail = caps - quotas
                elig = avail > 0
                if not elig.any():
                    break
                w_elig = q[elig]
                w_elig = w_elig / w_elig.sum() if w_elig.sum() > 0 else np.ones(elig.sum()) / elig.sum()
                exact_add = diff * w_elig
                int_add = np.floor(exact_add).astype(int)
                rem = int(diff - int_add.sum())
                if rem > 0:
                    order = np.argsort(-(exact_add - int_add))
                    for idx_o in order[:rem]:
                        int_add[idx_o] += 1
                int_add = np.minimum(avail[elig], int_add)
                if int_add.sum() == 0:
                    # Fallback: cong 1 cho pattern co q lon nhat con cho
                    best_i = np.where(elig)[0][np.argmax(q[elig])]
                    quotas[best_i] += 1
                else:
                    quotas[elig] += int_add
            else:
                need_sub = int(-diff)
                room = quotas - floors
                elig = room > 0
                if not elig.any():
                    break
                w_elig = q[elig]
                w_elig = w_elig / w_elig.sum() if w_elig.sum() > 0 else np.ones(elig.sum()) / elig.sum()
                exact_sub = need_sub * w_elig
                int_sub = np.floor(exact_sub).astype(int)
                rem = int(need_sub - int_sub.sum())
                if rem > 0:
                    order = np.argsort(-(exact_sub - int_sub))
                    for idx_o in order[:rem]:
                        int_sub[idx_o] += 1
                int_sub = np.minimum(room[elig], int_sub)
                if int_sub.sum() == 0:
                    best_i = np.where(elig)[0][np.argmax(room[elig])]
                    quotas[best_i] -= 1
                else:
                    quotas[elig] -= int_sub

        out = {p: 0 for p in self.pattern_names}
        for p, n_p in zip(active_pats, quotas):
            out[p] = int(n_p)
        return out

    def plan_epoch(self, epoch: int, decision=None) -> np.ndarray:
        """Chi so (vi tri trong train dataset) se huan luyen o epoch nay, da xao tron.
        decision=None => lay mau deu."""
        rng = np.random.default_rng([self.seed, int(epoch)])
        weights = self._extract_weights(decision)
        quotas = self.compute_quotas(weights)
        self.last_quotas = dict(quotas)

        chosen_parts = []
        for p in self.pattern_names:
            n_p = quotas.get(p, 0)
            if n_p <= 0:
                continue
            pool = self.pattern_indices[p]
            if n_p >= len(pool):
                sel = rng.permutation(pool)
            else:
                sel = rng.choice(pool, size=n_p, replace=False)
            chosen_parts.append(sel)

        if not chosen_parts:
            return np.empty(0, dtype=np.int64)

        all_indices = np.concatenate(chosen_parts).astype(np.int64)
        rng.shuffle(all_indices)
        return all_indices
