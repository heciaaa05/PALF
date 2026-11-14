"""Adaptive Scheduler (Spec v1 Muc 6): phan bo han ngach theo q_p (khong chan o N_p)
va lay mau co lap khi n_p > N_p de tang ty trong Pattern duoc uu tien."""
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
        """Tinh han ngach n_p cho tung Pattern theo Spec v1 Muc 6:
        - Tong sum(n_p) == n_e = round(budget_ratio * n_train)
        - Khi moi w_p bang nhau va n_e == n_train => n_p == N_p
        - Khong chan tran o N_p (cho phep n_p > N_p khi w_p cao)
        - San: n_p >= floor_p; phan chenh lech duoc chia lai theo ty le q_p."""
        if self.n_train == 0:
            return {p: 0 for p in self.pattern_names}

        n_e = int(round(self.cfg.budget_ratio * self.n_train))
        n_e = max(1, min(self.n_train, n_e))

        active_pats = [p for p in self.pattern_names if self.pattern_counts[p] > 0]
        if not active_pats:
            return {p: 0 for p in self.pattern_names}

        w_vals = np.array([max(0.0, float(weights.get(p, 1.0))) for p in active_pats], dtype=float)
        # Neu moi w_p bang nhau va n_e == n_train, moi Pattern lay dung N_p mau (toan bo train dung 1 lan)
        if n_e == self.n_train and np.allclose(w_vals, w_vals[0], atol=1e-12):
            return {p: self.pattern_counts[p] for p in self.pattern_names}

        # 1. Tinh ty le muc tieu q_p ~ w_p * pi_p
        raw_q = np.array([w_vals[i] * self.freqs[p] for i, p in enumerate(active_pats)], dtype=float)
        if raw_q.sum() <= 0:
            raw_q = np.array([self.freqs[p] for p in active_pats], dtype=float)
        q = raw_q / raw_q.sum()

        # 2. Tinh san floor_p (khong chan tran o N_p theo Spec v1 Muc 6)
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
        if floors.sum() > n_e:
            scaled = np.floor(floors * (n_e / floors.sum())).astype(int)
            floors = np.maximum(0, scaled)

        # 3. Khoi tao han ngach exact va ap san
        exact_n = n_e * q
        quotas = np.floor(exact_n).astype(int)
        rem_init = int(n_e - quotas.sum())
        if rem_init > 0:
            order = np.argsort(-(exact_n - quotas))
            for idx_o in order[:rem_init]:
                quotas[idx_o] += 1

        quotas = np.maximum(quotas, floors)

        # 4. Chia lai phan chenh lech sau khi nang san (lay tu cac Pattern con tren san theo ty le q_p)
        for _ in range(len(active_pats) + 4):
            diff = int(n_e - quotas.sum())
            if diff == 0:
                break
            if diff > 0:
                exact_add = diff * q
                int_add = np.floor(exact_add).astype(int)
                rem = int(diff - int_add.sum())
                if rem > 0:
                    order = np.argsort(-(exact_add - int_add))
                    for idx_o in order[:rem]:
                        int_add[idx_o] += 1
                quotas += int_add
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
        - Neu n_p <= N_p: chon ngau nhien khong lap n_p mau.
        - Neu n_p > N_p: lay toan bo N_p mau 1 lan roi chon them (n_p - N_p) mau co lap."""
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
            n_pool = len(pool)
            if n_pool == 0:
                continue
            if n_p == n_pool:
                sel = rng.permutation(pool)
            elif n_p < n_pool:
                sel = rng.choice(pool, size=n_p, replace=False)
            else:
                base_pass = rng.permutation(pool)
                extra = rng.choice(pool, size=n_p - n_pool, replace=True)
                sel = np.concatenate([base_pass, extra])
            chosen_parts.append(sel)

        if not chosen_parts:
            return np.empty(0, dtype=np.int64)

        all_indices = np.concatenate(chosen_parts).astype(np.int64)
        rng.shuffle(all_indices)
        return all_indices
