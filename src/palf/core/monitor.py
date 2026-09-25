import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from palf.data import PATTERNS


class LearningDynamicsMonitor:
    """Ghi loss tung sample theo epoch, tong hop theo pattern (kieu Dataset Cartography)."""

    def __init__(self, pat_train, pat_val):
        self.pat = np.asarray(pat_train)
        self.val_pat = np.asarray(pat_val)
        self.n = len(self.pat)
        self.loss_hist = []
        self.rows = []
        self._cur = np.full(self.n, np.nan, dtype=np.float32)

    def start_epoch(self):
        self._cur = np.full(self.n, np.nan, dtype=np.float32)

    def log_batch(self, idx, sample_loss):
        self._cur[idx.numpy()] = sample_loss.detach().cpu().numpy()

    def end_epoch(self, epoch, val_losses, grad_norms):
        self.loss_hist.append(self._cur.copy())
        for p, name in enumerate(PATTERNS):
            m = self.pat == p
            if m.sum() == 0:
                continue
            v = self._cur[m]
            vm = self.val_pat == p
            self.rows.append({
                "epoch": epoch,
                "pattern": name,
                "n": int(m.sum()),
                "train_loss_mean": float(np.nanmean(v)),
                "train_loss_std": float(np.nanstd(v)),
                "val_loss": float(val_losses[vm].mean()) if vm.any() else np.nan,
                "grad_norm": grad_norms.get(name, np.nan),
            })

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.rows)
        df.to_csv(os.path.join(out_dir, "epoch_pattern_log.csv"), index=False)
        h = np.stack(self.loss_hist)  # (epochs, n)
        carto = pd.DataFrame({
            "idx": np.arange(self.n),
            "pattern": [PATTERNS[p] for p in self.pat],
            "mean_loss": h.mean(0),
            "std_loss": h.std(0),
        })
        carto.to_csv(os.path.join(out_dir, "cartography.csv"), index=False)
        return df


def build_probe(ds, pat_train, k=256, seed=0):
    rng = np.random.default_rng(seed)
    probe = {}
    for p, name in enumerate(PATTERNS):
        ids = np.where(pat_train == p)[0]
        if len(ids) == 0:
            continue
        ids = rng.choice(ids, size=min(k, len(ids)), replace=False)
        probe[name] = (ds.tensors[0][ids], ds.tensors[1][ids])
    return probe


def probe_grad_norms(model, probe, device):
    """Gradient norm tren mot tap probe co dinh cho moi pattern."""
    model.train()
    out = {}
    for name, (x, y) in probe.items():
        model.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x.to(device)), y.to(device))
        loss.backward()
        sq = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
        out[name] = float(sq.sqrt())
    model.zero_grad(set_to_none=True)
    return out