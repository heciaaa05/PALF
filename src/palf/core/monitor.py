import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from palf.data.synthetic import PATTERNS


class LearningDynamicsMonitor:
    """Ghi loss tung sample theo epoch, tong hop theo pattern (kieu Dataset Cartography)."""

    def __init__(self, pat_train, pat_val, pattern_names=None, sample_index=None):
        self.pat = np.asarray(pat_train)
        self.val_pat = np.asarray(pat_val)
        self.pattern_names = list(pattern_names) if pattern_names is not None else PATTERNS
        self.n = len(self.pat)
        if sample_index is None:
            self.sample_index = np.arange(self.n, dtype=np.int64)
        else:
            self.sample_index = np.asarray(sample_index, dtype=np.int64)
        self.loss_hist = []
        self.rows = []
        self._cur = np.full(self.n, np.nan, dtype=np.float32)

    def start_epoch(self):
        self._cur = np.full(self.n, np.nan, dtype=np.float32)

    def log_batch(self, idx, sample_loss):
        self._cur[idx.numpy()] = sample_loss.detach().cpu().numpy()

    def end_epoch(self, epoch, val_losses, grad_norms):
        self.loss_hist.append(self._cur.copy())
        self._last_train_stats = {}
        for p, name in enumerate(self.pattern_names):
            m = self.pat == p
            if m.sum() == 0:
                continue
            v = self._cur[m]
            n_seen = int(np.isfinite(v).sum())
            if n_seen > 0:
                loss_mean = float(np.nanmean(v))
                loss_std = float(np.nanstd(v))
            else:
                loss_mean = float("nan")
                loss_std = float("nan")
            gn = float(grad_norms.get(name, np.nan))
            self._last_train_stats[name] = {
                "loss_mean": loss_mean,
                "loss_std": loss_std,
                "n_seen": n_seen,
                "grad_norm": gn,
            }
            vm = self.val_pat == p
            self.rows.append({
                "epoch": epoch,
                "pattern": name,
                "n": int(m.sum()),
                "n_seen": n_seen,
                "train_loss_mean": loss_mean,
                "train_loss_std": loss_std,
                "val_loss": float(val_losses[vm].mean()) if vm.any() else np.nan,
                "grad_norm": gn,
            })

    def get_last_train_stats(self):
        return dict(getattr(self, "_last_train_stats", {}))

    def save(self, out_dir, save_sample_losses=True):
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.rows)
        df.to_csv(os.path.join(out_dir, "epoch_pattern_log.csv"), index=False)
        h = np.stack(self.loss_hist)  # (epochs, n)
        pattern_labels_train = np.asarray([self.pattern_names[int(p)] for p in self.pat], dtype=str)
        finite_counts = np.isfinite(h).sum(axis=0)
        mean_loss = np.zeros(self.n, dtype=np.float64)
        std_loss = np.zeros(self.n, dtype=np.float64)
        has_obs = finite_counts > 0
        if has_obs.any():
            mean_loss[has_obs] = np.nanmean(h[:, has_obs], axis=0)
            std_loss[has_obs] = np.nanstd(h[:, has_obs], axis=0)
        if (~has_obs).any():
            for p_idx in range(len(self.pattern_names)):
                m_unseen = (~has_obs) & (self.pat == p_idx)
                if not m_unseen.any():
                    continue
                m_seen_pat = has_obs & (self.pat == p_idx)
                fallback_mean = float(mean_loss[m_seen_pat].mean()) if m_seen_pat.any() else 0.0
                mean_loss[m_unseen] = fallback_mean
                std_loss[m_unseen] = 0.0
        carto = pd.DataFrame({
            "idx": self.sample_index,
            "pattern": pattern_labels_train,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
        })
        carto.to_csv(os.path.join(out_dir, "cartography.csv"), index=False)
        if save_sample_losses:
            sample_loss_epochs = np.ascontiguousarray(h.T, dtype=np.float32)  # (n_train, epochs)
            np.save(os.path.join(out_dir, "sample_loss_epochs.npy"), sample_loss_epochs)
            np.save(os.path.join(out_dir, "sample_index.npy"), self.sample_index.astype(np.int64))
            np.save(os.path.join(out_dir, "pattern_labels_train.npy"), pattern_labels_train)
        return df


def build_probe(ds, pat_train, k=256, seed=0, pattern_names=None):
    if pattern_names is None:
        pattern_names = PATTERNS
    rng = np.random.default_rng(seed)
    probe = {}
    for p, name in enumerate(pattern_names):
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
