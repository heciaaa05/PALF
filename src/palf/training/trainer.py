import json
import os
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from palf.data.synthetic import prepare_data, PATTERNS
from palf.models import build_model, count_params, resolve_model_name
from palf.core.monitor import LearningDynamicsMonitor, build_probe, probe_grad_norms


def get_git_commit():
    try:
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def get_device(name):
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    if name == "cuda":
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Thiet bi '{name}' khong hop le. Chon: cpu, cuda, dml.")


@torch.no_grad()
def predict(model, ds, device, bs=512):
    model.eval()
    out = []
    for x, _, _, _ in DataLoader(ds, batch_size=bs):
        out.append(model(x.to(device)).cpu())
    return torch.cat(out).numpy()


def train_model(model_name, epochs=30, batch=128, lr=1e-3, device_name="cpu",
                seed=42, data=None, on_epoch=None, out_root="experiments/results",
                target_idx=0, patience=None, pattern_names=None,
                out_dir=None, use_seed_subdir=False, save_sample_losses=True,
                pattern_scheme=None, target_col=None):
    """patience=None (mac dinh): chay du `epochs`, khong doi so voi truoc day.
    patience=so nguyen duong: dung som neu val_loss khong cai thien sau
    `patience` epoch lien tiep, dung checkpoint tot nhat da luu."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    canonical_model = resolve_model_name(model_name)
    device = get_device(device_name)
    train, val, test, th = data[:4] if data is not None else prepare_data()
    meta = data[4] if (data is not None and len(data) >= 5 and isinstance(data[4], dict)) else {}
    th_dict = th if isinstance(th, dict) else {}

    if pattern_names is None:
        if "pattern_names" in meta:
            pattern_names = meta["pattern_names"]
        elif "pattern_names" in th_dict:
            pattern_names = th_dict["pattern_names"]
        else:
            pattern_names = PATTERNS

    if pattern_scheme is None:
        pattern_scheme = meta.get("pattern_scheme", th_dict.get("pattern_scheme", "synthetic"))

    if target_col is None:
        target_col = meta.get("target_col", th_dict.get("target_col", "y"))

    if "target_idx" in meta and target_idx == 0:
        target_idx = int(meta["target_idx"])
    elif "target_idx" in th_dict and target_idx == 0:
        target_idx = int(th_dict["target_idx"])

    target_mean = float(meta.get("target_mean", th_dict.get("target_mean", 0.0)))
    target_std = float(meta.get("target_std", th_dict.get("target_std", 1.0)))

    pat_train = train.tensors[2].numpy()
    pat_val = val.tensors[2].numpy()
    pat_test = test.tensors[2].numpy()
    sample_index_train = train.tensors[3].numpy().astype(np.int64)
    y_val = val.tensors[1].numpy()
    y_test = test.tensors[1].numpy()
    L, H = int(train.tensors[0].shape[1]), int(train.tensors[1].shape[1])

    input_dim = int(train.tensors[0].shape[-1])
    model = build_model(canonical_model, input_len=L, horizon=H, input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train, batch_size=batch, shuffle=True)
    monitor = LearningDynamicsMonitor(
        pat_train, pat_val, pattern_names=pattern_names, sample_index=sample_index_train
    )
    probe = build_probe(train, pat_train, pattern_names=pattern_names)

    history = []
    best = {"val": float("inf"), "state": None, "epoch": 0}
    t0 = time.time()
    epochs_since_improve = 0
    stopped_early = False
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        monitor.start_epoch()
        total, count = 0.0, 0
        for x, y, _, idx in loader:
            x, y = x.to(device), y.to(device)
            sample_loss = ((model(x) - y) ** 2).mean(1)
            loss = sample_loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            monitor.log_batch(idx, sample_loss)
            total += loss.item() * len(x)
            count += len(x)

        val_s = ((predict(model, val, device) - y_val) ** 2).mean(1)
        grads = probe_grad_norms(model, probe, device)
        monitor.end_epoch(epoch, val_s, grads)
        epoch_time_sec = time.time() - t_ep

        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "val_loss": float(val_s.mean()),
            "grad_norm": float(np.mean(list(grads.values()))),
            "epoch_time_sec": float(epoch_time_sec),
            "n_samples_processed": int(count),
            "elapsed": time.time() - t0,
        }
        history.append(row)

        improved = row["val_loss"] < best["val"]
        if improved:
            best = {"val": row["val_loss"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if on_epoch:
            on_epoch(row, epochs)

        epochs_run = epoch
        if patience is not None and epochs_since_improve >= patience:
            stopped_early = True
            break

    model.load_state_dict(best["state"])
    pred = predict(model, test, device)
    se = ((pred - y_test) ** 2).mean(1)
    ae = np.abs(pred - y_test).mean(1)

    pred_orig = pred * target_std + target_mean
    y_test_orig = y_test * target_std + target_mean
    se_orig = ((pred_orig - y_test_orig) ** 2).mean(1)
    ae_orig = np.abs(pred_orig - y_test_orig).mean(1)

    mse_norm = float(se.mean())
    rmse_norm = float(np.sqrt(mse_norm))
    mae_norm = float(ae.mean())

    mse_degC = float(se_orig.mean())
    rmse_degC = float(np.sqrt(mse_degC))
    mae_degC = float(ae_orig.mean())

    by_pattern = []
    for p, name in enumerate(pattern_names):
        m = pat_test == p
        if m.sum() == 0:
            continue
        p_mse_norm = float(se[m].mean())
        p_mse_degC = float(se_orig[m].mean())
        by_pattern.append({
            "pattern": name,
            "n": int(m.sum()),
            "mse": p_mse_norm,
            "rmse": float(np.sqrt(p_mse_norm)),
            "mae": float(ae[m].mean()),
            "mse_norm": p_mse_norm,
            "rmse_norm": float(np.sqrt(p_mse_norm)),
            "mae_norm": float(ae[m].mean()),
            "mse_degC": p_mse_degC,
            "rmse_degC": float(np.sqrt(p_mse_degC)),
            "mae_degC": float(ae_orig[m].mean()),
        })

    examples = {}
    for p, name in enumerate(pattern_names):
        ids = np.where(pat_test == p)[0]
        if len(ids) == 0:
            continue
        i = ids[len(ids) // 2]
        examples[name] = {"x": test.tensors[0][i, :, target_idx].numpy(), "y": y_test[i], "pred": pred[i]}

    if out_dir is None:
        if use_seed_subdir:
            out_dir = os.path.join(out_root, canonical_model.lower(), f"seed{seed}")
        else:
            out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    monitor.save(out_dir, save_sample_losses=save_sample_losses)

    git_commit = get_git_commit()
    run_config = {
        "model": canonical_model.lower(),
        "model_name": canonical_model,
        "seed": int(seed),
        "epochs": int(epochs),
        "epochs_run": int(epochs_run),
        "patience": patience,
        "stopped_early": bool(stopped_early),
        "batch": int(batch),
        "lr": float(lr),
        "device": str(device_name),
        "pattern_scheme": str(pattern_scheme),
        "pattern_names": list(pattern_names),
        "target_col": str(target_col),
        "target_idx": int(target_idx),
        "input_len": int(L),
        "horizon": int(H),
        "input_dim": int(input_dim),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "params": int(count_params(model)),
        "save_sample_losses": bool(save_sample_losses),
        "git_commit": git_commit,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    test_metrics = {
        "mae_norm": mae_norm,
        "rmse_norm": rmse_norm,
        "mse_norm": mse_norm,
        "mae_degC": mae_degC,
        "rmse_degC": rmse_degC,
        "mse_degC": mse_degC,
        "mae_orig": mae_degC,
        "rmse_orig": rmse_degC,
        "mse_orig": mse_degC,
        "normalized": {
            "mae": mae_norm,
            "rmse": rmse_norm,
            "mse": mse_norm,
        },
        "degC": {
            "mae": mae_degC,
            "rmse": rmse_degC,
            "mse": mse_degC,
        },
        "target_col": str(target_col),
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val"]),
        "n_test": int(len(test)),
        "by_pattern": by_pattern,
    }
    with open(os.path.join(out_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    return {
        "model_name": canonical_model,
        "params": count_params(model),
        "device": str(device),
        "elapsed": time.time() - t0,
        "best_epoch": best["epoch"],
        "best_val": best["val"],
        "stopped_early": stopped_early,
        "epochs_run": epochs_run,
        "history": hist_df,
        "pattern_log": pd.DataFrame(monitor.rows),
        "test": {
            "mse": mse_norm,
            "rmse": rmse_norm,
            "mae": mae_norm,
            "mse_degC": mse_degC,
            "rmse_degC": rmse_degC,
            "mae_degC": mae_degC,
            "by_pattern": pd.DataFrame(by_pattern),
        },
        "run_config": run_config,
        "test_metrics": test_metrics,
        "examples": examples,
        "out_dir": out_dir,
    }
