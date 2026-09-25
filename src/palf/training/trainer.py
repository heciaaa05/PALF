import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from palf.data import prepare_data, PATTERNS
from palf.models import build_model, count_params
from palf.core.monitor import LearningDynamicsMonitor, build_probe, probe_grad_norms


def get_device(name):
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, ds, device, bs=512):
    model.eval()
    out = []
    for x, _, _, _ in DataLoader(ds, batch_size=bs):
        out.append(model(x.to(device)).cpu())
    return torch.cat(out).numpy()


def train_model(model_name, epochs=30, batch=128, lr=1e-3, device_name="cpu",
                seed=42, data=None, on_epoch=None, out_root="experiments/results"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device(device_name)
    train, val, test, th = data if data is not None else prepare_data()

    pat_train = train.tensors[2].numpy()
    pat_val = val.tensors[2].numpy()
    pat_test = test.tensors[2].numpy()
    y_val = val.tensors[1].numpy()
    y_test = test.tensors[1].numpy()
    L, H = train.tensors[0].shape[1], train.tensors[1].shape[1]

    model = build_model(model_name, input_len=L, horizon=H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train, batch_size=batch, shuffle=True)
    monitor = LearningDynamicsMonitor(pat_train, pat_val)
    probe = build_probe(train, pat_train)

    history = []
    best = {"val": float("inf"), "state": None, "epoch": 0}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
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

        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "val_loss": float(val_s.mean()),
            "grad_norm": float(np.mean(list(grads.values()))),
            "elapsed": time.time() - t0,
        }
        history.append(row)
        if row["val_loss"] < best["val"]:
            best = {"val": row["val_loss"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if on_epoch:
            on_epoch(row, epochs)

    model.load_state_dict(best["state"])
    pred = predict(model, test, device)
    se = ((pred - y_test) ** 2).mean(1)
    ae = np.abs(pred - y_test).mean(1)

    by_pattern = []
    for p, name in enumerate(PATTERNS):
        m = pat_test == p
        if m.sum() == 0:
            continue
        by_pattern.append({"pattern": name, "n": int(m.sum()),
                           "mse": float(se[m].mean()), "mae": float(ae[m].mean())})

    examples = {}
    for p, name in enumerate(PATTERNS):
        ids = np.where(pat_test == p)[0]
        if len(ids) == 0:
            continue
        i = ids[len(ids) // 2]
        examples[name] = {"x": test.tensors[0][i, :, 0].numpy(), "y": y_test[i], "pred": pred[i]}

    out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    monitor.save(out_dir)

    return {
        "model_name": model_name,
        "params": count_params(model),
        "device": str(device),
        "elapsed": time.time() - t0,
        "best_epoch": best["epoch"],
        "best_val": best["val"],
        "history": hist_df,
        "pattern_log": pd.DataFrame(monitor.rows),
        "test": {"mse": float(se.mean()), "mae": float(ae.mean()),
                 "by_pattern": pd.DataFrame(by_pattern)},
        "examples": examples,
        "out_dir": out_dir,
    }