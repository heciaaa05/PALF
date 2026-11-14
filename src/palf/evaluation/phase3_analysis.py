"""Cong cu phan tich Phase 3 (Learning State Investigation) cho PALF.

Ho tro 2 che do:
- Che do rut gon ('reduced'): chi can epoch_pattern_log.csv (va history.csv neu co),
  dung duoc cho log synthetic cu.
- Che do day du ('full'): co sample_loss_epochs.npy va pattern_labels_train.npy,
  cho phep phan tich o muc su kien (run lien tiep) va kiem dinh doi chung hoan vi khoi.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import pandas as pd


CANDIDATE_VARS = [
    "loss_norm",
    "loss_trend",
    "learning_velocity",
    "prediction_stability_ps",
    "within_pattern_dispersion_dis",
    "grad_norm",
    "pattern_frequency",
    "generalization_gap",
    "forgetting_risk",
]


def chi2_sf(x, df):
    """Ham song sot P(Chi2(df) >= x) dung mpmath.gammainc."""
    if np.isnan(x) or df <= 0:
        return np.nan
    if x <= 0:
        return 1.0
    return float(mp.gammainc(df / 2.0, x / 2.0, regularized=True))


def f_sf(f_val, dfn, dfd):
    """Ham song sot P(F(dfn, dfd) >= f_val) dung mpmath.betainc."""
    if np.isnan(f_val) or dfn <= 0 or dfd <= 0:
        return np.nan
    if f_val <= 0:
        return 1.0
    x = dfd / (dfd + dfn * f_val)
    return float(mp.betainc(dfd / 2.0, dfn / 2.0, 0, x, regularized=True))


def discover_run_dirs(in_path):
    """Tim tat ca thu muc run co chua epoch_pattern_log.csv ben trong in_path."""
    in_path = os.path.abspath(in_path)
    if os.path.isfile(os.path.join(in_path, "epoch_pattern_log.csv")):
        return [in_path]

    found = []
    for root, dirs, files in os.walk(in_path):
        dirs.sort()
        if "epoch_pattern_log.csv" in files:
            found.append(root)
    return sorted(found)


def detect_run_mode(run_dir, requested_mode="auto"):
    """Phat hien che do 'full' hay 'reduced' cho mot thu muc run."""
    has_npy = (
        os.path.isfile(os.path.join(run_dir, "sample_loss_epochs.npy"))
        and os.path.isfile(os.path.join(run_dir, "pattern_labels_train.npy"))
    )
    if requested_mode == "reduced":
        return "reduced"
    if requested_mode == "full":
        if not has_npy:
            raise FileNotFoundError(
                f"Yeu cau che do 'full' nhung thieu sample_loss_epochs.npy hoac pattern_labels_train.npy tai {run_dir}"
            )
        return "full"
    return "full" if has_npy else "reduced"


def _infer_run_metadata(run_dir):
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return {
            "model": str(cfg.get("model", "unknown")),
            "seed": int(cfg.get("seed", 0)),
            "pattern_scheme": str(cfg.get("pattern_scheme", "unknown")),
        }

    parts = os.path.normpath(run_dir).split(os.sep)
    base = parts[-1]
    if base.lower().startswith("seed") and len(parts) >= 2:
        try:
            seed = int(base[4:])
        except ValueError:
            seed = 0
        model = parts[-2].lower()
    else:
        model = base.lower()
        seed = 0
    return {"model": model, "seed": seed, "pattern_scheme": "unknown"}


def compute_persistence_baseline_by_pattern(train_ds, pattern_names, target_idx=0):
    """Tinh MSE cua du bao persistence (y_hat = x[:, -1, target_idx]) theo tung Pattern."""
    x = train_ds.tensors[0].numpy()
    y = train_ds.tensors[1].numpy()
    pat = train_ds.tensors[2].numpy()
    last_val = x[:, -1:, target_idx]  # (n_train, 1)
    se = ((last_val - y) ** 2).mean(axis=1)
    out = {}
    for p, name in enumerate(pattern_names):
        m = pat == p
        out[name] = float(se[m].mean()) if m.any() else np.nan
    return out


def normalize_pattern_loss(df_log, normalization="epoch1", persistence_mse=None, eps=1e-8):
    """Muc 1: Chuan hoa loss theo tung Pattern.

    Tham so normalization:
    - 'epoch1': chia cho train_loss_mean cua chinh Pattern do o epoch nho nhat (epoch 1).
    - 'global_epoch1': chia cho trung binh co trong so cua toan tap o epoch 1.
    - 'persistence': chia cho MSE cua persistence baseline cua Pattern do (neu co).
    - 'raw': giu nguyen train_loss_mean.
    """
    out = df_log.copy()
    min_ep = out["epoch"].min()
    ep1 = out[out["epoch"] == min_ep]

    pat_ep1_map = ep1.set_index("pattern")["train_loss_mean"].to_dict()
    if "n" in ep1.columns and ep1["n"].sum() > 0:
        global_ep1 = float((ep1["train_loss_mean"] * ep1["n"]).sum() / ep1["n"].sum())
    else:
        global_ep1 = float(ep1["train_loss_mean"].mean())

    out["loss_raw"] = out["train_loss_mean"].astype(float)
    out["loss_norm_epoch1"] = out.apply(
        lambda r: float(r["train_loss_mean"]) / (pat_ep1_map.get(r["pattern"], np.nan) + eps),
        axis=1,
    )
    out["loss_norm_global_epoch1"] = out["train_loss_mean"] / (global_ep1 + eps)

    if persistence_mse is not None:
        out["loss_norm_persistence"] = out.apply(
            lambda r: float(r["train_loss_mean"]) / (persistence_mse.get(r["pattern"], np.nan) + eps)
            if r["pattern"] in persistence_mse and not np.isnan(persistence_mse[r["pattern"]])
            else np.nan,
            axis=1,
        )
    else:
        out["loss_norm_persistence"] = np.nan

    if normalization == "epoch1":
        out["loss_norm"] = out["loss_norm_epoch1"]
    elif normalization == "global_epoch1":
        out["loss_norm"] = out["loss_norm_global_epoch1"]
    elif normalization == "persistence":
        if persistence_mse is not None:
            out["loss_norm"] = out["loss_norm_persistence"]
        else:
            out["loss_norm"] = out["loss_norm_epoch1"]
    elif normalization == "raw":
        out["loss_norm"] = out["loss_raw"]
    else:
        raise ValueError(f"Che do chuan hoa '{normalization}' khong hop le.")

    return out


def _linear_slope(y):
    """He so goc hoi quy tuyen tinh cua chuoi y theo chi so t = 0..len(y)-1."""
    m = len(y)
    if m <= 1:
        return 0.0
    t = np.arange(m, dtype=float)
    t_mean = t.mean()
    y_arr = np.asarray(y, dtype=float)
    denom = ((t - t_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((t - t_mean) * (y_arr - y_arr.mean())).sum() / denom)


def compute_candidate_variables(df_log, W=5, k=3, normalization="epoch1", persistence_mse=None, eps=1e-8):
    """Muc 2: Tinh cac bien trang thai ung vien theo (run, Pattern, epoch)."""
    df = normalize_pattern_loss(df_log, normalization=normalization, persistence_mse=persistence_mse, eps=eps)

    if "model" not in df.columns:
        df["model"] = "model"
    if "seed" not in df.columns:
        df["seed"] = 0

    # Tinh pattern_frequency theo tung (model, seed, epoch)
    tot_n = df.groupby(["model", "seed", "epoch"])["n"].transform("sum")
    df["pattern_frequency"] = np.where(tot_n > 0, df["n"] / tot_n, np.nan)

    # Tinh within_pattern_dispersion_dis va generalization_gap
    df["within_pattern_dispersion_dis"] = df["train_loss_std"] / (df["train_loss_mean"].abs() + eps)
    if "val_loss" in df.columns:
        df["generalization_gap"] = df["val_loss"] - df["train_loss_mean"]
    else:
        df["generalization_gap"] = np.nan

    groups = []
    for (model, seed, pat), grp in df.groupby(["model", "seed", "pattern"], sort=False):
        g = grp.sort_values("epoch").copy()
        losses = g["loss_norm"].to_numpy(dtype=float)
        raw_losses = g["train_loss_mean"].to_numpy(dtype=float)
        n_ep = len(losses)

        lt = np.zeros(n_ep, dtype=float)
        lv = np.zeros(n_ep, dtype=float)
        ps = np.zeros(n_ep, dtype=float)
        fr = np.zeros(n_ep, dtype=float)
        delta_future_norm = np.full(n_ep, np.nan, dtype=float)
        delta_future_raw = np.full(n_ep, np.nan, dtype=float)

        best_so_far = np.inf
        for i in range(n_ep):
            win = losses[max(0, i - W + 1) : i + 1]
            lt[i] = _linear_slope(win)
            lv[i] = (losses[i - 1] - losses[i]) if i >= 1 else 0.0
            if len(win) >= 2:
                diffs = np.diff(win)
                ps[i] = float(np.std(diffs)) if len(diffs) >= 2 else float(np.abs(diffs[0]))
            else:
                ps[i] = 0.0
            best_so_far = min(best_so_far, losses[i])
            fr[i] = max(0.0, float(losses[i] - best_so_far))

            if i + k < n_ep:
                delta_future_norm[i] = float(losses[i + k] - losses[i])
                delta_future_raw[i] = float(raw_losses[i + k] - raw_losses[i])

        g["loss_trend"] = lt
        g["learning_velocity"] = lv
        g["prediction_stability_ps"] = ps
        g["forgetting_risk"] = fr
        g[f"future_delta_loss_k{k}"] = delta_future_norm
        g[f"future_delta_raw_k{k}"] = delta_future_raw
        groups.append(g)

    return pd.concat(groups, ignore_index=True)


def compute_redundancy_matrix(df_vars, variables=None, threshold=0.8):
    """Muc 3: Ma tran tuong quan hang Spearman giua cac bien va danh dau cap |rho| > threshold."""
    if variables is None:
        variables = [v for v in CANDIDATE_VARS if v in df_vars.columns]

    sub = df_vars[variables].copy()
    valid_cols = [c for c in variables if sub[c].notna().sum() >= 3 and sub[c].nunique(dropna=True) > 1]
    corr = sub[valid_cols].rank(method="average").corr(method="pearson")
    # Reindex to include all requested variables
    corr = corr.reindex(index=variables, columns=variables)

    pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            v1, v2 = variables[i], variables[j]
            rho = corr.loc[v1, v2]
            if pd.notna(rho):
                pairs.append({
                    "var_1": v1,
                    "var_2": v2,
                    "spearman_rho": float(rho),
                    "abs_rho": float(abs(rho)),
                    "is_redundant_gt_0p8": bool(abs(rho) > threshold),
                })
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    return corr, pairs_df


def extract_contiguous_runs(labels):
    """Tach chuoi nhan 1D thanh danh sach cac su kien (run lien tiep cung nhan).

    Tra ve list cac dict: {'run_id', 'pattern', 'start', 'end', 'length'}.
    """
    arr = np.asarray(labels)
    n = len(arr)
    if n == 0:
        return []
    runs = []
    start = 0
    run_id = 0
    for i in range(1, n + 1):
        if i == n or arr[i] != arr[start]:
            runs.append({
                "run_id": run_id,
                "pattern": str(arr[start]),
                "start": int(start),
                "end": int(i),
                "length": int(i - start),
            })
            run_id += 1
            start = i
    return runs


def build_event_run_table(sample_loss_epochs, pattern_labels_train, W=5, eps=1e-8):
    """Gom sample_loss_epochs (n_train, epochs) theo tung su kien (run lien tiep cung nhan)."""
    loss_mat = np.asarray(sample_loss_epochs, dtype=float)
    labels = np.asarray(pattern_labels_train)
    runs = extract_contiguous_runs(labels)

    # Tinh epoch 1 mean loss theo tung Pattern de chuan hoa
    pat_ep1 = {}
    for p in np.unique(labels):
        m = labels == p
        pat_ep1[str(p)] = float(loss_mat[m, 0].mean()) if m.any() else 1.0

    rows = []
    for r in runs:
        s, e = r["start"], r["end"]
        pat = r["pattern"]
        sub = loss_mat[s:e, :]  # (length, epochs)
        traj_raw = sub.mean(axis=0)  # (epochs,)
        traj_norm = traj_raw / (pat_ep1.get(pat, 1.0) + eps)

        diffs = np.diff(traj_norm)
        lv_mean = float((traj_norm[:-1] - traj_norm[1:]).mean()) if len(traj_norm) >= 2 else 0.0
        lt_slope = _linear_slope(traj_norm)
        ps_val = float(np.std(diffs)) if len(diffs) >= 2 else 0.0

        # Within-run dispersion across windows (for runs with length >= 2; else temporal std/mean)
        if sub.shape[0] >= 2:
            dis_val = float((sub.std(axis=0) / (sub.mean(axis=0) + eps)).mean())
        else:
            dis_val = float(traj_raw.std() / (traj_raw.mean() + eps))

        cum_min = np.minimum.accumulate(traj_norm)
        forgetting_val = float(np.maximum(0.0, traj_norm - cum_min).mean())

        rows.append({
            "run_id": r["run_id"],
            "pattern": pat,
            "start": s,
            "end": e,
            "length": r["length"],
            "run_loss_raw_mean": float(traj_raw.mean()),
            "run_loss_norm_mean": float(traj_norm.mean()),
            "run_loss_norm_end": float(traj_norm[-1]),
            "run_loss_trend": lt_slope,
            "run_learning_velocity": lv_mean,
            "run_prediction_stability_ps": ps_val,
            "run_within_dispersion_dis": dis_val,
            "run_forgetting_risk": forgetting_val,
        })
    return pd.DataFrame(rows)


def kruskal_wallis_stat(groups):
    """Tinh thong ke Kruskal-Wallis H, p-value, epsilon^2 va eta^2_H tren danh sach cac mang 1D."""
    clean_groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    clean_groups = [g[~np.isnan(g)] for g in clean_groups]
    clean_groups = [g for g in clean_groups if len(g) > 0]
    k = len(clean_groups)
    if k < 2:
        return {"H": np.nan, "p_value": np.nan, "epsilon_sq": np.nan, "eta_sq_H": np.nan, "n_total": 0, "k_groups": k}

    all_vals = np.concatenate(clean_groups)
    N = len(all_vals)
    if N < k or N <= 1:
        return {"H": np.nan, "p_value": np.nan, "epsilon_sq": np.nan, "eta_sq_H": np.nan, "n_total": N, "k_groups": k}

    ranks = pd.Series(all_vals).rank(method="average").to_numpy(dtype=float)
    # Tie correction
    _, counts = np.unique(all_vals, return_counts=True)
    tie_sum = float(np.sum(counts ** 3 - counts))
    tie_factor = 1.0 - tie_sum / (N ** 3 - N) if (N ** 3 - N) > 0 else 1.0
    if tie_factor <= 0:
        return {"H": 0.0, "p_value": 1.0, "epsilon_sq": 0.0, "eta_sq_H": 0.0, "n_total": N, "k_groups": k}

    idx = 0
    sum_sq = 0.0
    for g in clean_groups:
        n_i = len(g)
        r_i = ranks[idx : idx + n_i]
        sum_sq += (r_i.sum() ** 2) / n_i
        idx += n_i

    H = (12.0 / (N * (N + 1.0))) * sum_sq - 3.0 * (N + 1.0)
    H = max(0.0, float(H / tie_factor))
    df = k - 1
    p_val = chi2_sf(H, df)
    eps_sq = float(H / (N - 1.0)) if N > 1 else np.nan
    eta_sq = float(max(0.0, (H - k + 1.0) / (N - k))) if N > k else np.nan
    return {
        "H": H,
        "p_value": p_val,
        "epsilon_sq": eps_sq,
        "eta_sq_H": eta_sq,
        "n_total": int(N),
        "k_groups": int(k),
    }


def compute_pattern_differences_on_runs(run_df, metrics=None, n_boot=200, seed=42):
    """Muc 4 (che do day du): Kruskal-Wallis + bootstrap tren don vi SU KIEN (run)."""
    if metrics is None:
        metrics = [
            "run_loss_raw_mean",
            "run_loss_norm_mean",
            "run_loss_norm_end",
            "run_loss_trend",
            "run_learning_velocity",
            "run_prediction_stability_ps",
            "run_within_dispersion_dis",
            "run_forgetting_risk",
        ]
    rng = np.random.default_rng(seed)
    patterns = sorted(run_df["pattern"].unique())
    rows = []

    for m in metrics:
        if m not in run_df.columns:
            continue
        groups = [run_df.loc[run_df["pattern"] == p, m].dropna().to_numpy(dtype=float) for p in patterns]
        kw = kruskal_wallis_stat(groups)

        # Bootstrap tren cac run trong tung nhom Pattern
        boot_eps = []
        for _ in range(n_boot):
            b_groups = [
                g[rng.integers(0, len(g), size=len(g))] if len(g) > 0 else g
                for g in groups
            ]
            b_kw = kruskal_wallis_stat(b_groups)
            if not np.isnan(b_kw["epsilon_sq"]):
                boot_eps.append(b_kw["epsilon_sq"])

        ci_low = float(np.percentile(boot_eps, 2.5)) if boot_eps else np.nan
        ci_high = float(np.percentile(boot_eps, 97.5)) if boot_eps else np.nan

        row = {
            "sampling_unit": "event_run",
            "variable": m,
            "n_runs_total": kw["n_total"],
            "k_patterns": kw["k_groups"],
            "kruskal_H": kw["H"],
            "p_value": kw["p_value"],
            "epsilon_sq": kw["epsilon_sq"],
            "eta_sq_H": kw["eta_sq_H"],
            "epsilon_sq_ci95_low": ci_low,
            "epsilon_sq_ci95_high": ci_high,
        }
        for p, g in zip(patterns, groups):
            row[f"median_{p}"] = float(np.median(g)) if len(g) > 0 else np.nan
            row[f"n_runs_{p}"] = int(len(g))
        rows.append(row)

    return pd.DataFrame(rows)


def compute_pattern_differences_epoch_blocks(df_vars, variables=None, block_size=5, n_boot=200, seed=42):
    """Muc 4 (ho tro ca reduced va full): Kruskal-Wallis + block bootstrap theo khoi epoch."""
    if variables is None:
        variables = [v for v in CANDIDATE_VARS if v in df_vars.columns]
    rng = np.random.default_rng(seed)
    patterns = sorted(df_vars["pattern"].unique())
    epochs = sorted(df_vars["epoch"].unique())
    n_ep = len(epochs)

    rows = []
    for v in variables:
        sub = df_vars[["pattern", "epoch", "seed", v]].dropna()
        if sub.empty:
            continue
        groups = [sub.loc[sub["pattern"] == p, v].to_numpy(dtype=float) for p in patterns]
        kw = kruskal_wallis_stat(groups)

        # Block bootstrap theo khoi epoch lien tiep
        boot_eps = []
        if n_ep >= 2:
            b_len = min(block_size, n_ep)
            n_blocks = max(1, int(np.ceil(n_ep / b_len)))
            for _ in range(n_boot):
                starts = rng.integers(0, max(1, n_ep - b_len + 1), size=n_blocks)
                sampled_epochs = []
                for st in starts:
                    sampled_epochs.extend(epochs[st : st + b_len])
                sampled_epochs = sampled_epochs[:n_ep]
                b_groups = []
                for p in patterns:
                    p_sub = sub[sub["pattern"] == p]
                    vals = []
                    for ep in sampled_epochs:
                        vals.extend(p_sub.loc[p_sub["epoch"] == ep, v].to_numpy(dtype=float))
                    b_groups.append(np.asarray(vals, dtype=float))
                b_kw = kruskal_wallis_stat(b_groups)
                if not np.isnan(b_kw["epsilon_sq"]):
                    boot_eps.append(b_kw["epsilon_sq"])

        ci_low = float(np.percentile(boot_eps, 2.5)) if boot_eps else np.nan
        ci_high = float(np.percentile(boot_eps, 97.5)) if boot_eps else np.nan

        row = {
            "sampling_unit": "epoch_block",
            "variable": v,
            "n_obs_total": kw["n_total"],
            "k_patterns": kw["k_groups"],
            "kruskal_H": kw["H"],
            "p_value": kw["p_value"],
            "epsilon_sq": kw["epsilon_sq"],
            "eta_sq_H": kw["eta_sq_H"],
            "epsilon_sq_ci95_low": ci_low,
            "epsilon_sq_ci95_high": ci_high,
        }
        for p, g in zip(patterns, groups):
            row[f"median_{p}"] = float(np.median(g)) if len(g) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def compute_seed_stability(df_vars, variables=None):
    """Muc 5: Thu hang Pattern theo tung bien va tuong quan hang giua cac seed."""
    if variables is None:
        variables = [v for v in CANDIDATE_VARS if v in df_vars.columns]

    rank_rows = []
    stability_rows = []

    for model, m_grp in df_vars.groupby("model"):
        seeds = sorted(m_grp["seed"].unique())
        patterns = sorted(m_grp["pattern"].unique())

        for v in variables:
            seed_rank_Vectors = {}
            for s in seeds:
                s_grp = m_grp[m_grp["seed"] == s]
                means = s_grp.groupby("pattern")[v].mean().reindex(patterns)
                if means.isna().all():
                    continue
                ranks = means.rank(ascending=False, method="average")
                seed_rank_Vectors[s] = ranks
                for p in patterns:
                    rank_rows.append({
                        "model": model,
                        "seed": int(s),
                        "variable": v,
                        "pattern": p,
                        "mean_value": float(means.loc[p]) if pd.notna(means.loc[p]) else np.nan,
                        "rank": float(ranks.loc[p]) if pd.notna(ranks.loc[p]) else np.nan,
                    })

            valid_seeds = sorted(seed_rank_Vectors.keys())
            if len(valid_seeds) >= 2:
                pair_rhos = []
                for i in range(len(valid_seeds)):
                    for j in range(i + 1, len(valid_seeds)):
                        r1 = seed_rank_Vectors[valid_seeds[i]]
                        r2 = seed_rank_Vectors[valid_seeds[j]]
                        if r1.nunique() <= 1 or r2.nunique() <= 1:
                            rho = 1.0 if (r1.nunique() <= 1 and r2.nunique() <= 1) else np.nan
                        else:
                            rho = float(r1.rank(method="average").corr(r2.rank(method="average"), method="pearson"))
                        if not np.isnan(rho):
                            pair_rhos.append(rho)
                # Tinh Kendall's W
                rank_mat = np.vstack([seed_rank_Vectors[s].to_numpy(dtype=float) for s in valid_seeds])
                m_seeds, k_pats = rank_mat.shape
                col_sums = rank_mat.sum(axis=0)
                S_var = float(np.sum((col_sums - col_sums.mean()) ** 2))
                denom = (m_seeds ** 2) * (k_pats ** 3 - k_pats) / 12.0
                kendall_w = S_var / denom if denom > 0 else np.nan

                stability_rows.append({
                    "model": model,
                    "variable": v,
                    "n_seeds": len(valid_seeds),
                    "spearman_rank_mean": float(np.mean(pair_rhos)) if pair_rhos else np.nan,
                    "spearman_rank_std": float(np.std(pair_rhos)) if pair_rhos else np.nan,
                    "spearman_rank_min": float(np.min(pair_rhos)) if pair_rhos else np.nan,
                    "kendall_W": float(kendall_w),
                })
            else:
                stability_rows.append({
                    "model": model,
                    "variable": v,
                    "n_seeds": len(valid_seeds),
                    "spearman_rank_mean": np.nan,
                    "spearman_rank_std": np.nan,
                    "spearman_rank_min": np.nan,
                    "kendall_W": np.nan,
                })

    return pd.DataFrame(rank_rows), pd.DataFrame(stability_rows)


def _ols_r2_and_rss(X, y):
    """Fit OLS y ~ X (co cot hang so) va tra ve (r2, rss, beta)."""
    n = len(y)
    X_design = np.column_stack([np.ones(n, dtype=float), X])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    return r2, rss, beta


def compute_predictive_test(df_vars, k=3, variables=None, controls_epoch=False):
    """Muc 6: Kiem dinh bien X(e) co du doan Delta_loss(e -> e+k) ngoai L(e) hay khong.
    Neu controls_epoch=True: tru trung binh theo epoch (epoch fixed effects) khoi L(e), X(e) va Delta_loss."""
    target_col = f"future_delta_loss_k{k}"
    if target_col not in df_vars.columns:
        raise ValueError(f"Khong tim thay cot '{target_col}' trong df_vars.")

    if variables is None:
        variables = [v for v in CANDIDATE_VARS if v in df_vars.columns and v != "loss_norm"]

    rows = []
    for v in variables:
        cols_need = ["loss_norm", v, target_col]
        if controls_epoch and "epoch" in df_vars.columns:
            cols_need = ["epoch"] + cols_need
        sub = df_vars[cols_need].dropna().copy()
        n = len(sub)
        if n < 6 or sub[v].nunique() <= 1:
            rows.append({
                "variable": v,
                "horizon_k": int(k),
                "controls_epoch": bool(controls_epoch),
                "n_obs": int(n),
                "r2_baseline_loss_only": np.nan,
                "r2_with_variable": np.nan,
                "delta_r2": np.nan,
                "beta_loss_norm": np.nan,
                "beta_variable": np.nan,
                "F_stat": np.nan,
                "p_value": np.nan,
            })
            continue

        n_ep_groups = 1
        if controls_epoch and "epoch" in sub.columns:
            n_ep_groups = int(sub["epoch"].nunique())
            for c_dem in ("loss_norm", v, target_col):
                sub[c_dem] = sub[c_dem] - sub.groupby("epoch")[c_dem].transform("mean")

        L_cur = sub["loss_norm"].to_numpy(dtype=float)
        X_var = sub[v].to_numpy(dtype=float)
        y_delta = sub[target_col].to_numpy(dtype=float)

        r2_0, rss_0, _ = _ols_r2_and_rss(L_cur.reshape(-1, 1), y_delta)
        r2_1, rss_1, beta_1 = _ols_r2_and_rss(np.column_stack([L_cur, X_var]), y_delta)

        delta_r2 = max(0.0, float(r2_1 - r2_0))
        dfd = n - (2 + n_ep_groups) if controls_epoch else (n - 3)
        if rss_1 > 0 and dfd > 0:
            f_stat = max(0.0, float(((rss_0 - rss_1) / 1.0) / (rss_1 / dfd)))
            p_val = f_sf(f_stat, 1, dfd)
        else:
            f_stat = 0.0
            p_val = 1.0

        rows.append({
            "variable": v,
            "horizon_k": int(k),
            "controls_epoch": bool(controls_epoch),
            "n_obs": int(n),
            "r2_baseline_loss_only": float(r2_0),
            "r2_with_variable": float(r2_1),
            "delta_r2": delta_r2,
            "beta_loss_norm": float(beta_1[1]),
            "beta_variable": float(beta_1[2]),
            "F_stat": f_stat,
            "p_value": p_val,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("delta_r2", ascending=False).reset_index(drop=True)
    return out


def block_permute_pattern_labels(labels, rng=None):
    """Muc 7: Hoan vi nhan Pattern theo KHOI (run lien tiep cung nhan),
    giu nguyen phan phoi do dai doan va tong so cua so moi nhan."""
    if rng is None:
        rng = np.random.default_rng()
    runs = extract_contiguous_runs(labels)
    if not runs:
        return np.asarray(labels).copy()

    order = rng.permutation(len(runs))
    pieces = [np.full(runs[idx]["length"], runs[idx]["pattern"], dtype=object) for idx in order]
    return np.concatenate(pieces).astype(str)


def compute_null_permutation_control(sample_loss_epochs, pattern_labels_train, n_perm=50, W=5, seed=42):
    """Muc 7 (che do day du): So sanh hieu ung that voi phan phoi Null tu hoan vi khoi
    va doi chung thu hai dua tren phan vi do bien dong mau (sample volatility)."""
    rng = np.random.default_rng(seed)
    real_run_df = build_event_run_table(sample_loss_epochs, pattern_labels_train, W=W)
    real_kw_df = compute_pattern_differences_on_runs(real_run_df, n_boot=0, seed=seed)

    metrics = list(real_kw_df["variable"])
    null_eps = {m: [] for m in metrics}
    null_H = {m: [] for m in metrics}

    loss_mat = np.asarray(sample_loss_epochs, dtype=float)
    labels_arr = np.asarray(pattern_labels_train)
    runs = extract_contiguous_runs(labels_arr)
    n_runs = len(runs)

    # Tinh san cac dac trung cua tung run (khong phu thuoc nhan, tru loss_norm_mean/end)
    base_df = real_run_df.copy()

    for _ in range(n_perm):
        # Hoan vi nhan giua cac khoi run (giu nguyen so luong run cua moi nhan)
        perm_patterns = rng.permutation(base_df["pattern"].to_numpy())
        perm_df = base_df.copy()
        perm_df["pattern"] = perm_patterns

        for m in metrics:
            groups = [perm_df.loc[perm_df["pattern"] == p, m].to_numpy(dtype=float) for p in sorted(perm_df["pattern"].unique())]
            kw = kruskal_wallis_stat(groups)
            null_eps[m].append(kw["epsilon_sq"])
            null_H[m].append(kw["H"])

    summary_rows = []
    for _, r in real_kw_df.iterrows():
        m = r["variable"]
        arr_eps = np.asarray(null_eps[m], dtype=float)
        valid_eps = arr_eps[~np.isnan(arr_eps)]
        real_val = float(r["epsilon_sq"])
        if len(valid_eps) > 0 and not np.isnan(real_val):
            emp_p = float((np.sum(valid_eps >= real_val) + 1.0) / (len(valid_eps) + 1.0))
            null_mean = float(np.mean(valid_eps))
            null_std = float(np.std(valid_eps))
            null_p95 = float(np.percentile(valid_eps, 95))
            exceeds = bool(real_val > null_p95)
        else:
            emp_p = np.nan
            null_mean = np.nan
            null_std = np.nan
            null_p95 = np.nan
            exceeds = False
        summary_rows.append({
            "variable": m,
            "real_kruskal_H": float(r["kruskal_H"]),
            "real_epsilon_sq": real_val,
            "null_epsilon_sq_mean": null_mean,
            "null_epsilon_sq_std": null_std,
            "null_epsilon_sq_p95": null_p95,
            "empirical_p_value": emp_p,
            "exceeds_null_p95": exceeds,
            "n_permutations": int(n_perm),
        })

    # Doi chung thu hai: gan nhan chi dua tren do lon dao dong loss cua tung mau theo epoch
    sample_std = loss_mat.std(axis=1)
    unique_pats, pat_counts = np.unique(labels_arr, return_counts=True)
    order_pats = np.argsort(pat_counts)  # tu hiem den pho bien
    sorted_pats = unique_pats[order_pats]
    cum_counts = np.cumsum(pat_counts[order_pats])

    rank_idx = np.argsort(np.argsort(-sample_std))  # mau dao dong manh nhat xep truoc
    vol_labels = np.empty(len(labels_arr), dtype=object)
    prev = 0
    for p_name, c_end in zip(sorted_pats, cum_counts):
        mask = (rank_idx >= prev) & (rank_idx < c_end)
        vol_labels[mask] = str(p_name)
        prev = c_end

    vol_run_df = build_event_run_table(sample_loss_epochs, vol_labels.astype(str), W=W)
    vol_kw_df = compute_pattern_differences_on_runs(vol_run_df, n_boot=0, seed=seed)
    vol_kw_df["control_type"] = "volatility_quantile_pseudo_labels"

    return pd.DataFrame(summary_rows), vol_kw_df


def _df_to_markdown(df):
    if df is None or df.empty:
        return "*(Trống)*"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("NaN" if np.isnan(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def _save_plots(df_vars, corr, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Normalized loss curves per pattern (averaged across seeds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    grouped = df_vars.groupby(["pattern", "epoch"])[["train_loss_mean", "loss_norm"]].mean().reset_index()
    for pat, grp in grouped.groupby("pattern"):
        axes[0].plot(grp["epoch"], grp["train_loss_mean"], label=pat)
        axes[1].plot(grp["epoch"], grp["loss_norm"], label=pat)
    axes[0].set_title("Raw Train Loss by Pattern")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[1].set_title("Normalized Loss L(p, e) / L(p, 1)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Normalized Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "1_normalized_loss_curves.png"), dpi=150)
    plt.close(fig)

    # 2. Spearman correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mat = corr.to_numpy(dtype=float)
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spearman Correlation Matrix of Candidate State Variables")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "3_spearman_heatmap.png"), dpi=150)
    plt.close(fig)


def _try_get_jena_persistence_mse(patterns):
    """Neu tap pattern la weather_v1 va co du lieu Jena, tinh persistence MSE tren train."""
    from palf.data.patterns_weather import WEATHER_PATTERNS
    if set(patterns) == set(WEATHER_PATTERNS) and os.path.isfile("data/raw/jena_climate_2009_2016.csv"):
        try:
            from palf.data.weather import prepare_weather_data
            tr_ds, _, _, _, meta = prepare_weather_data(pattern_scheme="weather_v1")
            return compute_persistence_baseline_by_pattern(
                tr_ds, WEATHER_PATTERNS, target_idx=meta["target_idx"]
            )
        except Exception:
            return None
    return None


def run_phase3_analysis(
    in_path,
    out_dir,
    mode="auto",
    normalization="epoch1",
    W=5,
    k=3,
    n_boot=200,
    n_perm=50,
    seed=42,
):
    """Chay toan bo quy trinh phan tich Phase 3 cho 1 thu muc run hoac nhieu run/backbone."""
    run_dirs = discover_run_dirs(in_path)
    if not run_dirs:
        raise FileNotFoundError(f"Khong tim thay file epoch_pattern_log.csv nao trong '{in_path}'.")

    os.makedirs(out_dir, exist_ok=True)
    run_modes = {rd: detect_run_mode(rd, requested_mode=mode) for rd in run_dirs}
    effective_mode = "full" if all(m == "full" for m in run_modes.values()) else "reduced"

    all_logs = []
    full_runs_data = []

    for rd in run_dirs:
        meta = _infer_run_metadata(rd)
        df_log = pd.read_csv(os.path.join(rd, "epoch_pattern_log.csv"))
        df_log["model"] = meta["model"]
        df_log["seed"] = meta["seed"]
        df_log["run_dir"] = rd
        all_logs.append(df_log)

        if run_modes[rd] == "full":
            loss_ep = np.load(os.path.join(rd, "sample_loss_epochs.npy"))
            pat_lab = np.load(os.path.join(rd, "pattern_labels_train.npy"))
            full_runs_data.append((rd, meta, loss_ep, pat_lab))

    combined_log = pd.concat(all_logs, ignore_index=True)
    persistence_mse = _try_get_jena_persistence_mse(combined_log["pattern"].unique())

    # Muc 1 & Muc 2: Chuan hoa loss va tinh cac bien ung vien
    df_vars = compute_candidate_variables(
        combined_log, W=W, k=k, normalization=normalization, persistence_mse=persistence_mse
    )
    df_vars.to_csv(os.path.join(out_dir, "1_normalized_loss.csv"), index=False)
    df_vars.to_csv(os.path.join(out_dir, "2_candidate_variables.csv"), index=False)

    # Muc 3: Du thua (Spearman matrix)
    corr_df, redundant_pairs_df = compute_redundancy_matrix(df_vars)
    corr_df.to_csv(os.path.join(out_dir, "3_spearman_matrix.csv"))
    redundant_pairs_df.to_csv(os.path.join(out_dir, "3_redundant_pairs.csv"), index=False)

    # Muc 4: Khac biet giua cac Pattern
    kw_epoch_df = compute_pattern_differences_epoch_blocks(
        df_vars, block_size=W, n_boot=n_boot, seed=seed
    )
    kw_epoch_df.to_csv(os.path.join(out_dir, "4_kruskal_wallis_epoch_blocks.csv"), index=False)

    kw_runs_df = None
    null_perm_df = None
    vol_ctrl_df = None

    if effective_mode == "full" and full_runs_data:
        # Trung binh ma tran sample_loss_epochs qua cac seed cua cung tap train (hoac dung seed dau tien neu khac kich thuoc)
        shapes = {arr.shape for _, _, arr, _ in full_runs_data}
        if len(shapes) == 1:
            mean_loss_ep = np.mean([arr for _, _, arr, _ in full_runs_data], axis=0)
            pat_lab = full_runs_data[0][3]
        else:
            _, _, mean_loss_ep, pat_lab = full_runs_data[0]

        event_run_df = build_event_run_table(mean_loss_ep, pat_lab, W=W)
        event_run_df.to_csv(os.path.join(out_dir, "4_event_runs_table.csv"), index=False)

        kw_runs_df = compute_pattern_differences_on_runs(
            event_run_df, n_boot=n_boot, seed=seed
        )
        kw_runs_df.to_csv(os.path.join(out_dir, "4_kruskal_wallis_runs.csv"), index=False)

        # Muc 7: Nhom doi chung (Block permutation + Volatility quantile control)
        null_perm_df, vol_ctrl_df = compute_null_permutation_control(
            mean_loss_ep, pat_lab, n_perm=n_perm, W=W, seed=seed
        )
        null_perm_df.to_csv(os.path.join(out_dir, "7_null_permutation_summary.csv"), index=False)
        vol_ctrl_df.to_csv(os.path.join(out_dir, "7_volatility_control.csv"), index=False)

    # Muc 5: On dinh giua cac seed
    ranks_df, stability_df = compute_seed_stability(df_vars)
    ranks_df.to_csv(os.path.join(out_dir, "5_pattern_ranks_by_seed.csv"), index=False)
    stability_df.to_csv(os.path.join(out_dir, "5_seed_rank_stability.csv"), index=False)

    # Muc 6: Phep thu du doan
    pred_df = compute_predictive_test(df_vars, k=k)
    pred_df.to_csv(os.path.join(out_dir, "6_predictive_regression.csv"), index=False)

    # Luu hinh va config
    _save_plots(df_vars, corr_df, out_dir)

    cfg_used = {
        "in_path": os.path.abspath(in_path),
        "out_dir": os.path.abspath(out_dir),
        "mode_requested": mode,
        "mode_effective": effective_mode,
        "n_run_dirs": len(run_dirs),
        "run_dirs": run_dirs,
        "run_modes": run_modes,
        "normalization": normalization,
        "W": int(W),
        "k": int(k),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "seed": int(seed),
    }
    with open(os.path.join(out_dir, "analysis_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_used, f, indent=2)

    # Tao report.md tom tat cac bang so lieu (khong ket luan chon bien thay nguoi dung)
    report_lines = [
        "# Báo cáo Phân tích Phase 3 (Learning State Investigation)",
        "",
        f"- **Chế độ chạy (`mode_effective`):** `{effective_mode}`",
        f"- **Số thư mục run đầu vào:** `{len(run_dirs)}`",
        f"- **Chuẩn hóa (`normalization`):** `{normalization}` | **Cửa sổ `W`:** `{W}` | **Độ trễ dự đoán `k`:** `{k}`",
        "",
        "## Mục 3 — Các cặp biến có tương quan hạng Spearman |ρ| > 0.8",
        "",
    ]
    high_red = redundant_pairs_df[redundant_pairs_df["is_redundant_gt_0p8"]] if not redundant_pairs_df.empty else pd.DataFrame()
    if not high_red.empty:
        report_lines.append(_df_to_markdown(high_red))
    else:
        report_lines.append("Không có cặp biến nào có |ρ| > 0.8.")

    report_lines.extend([
        "",
        "## Mục 4 — Khác biệt giữa các Pattern (Kruskal-Wallis theo khối Epoch)",
        "",
        _df_to_markdown(kw_epoch_df),
    ])
    if kw_runs_df is not None:
        report_lines.extend([
            "",
            "## Mục 4 (Full Mode) — Khác biệt giữa các Pattern trên đơn vị SỰ KIỆN (`event_run`)",
            "",
            _df_to_markdown(kw_runs_df),
        ])

    report_lines.extend([
        "",
        "## Mục 5 — Độ ổn định hạng Pattern giữa các Seed",
        "",
        _df_to_markdown(stability_df),
        "",
        f"## Mục 6 — Phép thử dự đoán Δloss sau k={k} epoch (ngoài thông tin từ loss hiện tại)",
        "",
        _df_to_markdown(pred_df),
    ])

    if null_perm_df is not None:
        report_lines.extend([
            "",
            "## Mục 7 (Full Mode) — Kiểm định đối chứng hoán vị khối (Block Permutation Null Distribution)",
            "",
            _df_to_markdown(null_perm_df),
        ])

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return {
        "mode_effective": effective_mode,
        "config": cfg_used,
        "candidate_variables": df_vars,
        "spearman_matrix": corr_df,
        "redundant_pairs": redundant_pairs_df,
        "kruskal_epoch_blocks": kw_epoch_df,
        "kruskal_runs": kw_runs_df,
        "pattern_ranks": ranks_df,
        "seed_stability": stability_df,
        "predictive_test": pred_df,
        "null_permutation": null_perm_df,
        "volatility_control": vol_ctrl_df,
    }


def _smooth_series(vals: np.ndarray, method: str = "raw") -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    if method == "raw" or len(arr) == 0:
        return arr.copy()
    if method == "ma3":
        return pd.Series(arr).rolling(window=3, min_periods=1).mean().to_numpy(dtype=float)
    if method == "ema":
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = 0.5 * out[i - 1] + 0.5 * arr[i]
        return out
    raise ValueError(f"Unknown smoothing method: {method}")


def build_tracker_aligned_table(combined_log: pd.DataFrame, W: int = 3, k: int = 3, smooth: str = "raw") -> pd.DataFrame:
    """Tao bang bien trang thai khop 100% voi LearningStateTracker (loss_ratio, velocity, instability, loss_trend)
    va dev_ma3 (do lech khoi trung binh truot 3 epoch) de kiem dinh (a), (b), (c) va xuat predictive_coef.json."""
    df = combined_log.sort_values(["model", "seed", "pattern", "epoch"]).reset_index(drop=True)
    parts = []

    for (model, seed, pat), grp in df.groupby(["model", "seed", "pattern"], sort=False):
        grp = grp.sort_values("epoch").copy()
        raw_loss = grp["train_loss_mean"].to_numpy(dtype=float)
        ep1 = raw_loss[0] if (len(raw_loss) > 0 and raw_loss[0] > 0) else 1.0
        lr_raw = raw_loss / ep1
        lr = _smooth_series(lr_raw, method=smooth)

        n = len(lr)
        vel = np.full(n, np.nan, dtype=float)
        instab = np.full(n, np.nan, dtype=float)
        ltrend = np.full(n, np.nan, dtype=float)
        ma3 = pd.Series(lr_raw).rolling(window=3, min_periods=1).mean().to_numpy(dtype=float)
        dev_ma3 = lr_raw - ma3
        delta_k = np.full(n, np.nan, dtype=float)

        for i in range(n):
            if i >= W:
                vel[i] = (lr[i - W] - lr[i]) / float(W)
            if i + 1 >= W:
                win = lr[i - W + 1 : i + 1]
                instab[i] = float(np.std(win, ddof=0))
                ltrend[i] = _linear_slope(win)
            if i + k < n:
                delta_k[i] = lr[i + k] - lr[i]

        grp["loss_ratio"] = lr
        grp["loss_norm"] = lr
        grp["velocity"] = vel
        grp["instability"] = instab
        grp["prediction_stability_ps"] = instab
        grp["loss_trend"] = ltrend
        grp["dev_ma3"] = dev_ma3
        grp[f"future_delta_loss_k{k}"] = delta_k
        parts.append(grp)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fit_predictive_coef_dict(
    df_aligned: pd.DataFrame,
    features=("loss_ratio", "instability"),
    k: int = 3,
    fit_scope: str = "pooled",
    fit_seeds=(0, 1, 2, 3, 4),
    controls_epoch: bool = True,
) -> dict:
    """Uoc luong he so hoi quy Delta_loss_k ~ features (da chuan hoa z-score tren tap fit)
    co kiem soat hieu ung co dinh theo epoch, xuat dung dinh dang predictive_coef.json (Phan 1 Buoc 2)."""
    target_col = f"future_delta_loss_k{k}"
    sub = df_aligned.copy()
    if fit_scope != "pooled":
        sub = sub[sub["model"] == fit_scope]
    if fit_seeds is not None:
        sub = sub[sub["seed"].isin(list(fit_seeds))]

    cols = ["epoch", target_col] + list(features)
    sub = sub[cols].dropna().copy()
    if len(sub) < 6:
        raise ValueError(f"Khong du quan sat de fit predictive_coef cho scope={fit_scope}, seeds={fit_seeds}")

    standardize = {}
    Z_cols = []
    for f in features:
        arr = sub[f].to_numpy(dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=0))
        if sd <= 1e-12:
            sd = 1.0
        standardize[f] = {"mean": mu, "std": sd}
        sub[f"_z_{f}"] = (arr - mu) / sd
        Z_cols.append(f"_z_{f}")

    y_raw = sub[target_col].to_numpy(dtype=float)
    intercept = float(y_raw.mean())

    if controls_epoch:
        y_fit = (sub[target_col] - sub.groupby("epoch")[target_col].transform("mean")).to_numpy(dtype=float)
        X_fit = np.column_stack([
            (sub[zc] - sub.groupby("epoch")[zc].transform("mean")).to_numpy(dtype=float)
            for zc in Z_cols
        ])
        beta, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    else:
        X_design = np.column_stack([np.ones(len(sub)), sub[Z_cols].to_numpy(dtype=float)])
        b_full, _, _, _ = np.linalg.lstsq(X_design, y_raw, rcond=None)
        intercept = float(b_full[0])
        beta = b_full[1:]

    coef_dict = {f: float(b) for f, b in zip(features, beta)}
    return {
        "target": f"delta_loss_k{k}",
        "features": list(features),
        "standardize": standardize,
        "intercept": intercept,
        "coef": coef_dict,
        "fit_scope": str(fit_scope),
        "fit_seeds": [int(s) for s in fit_seeds],
        "controls_epoch": bool(controls_epoch),
    }


def _eval_cross_seed_oos(df_aligned: pd.DataFrame, var_name: str, train_seeds, test_seeds, k: int = 3) -> dict:
    """Huan luyen mo hinh (co kiem soat epoch) tren train_seeds, danh gia R2 ngoai mau tren test_seeds."""
    target_col = f"future_delta_loss_k{k}"
    sub = df_aligned[["model", "seed", "epoch", "loss_ratio", var_name, target_col]].dropna().copy()
    # Demean theo (model, epoch) tren tung tap train/test doc lap de loai trung binh epoch
    tr = sub[sub["seed"].isin(train_seeds)].copy()
    te = sub[sub["seed"].isin(test_seeds)].copy()
    if len(tr) < 6 or len(te) < 6:
        return {"r2_oos_loss_only": np.nan, "r2_oos_full": np.nan, "delta_r2_oos": np.nan}

    for df_part in (tr, te):
        for c in ("loss_ratio", var_name, target_col):
            df_part[c] = df_part[c] - df_part.groupby(["model", "epoch"])[c].transform("mean")

    X0_tr = np.column_stack([np.ones(len(tr)), tr["loss_ratio"].to_numpy(dtype=float)])
    X1_tr = np.column_stack([np.ones(len(tr)), tr["loss_ratio"].to_numpy(dtype=float), tr[var_name].to_numpy(dtype=float)])
    y_tr = tr[target_col].to_numpy(dtype=float)

    b0, _, _, _ = np.linalg.lstsq(X0_tr, y_tr, rcond=None)
    b1, _, _, _ = np.linalg.lstsq(X1_tr, y_tr, rcond=None)

    X0_te = np.column_stack([np.ones(len(te)), te["loss_ratio"].to_numpy(dtype=float)])
    X1_te = np.column_stack([np.ones(len(te)), te["loss_ratio"].to_numpy(dtype=float), te[var_name].to_numpy(dtype=float)])
    y_te = te[target_col].to_numpy(dtype=float)

    tss_te = float(np.sum((y_te - y_te.mean()) ** 2))
    if tss_te <= 0:
        return {"r2_oos_loss_only": 0.0, "r2_oos_full": 0.0, "delta_r2_oos": 0.0}

    rss0_te = float(np.sum((y_te - X0_te @ b0) ** 2))
    rss1_te = float(np.sum((y_te - X1_te @ b1) ** 2))
    r2_0 = 1.0 - rss0_te / tss_te
    r2_1 = 1.0 - rss1_te / tss_te
    return {
        "r2_oos_loss_only": float(r2_0),
        "r2_oos_full": float(r2_1),
        "delta_r2_oos": float(r2_1 - r2_0),
    }


def run_g3_followup_analysis(
    in_path: str = "experiments/results/baseline/jena",
    out_dir: str = "experiments/analysis/phase3_real",
    W: int = 3,
    k: int = 3,
) -> dict:
    """Thuc hien toan bo Phan 4.1 (a, b, c), 4.2 (xuat 3 file predictive_coef*.json)
    va 4.3 (viet g3_followup.md theo nguong dat truoc)."""
    run_dirs = discover_run_dirs(in_path)
    if not run_dirs:
        raise FileNotFoundError(f"Khong tim thay epoch_pattern_log.csv trong '{in_path}'")
    os.makedirs(out_dir, exist_ok=True)

    logs = []
    for rd in run_dirs:
        meta = _infer_run_metadata(rd)
        df_l = pd.read_csv(os.path.join(rd, "epoch_pattern_log.csv"))
        df_l["model"] = meta["model"]
        df_l["seed"] = meta["seed"]
        logs.append(df_l)
    combined_log = pd.concat(logs, ignore_index=True)

    df_raw = build_tracker_aligned_table(combined_log, W=W, k=k, smooth="raw")
    df_ma3 = build_tracker_aligned_table(combined_log, W=W, k=k, smooth="ma3")
    df_ema = build_tracker_aligned_table(combined_log, W=W, k=k, smooth="ema")

    vars_check = ["instability", "loss_trend", "velocity"]
    models = sorted(df_raw["model"].unique())

    # (a) Truoc va sau kiem soat epoch tren tung backbone va pooled
    rows_a = []
    for scope in models + ["pooled"]:
        sub_df = df_raw if scope == "pooled" else df_raw[df_raw["model"] == scope]
        res_no = compute_predictive_test(sub_df, k=k, variables=vars_check, controls_epoch=False).set_index("variable")
        res_fe = compute_predictive_test(sub_df, k=k, variables=vars_check, controls_epoch=True).set_index("variable")
        for v in vars_check:
            rows_a.append({
                "scope": scope,
                "variable": v,
                "delta_r2_before_fe": float(res_no.loc[v, "delta_r2"]),
                "beta_before_fe": float(res_no.loc[v, "beta_variable"]),
                "p_before_fe": float(res_no.loc[v, "p_value"]),
                "delta_r2_after_fe": float(res_fe.loc[v, "delta_r2"]),
                "beta_after_fe": float(res_fe.loc[v, "beta_variable"]),
                "p_after_fe": float(res_fe.loc[v, "p_value"]),
            })
    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(os.path.join(out_dir, "g3_check_a_epoch_control.csv"), index=False)

    # (b-i) Loc nhieu (ma3 va ema) co kiem soat epoch
    rows_b_smooth = []
    for sm_name, sm_df in [("raw", df_raw), ("ma3", df_ma3), ("ema", df_ema)]:
        for scope in models + ["pooled"]:
            sub_df = sm_df if scope == "pooled" else sm_df[sm_df["model"] == scope]
            res_fe = compute_predictive_test(sub_df, k=k, variables=vars_check, controls_epoch=True).set_index("variable")
            for v in vars_check:
                rows_b_smooth.append({
                    "smoothing": sm_name,
                    "scope": scope,
                    "variable": v,
                    "delta_r2_after_fe": float(res_fe.loc[v, "delta_r2"]),
                    "beta_after_fe": float(res_fe.loc[v, "beta_variable"]),
                    "p_after_fe": float(res_fe.loc[v, "p_value"]),
                })
    df_b_smooth = pd.DataFrame(rows_b_smooth)
    df_b_smooth.to_csv(os.path.join(out_dir, "g3_check_b_smoothing.csv"), index=False)

    # (b-ii) Cross-seed out-of-sample R2 (0-2 -> 3-4 va 3-4 -> 0-2)
    rows_b_oos = []
    for scope in models + ["pooled"]:
        sub_df = df_raw if scope == "pooled" else df_raw[df_raw["model"] == scope]
        for v in vars_check:
            fwd = _eval_cross_seed_oos(sub_df, v, [0, 1, 2], [3, 4], k=k)
            bwd = _eval_cross_seed_oos(sub_df, v, [3, 4], [0, 1, 2], k=k)
            rows_b_oos.append({
                "scope": scope,
                "variable": v,
                "r2_oos_012_to_34_loss": fwd["r2_oos_loss_only"],
                "r2_oos_012_to_34_full": fwd["r2_oos_full"],
                "delta_r2_oos_012_to_34": fwd["delta_r2_oos"],
                "r2_oos_34_to_012_loss": bwd["r2_oos_loss_only"],
                "r2_oos_34_to_012_full": bwd["r2_oos_full"],
                "delta_r2_oos_34_to_012": bwd["delta_r2_oos"],
            })
    df_b_oos = pd.DataFrame(rows_b_oos)
    df_b_oos.to_csv(os.path.join(out_dir, "g3_check_b_cross_seed_oos.csv"), index=False)

    # (c) Kiem quay ve trung binh truc tiep (dev_ma3 = loss_ratio - ma3)
    target_col = f"future_delta_loss_k{k}"
    rows_c = []
    for scope in models + ["pooled"]:
        sub_df = df_raw if scope == "pooled" else df_raw[df_raw["model"] == scope]
        res_dev = compute_predictive_test(sub_df, k=k, variables=["dev_ma3"], controls_epoch=True).iloc[0]
        # Kiem tra phan du Delta R2 cua instability/loss_trend/velocity SAU KHI da dua dev_ma3 vao mo hinh co so
        for v in vars_check:
            sub_c = sub_df[["epoch", "loss_ratio", "dev_ma3", v, target_col]].dropna().copy()
            for c_dem in ("loss_ratio", "dev_ma3", v, target_col):
                sub_c[c_dem] = sub_c[c_dem] - sub_c.groupby("epoch")[c_dem].transform("mean")
            X_base_dev = np.column_stack([sub_c["loss_ratio"].to_numpy(dtype=float), sub_c["dev_ma3"].to_numpy(dtype=float)])
            X_full_dev = np.column_stack([X_base_dev, sub_c[v].to_numpy(dtype=float)])
            y_c = sub_c[target_col].to_numpy(dtype=float)
            r2_bd, _, _ = _ols_r2_and_rss(X_base_dev, y_c)
            r2_fd, _, b_fd = _ols_r2_and_rss(X_full_dev, y_c)
            rows_c.append({
                "scope": scope,
                "beta_dev_ma3": float(res_dev["beta_variable"]),
                "delta_r2_dev_ma3": float(res_dev["delta_r2"]),
                "variable": v,
                "delta_r2_given_dev_ma3": max(0.0, float(r2_fd - r2_bd)),
                "beta_given_dev_ma3": float(b_fd[3]),
            })
    df_c = pd.DataFrame(rows_c)
    df_c.to_csv(os.path.join(out_dir, "g3_check_c_mean_reversion.csv"), index=False)

    # Phan 4.2: Xuat 3 ban predictive_coef*.json + ban mac dinh predictive_coef.json
    coef_pooled = fit_predictive_coef_dict(df_raw, features=("loss_ratio", "instability"), k=k, fit_scope="pooled", fit_seeds=(0, 1, 2, 3, 4), controls_epoch=True)
    coef_gru = fit_predictive_coef_dict(df_raw, features=("loss_ratio", "instability"), k=k, fit_scope="gru", fit_seeds=(0, 1, 2, 3, 4), controls_epoch=True)
    coef_gru34 = fit_predictive_coef_dict(df_raw, features=("loss_ratio", "instability"), k=k, fit_scope="gru", fit_seeds=(3, 4), controls_epoch=True)

    for fname, cdict in [
        ("predictive_coef_pooled.json", coef_pooled),
        ("predictive_coef_gru.json", coef_gru),
        ("predictive_coef_gru_seeds34.json", coef_gru34),
        ("predictive_coef.json", coef_gru34),
    ]:
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            json.dump(cdict, f, indent=2)

    # Phan 4.3: Danh gia theo nguong dat truoc cho instability (prediction_stability_ps)
    sub_instab = df_a[(df_a["variable"] == "instability") & (df_a["scope"].isin(models))]
    n_pass_dr2 = int((sub_instab["delta_r2_after_fe"] >= 0.02).sum())
    signs = np.sign(sub_instab["beta_after_fe"].to_numpy(dtype=float))
    same_sign = bool(len(signs) == 4 and (np.all(signs > 0) or np.all(signs < 0)))
    keep_learning_state = bool(n_pass_dr2 >= 3 and same_sign)
    verdict_str = (
        "GIỮ Learning State {loss_ratio, instability} (đạt ngưỡng đặt trước)"
        if keep_learning_state
        else "LUI VỀ PHƯƠNG ÁN DỰ PHÒNG (không đạt ngưỡng ΔR² ≥ 0.02 ở ≥ 3/4 backbone và cùng dấu ở cả 4)"
    )

    md_lines = [
        "# Báo cáo G3 Follow-up: Kiểm soát Epoch, Lọc Nhiễu & Kiểm định Mean-Reversion (Prompt 7 — Phần 4)",
        "",
        "## 1. Kết luận theo ngưỡng đặt trước (Pre-registered Decision Rule)",
        "",
        "- **Ngưỡng đặt trước (Prompt 7 Phần 4.3):** *Learning State được giữ nếu $\\Delta R^2$ sau kiểm soát epoch $\\ge 0.02$ ở ít nhất 3/4 backbone và dấu $\\beta$ cùng chiều ở cả 4. Nếu không, báo \"lui về phương án dự phòng\".*",
        f"- **Kết quả của `instability` (`prediction_stability_ps`):** Số backbone có $\\Delta R^2 \\ge 0.02$ sau kiểm soát epoch = **{n_pass_dr2}/4** | Dấu $\\beta$ cùng chiều ở cả 4 backbone = **{same_sign}**.",
        f"- **Quyết định chính thức:** **{verdict_str}**.",
        "",
        "## 2. Kiểm tra (a) — Trước và Sau khi Kiểm soát Hiệu ứng Cố định theo Epoch (`controls_epoch`)",
        "",
        _df_to_markdown(df_a),
        "",
        "## 3. Kiểm tra (b) — Lọc nhiễu (`ma3`, `ema`) và Tái lập ngoài mẫu giữa các Seed (`0-2 ↔ 3-4`)",
        "",
        "### 3.1. Trên đường cong loss đã làm mượt (có kiểm soát epoch)",
        "",
        _df_to_markdown(df_b_smooth[df_b_smooth["scope"].isin(["gru", "pooled"])]),
        "",
        "### 3.2. $R^2$ ngoài mẫu chéo giữa các Seed (`Seed 0-2 -> 3-4` và `Seed 3-4 -> 0-2`)",
        "",
        _df_to_markdown(df_b_oos),
        "",
        "## 4. Kiểm tra (c) — Kiểm định trực tiếp hiện tượng \"Quay về trung bình\" (`dev_ma3 = loss_ratio - ma3`)",
        "",
        _df_to_markdown(df_c),
        "",
        "## 5. Hệ số xuất ra trong `predictive_coef.json` (Bản GRU học trên Seed 3–4, dùng cho Seed 0–2)",
        "",
        "```json",
        json.dumps(coef_gru34, indent=2),
        "```",
    ]
    g3_path = os.path.join(out_dir, "g3_followup.md")
    with open(g3_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {
        "df_a": df_a,
        "df_b_smooth": df_b_smooth,
        "df_b_oos": df_b_oos,
        "df_c": df_c,
        "coef_pooled": coef_pooled,
        "coef_gru": coef_gru,
        "coef_gru34": coef_gru34,
        "keep_learning_state": keep_learning_state,
        "verdict": verdict_str,
        "report_path": g3_path,
    }
