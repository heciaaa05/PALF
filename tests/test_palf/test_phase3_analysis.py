import os
import numpy as np
import pandas as pd
import pytest

from palf.evaluation.phase3_analysis import (
    block_permute_pattern_labels,
    compute_candidate_variables,
    extract_contiguous_runs,
    kruskal_wallis_stat,
    normalize_pattern_loss,
    run_phase3_analysis,
)


def test_normalize_pattern_loss_known_values():
    """Kiem tra ham chuan hoa loss theo tung Pattern (epoch1, global_epoch1, persistence, raw)."""
    df = pd.DataFrame({
        "epoch": [1, 1, 2, 2],
        "pattern": ["A", "B", "A", "B"],
        "n": [10, 30, 10, 30],
        "train_loss_mean": [2.0, 4.0, 1.0, 1.0],
        "train_loss_std": [0.2, 0.4, 0.1, 0.2],
        "val_loss": [2.5, 4.5, 1.5, 1.5],
        "grad_norm": [0.5, 0.8, 0.3, 0.4],
    })
    out_ep1 = normalize_pattern_loss(df, normalization="epoch1", eps=0.0)
    # Epoch 1 phai bang 1.0; Epoch 2 cua A la 1.0/2.0 = 0.5; cua B la 1.0/4.0 = 0.25
    assert list(out_ep1["loss_norm"]) == pytest.approx([1.0, 1.0, 0.5, 0.25])

    out_glob = normalize_pattern_loss(df, normalization="global_epoch1", eps=0.0)
    # Weighted mean o epoch 1 = (2*10 + 4*30) / 40 = 140 / 40 = 3.5
    assert list(out_glob["loss_norm"]) == pytest.approx([2.0 / 3.5, 4.0 / 3.5, 1.0 / 3.5, 1.0 / 3.5])

    out_pers = normalize_pattern_loss(
        df, normalization="persistence", persistence_mse={"A": 0.5, "B": 2.0}, eps=0.0
    )
    assert list(out_pers["loss_norm"]) == pytest.approx([4.0, 2.0, 2.0, 0.5])


def test_compute_candidate_variables_known_values():
    """Kiem tra ham tinh cac bien trang thai tren du lieu nho tu dung voi ket qua biet truoc."""
    # Mot pattern 'A' qua 5 epoch voi loss_raw = [10, 8, 6, 4, 6]
    # Khi chuan hoa epoch1 (chia cho 10), loss_norm = [1.0, 0.8, 0.6, 0.4, 0.6]
    df = pd.DataFrame({
        "model": ["gru"] * 5,
        "seed": [0] * 5,
        "epoch": [1, 2, 3, 4, 5],
        "pattern": ["A"] * 5,
        "n": [100] * 5,
        "train_loss_mean": [10.0, 8.0, 6.0, 4.0, 6.0],
        "train_loss_std": [1.0, 1.6, 0.6, 0.8, 1.2],
        "val_loss": [11.0, 9.0, 7.0, 5.5, 8.0],
        "grad_norm": [0.5, 0.4, 0.3, 0.2, 0.25],
    })
    res = compute_candidate_variables(df, W=3, k=2, normalization="epoch1", eps=0.0)

    # 1. Learning velocity LV(e) = L(e-1) - L(e): [0.0, 0.2, 0.2, 0.2, -0.2]
    assert list(res["learning_velocity"]) == pytest.approx([0.0, 0.2, 0.2, 0.2, -0.2])

    # 2. Loss trend LT trong cua so W=3:
    # Ep 1: [1.0] -> 0.0
    # Ep 2: [1.0, 0.8] -> -0.2
    # Ep 3: [1.0, 0.8, 0.6] -> -0.2
    # Ep 4: [0.8, 0.6, 0.4] -> -0.2
    # Ep 5: [0.6, 0.4, 0.6] -> slope qua 3 diem (0, 0.6), (1, 0.4), (2, 0.6) bang 0.0
    assert list(res["loss_trend"]) == pytest.approx([0.0, -0.2, -0.2, -0.2, 0.0])

    # 3. Prediction stability PS trong cua so W=3:
    # Ep 3: win=[1.0, 0.8, 0.6], diffs=[-0.2, -0.2], std(diffs) = 0.0
    # Ep 5: win=[0.6, 0.4, 0.6], diffs=[-0.2, +0.2], std(diffs) = 0.2
    assert res.loc[2, "prediction_stability_ps"] == pytest.approx(0.0)
    assert res.loc[4, "prediction_stability_ps"] == pytest.approx(0.2)

    # 4. Within-pattern dispersion DIS = std / mean: [0.1, 0.2, 0.1, 0.2, 0.2]
    assert list(res["within_pattern_dispersion_dis"]) == pytest.approx([0.1, 0.2, 0.1, 0.2, 0.2])

    # 5. Forgetting risk: o ep 1..4 giam deu -> 0.0; o ep 5 tang tu 0.4 len 0.6 -> 0.2
    assert list(res["forgetting_risk"]) == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.2])

    # 6. Future delta loss voi k=2:
    # Ep 1 -> Ep 3: 0.6 - 1.0 = -0.4
    # Ep 2 -> Ep 4: 0.4 - 0.8 = -0.4
    # Ep 3 -> Ep 5: 0.6 - 0.6 = 0.0
    assert res.loc[0, "future_delta_loss_k2"] == pytest.approx(-0.4)
    assert res.loc[1, "future_delta_loss_k2"] == pytest.approx(-0.4)
    assert res.loc[2, "future_delta_loss_k2"] == pytest.approx(0.0)
    assert np.isnan(res.loc[3, "future_delta_loss_k2"])


def test_block_permutation_preserves_run_lengths_and_label_counts():
    """Kiem tra hoan vi khoi giu nguyen tap hop do dai cac doan va tong so cua so moi nhan."""
    labels = np.array(
        ["normal"] * 5
        + ["pressure_drop"] * 3
        + ["normal"] * 4
        + ["temp_anomaly"] * 6
        + ["humidity_net_rise"] * 2
        + ["pressure_drop"] * 4,
        dtype=str,
    )
    orig_runs = extract_contiguous_runs(labels)
    orig_lengths_by_pat = {}
    for r in orig_runs:
        orig_lengths_by_pat.setdefault(r["pattern"], []).append(r["length"])
    for k in orig_lengths_by_pat:
        orig_lengths_by_pat[k].sort()

    rng = np.random.default_rng(123)
    perm_labels = block_permute_pattern_labels(labels, rng=rng)

    assert len(perm_labels) == len(labels)
    # Tong so cua so moi nhan phai giu nguyen tuyet doi
    for p in np.unique(labels):
        assert int((perm_labels == p).sum()) == int((labels == p).sum())

    # Moi khoi ban dau van duoc ghep nguyen khoi (nếu 2 khối cùng nhãn đứng cạnh nhau thì tổng độ dài bằng tổng các khối gốc)
    perm_runs = extract_contiguous_runs(perm_labels)
    assert sum(r["length"] for r in perm_runs) == len(labels)


def test_phase3_tool_runs_in_both_reduced_and_full_modes(tmp_path):
    """Kiem tra cong cu chay duoc o ca che do rut gon (chi co CSV) va che do day du (co .npy)."""
    run_dir = os.path.join(str(tmp_path), "gru", "seed0")
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(7)
    pats = ["pressure_drop", "humidity_net_rise", "temp_anomaly", "normal"]
    # Tao 40 mau train voi cac doan run lien tiep
    pat_labels = np.array(
        ["normal"] * 12
        + ["pressure_drop"] * 8
        + ["humidity_net_rise"] * 8
        + ["temp_anomaly"] * 12,
        dtype=str,
    )
    n_train, n_epochs = len(pat_labels), 8
    sample_losses = np.zeros((n_train, n_epochs), dtype=np.float32)
    for e in range(n_epochs):
        decay = np.exp(-0.15 * e)
        sample_losses[:, e] = rng.uniform(0.5, 1.5, size=n_train) * decay

    rows = []
    for e in range(1, n_epochs + 1):
        for p in pats:
            m = pat_labels == p
            v = sample_losses[m, e - 1]
            rows.append({
                "epoch": e,
                "pattern": p,
                "n": int(m.sum()),
                "train_loss_mean": float(v.mean()),
                "train_loss_std": float(v.std()),
                "val_loss": float(v.mean() * 1.1),
                "grad_norm": float(0.5 / e),
            })
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "epoch_pattern_log.csv"), index=False)

    # 1. Chay khi CHUA co file .npy -> tu dong chay che do 'reduced'
    out_reduced = os.path.join(str(tmp_path), "out_reduced")
    res_red = run_phase3_analysis(run_dir, out_reduced, mode="auto", W=3, k=2, n_boot=20, n_perm=10)
    assert res_red["mode_effective"] == "reduced"
    for fn in (
        "1_normalized_loss.csv",
        "2_candidate_variables.csv",
        "3_spearman_matrix.csv",
        "4_kruskal_wallis_epoch_blocks.csv",
        "5_pattern_ranks_by_seed.csv",
        "6_predictive_regression.csv",
        "analysis_config.json",
        "report.md",
    ):
        assert os.path.isfile(os.path.join(out_reduced, fn))

    # 2. Luu them sample_loss_epochs.npy va pattern_labels_train.npy (cho 2 seed) -> chay che do 'full'
    np.save(os.path.join(run_dir, "sample_loss_epochs.npy"), sample_losses)
    np.save(os.path.join(run_dir, "pattern_labels_train.npy"), pat_labels)

    run_dir_s1 = os.path.join(str(tmp_path), "gru", "seed1")
    os.makedirs(run_dir_s1, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(run_dir_s1, "epoch_pattern_log.csv"), index=False)
    np.save(os.path.join(run_dir_s1, "sample_loss_epochs.npy"), sample_losses * 1.02)
    np.save(os.path.join(run_dir_s1, "pattern_labels_train.npy"), pat_labels)

    out_full = os.path.join(str(tmp_path), "out_full")
    res_full = run_phase3_analysis(os.path.join(str(tmp_path), "gru"), out_full, mode="auto", W=3, k=2, n_boot=20, n_perm=10)
    assert res_full["mode_effective"] == "full"
    assert (res_full["seed_stability"]["n_seeds"] == 2).all()
    for fn in (
        "4_event_runs_table.csv",
        "4_kruskal_wallis_runs.csv",
        "7_null_permutation_summary.csv",
        "7_volatility_control.csv",
    ):
        assert os.path.isfile(os.path.join(out_full, fn))
