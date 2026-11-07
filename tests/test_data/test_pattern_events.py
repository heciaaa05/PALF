import numpy as np
import pandas as pd
import pytest

from palf.data.pattern_events import (
    count_mask_runs,
    label_run_stats_by_split,
    mask_run_stats,
    split_mask_runs,
)


def test_empty_case():
    """Truong hop rong: mang rong, danh sach rong, hoac toan False."""
    assert len(count_mask_runs([])) == 0
    assert len(count_mask_runs(np.array([], dtype=bool))) == 0
    assert len(count_mask_runs([False, False, False])) == 0

    st = mask_run_stats([])
    assert st["n_windows"] == 0
    assert st["n_runs"] == 0
    assert np.isnan(st["mean_run_length"])
    assert st["max_run_length"] == 0


def test_single_run():
    """Mot run: mot phan tu True, nhieu phan tu True lien tiep, giua cac False."""
    # 1 phan tu duy nhat
    lens1 = count_mask_runs([True])
    assert list(lens1) == [1]

    # Nhieu phan tu lien tiep
    lens2 = count_mask_runs([True, True, True, True])
    assert list(lens2) == [4]

    # Mot run o giua cac False
    lens3 = count_mask_runs([False, False, True, True, True, False])
    assert list(lens3) == [3]

    st = mask_run_stats([False, True, True, True, False])
    assert st["n_windows"] == 3
    assert st["n_runs"] == 1
    assert st["mean_run_length"] == pytest.approx(3.0)
    assert st["max_run_length"] == 3


def test_run_touching_boundary():
    """Run dinh bien: bat dau tu dau mang (index 0), cham cuoi mang (index n-1), hoac ca hai."""
    # Run bat dau tu dau mang
    lens_start = count_mask_runs([True, True, True, False, False])
    assert list(lens_start) == [3]

    # Run cham cuoi mang
    lens_end = count_mask_runs([False, False, True, True])
    assert list(lens_end) == [2]

    # Run o ca hai bien (dau va cuoi) ngan cach boi False
    lens_both = count_mask_runs([True, True, False, False, True])
    assert list(lens_both) == [2, 1]

    # Toan bo mang la True (cham ca dau lan cuoi)
    lens_all = count_mask_runs([True, True, True])
    assert list(lens_all) == [3]


def test_run_boundary_isolation_across_splits():
    """Kiem tra run khong bi tran (leak) hoac noi gop qua ranh gioi giua cac tap va purge gap."""
    # Gia su mang co train: [0, 4), purge gap: [4, 6), val: [6, 10)
    # Mask co True o cuoi train (index 2, 3), trong gap (index 4, 5) va o dau val (index 6, 7).
    mask = np.array([
        False, False, True, True,  # train: run dai 2 (cham bien cuoi train)
        True, True,                # purge gap: khong duoc tinh vao train hay val
        True, True, False, False,  # val: run dai 2 (cham bien dau val)
    ])
    spans = {
        "train": (0, 4),
        "val": (6, 10),
    }

    df_runs = split_mask_runs(mask, spans)
    assert len(df_runs) == 2

    train_row = df_runs[df_runs["split"] == "train"].iloc[0]
    assert train_row["n_windows"] == 2
    assert train_row["n_runs"] == 1
    assert train_row["mean_run_length"] == pytest.approx(2.0)
    assert train_row["max_run_length"] == 2

    val_row = df_runs[df_runs["split"] == "val"].iloc[0]
    assert val_row["n_windows"] == 2
    assert val_row["n_runs"] == 1
    assert val_row["mean_run_length"] == pytest.approx(2.0)
    assert val_row["max_run_length"] == 2


def test_label_run_stats_by_split():
    """Kiem tra thong ke nhan theo tap."""
    labels = pd.Series(["spike", "spike", "normal", "drop", "drop", "normal", "spike"])
    spans = {"train": (0, 3), "val": (3, 7)}
    df = label_run_stats_by_split(labels, spans, ["spike", "drop", "normal"])

    # Train: spike=2 (1 run len 2), drop=0, normal=1 (1 run len 1)
    tr_spike = df[(df["split"] == "train") & (df["label"] == "spike")].iloc[0]
    assert tr_spike["n_windows"] == 2
    assert tr_spike["n_runs"] == 1
    assert tr_spike["mean_run_length"] == pytest.approx(2.0)

    # Val: drop=2 (1 run len 2), normal=1, spike=1 (1 run len 1)
    va_spike = df[(df["split"] == "val") & (df["label"] == "spike")].iloc[0]
    assert va_spike["n_windows"] == 1
    assert va_spike["n_runs"] == 1
