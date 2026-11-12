"""Script chay quan sat / baseline tren du lieu Jena Climate (hoac synthetic).

Quy uoc dau ra (Prompt 5 - Phan A):
    experiments/results/baseline/jena/<model>/seed<k>/
        history.csv
        epoch_pattern_log.csv
        cartography.csv
        sample_loss_epochs.npy
        sample_index.npy
        pattern_labels_train.npy
        run_config.json
        test_metrics.json
"""
import argparse
import os
import sys
from torch.utils.data import TensorDataset

from palf.data.synthetic import prepare_data
from palf.data.weather import prepare_weather_data
from palf.models import resolve_model_name
from palf.training.trainer import train_model


def _slice_dataset(ds, n):
    if n is None or n >= len(ds):
        return ds
    t0, t1, t2, t3 = ds.tensors
    return TensorDataset(t0[:n], t1[:n], t2[:n], t3[:n])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Chay baseline (random sampling) va ghi log quan sat Learning Dynamics."
    )
    parser.add_argument(
        "--model", type=str, default="gru",
        help="Ten backbone: gru, tcn, transformer, patchtst (khong phan biet hoa/thuong)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed khoi tao trong so va thu tu xao tron batch (mac dinh: 0)."
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Danh sach seed cach nhau boi dau phay hoac khoang trang (vd: '0,1,2,3,4'). Neu dat se ghi de --seed."
    )
    parser.add_argument("--epochs", type=int, default=30, help="So epoch huan luyen (mac dinh: 30).")
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=128, help="Batch size (mac dinh: 128).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (mac dinh: 1e-3).")
    parser.add_argument("--device", type=str, default="cpu", help="Thiet bi: cpu, cuda, dml (mac dinh: cpu).")
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early stopping patience (mac dinh: None = khong dung som)."
    )
    parser.add_argument(
        "--mode", type=str, default="baseline", choices=["baseline", "palf"],
        help="Che do huan luyen. Chi cho phep 'baseline'; 'palf' bi chan o giai doan nay."
    )
    parser.add_argument(
        "--dataset", "--data", dest="dataset", type=str, default="jena",
        choices=["jena", "weather", "synthetic"],
        help="Nguon du lieu: jena (mac dinh) hoac synthetic."
    )
    parser.add_argument(
        "--pattern-scheme", type=str, default="weather_v1",
        help="Quy uoc nhan Pattern cho Jena (mac dinh: weather_v1)."
    )
    parser.add_argument(
        "--target-col", type=str, default="T (degC)",
        help="Cot muc tieu tren Jena (mac dinh: 'T (degC)')."
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Duong dan file CSV Jena neu co san.")
    parser.add_argument(
        "--out-root", type=str, default="experiments/results/baseline/jena",
        help="Thu muc goc luu ket qua (mac dinh: experiments/results/baseline/jena)."
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Thu muc dau ra cu the (neu khong truyen se dung <out-root>/<model>/seed<k>)."
    )
    parser.add_argument(
        "--save-sample-losses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Luu sample_loss_epochs.npy, sample_index.npy, pattern_labels_train.npy (mac dinh: True)."
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Gioi han so mau train/val/test de chay thu nhanh (mac dinh: None = dung toan bo)."
    )
    return parser.parse_args(argv)


def _parse_seed_list(args):
    if args.seeds is None:
        return [int(args.seed)]
    raw = args.seeds.replace(",", " ").split()
    return [int(s) for s in raw if s.strip()]


def run_from_args(args):
    if args.mode == "palf":
        raise RuntimeError(
            "Che do '--mode palf' bi chan o giai doan nay. Chi chay '--mode baseline'."
        )

    canonical_model = resolve_model_name(args.model)
    model_lower = canonical_model.lower()

    if args.dataset in ("jena", "weather"):
        train_ds, val_ds, test_ds, th, meta = prepare_weather_data(
            target_col=args.target_col,
            csv_path=args.csv_path,
            pattern_scheme=args.pattern_scheme,
        )
        target_idx = meta["target_idx"]
        pattern_scheme = meta["pattern_scheme"]
        target_col = meta["target_col"]
    else:
        train_ds, val_ds, test_ds, th = prepare_data()
        meta = {"pattern_scheme": "synthetic", "target_col": "y", "target_idx": 0}
        target_idx = 0
        pattern_scheme = "synthetic"
        target_col = "y"

    if args.subset is not None:
        train_ds = _slice_dataset(train_ds, args.subset)
        val_ds = _slice_dataset(val_ds, max(16, args.subset // 4))
        test_ds = _slice_dataset(test_ds, max(16, args.subset // 4))

    data_tuple = (train_ds, val_ds, test_ds, th, meta)
    seeds = _parse_seed_list(args)
    results = []

    for seed in seeds:
        if args.out_dir is not None and len(seeds) == 1:
            run_out_dir = args.out_dir
        elif args.out_dir is not None:
            run_out_dir = os.path.join(args.out_dir, f"seed{seed}")
        else:
            run_out_dir = os.path.join(args.out_root, model_lower, f"seed{seed}")

        print(
            f"[Baseline] model={canonical_model} ({model_lower}) | seed={seed} | "
            f"epochs={args.epochs} | batch={args.batch} | lr={args.lr} | "
            f"device={args.device} | out_dir={run_out_dir}",
            flush=True,
        )

        def _on_epoch(row, total_ep):
            print(
                f"  Epoch {row['epoch']:02d}/{total_ep:02d} | "
                f"train_loss={row['train_loss']:.4f} | val_loss={row['val_loss']:.4f} | "
                f"epoch_time={row['epoch_time_sec']:.2f}s | elapsed={row['elapsed']:.1f}s",
                flush=True,
            )

        res = train_model(
            model_name=canonical_model,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            device_name=args.device,
            seed=seed,
            data=data_tuple,
            on_epoch=_on_epoch,
            out_root=args.out_root,
            out_dir=run_out_dir,
            use_seed_subdir=True,
            target_idx=target_idx,
            patience=args.patience,
            save_sample_losses=args.save_sample_losses,
            pattern_scheme=pattern_scheme,
            target_col=target_col,
        )
        tm = res["test_metrics"]
        print(
            f"[Done] seed={seed} | best_epoch={res['best_epoch']} | "
            f"best_val={res['best_val']:.4f} | "
            f"test MAE={tm['mae_degC']:.4f} degC (norm {tm['mae_norm']:.4f}) | "
            f"test RMSE={tm['rmse_degC']:.4f} degC (norm {tm['rmse_norm']:.4f}) | "
            f"total_time={res['elapsed']:.1f}s",
            flush=True,
        )
        results.append(res)

    return results


def main(argv=None):
    args = parse_args(argv)
    try:
        run_from_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
