"""Script chay quan sat / baseline va PALF tren du lieu Jena Climate (hoac synthetic).

Quy uoc dau ra:
  - Baseline:
      experiments/results/baseline/jena/<model>/seed<k>/
  - PALF:
      experiments/results/palf/jena/<config_tag>/<model>/seed<k>/
"""
import argparse
import dataclasses
import json
import os
import sys
from torch.utils.data import TensorDataset

from palf.core.config import PalfConfig
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
        description="Chay baseline hoac PALF va ghi log quan sat Learning Dynamics."
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
        help="Che do huan luyen: 'baseline' hoac 'palf' (mac dinh: baseline)."
    )
    # Cac co PALF (Phase 5-6 v1)
    parser.add_argument(
        "--policy", type=str, default="predictive",
        choices=["uniform", "loss", "loss_abs", "state", "predictive"],
        help="Chinh sach PALF: uniform, loss, loss_abs, state, predictive (mac dinh: predictive)."
    )
    parser.add_argument(
        "--predictive-coef", dest="predictive_coef_path", type=str,
        default="experiments/analysis/phase3_real/predictive_coef.json",
        help="Duong dan file JSON he so hoi quy cho policy='predictive'."
    )
    parser.add_argument(
        "--budget-ratio", dest="budget_ratio", type=float, default=1.0,
        help="Ty le ngan sach mau moi epoch so voi n_train (mac dinh: 1.0)."
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="Do manh phan ung trong so Policy (mac dinh: 1.0)."
    )
    parser.add_argument(
        "--w-min", dest="w_min", type=float, default=0.25,
        help="Trong so san w_min (mac dinh: 0.25)."
    )
    parser.add_argument(
        "--w-max", dest="w_max", type=float, default=3.0,
        help="Trong so tran w_max (mac dinh: 3.0)."
    )
    parser.add_argument(
        "--ema-beta", dest="ema_beta", type=float, default=0.5,
        help="He so lam muot EMA theo epoch (mac dinh: 0.5)."
    )
    parser.add_argument(
        "--warmup", "--warmup-epochs", dest="warmup", type=int, default=3,
        help="So epoch khoi dong (warm-up) giu w=1 (mac dinh: 3)."
    )
    parser.add_argument(
        "--update-every", dest="update_every", type=int, default=1,
        help="Cap nhat trong so Policy moi bao nhieu epoch (mac dinh: 1)."
    )
    parser.add_argument(
        "--window", type=int, default=3,
        help="Cua so epoch tinh velocity / instability (mac dinh: 3)."
    )
    parser.add_argument(
        "--config-tag", dest="config_tag", type=str, default=None,
        help="Tag thu muc cau hinh PALF (mac dinh: tu dong sinh tu PalfConfig, vd: state_b1.0_a1.0)."
    )
    parser.add_argument(
        "--shuffle-weights", dest="shuffle_weights", action="store_true", default=False,
        help="Prompt 10 (doi chung A0w): hoan vi ngau nhien trong so tinh duoc giua cac Pattern moi epoch "
             "(truoc EMA/rao chan/Scheduler), giu nguyen phan bo trong so nhung mat thong tin Pattern."
    )
    parser.add_argument(
        "--shuffle-seed", dest="shuffle_seed", type=int, default=None,
        help="Seed RNG rieng cho hoan vi trong so (mac dinh: 100000 + seed huan luyen; doc lap voi RNG huan luyen)."
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
        "--out-root", type=str, default=None,
        help="Thu muc goc luu ket qua (mac dinh: experiments/results/baseline/jena hoac experiments/results/palf/jena)."
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Thu muc dau ra cu the (neu khong truyen se dung quy uoc theo mode)."
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


def build_palf_config(args) -> PalfConfig:
    return PalfConfig(
        policy=args.policy,
        warmup_epochs=args.warmup,
        update_every=args.update_every,
        alpha=args.alpha,
        w_min=args.w_min,
        w_max=args.w_max,
        ema_beta=args.ema_beta,
        budget_ratio=args.budget_ratio,
        window=args.window,
        predictive_coef_path=args.predictive_coef_path,
        shuffle_weights=bool(getattr(args, "shuffle_weights", False)),
        shuffle_seed=getattr(args, "shuffle_seed", None),
    )


def run_from_args(args):
    canonical_model = resolve_model_name(args.model)
    model_lower = canonical_model.lower()

    palf_cfg = build_palf_config(args) if args.mode == "palf" else None
    config_tag = (
        args.config_tag
        if (args.mode == "palf" and args.config_tag)
        else (palf_cfg.config_tag() if palf_cfg is not None else "baseline")
    )

    if args.out_root is not None:
        out_root = args.out_root
    elif args.subset is not None:
        sub_mode = "palf" if args.mode == "palf" else "baseline"
        out_root = os.path.join("experiments", "results", "smoke", sub_mode, "jena")
    elif args.mode == "palf":
        out_root = os.path.join("experiments", "results", "palf", "jena")
    else:
        out_root = os.path.join("experiments", "results", "baseline", "jena")

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
        elif args.mode == "palf":
            run_out_dir = os.path.join(out_root, config_tag, model_lower, f"seed{seed}")
        else:
            run_out_dir = os.path.join(out_root, model_lower, f"seed{seed}")

        mode_label = f"PALF ({config_tag})" if args.mode == "palf" else "Baseline"
        print(
            f"[{mode_label}] model={canonical_model} ({model_lower}) | seed={seed} | "
            f"epochs={args.epochs} | batch={args.batch} | lr={args.lr} | "
            f"device={args.device} | out_dir={run_out_dir}",
            flush=True,
        )

        def _on_epoch(row, total_ep):
            ov_str = f" (ov={row.get('overhead_sec', 0.0):.3f}s)" if args.mode == "palf" else ""
            print(
                f"  Epoch {row['epoch']:02d}/{total_ep:02d} | "
                f"train_loss={row['train_loss']:.4f} | val_loss={row['val_loss']:.4f} | "
                f"n={row['n_samples_processed']} | "
                f"epoch_time={row['epoch_time_sec']:.2f}s{ov_str} | elapsed={row['elapsed']:.1f}s",
                flush=True,
            )

        run_palf_cfg = palf_cfg
        if palf_cfg is not None and palf_cfg.shuffle_weights:
            s_seed = args.shuffle_seed if args.shuffle_seed is not None else 100000 + int(seed)
            run_palf_cfg = dataclasses.replace(palf_cfg, shuffle_seed=int(s_seed))

        res = train_model(
            model_name=canonical_model,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            device_name=args.device,
            seed=seed,
            data=data_tuple,
            on_epoch=_on_epoch,
            out_root=out_root,
            out_dir=run_out_dir,
            use_seed_subdir=True,
            target_idx=target_idx,
            patience=args.patience,
            save_sample_losses=args.save_sample_losses,
            pattern_scheme=pattern_scheme,
            target_col=target_col,
            mode=args.mode,
            palf_config=run_palf_cfg,
        )
        if args.mode == "palf" and args.config_tag:
            res["run_config"]["config_tag"] = config_tag
            with open(os.path.join(run_out_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(res["run_config"], f, indent=2)

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
