"""CLI script chay phan tich Phase 3 (Learning State Investigation) cho 1 thu muc log
hoac gop nhieu backbone/seed.

Vi du:
1. Chay cho 1 thu muc log (hoac 1 backbone gom nhieu seed):
   python experiments/scripts/analyze_phase3.py --in-dir experiments/results/baseline/jena/gru --out-dir experiments/results/phase3/jena_gru

2. Chay gop cho nhieu backbone:
   python experiments/scripts/analyze_phase3.py --in-dirs experiments/results/GRU experiments/results/Transformer --out-dir experiments/results/phase3/synthetic_combined
"""
import argparse
import os
import pandas as pd

from palf.evaluation.phase3_analysis import (
    discover_run_dirs,
    run_g3_followup_analysis,
    run_phase3_analysis,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Cong cu phan tich Phase 3 (Learning State Investigation) - ho tro reduced & full mode."
    )
    parser.add_argument(
        "--in-dir", "--results-root", dest="in_dir", type=str, default="experiments/results/baseline/jena",
        help="Thu muc log dau vao (co the la 1 run, 1 backbone co seed0..k, hoac thu muc goc chua nhieu backbone)."
    )
    parser.add_argument(
        "--g3-followup", action="store_true",
        help="Chay kiem tra G3 Follow-up (kiem soat epoch, loc nhieu, mean-reversion) va xuat predictive_coef.json + g3_followup.md."
    )
    parser.add_argument(
        "--in-dirs", nargs="+", default=None,
        help="Danh sach nhieu thu muc log dau vao de chay phan tich tung thu muc + chay gop."
    )
    parser.add_argument(
        "--out-dir", type=str, default="experiments/results/phase3/jena_gru",
        help="Thu muc luu ket qua phan tich Phase 3."
    )
    parser.add_argument(
        "--mode", type=str, default="auto", choices=["auto", "reduced", "full"],
        help="Che do phan tich: auto (tu phat hien), reduced (chi dung CSV), full (dung sample_loss_epochs.npy)."
    )
    parser.add_argument(
        "--normalization", type=str, default="epoch1",
        choices=["epoch1", "global_epoch1", "persistence", "raw"],
        help="Phuong phap chuan hoa loss theo Pattern (mac dinh: epoch1)."
    )
    parser.add_argument("--W", type=int, default=5, help="Do dai cua so epoch W cho LT va PS (mac dinh: 5).")
    parser.add_argument("--k", type=int, default=3, help="So epoch tuong lai k cho phep thu du doan (mac dinh: 3).")
    parser.add_argument("--n-boot", type=int, default=200, help="So vong lap bootstrap (mac dinh: 200).")
    parser.add_argument("--n-perm", type=int, default=100, help="So lan hoan vi khoi cho nhom doi chung (mac dinh: 100).")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngau nhien (mac dinh: 42).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.g3_followup:
        res_g3 = run_g3_followup_analysis(
            in_path=args.in_dir,
            out_dir=args.out_dir,
            W=3,
            k=args.k,
        )
        print(
            f"[G3 Follow-up] DONE | verdict='{res_g3['verdict']}' | report={res_g3['report_path']}",
            flush=True,
        )
        return res_g3

    targets = args.in_dirs if args.in_dirs is not None else [args.in_dir]

    if len(targets) == 1:
        in_path = targets[0]
        res = run_phase3_analysis(
            in_path=in_path,
            out_dir=args.out_dir,
            mode=args.mode,
            normalization=args.normalization,
            W=args.W,
            k=args.k,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        print(
            f"[Phase 3 Analysis] DONE | in_dir={in_path} | "
            f"mode_effective={res['mode_effective'].upper()} | "
            f"n_runs={res['config']['n_run_dirs']} | out_dir={args.out_dir}",
            flush=True,
        )
        return res

    # Truong hop chay gop nhieu thu muc / nhieu backbone
    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []
    for t_dir in targets:
        name = os.path.basename(os.path.normpath(t_dir)).lower()
        sub_out = os.path.join(args.out_dir, name)
        res = run_phase3_analysis(
            in_path=t_dir,
            out_dir=sub_out,
            mode=args.mode,
            normalization=args.normalization,
            W=args.W,
            k=args.k,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        print(
            f"[Phase 3 Analysis] Sub-target '{name}' DONE | "
            f"mode_effective={res['mode_effective'].upper()} | "
            f"n_runs={res['config']['n_run_dirs']} | out_dir={sub_out}",
            flush=True,
        )
        summary_rows.append({
            "target": name,
            "in_path": os.path.abspath(t_dir),
            "mode_effective": res["mode_effective"],
            "n_runs": res["config"]["n_run_dirs"],
            "n_redundant_pairs": int(res["redundant_pairs"]["is_redundant_gt_0p8"].sum())
            if not res["redundant_pairs"].empty else 0,
            "out_dir": os.path.abspath(sub_out),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.out_dir, "multi_backbone_summary.csv"), index=False)
    print(f"[Phase 3 Analysis] Multi-backbone summary saved to {args.out_dir}", flush=True)
    return summary_df


if __name__ == "__main__":
    main()
