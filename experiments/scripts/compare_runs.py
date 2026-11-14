"""Script CLI so sanh cong bang Baseline vs PALF theo Giao thuc Muc 7 (Phan 4.2).

Vi du chay:
    python experiments/scripts/compare_runs.py --models gru,tcn,transformer,patchtst --seeds 0,1,2,3,4
"""
import argparse
import os
import sys

from palf.evaluation.compare_runs import run_comparison_protocol


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="So sanh cong bang Baseline (A0) va cac cau hinh PALF (A0r, A1, A2, A2r) theo Muc 7."
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default="experiments/results/baseline/jena",
        help="Thu muc goc chua ket qua baseline A0 (mac dinh: experiments/results/baseline/jena).",
    )
    parser.add_argument(
        "--palf-root",
        type=str,
        default="experiments/results/palf/jena",
        help="Thu muc goc chua ket qua PALF theo config_tag (mac dinh: experiments/results/palf/jena).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/analysis/comparison",
        help="Thu muc xuat bang CSV, hinh PNG va comparison_report.md.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gru",
        help="Danh sach backbone cach nhau boi dau phay (mac dinh: gru).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Danh sach seed cach nhau boi dau phay (mac dinh: 0,1,2 theo Prompt 7 Phan 5.2).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    models = tuple(m.strip().lower() for m in args.models.replace(",", " ").split() if m.strip())
    seeds = tuple(int(s.strip()) for s in args.seeds.replace(",", " ").split() if s.strip())

    res = run_comparison_protocol(
        baseline_root=args.baseline_root,
        palf_root=args.palf_root,
        out_dir=args.out_dir,
        models=models,
        seeds=seeds,
    )

    print(f"[CompareRuns] Tong so lan chay tim thay: {len(res['found_runs'])}", flush=True)
    print(f"[CompareRuns] Da luu bao cao tai: {res['report_path']}", flush=True)
    if not res["missing_df"].empty:
        n_miss = len(res["missing_df"])
        print(
            f"[CompareRuns] Luu y: Co {n_miss} to hop (backbone, cau hinh) con thieu hoac chua du seed "
            f"(chi tiet trong {os.path.join(args.out_dir, 'missing_configs.csv')}).",
            flush=True,
        )
    return res


if __name__ == "__main__":
    main()
