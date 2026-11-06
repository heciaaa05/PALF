import argparse
import json
import os

import numpy as np
import pandas as pd

from palf.data.weather import download_jena
from palf.data.pattern_features import (
    CHANNELS, load_hourly_with_time, window_split, build_feature_table,
)
from palf.data.pattern_labels import (
    RULE_ORDER, rule_specs, add_raw_mean, fit_thresholds, fire_matrix,
    assign_labels, overlap_counts, label_share_table, check_criteria,
)
from pattern_stats import md_table

MODES = (("A", "Cach A - dac trung phan du so voi khi hau"),
         ("B", "Cach B - dac trung tho (L boi chu ky ngay)"))


def run_mode(mode, title, feats, spans, a, out_dir):
    spec = rule_specs(mode, a.spike_channel, a.q_spike, a.q_low, a.q_high)
    th = fit_thresholds(feats, spec, spans["train"])
    fires = fire_matrix(feats, th)
    labels = assign_labels(fires, a.priority)
    tab = label_share_table(labels, spans, a.priority)
    c2, c3, c4, warn = check_criteria(
        tab, a.c2_min_share, a.c2_target_share, a.c2_min_count,
        a.c3_max_normal, a.c3_target_normal, a.c4_max_rel_dev)

    counts = overlap_counts(fires, spans["train"])
    diag = pd.Series(np.diag(counts.to_numpy()), index=counts.index).clip(lower=1)
    cond = counts.div(diag, axis=0)

    rows = []
    for name in th:
        row = {"rule": name}
        for s, (lo, hi) in spans.items():
            row[f"share_{s}"] = float(fires[name].iloc[lo:hi].mean())
        rows.append(row)
    firing = pd.DataFrame(rows)
    th_df = pd.DataFrame(th).T.reset_index().rename(columns={"index": "rule"})

    d = os.path.join(out_dir, mode)
    os.makedirs(d, exist_ok=True)
    th_df.to_csv(os.path.join(d, "thresholds.csv"), index=False)
    firing.to_csv(os.path.join(d, "rule_firing.csv"), index=False)
    tab.to_csv(os.path.join(d, "label_counts.csv"), index=False)
    counts.to_csv(os.path.join(d, "overlap_counts.csv"))
    cond.to_csv(os.path.join(d, "overlap_conditional.csv"))
    c2.to_csv(os.path.join(d, "criteria_c2.csv"), index=False)
    c3.to_csv(os.path.join(d, "criteria_c3.csv"), index=False)
    c4.to_csv(os.path.join(d, "criteria_c4.csv"), index=False)
    with open(os.path.join(d, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(th, f, indent=2, default=float)

    lines = [
        f"## {title}", "",
        "Thu tu uu tien: " + " > ".join(a.priority) + " > normal", "",
        "Nguong (chi tu train):", "", md_table(th_df), "",
        "Ty le moi luat kich hoat (truoc uu tien, moi luat doc lap):", "", md_table(firing), "",
        "Nhan sau khi gan theo uu tien:", "", md_table(tab), "",
        "Chong lan, so cua so train thoa dong thoi (duong cheo = so cua so cua luat):", "",
        md_table(counts.reset_index().rename(columns={"index": "rule"}), "{:.0f}"), "",
        "Xac suat co dieu kien P(cot | hang) tren train:", "",
        md_table(cond.reset_index().rename(columns={"index": "rule"}), "{:.3f}"), "",
        "### Tieu chi C2 (moi nhan khac Normal)", "", md_table(c2), "",
        "### Tieu chi C3 (ty le Normal)", "", md_table(c3), "",
        "### Tieu chi C4 (lech ty le giua cac tap)", "", md_table(c4), "",
    ]
    if warn:
        lines += ["Canh bao:"] + [f"- {w}" for w in warn] + [""]
    else:
        lines += ["Khong co tieu chi nao khong dat.", ""]
    return lines, tab, cond, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--subsample", type=int, default=6)
    ap.add_argument("--clim", choices=["hour", "month_hour"], default="month_hour")
    ap.add_argument("--purge", type=int, default=None)
    ap.add_argument("--spike-channel", default="T", choices=list(CHANNELS))
    ap.add_argument("--q-spike", type=float, default=0.99)
    ap.add_argument("--q-low", type=float, default=0.10)
    ap.add_argument("--q-high", type=float, default=0.90)
    ap.add_argument("--priority", nargs="+", default=RULE_ORDER, choices=RULE_ORDER)
    ap.add_argument("--c2-min-share", type=float, default=0.02)
    ap.add_argument("--c2-target-share", type=float, default=0.05)
    ap.add_argument("--c2-min-count", type=int, default=300)
    ap.add_argument("--c3-max-normal", type=float, default=0.70)
    ap.add_argument("--c3-target-normal", type=float, default=0.60)
    ap.add_argument("--c4-max-rel-dev", type=float, default=0.5)
    ap.add_argument("--out", default="experiments/pattern_stats/jena")
    a = ap.parse_args()

    out_dir = os.path.join(a.out, "labels")
    os.makedirs(out_dir, exist_ok=True)

    df, n_wv_err = load_hourly_with_time(a.csv or download_jena(), a.subsample)
    df = df[list(CHANNELS.values())]
    sp = window_split(len(df), a.L, a.H, purge=a.purge)
    n_w, spans = sp["n_windows"], sp["spans"]
    feats, _ = build_feature_table(df, CHANNELS, a.L, n_w, sp["train_rows"], a.clim)
    feats = add_raw_mean(feats, df, CHANNELS, a.L, n_w)

    tr, va, te = spans["train"], spans["val"], spans["test"]
    report = [
        "# Pattern stats - Jena (buoc 2: nhan thu, chong lan, tieu chi C2-C4)", "",
        "Chi do, khong chon thay nguoi dung. Nguong chi tinh tren train; nhan chi dung dac trung cua X.", "",
        f"- Cua so: train {tr[1] - tr[0]}, val {va[1] - va[0]}, test {te[1] - te[0]} "
        f"(tong {n_w}, purge {sp['gap']})",
        f"- Phan vi: spike {a.q_spike}, thap {a.q_low}, cao {a.q_high}; kenh spike: {a.spike_channel}",
        "- Cach A: dac trung phan du; anom = mean(x - khi hau). "
        "Cach B: dac trung tho; heat/cold dung trung binh tho cua T (khong tru khi hau).",
        f"- C2: share_train >= {a.c2_min_share:.0%} va so cua so nho nhat qua 3 tap >= {a.c2_min_count}; "
        f"C3: Normal <= {a.c3_max_normal:.0%} o ca 3 tap; "
        f"C4: do lech tuong doi so voi train <= {a.c4_max_rel_dev} "
        "(dung sai C4 la mac dinh cua script, prompt khong neu).", "",
    ]
    all_warn = {}
    for mode, title in MODES:
        lines, tab, cond, warn = run_mode(mode, title, feats, spans, a, out_dir)
        report += lines
        all_warn[mode] = warn
        print(f"\n===== {title} =====")
        print(tab.round(4).to_string(index=False))
        print("\nP(cot | hang) tren train:")
        print(cond.round(3).to_string())
        for w in warn:
            print("[CANH BAO]", w)

    with open(os.path.join(a.out, "report_labels.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    cfg = dict(vars(a))
    cfg.update({"n_windows": n_w, "gap": sp["gap"], "spans": spans,
                "n_wv_sensor_errors_set_to_zero": n_wv_err})
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=float)
    print("\nDa luu vao:", a.out)


if __name__ == "__main__":
    main()