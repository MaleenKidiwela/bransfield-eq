"""
Sweep DD-anomaly-ratio thresholds on a post-filtered picks subdir, validate
each threshold against mag07 manual catalog, plot precision/recall/F1.

The input subdir must have CSVs with the `dd_anomaly_ratio` column added by
scripts/13_dd_post_filter.py.

Usage:
    python scripts/14_dd_filter_sweep.py \
        --picks-subdir picks_dd_filtered \
        --start 2019-02-04 --end 2019-02-14 \
        --label "PhaseNet instance @ 0.1 + DD filter"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def filter_to_temp_subdir(src_subdir: str, ratio_thresh: float) -> str:
    """Create a temp subdir holding only picks with anomaly ratio >= thresh."""
    src = REPO / "catalogs" / src_subdir
    tmp_name = f"_tmp_filter_r{ratio_thresh:.2f}"
    tmp = REPO / "catalogs" / tmp_name
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for sta_dir in sorted(src.iterdir()):
        if not sta_dir.is_dir():
            continue
        out_sta = tmp / sta_dir.name
        out_sta.mkdir(parents=True, exist_ok=True)
        for csv in sorted(sta_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pd.DataFrame().to_csv(out_sta / csv.name, index=False)
                continue
            if df.empty or "dd_anomaly_ratio" not in df.columns:
                df.to_csv(out_sta / csv.name, index=False)
                continue
            keep = df[df.dd_anomaly_ratio >= ratio_thresh].copy()
            keep.to_csv(out_sta / csv.name, index=False)
    return tmp_name


def validate(subdir: str, start: str, end: str) -> dict:
    out = subprocess.check_output([
        sys.executable, str(REPO / "scripts" / "05_validate_picks.py"),
        "--picks-subdir", subdir, "--manual-source", "mag07",
        "--start", start, "--end", end,
    ], cwd=str(REPO), stderr=subprocess.STDOUT, text=True)
    out = "\n".join(l for l in out.splitlines()
                    if "FutureWarning" not in l and "manual.at[" not in l)
    import re
    pm = re.search(r"P: TP=(\d+) FP=(\d+) FN=(\d+) +precision=([\d.]+) +recall=([\d.]+)", out)
    sm = re.search(r"S: TP=(\d+) FP=(\d+) FN=(\d+) +precision=([\d.]+) +recall=([\d.]+)", out)
    tot = re.search(r"(\d+) PhaseNet picks", out)
    return {
        "total_picks": int(tot.group(1)) if tot else 0,
        "P_TP": int(pm.group(1)) if pm else 0, "P_FP": int(pm.group(2)) if pm else 0,
        "P_FN": int(pm.group(3)) if pm else 0,
        "P_precision": float(pm.group(4)) if pm else 0.0,
        "P_recall": float(pm.group(5)) if pm else 0.0,
        "S_TP": int(sm.group(1)) if sm else 0, "S_FP": int(sm.group(2)) if sm else 0,
        "S_FN": int(sm.group(3)) if sm else 0,
        "S_precision": float(sm.group(4)) if sm else 0.0,
        "S_recall": float(sm.group(5)) if sm else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks-subdir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.0, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
    args = ap.parse_args()
    label = args.label or args.picks_subdir

    print(f"=== DD-anomaly threshold sweep on '{args.picks_subdir}' ===")
    print(f"  date range: {args.start} → {args.end}")
    print(f"  ratios: {args.ratios}")
    print()
    rows = []
    for r in args.ratios:
        if r == 0.0:
            # Baseline = all picks (no filter)
            tmp = args.picks_subdir
        else:
            print(f"--- ratio ≥ {r:.2f} (filtering) ---", flush=True)
            tmp = filter_to_temp_subdir(args.picks_subdir, r)
        stats = validate(tmp, args.start, args.end)
        stats["ratio"] = r
        rows.append(stats)
        print(f"  ratio={r:.2f}  total={stats['total_picks']:>7,}  "
              f"P-rec={stats['P_recall']:.3f}  P-prec={stats['P_precision']:.4f}  "
              f"S-rec={stats['S_recall']:.3f}  S-prec={stats['S_precision']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = REPO / "catalogs" / f"dd_filter_sweep_{args.picks_subdir}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Cleanup temp dirs
    for r in args.ratios:
        if r == 0.0:
            continue
        tmp = REPO / "catalogs" / f"_tmp_filter_r{r:.2f}"
        if tmp.exists() and tmp.is_dir():
            shutil.rmtree(tmp)

    # Plot precision/recall/F1 vs threshold
    df["P_F1"] = 2 * df.P_precision * df.P_recall / (df.P_precision + df.P_recall + 1e-12)
    df["S_F1"] = 2 * df.S_precision * df.S_recall / (df.S_precision + df.S_recall + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # P recall + precision vs ratio
    axes[0].plot(df.ratio, df.P_recall, "o-", color="red", label="P recall")
    axes[0].plot(df.ratio, df.S_recall, "o-", color="blue", label="S recall")
    axes[0].set_xlabel("DD anomaly ratio threshold"); axes[0].set_ylabel("Recall")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_ylim(0, 1.05); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Recall vs DD anomaly ratio threshold")
    # Precision (log scale)
    axes[1].semilogy(df.ratio, df.P_precision, "o-", color="red", label="P precision")
    axes[1].semilogy(df.ratio, df.S_precision, "o-", color="blue", label="S precision")
    axes[1].set_xlabel("DD anomaly ratio threshold"); axes[1].set_ylabel("Precision (log)")
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    axes[1].set_title("Precision vs DD anomaly ratio threshold")
    # Pick volume
    axes[2].semilogy(df.ratio, df.total_picks, "o-", color="black")
    axes[2].set_xlabel("DD anomaly ratio threshold"); axes[2].set_ylabel("Total picks (log)")
    axes[2].set_xscale("symlog", linthresh=1)
    axes[2].grid(alpha=0.3, which="both")
    axes[2].set_title("Pick volume vs DD anomaly ratio threshold")
    fig.suptitle(f"{label} — {args.start} → {args.end}", fontsize=11)
    fig.tight_layout()
    out_png = REPO / "figures" / f"dd_filter_sweep_{args.picks_subdir}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
