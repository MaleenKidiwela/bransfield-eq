"""
Inspect the DD anomaly-ratio distribution for picks that match manual picks
(TPs) vs those that don't (FPs). Tests the hypothesis: does the DD signal
energy at a pick time discriminate real arrivals from spurious ones?

Reads picks_dd_filtered/ (output of 13_dd_post_filter.py) and matches each
pick to the mag07 manual catalog. Outputs:
  - figures/anomaly_score_tp_vs_fp.png — histograms / KDEs
  - catalogs/anomaly_score_summary.csv — TP vs FP distribution stats
  - For manual picks NOT matched by PhaseNet: report missed-event count
    (we can't compute anomaly there from picks alone, but we know the count)

Tolerances follow scripts/05_validate_picks.py: P ±0.5s, S ±1.0s.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

P_TOL_S = 0.5
S_TOL_S = 1.0


def load_filtered_picks(subdir: str, start: str, end: str) -> pd.DataFrame:
    root = REPO / "catalogs" / subdir
    rows = []
    start_pd = pd.Timestamp(start, tz="UTC")
    end_pd = pd.Timestamp(end, tz="UTC")
    for sta_dir in sorted(root.iterdir()):
        if not sta_dir.is_dir():
            continue
        try:
            net, sta = sta_dir.name.split(".")
        except ValueError:
            continue
        for csv in sorted(sta_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if df.empty:
                continue
            df = df.copy()
            df["t"] = pd.to_datetime(df["time"], utc=True)
            df = df[(df.t >= start_pd) & (df.t < end_pd)]
            if df.empty: continue
            df["network"] = net
            df["station"] = sta
            df["phase"] = df["phase"].str.upper().str[0]
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_manual(start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / "catalogs" / "manual_picks.csv",
                     parse_dates=["pick_time"])
    df = df[df.source_file == "nllmaleen_mag07_202210.out"].copy()
    df["t"] = pd.to_datetime(df.pick_time, utc=True)
    df["phase"] = df["phase"].str.upper().str[0]
    start_pd = pd.Timestamp(start, tz="UTC")
    end_pd = pd.Timestamp(end, tz="UTC")
    df = df[(df.t >= start_pd) & (df.t < end_pd)]
    return df[["network", "station", "phase", "t", "uncertainty_s"]]


def match_picks_to_manual(picks: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Add a 'matches_manual' column to picks. True if any manual pick of same
    phase on same station within tolerance."""
    picks = picks.copy()
    picks["matches_manual"] = False
    for (net, sta, phase), grp in picks.groupby(["network", "station", "phase"]):
        man = manual[(manual.network == net) & (manual.station == sta)
                     & (manual.phase == phase)]
        if man.empty:
            continue
        tol = pd.Timedelta(seconds=(P_TOL_S if phase == "P" else S_TOL_S))
        man_t = man.t.values
        pick_t = grp.t.values
        # For each pick, check if any manual within tol
        for i, t in enumerate(pick_t):
            diffs = np.abs(man_t - t)
            if (diffs <= tol).any():
                picks.loc[grp.index[i], "matches_manual"] = True
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks-subdir", default="picks_dd_filtered")
    ap.add_argument("--start", default="2019-02-04")
    ap.add_argument("--end", default="2019-02-14")
    args = ap.parse_args()

    print(f"Loading filtered picks from {args.picks_subdir} ({args.start} → {args.end}) ...")
    picks = load_filtered_picks(args.picks_subdir, args.start, args.end)
    print(f"  {len(picks):,} picks  cols={list(picks.columns)[:8]}...")

    if "dd_anomaly_ratio" not in picks.columns:
        print(f"  ERROR: no dd_anomaly_ratio column in {args.picks_subdir}")
        return

    print(f"Loading manual mag07 picks ({args.start} → {args.end}) ...")
    manual = load_manual(args.start, args.end)
    print(f"  {len(manual):,} manual picks across "
          f"{manual[['network','station']].drop_duplicates().shape[0]} stations")

    print("Matching picks to manual catalog ...")
    picks = match_picks_to_manual(picks, manual)
    n_tp = picks.matches_manual.sum()
    n_fp = (~picks.matches_manual).sum()
    print(f"  TP (matches a manual): {n_tp:,}")
    print(f"  FP (no manual within tol): {n_fp:,}")

    # Per-phase stats
    print("\n=== Anomaly ratio statistics (TP vs FP by phase) ===")
    for ph in ("P", "S"):
        sub = picks[picks.phase == ph]
        tp = sub[sub.matches_manual].dd_anomaly_ratio
        fp = sub[~sub.matches_manual].dd_anomaly_ratio
        if tp.empty or fp.empty:
            continue
        print(f"  Phase {ph}:")
        print(f"    TP  n={len(tp):>5,}  median={tp.median():.2f}  "
              f"p25={tp.quantile(0.25):.2f}  p75={tp.quantile(0.75):.2f}")
        print(f"    FP  n={len(fp):>5,}  median={fp.median():.2f}  "
              f"p25={fp.quantile(0.25):.2f}  p75={fp.quantile(0.75):.2f}")
        print(f"    TP/FP median ratio = {tp.median()/fp.median():.2f}")
        # AUC of separability: fraction of FP with ratio < TP median
        threshold = tp.median()
        fp_below = (fp < threshold).sum() / len(fp)
        print(f"    Fraction of FPs below TP median ({threshold:.2f}): {fp_below:.2%}")

    # Plot KDE-style histograms
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.logspace(-1, 3, 60)
    for ax, ph in zip(axes, ("P", "S")):
        sub = picks[picks.phase == ph]
        tp = sub[sub.matches_manual].dd_anomaly_ratio.replace([np.inf,-np.inf], np.nan).dropna()
        fp = sub[~sub.matches_manual].dd_anomaly_ratio.replace([np.inf,-np.inf], np.nan).dropna()
        if not tp.empty:
            ax.hist(tp.clip(lower=0.1, upper=1000), bins=bins, alpha=0.6, color="green",
                    density=True, label=f"TP (n={len(tp):,})")
        if not fp.empty:
            ax.hist(fp.clip(lower=0.1, upper=1000), bins=bins, alpha=0.4, color="red",
                    density=True, label=f"FP (n={len(fp):,})")
        ax.set_xscale("log")
        ax.set_xlabel("DD anomaly ratio (denoised RMS / station-day baseline)")
        ax.set_ylabel("Density")
        ax.set_title(f"{ph} picks — TP vs FP anomaly score")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        ax.axvline(1.0, color="black", lw=0.6, ls=":", alpha=0.5)
    fig.suptitle(f"DD anomaly ratio: PhaseNet 'instance' picks on {args.start} → {args.end}",
                 fontsize=11)
    fig.tight_layout()
    out = REPO / "figures" / f"anomaly_score_tp_vs_fp_{args.picks_subdir}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
