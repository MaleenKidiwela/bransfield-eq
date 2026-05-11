"""
Phase 4: held-out evaluation on August 2019.

Runs four pickers on Aug 2019 (idempotent — skips days already done) and
validates each against the high-confidence mag07 manual catalog. Aggregates
per-day TP/FP/FN, plots a per-day bar chart, and writes catalogs/aug2019_eval.csv.

Pickers:
  A. PhaseNet `instance` @ 0.1               → catalogs/picks/
  B. OBSTransformer `obst2024` @ 0.5         → catalogs/picks_obst_05/
  C. PhaseNet fine-tuned (ckpt:...) @ 0.1    → catalogs/picks_pn_ft/
  D. (optional, --with-dd) C on DeepDenoiser-cleaned input → catalogs/picks_pn_ft_dd/

Decision rule:
  C > A on aggregate P recall  → fine-tune helped
  D > C on aggregate recall    → denoising at inference time helps too
  Pick volume not collapsed    → not just a precision tradeoff
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CKPT_DEFAULT = REPO / "models" / "phasenet_bransfield_v1" / "best.ckpt"


def run_picker(model: str, weights: str, out_subdir: str,
               p_thresh: float, s_thresh: float,
               start: str, end: str, workers: int) -> None:
    """Invoke 03_run_phasenet.py for a date range. Idempotent."""
    cmd = [
        sys.executable, str(REPO / "scripts" / "03_run_phasenet.py"),
        "--model", model, "--weights", weights, "--out-subdir", out_subdir,
        "--start", start, "--end", end,
        "--p-thresh", str(p_thresh), "--s-thresh", str(s_thresh),
        "--workers", str(workers),
    ]
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(REPO))
    if rc.returncode != 0:
        print(f"    [ERR] picker exited {rc.returncode}")


def validate_subdir(picks_subdir: str, start: str, end: str) -> dict:
    """Run scripts/05_validate_picks.py for the date range, parse output."""
    cmd = [
        sys.executable, str(REPO / "scripts" / "05_validate_picks.py"),
        "--picks-subdir", picks_subdir,
        "--manual-source", "mag07",
        "--start", start, "--end", end,
    ]
    out = subprocess.check_output(cmd, cwd=str(REPO),
                                   stderr=subprocess.STDOUT, text=True)
    out = "\n".join(l for l in out.splitlines()
                    if "FutureWarning" not in l and "manual.at[" not in l)
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
    ap.add_argument("--ckpt", default=str(CKPT_DEFAULT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--with-dd", action="store_true",
                    help="Also run picker D (PhaseNet fine-tuned on DD-cleaned input)")
    ap.add_argument("--start", default="2019-08-01")
    ap.add_argument("--end", default="2019-09-01")
    ap.add_argument("--skip-pick", action="store_true",
                    help="Only validate (assume picks already done)")
    args = ap.parse_args()

    pickers = [
        # (id, model, weights, out_subdir, p_thr, s_thr)
        ("A", "PhaseNet",       "instance",   "picks",          0.1, 0.1),
        ("B", "OBSTransformer", "obst2024",   "picks_obst_05",  0.5, 0.5),
        ("C", "PhaseNet",       f"ckpt:{args.ckpt}", "picks_pn_ft", 0.1, 0.1),
    ]
    if args.with_dd:
        pickers.append(("D", "PhaseNet",
                        f"ckpt:{args.ckpt}", "picks_pn_ft_dd", 0.1, 0.1))
        # NOTE: D would need a separate denoised-input pipeline (not implemented in
        # 03_run_phasenet.py). Kept as a stub; actual D run would require a
        # second script. For v1 we focus on A / B / C.

    print(f"=== Picking {args.start} → {args.end} ===")
    if not args.skip_pick:
        for pid, model, weights, sd, pt, st in pickers:
            print(f"\n--- Picker {pid} ({model} {weights} @ {pt}/{st}) → {sd}/ ---")
            try:
                run_picker(model, weights, sd, pt, st, args.start, args.end, args.workers)
            except Exception as e:
                print(f"  [ERR] {e}")

    print(f"\n=== Validating {args.start} → {args.end} (mag07 trusted) ===")
    rows = []
    for pid, model, weights, sd, pt, st in pickers:
        try:
            stats = validate_subdir(sd, args.start, args.end)
        except subprocess.CalledProcessError as e:
            print(f"  [skip {pid}] validation failed: {e}")
            continue
        stats.update({"picker_id": pid, "model": model, "weights": weights,
                      "subdir": sd, "p_thresh": pt, "s_thresh": st})
        rows.append(stats)
        print(f"  {pid}  {model:<14} {weights:<24}  total={stats['total_picks']:>7,}  "
              f"P-rec={stats['P_recall']:.3f}  S-rec={stats['S_recall']:.3f}")
    df = pd.DataFrame(rows)
    out_csv = REPO / "catalogs" / "aug2019_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")

    # Bar chart
    if df.empty:
        print("  no rows to plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(df))
    labels = [f"{r.picker_id}: {r.model[:6]} {r.weights[:18]}"
              for r in df.itertuples()]
    axes[0].bar(x, df.P_recall.values, color=["#1f77b4", "#9467bd", "#2ca02c", "#d62728"][:len(df)])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("P recall (mag07, Aug 2019)"); axes[0].set_ylim(0, 1)
    for i, v in enumerate(df.P_recall.values):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_title(f"P recall — {df.iloc[0].P_TP + df.iloc[0].P_FN} manual P picks")
    axes[1].bar(x, df.S_recall.values, color=["#1f77b4", "#9467bd", "#2ca02c", "#d62728"][:len(df)])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("S recall (mag07, Aug 2019)"); axes[1].set_ylim(0, 1)
    for i, v in enumerate(df.S_recall.values):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].set_title(f"S recall — {df.iloc[0].S_TP + df.iloc[0].S_FN} manual S picks")
    fig.suptitle(f"August 2019 held-out eval ({args.start} → {args.end})")
    fig.tight_layout()
    fig_path = REPO / "figures" / "aug2019_recall.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    print(f"  wrote {fig_path}")


if __name__ == "__main__":
    main()
