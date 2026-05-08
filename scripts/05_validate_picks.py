"""
Stage 1.e — validate PhaseNet picks against manual ground truth.

Reads:
  - catalogs/manual_picks.csv     (from src/bransfield_eq/manual_picks.py)
  - catalogs/picks/<NET>.<STA>/*.csv  (from scripts/03_run_phasenet.py)

For each (network, station, phase), greedily match each manual pick to the
nearest PhaseNet pick within a tolerance window. Reports:
  - per-station precision, recall, F1, mean/std time residual
  - overall + per-phase aggregates
  - confusion-matrix style summary CSV at catalogs/validation_report.csv

Definitions (per station-phase):
  TP  manual pick has a PhaseNet pick within tolerance
  FN  manual pick has none
  FP  PhaseNet pick has no manual within tolerance
  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)

Usage:
    python scripts/05_validate_picks.py
    python scripts/05_validate_picks.py --p-tol 0.5 --s-tol 1.0
    python scripts/05_validate_picks.py --network ZX
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bransfield_eq.config import PICK_DIR, REPO  # noqa: E402

MANUAL_CSV = REPO / "catalogs" / "manual_picks.csv"
REPORT_CSV = REPO / "catalogs" / "validation_report.csv"
PER_PICK_CSV = REPO / "catalogs" / "validation_per_pick.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-tol", type=float, default=0.5,
                   help="P-pick match tolerance in seconds (default 0.5)")
    p.add_argument("--s-tol", type=float, default=1.0,
                   help="S-pick match tolerance in seconds (default 1.0)")
    p.add_argument("--network", default=None,
                   help="restrict to one network code")
    p.add_argument("--start", default=None,
                   help="restrict to picks after this time (ISO)")
    p.add_argument("--end", default=None,
                   help="restrict to picks before this time (ISO)")
    return p.parse_args()


def load_phasenet_picks() -> pd.DataFrame:
    """Concatenate all per-day PhaseNet pick CSVs into one DataFrame."""
    files = list(PICK_DIR.glob("*/*.csv")) if PICK_DIR.exists() else []
    if not files:
        return pd.DataFrame(columns=["time", "trace_id", "phase", "prob"])
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # trace_id format: NET.STA.LOC.CHAN — split into columns
    parts = df["trace_id"].str.split(".", expand=True)
    df["network"] = parts[0]
    df["station"] = parts[1]
    df["channel"] = parts[3] if parts.shape[1] > 3 else None
    df["t"] = pd.to_datetime(df["time"], utc=True)
    df["phase"] = df["phase"].str.upper().str[0]   # PhaseNet emits "P"/"S"
    return df[["network", "station", "channel", "phase", "t", "prob"]]


def load_manual_picks() -> pd.DataFrame:
    if not MANUAL_CSV.exists():
        raise SystemExit(f"Missing {MANUAL_CSV}. Run "
                         "PYTHONPATH=src python -m bransfield_eq.manual_picks first.")
    df = pd.read_csv(MANUAL_CSV)
    df["t"] = pd.to_datetime(df["pick_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["t", "phase", "station"])
    df["phase"] = df["phase"].str.upper().str[0]
    return df[["network", "station", "phase", "t", "magnitude", "event_id"]]


def match_one_group(manual: pd.DataFrame, ml: pd.DataFrame,
                    tol_s: float) -> tuple[pd.DataFrame, int, int]:
    """
    Greedy nearest-time matching within `tol_s` seconds.
    Returns: per-manual-pick result table, n_unmatched_ml (FP), total_ml.
    """
    manual = manual.sort_values("t").reset_index(drop=True).copy()
    ml = ml.sort_values("t").reset_index(drop=True).copy()
    manual["matched"] = False
    manual["ml_t"] = pd.NaT
    manual["ml_prob"] = np.nan
    manual["residual_s"] = np.nan

    if ml.empty:
        return manual, 0, 0

    ml_used = np.zeros(len(ml), dtype=bool)
    ml_t_ns = ml["t"].astype("int64").values
    for i, row in manual.iterrows():
        man_t_ns = row["t"].value
        # binary search for the closest unused ML pick
        idx = np.searchsorted(ml_t_ns, man_t_ns)
        candidates = []
        for j in (idx - 1, idx):
            if 0 <= j < len(ml) and not ml_used[j]:
                candidates.append((abs(ml_t_ns[j] - man_t_ns) / 1e9, j))
        if not candidates:
            continue
        dt, j = min(candidates)
        if dt <= tol_s:
            ml_used[j] = True
            manual.at[i, "matched"] = True
            manual.at[i, "ml_t"] = ml.iloc[j]["t"]
            manual.at[i, "ml_prob"] = ml.iloc[j]["prob"]
            manual.at[i, "residual_s"] = (ml.iloc[j]["t"] - row["t"]).total_seconds()
    n_fp = (~ml_used).sum()
    return manual, int(n_fp), len(ml)


def main() -> None:
    args = parse_args()
    print("Loading manual picks ...")
    manual = load_manual_picks()
    print(f"  {len(manual)} manual picks across {manual.station.nunique()} stations.")

    print("Loading PhaseNet picks ...")
    ml = load_phasenet_picks()
    print(f"  {len(ml)} PhaseNet picks across {ml.station.nunique() if len(ml) else 0} stations.")

    if args.network:
        manual = manual[manual.network == args.network]
        ml = ml[ml.network == args.network]
    if args.start:
        t0 = pd.to_datetime(args.start, utc=True)
        manual = manual[manual.t >= t0]; ml = ml[ml.t >= t0]
    if args.end:
        t1 = pd.to_datetime(args.end, utc=True)
        manual = manual[manual.t <= t1]; ml = ml[ml.t <= t1]

    if ml.empty:
        print("\n  [warn] No PhaseNet picks on disk — skipping match. "
              "Run scripts/03_run_phasenet.py first.\n"
              "  Manual-pick stats only:")
        manual.groupby(["network", "station", "phase"]).size().to_csv(REPORT_CSV)
        print(f"  Wrote {REPORT_CSV.relative_to(REPO)} (manual-only counts)")
        return

    rows = []
    per_pick_frames = []
    for (net, sta, phase), m_grp in manual.groupby(["network", "station", "phase"], dropna=False):
        tol = args.p_tol if phase == "P" else args.s_tol
        m_grp_ml = ml[(ml.network == net) & (ml.station == sta) & (ml.phase == phase)]
        matched, n_fp, n_ml = match_one_group(m_grp, m_grp_ml, tol)
        per_pick_frames.append(matched.assign(network=net, station=sta, phase=phase))

        tp = matched["matched"].sum()
        fn = len(matched) - tp
        fp = n_fp
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec > 0) else np.nan
        res = matched.loc[matched["matched"], "residual_s"]
        rows.append({
            "network": net, "station": sta, "phase": phase,
            "n_manual": len(matched), "n_ml": n_ml,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
            "residual_mean_s": res.mean() if len(res) else np.nan,
            "residual_std_s": res.std() if len(res) else np.nan,
            "tol_s": tol,
        })
    rep = pd.DataFrame(rows)
    rep.to_csv(REPORT_CSV, index=False)
    pd.concat(per_pick_frames, ignore_index=True).to_csv(PER_PICK_CSV, index=False)

    # Aggregates
    print("\n=== Per-phase aggregates ===")
    for phase, grp in rep.groupby("phase"):
        tp = grp.tp.sum(); fp = grp.fp.sum(); fn = grp.fn.sum()
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        print(f"  {phase}: TP={tp} FP={fp} FN={fn}  precision={prec:.3f}  recall={rec:.3f}")
    print(f"\n  Wrote {REPORT_CSV.relative_to(REPO)} ({len(rep)} rows: net × sta × phase)")
    print(f"  Wrote {PER_PICK_CSV.relative_to(REPO)} ({sum(len(f) for f in per_pick_frames)} rows: per-manual-pick)")


if __name__ == "__main__":
    main()
