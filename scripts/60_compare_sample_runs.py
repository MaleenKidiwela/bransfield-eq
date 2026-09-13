"""Side-by-side of NLLoc sample runs (same obs, different travel-time grids):
depth distribution, fraction pinned at the grid top, rms, and per-event depth shift.
Uses script 52's loader so the event mapping is identical to the S-P scorer."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="output dirs under nlloc/output, e.g. abtest_ORCA_v4 abtest_ORCA_v5")
    ap.add_argument("--obs-order", default="nlloc/obs/year_v4.event_order.csv")
    ap.add_argument("--n-first", type=int, default=2000)
    a = ap.parse_args()
    from importlib import import_module
    s52 = import_module("52_sp_depth_check")
    cats = {}
    for r in a.runs:
        df = s52.load_hypdir(REPO / "nlloc" / "output" / r, REPO / a.obs_order, a.n_first)
        cats[r] = df
        z = df.depth_km
        print(f"{r:22s} n={len(df):5d}  top-pinned(z<0.05) {(z < 0.05).mean()*100:5.1f}%  "
              f"depth p10/50/90 {z.quantile(.1):.2f}/{z.median():.2f}/{z.quantile(.9):.2f} km")
    if len(a.runs) == 2:
        A, B = (cats[r] for r in a.runs)
        j = A.merge(B, on="event_idx", suffixes=("_a", "_b"))
        dz = j.depth_km_b - j.depth_km_a
        print(f"\n{a.runs[1]} minus {a.runs[0]} on {len(j)} common events: depth shift median {dz.median():+.2f} km, p10/p90 {dz.quantile(.1):+.2f}/{dz.quantile(.9):+.2f}")
        for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 6), (6, 9), (9, 15), (15, 40)]:
            k = (j.depth_km_a >= lo) & (j.depth_km_a < hi)
            if k.sum(): print(f"   {a.runs[0]} depth {lo:>2}-{hi:<2} km  n={k.sum():4d}  -> {a.runs[1]} median {j.depth_km_b[k].median():5.2f}  (dz {dz[k].median():+.2f})")
        h = np.hypot(j.lon_b - j.lon_a, j.lat_b - j.lat_a) * 111.19 if "lon_a" in j else None
        if h is not None: print(f"epicentre shift median {np.median(h):.2f} km, p90 {np.percentile(h, 90):.2f}")


if __name__ == "__main__":
    main()
