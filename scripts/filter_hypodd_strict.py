"""Apply a strict post-hoc filter to a hypoDD catalog and write the surviving
events to a new CSV.

Filter (defaults):
    nctp >= 12   # >=12 catalog P obs used per event
    ncts >= 4    # >=4 catalog S obs
    rct  <  0.15 # mean CT residual < 0.15 s
    physical == True  # depth >= median seafloor depth (if column present)
    dist_to_centroid_km < 2 km (if column present -- bridge filter)

Usage:
    python scripts/filter_hypodd_strict.py --label stage_b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--nctp", type=int, default=12)
    ap.add_argument("--ncts", type=int, default=4)
    ap.add_argument("--rct-max", type=float, default=0.15)
    ap.add_argument("--dist-max", type=float, default=2.0,
                    help="Max distance-to-sub-region-centroid (km) for Stage B/C "
                         "events. Ignored if column absent.")
    args = ap.parse_args()

    inp = REPO / "catalogs" / f"hypodd_{args.label}.csv"
    out = REPO / "catalogs" / f"hypodd_{args.label}_strict.csv"
    df = pd.read_csv(inp)
    n0 = len(df)
    print(f"loaded {n0:,} events from {inp.name}")

    keep = pd.Series(True, index=df.index)
    for col, thr, op in [
        ("nctp", args.nctp, ">="),
        ("ncts", args.ncts, ">="),
        ("rct", args.rct_max, "<"),
    ]:
        if col not in df.columns:
            print(f"  [warn] no {col} column, skipping")
            continue
        if op == ">=":
            m = df[col] >= thr
        else:
            m = df[col] < thr
        print(f"  {col} {op} {thr}: {m.sum():,} pass / {(~m).sum():,} fail")
        keep &= m

    if "physical" in df.columns:
        m = df["physical"].astype(bool)
        print(f"  physical == True: {m.sum():,} pass")
        keep &= m
    if "dist_to_centroid_km" in df.columns:
        m = df["dist_to_centroid_km"] < args.dist_max
        print(f"  dist_to_centroid < {args.dist_max} km: {m.sum():,} pass")
        keep &= m

    df_strict = df[keep].reset_index(drop=True)
    df_strict.to_csv(out, index=False)
    print(f"\nkept {len(df_strict):,} of {n0:,} ({len(df_strict)/n0*100:.1f}%)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
