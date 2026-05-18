"""Filter NLLoc catalog to a 'reliable' subset using standard QC cuts.

Reports counts at several tiers and writes the strict (paper-defensible)
subset by default. Tiers are cumulative:

  loose:    gap<200, RMS<0.7, Nphs>=4
  standard: gap<180, RMS<0.5, Nphs>=6     (the 'HQ' tier used elsewhere)
  strict:   gap<120, RMS<0.3, Nphs>=8,
            sigma_x<1km, sigma_y<1km, sigma_z<2km,
            depth_below_seafloor > 0.2 km, not edge-pinned, inside ZX hull

All tiers also require interior-of-grid (not boundary-pinned).

Writes:
    catalogs/nlloc_<label>_reliable.csv      (strict tier by default)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"

GRID_LAYOUTS = {
    "ORCA":    ((-29.8, 30.2), (-20.0, 20.0)),
    "ORCA_v2": ((-150.0, 80.0), (-110.0, 70.0)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only_no_shots_v2_vpvs210")
    ap.add_argument("--tt-prefix", default="ORCA_v2",
                    choices=sorted(GRID_LAYOUTS.keys()))
    ap.add_argument("--tier", default="standard",
                    choices=["loose", "standard", "strict"])
    args = ap.parse_args()

    df = pd.read_csv(REPO / "catalogs" / f"nlloc_{args.label}.csv")
    (gx0, gx1), (gy0, gy1) = GRID_LAYOUTS[args.tt_prefix]
    df["on_boundary"] = (
        (df.nlloc_x_km - gx0 < 0.5) | (gx1 - df.nlloc_x_km < 0.5) |
        (df.nlloc_y_km - gy0 < 0.5) | (gy1 - df.nlloc_y_km < 0.5)
    )

    # Station-hull mask (ZX OBS convex hull)
    st = pd.read_csv(ST)
    zx = st[st.network == "ZX"][["longitude", "latitude"]].to_numpy()
    hull = ConvexHull(zx)
    hull_path = MplPath(zx[hull.vertices])
    df["in_hull"] = hull_path.contains_points(np.c_[df.lon, df.lat])

    n = len(df)
    def report(mask, name):
        print(f"  {name:30s} {mask.sum():,} ({100*mask.mean():.1f}%)")

    print(f"input: {n:,} events ({args.label})")

    loose    = ~df.on_boundary & (df.gap_deg < 200) & (df.rms_s < 0.7) & (df.n_phases >= 4)
    standard = ~df.on_boundary & (df.gap_deg < 180) & (df.rms_s < 0.5) & (df.n_phases >= 6)
    strict   = (~df.on_boundary &
                df.in_hull &
                (df.gap_deg < 120) &
                (df.rms_s < 0.3) &
                (df.n_phases >= 8) &
                (df.sigma_x_km < 1.0) &
                (df.sigma_y_km < 1.0) &
                (df.sigma_z_km < 2.0) &
                (df.depth_km > 0.2))

    report(loose,    "loose    (gap<200, RMS<0.7, N>=4)")
    report(standard, "standard (gap<180, RMS<0.5, N>=6)")
    report(strict,   "strict   (gap<120, RMS<0.3, N>=8, σ<1/1/2 km, hull, bsf>0.2)")

    chosen = {"loose": loose, "standard": standard, "strict": strict}[args.tier]
    out = df[chosen].copy()
    out_path = REPO / "catalogs" / f"nlloc_{args.label}_reliable.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out):,} {args.tier} events -> {out_path}")


if __name__ == "__main__":
    main()
