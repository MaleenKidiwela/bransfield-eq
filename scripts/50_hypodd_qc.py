"""QC a hypoDD relocated catalogue: physical plausibility against LOCAL bathymetry,
and a minimum-links cut.

Why
---
hypoDD runs on a flat seafloor datum (it flattens every OBS to elevation 0), so
its own 'depth >= 0' test says nothing about the water column: under the Orca
edifice the local seafloor is as shallow as 0.35 km BSL. The physical test is
depth_bsl vs the 30 m bathymetry at the event's own position -- the same test
the NLLoc tiers use (script 40).

hypoDD also 'relocates' events that have almost no differential-time links; on
the v4 baseline the 10 events that moved >10 km had a median of 3 catalogue
links against 336 for the population. A relocation with a handful of links is
not a relocation. Standard practice is a minimum-links cut.

Nothing is deleted: the full catalogue is written with flags; a `_qc` file holds
the subset that passes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ORCA_BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"


def local_water_depth_km(lat, lon):
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    o = xr.open_dataset(ORCA_BATHY)
    la = np.asarray(o.latitude.values, float); lo = np.asarray(o.longitude.values, float)
    z = np.asarray(o["data"].values, float)
    if z.shape != (len(la), len(lo)):
        z = z.T
    f = RegularGridInterpolator((la, lo), np.clip(-z / 1000.0, 0.0, None),
                                bounds_error=False, fill_value=np.nan)
    return f(np.c_[np.asarray(lat), np.asarray(lon)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="e.g. year_v4_1d")
    ap.add_argument("--min-links", type=int, default=20,
                    help="minimum nctp+ncts (+nccp+nccs) to count as relocated")
    ap.add_argument("--seafloor-tol-km", type=float, default=0.2,
                    help="half a 0.4 km grid cell; events more than this above the "
                         "local seafloor are flagged not physical")
    args = ap.parse_args()

    src = REPO / "catalogs" / f"hypodd_{args.label}.csv"
    df = pd.read_csv(src)
    if "depth_bsl_km" not in df.columns:
        raise SystemExit(f"{src} has no depth_bsl_km -- rerun script 24 (it stamps the datum)")

    w = local_water_depth_km(df.lat.values, df.lon.values)
    df["local_water_km"] = w
    df["depth_bsf_local_km"] = df.depth_bsl_km - w
    in_foot = np.isfinite(w)
    df["physical_local"] = np.where(in_foot, df.depth_bsf_local_km > -args.seafloor_tol_km, True)
    links = df[["nctp", "ncts"]].sum(axis=1)
    for c in ("nccp", "nccs"):
        if c in df.columns:
            links = links + df[c].clip(lower=0)
    df["n_links"] = links
    df["well_linked"] = df.n_links >= args.min_links
    df["hypodd_qc_pass"] = df.physical_local & df.well_linked

    n = len(df)
    print(f"{src.name}: {n:,} relocated events")
    print(f"  inside bathymetry footprint:        {int(in_foot.sum()):,}")
    print(f"  NOT physical (>{args.seafloor_tol_km} km above local seafloor): "
          f"{int((~df.physical_local).sum()):,} ({(~df.physical_local).mean()*100:.2f}%)")
    print(f"  poorly linked (< {args.min_links} links):    {int((~df.well_linked).sum()):,} "
          f"({(~df.well_linked).mean()*100:.2f}%)   n_links p10/p50 {df.n_links.quantile(.1):.0f}/{df.n_links.median():.0f}")
    print(f"  PASS both:                          {int(df.hypodd_qc_pass.sum()):,} ({df.hypodd_qc_pass.mean()*100:.1f}%)")
    b = df.depth_bsf_local_km[in_foot & df.hypodd_qc_pass]
    print(f"  QC-pass depth below LOCAL seafloor p10/50/90: {b.quantile(.1):.2f}/{b.median():.2f}/{b.quantile(.9):.2f} km")

    df.to_csv(src, index=False)                                   # flags added in place
    out = REPO / "catalogs" / f"hypodd_{args.label}_qc.csv"
    df[df.hypodd_qc_pass].to_csv(out, index=False)
    print(f"wrote flags into {src.name}; QC subset -> {out.name}")


if __name__ == "__main__":
    main()
