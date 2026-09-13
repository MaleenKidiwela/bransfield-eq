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
    "ORCA_v3": ((-150.0, 80.0), (-110.0, 70.0)),
    "ORCA_v4": ((-150.0, 80.0), (-110.0, 70.0)),
    "ORCA_v5": ((-150.0, 80.0), (-110.0, 70.0)),
}
# Deepest travel-time node: ORCA 126 nodes x 0.2 km = 25.0; ORCA_v2 64 x 0.4 = 25.2.
GRID_Z = {"ORCA": (0.0, 25.0), "ORCA_v2": (0.0, 25.2), "ORCA_v3": (0.0, 25.2), "ORCA_v4": (0.0, 25.2), "ORCA_v5": (0.0, 25.2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only_no_shots_v2_vpvs210")
    ap.add_argument("--tt-prefix", default="ORCA_v2",
                    choices=sorted(GRID_LAYOUTS.keys()))
    ap.add_argument("--depth-datum", default=None, choices=["seafloor", "sealevel"],
                    help="datum of depth_km. Default: the catalogue's depth_datum column.")
    ap.add_argument("--tier", default="standard",
                    choices=["loose", "standard", "strict"])
    args = ap.parse_args()

    df = pd.read_csv(REPO / "catalogs" / f"nlloc_{args.label}.csv")
    (gx0, gx1), (gy0, gy1) = GRID_LAYOUTS[args.tt_prefix]
    # The boundary test used to check x and y only, so DEPTH-pinned events passed every
    # tier untagged: 13.2% of the old "reliable" file sat at depth < 0.05 km (top of grid,
    # with artificially small sigma_z) and 2.3% at the bottom face. z is now included.
    gz0, gz1 = GRID_Z[args.tt_prefix]
    df["on_boundary"] = (
        (df.nlloc_x_km - gx0 < 0.5) | (gx1 - df.nlloc_x_km < 0.5) |
        (df.nlloc_y_km - gy0 < 0.5) | (gy1 - df.nlloc_y_km < 0.5) |
        (df.depth_km - gz0 < 0.5) | (gz1 - df.depth_km < 0.5)
    )
    # NLLoc writes a full GEOGRAPHIC line even when it REJECTS a solution, so rejected
    # events used to enter every tier. Script 31 now records the status.
    if "nlloc_status" in df.columns:
        n_rej = (df.nlloc_status != "LOCATED").sum()
        if n_rej:
            print(f"  dropping {n_rej:,} non-LOCATED solutions")
        df = df[df.nlloc_status == "LOCATED"].copy()

    # ---- depth below the LOCAL seafloor, for the strict tier's bsf test ----
    # On the sheared v1-v3 grids depth_km already IS below-seafloor. On the un-sheared
    # ORCA_v4+ grid depth_km is below SEA LEVEL, so 'depth_km > 0.2' would be trivially
    # true and the test would silently stop doing anything. Compute bsf explicitly.
    datum = args.depth_datum
    if datum is None:
        if "depth_datum" in df.columns and df["depth_datum"].nunique() == 1:
            datum = str(df["depth_datum"].iloc[0])
        else:
            raise SystemExit("cannot determine depth datum: pass --depth-datum "
                             "(catalogue has no depth_datum column)")
    if datum == "seafloor":
        df["depth_bsf_km"] = df.depth_km
    elif datum == "sealevel":
        wsurf = REPO / "nlloc" / "model" / f"{args.tt_prefix}.water_depth_km.npy"
        if not wsurf.exists():
            raise SystemExit(f"sealevel datum needs the water-depth surface {wsurf}")
        w = np.load(wsurf)                       # (nx, ny) in Stingray km, from script 41
        hdr = (REPO / "nlloc" / "model" / f"{args.tt_prefix}.P.mod.hdr").read_text().split()
        hx0, hy0, hdx = float(hdr[3]), float(hdr[4]), float(hdr[6])
        # Bilinear interpolation at the exact hypocentre, NOT nearest grid node. On the
        # caldera walls the seafloor changes by hundreds of metres within one 0.4 km
        # cell; nearest-node lookup let 67 events (0.7% of standard) sit >0.2 km above
        # their true local seafloor while passing the test. Verified against the 30 m
        # Orca bathymetry independently.
        from scipy.interpolate import RegularGridInterpolator
        gx = hx0 + np.arange(w.shape[0]) * hdx
        gy = hy0 + np.arange(w.shape[1]) * hdx
        fw = RegularGridInterpolator((gx, gy), w, method="linear",
                                     bounds_error=False, fill_value=None)
        xq = np.clip(df.nlloc_x_km.values, gx[0], gx[-1])
        yq = np.clip(df.nlloc_y_km.values, gy[0], gy[-1])
        coarse = fw(np.c_[xq, yq])
        # Even bilinear on the 0.4 km surface left 49 events >0.2 km above the true
        # seafloor (worst +0.28 km) on the caldera walls. Use the 30 m Orca bathymetry
        # directly wherever it covers the event; the coarse surface only outside its
        # footprint, where the basin floor is smooth and 0.4 km is adequate.
        orca = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
        fine = np.full(len(df), np.nan)
        if orca.exists():
            import xarray as xr
            o = xr.open_dataset(orca)
            olat = np.asarray(o.latitude.values, float); olon = np.asarray(o.longitude.values, float)
            oz = np.asarray(o["data"].values, float)
            if oz.shape != (len(olat), len(olon)):
                oz = oz.T
            fo = RegularGridInterpolator((olat, olon), np.clip(-oz / 1000.0, 0.0, None),
                                         bounds_error=False, fill_value=np.nan)
            fine = fo(np.c_[df.lat.values, df.lon.values])
        n_fine = int(np.isfinite(fine).sum())
        df["local_water_km"] = np.where(np.isfinite(fine), fine, coarse)
        print(f"  local seafloor: {n_fine:,} events from 30 m Orca bathymetry, "
              f"{len(df) - n_fine:,} from the 0.4 km surface")
        df["depth_bsf_km"] = df.depth_km - df["local_water_km"]
    else:
        raise SystemExit(f"unknown depth datum {datum!r}")
    print(f"  depth datum: {datum}   bsf median {df.depth_bsf_km.median():.2f} km")

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

    # Physical plausibility: an event ABOVE the local seafloor is in the water column
    # and cannot be an earthquake. On the un-sheared v4 grid 25% of the standard tier
    # (2,842 events) had bsf < 0; they carry rms 0.298 vs 0.199 and an artificially
    # tight sigma_z 0.30 vs 0.77 - a forced minimum, i.e. picks inconsistent with any
    # sub-seafloor source. Tolerance is half a grid cell (0.4 km / 2), the resolution
    # at which NLLoc can place a hypocentre against the seafloor; strict uses +0.2.
    SEAFLOOR_TOL_KM = 0.2
    below_seafloor = df.depth_bsf_km > -SEAFLOOR_TOL_KM
    loose    = ~df.on_boundary & below_seafloor & (df.gap_deg < 200) & (df.rms_s < 0.7) & (df.n_phases >= 4)
    standard = ~df.on_boundary & below_seafloor & (df.gap_deg < 180) & (df.rms_s < 0.5) & (df.n_phases >= 6)
    strict   = (~df.on_boundary &
                df.in_hull &
                (df.gap_deg < 120) &
                (df.rms_s < 0.3) &
                (df.n_phases >= 8) &
                (df.sigma_x_km < 1.0) &
                (df.sigma_y_km < 1.0) &
                (df.sigma_z_km < 2.0) &
                (df.depth_bsf_km > 0.2))   # below the LOCAL seafloor, datum-aware

    report(loose,    "loose    (gap<200, RMS<0.7, N>=4, bsf>-0.2)")
    report(standard, "standard (gap<180, RMS<0.5, N>=6, bsf>-0.2)")
    report(strict,   "strict   (gap<120, RMS<0.3, N>=8, σ<1/1/2 km, hull, bsf>0.2)")

    chosen = {"loose": loose, "standard": standard, "strict": strict}[args.tier]
    out = df[chosen].copy()
    # Every tier used to write "_reliable.csv", so a --tier loose run silently
    # overwrote the strict/standard file and downstream could not tell them apart.
    out["qc_tier"] = args.tier
    out_path = REPO / "catalogs" / f"nlloc_{args.label}_{args.tier}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out):,} {args.tier} events -> {out_path}")


if __name__ == "__main__":
    main()
