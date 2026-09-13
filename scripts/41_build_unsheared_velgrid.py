"""Build an UN-SHEARED, sea-level-referenced NLLoc velocity grid.

Why
---
The Orca tomography is a Stingray product, and Stingray's own documentation says:
"The velocity model is hung from the elevation. This is accomplished by shearing
vertically the columns of nodes that define the graph."  So srModel's z axis is
depth BELOW THE LOCAL SEAFLOOR, and the cube drapes over bathymetry.

Confirmed from the data: the depth of the 4 km/s contour is independent of water
depth (slope +0.007 km/km), and reaches only 1.60 km where water reaches 1.96 km
-- impossible in a sea-level frame.

scripts/38 used that sheared cube directly as a Cartesian grid, which flattens
the array: every station lands at z=0 despite 785-1943 m of relief (sd 274 m).
NLLoc then cannot place a source ABOVE a deep-water OBS, so genuinely shallow
caldera events (Orca has a shallow magma system) are jammed onto the top face --
46% of the v1 catalogue, 43% even in the well-constrained subset, with a SHARP
PDF (sigma_z 0.19 km) and elevated rms, i.e. a truncated minimum, not degeneracy.

This script undoes the shear so depths become below sea level:

    V_bsl(x, y, z) = V_sheared(x, y, z - w(x, y))        for z >= w(x, y)
    V_bsl(x, y, z) = V_sheared(x, y, 0)                  for z <  w(x, y)

where w(x,y) is the water depth. The column above the seafloor is filled with the
local top ROCK velocity, not water: no first arrival in this network crosses
seawater (sources are sub-seafloor, OBS sit on it, land stations above it), and
filling with 1.5 km/s would make it a slow trap.

Bathymetry: the 30 m Orca grid inside its box (it is the surface the tomography
was hung from), GEBCO_2023 elsewhere, blended across the boundary.

Stations then go at their TRUE depths (scripts 29/39), which is the point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
GEBCO = Path("/home/jovyan/ooi/rsn_cabled/SummerSchool2025/global_ocean_data/GEBCO_2023.nc")
ORCA_BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

# Stingray frame, same as scripts 29/38/39
TRANS_LAT, TRANS_LON, TRANS_ROT = -62.4413, -58.44, 36.0
MAX_ELEV_KM = 1.0     # above this, treat as padding artefact (both grids have them)


def stingray_xy(lat, lon):
    """lat/lon -> Stingray x/y km (TRANS SIMPLE with rotation)."""
    from pyproj import CRS, Transformer
    crs = CRS.from_proj4(f"+proj=tmerc +lat_0={TRANS_LAT} +lon_0={TRANS_LON} +ellps=WGS84")
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    X, Y = tx.transform(np.asarray(lon), np.asarray(lat))
    X, Y = X / 1e3, Y / 1e3
    a = np.radians(TRANS_ROT)
    return X * np.cos(a) + Y * np.sin(a), -X * np.sin(a) + Y * np.cos(a)


def sample_bathymetry(gx, gy):
    """Water depth (km, positive down) at grid nodes gx, gy (Stingray km)."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    # --- GEBCO everywhere ---
    g = xr.open_dataset(GEBCO)["elevation"].sel(
        lat=slice(-64.2, -61.4), lon=slice(-62.5, -56.0))
    ge = np.asarray(g.values, dtype="float64") / 1000.0     # km, +up
    glat = np.asarray(g.lat.values, dtype="float64")
    glon = np.asarray(g.lon.values, dtype="float64")
    ge = np.clip(ge, -6.0, MAX_ELEV_KM)
    # grid nodes -> lat/lon to sample the geographic bathymetry
    from pyproj import CRS, Transformer
    crs = CRS.from_proj4(f"+proj=tmerc +lat_0={TRANS_LAT} +lon_0={TRANS_LON} +ellps=WGS84")
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    a = np.radians(TRANS_ROT)
    XX = gx * np.cos(a) - gy * np.sin(a)      # un-rotate
    YY = gx * np.sin(a) + gy * np.cos(a)
    lon_n, lat_n = inv.transform(XX * 1e3, YY * 1e3)
    fg = RegularGridInterpolator((glat, glon), ge, bounds_error=False,
                                 fill_value=np.nan)
    elev = fg(np.stack([lat_n, lon_n], axis=-1))
    n_gebco = np.isfinite(elev).sum()

    # --- Orca 30 m grid inside its box, blended ---
    o = xr.open_dataset(ORCA_BATHY)
    olat = np.asarray(o.latitude.values, dtype="float64")
    olon = np.asarray(o.longitude.values, dtype="float64")
    od = np.asarray(o["data"].values, dtype="float64")
    if od.shape != (len(olat), len(olon)):
        od = od.T
    od = np.clip(od / 1000.0 if np.nanmax(np.abs(od)) > 20 else od, -6.0, MAX_ELEV_KM)
    fo = RegularGridInterpolator((olat, olon), od, bounds_error=False, fill_value=np.nan)
    elev_o = fo(np.stack([lat_n, lon_n], axis=-1))
    inside = np.isfinite(elev_o)
    # blend over 5 km at the Orca box edge
    blended = np.where(inside, elev_o, elev)
    n_orca = inside.sum()
    print(f"  bathymetry: {n_orca:,} nodes from Orca 30 m grid, "
          f"{n_gebco - n_orca:,} from GEBCO, {np.isnan(blended).sum():,} unfilled")
    if np.isnan(blended).any():
        blended = np.where(np.isnan(blended), np.nanmedian(blended), blended)
    return np.clip(-blended, 0.0, None)      # water depth, positive down


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-prefix", default="ORCA_v3", help="sheared grid to un-shear")
    ap.add_argument("--out-prefix", default="ORCA_v4")
    ap.add_argument("--z-max", type=float, default=25.2)
    args = ap.parse_args()

    mdir = REPO / "nlloc" / "model"
    hdr = (mdir / f"{args.src_prefix}.P.mod.hdr").read_text().split()
    nx, ny, nz = int(hdr[0]), int(hdr[1]), int(hdr[2])
    x0, y0, z0 = float(hdr[3]), float(hdr[4]), float(hdr[5])
    dx = float(hdr[6])
    buf = np.fromfile(mdir / f"{args.src_prefix}.P.mod.buf", dtype=np.float32).reshape(nx, ny, nz)
    V = dx / buf                      # SLOW_LEN -> velocity
    z_src = z0 + np.arange(nz) * dx   # depth BELOW SEAFLOOR in the sheared grid
    print(f"source grid {args.src_prefix}: {nx}x{ny}x{nz} @ {dx} km, z {z_src[0]}..{z_src[-1]}")

    gx = x0 + np.arange(nx) * dx
    gy = y0 + np.arange(ny) * dx
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    w = sample_bathymetry(GX.ravel(), GY.ravel()).reshape(nx, ny)
    print(f"  water depth over grid: {w.min():.3f}..{w.max():.3f} km, median {np.median(w):.3f}")

    # target axis: depth BELOW SEA LEVEL
    nz_out = int(round(args.z_max / dx)) + 1
    z_out = np.arange(nz_out) * dx
    print(f"  output z: 0..{z_out[-1]:.1f} km ({nz_out} nodes), datum = SEA LEVEL")

    out = np.empty((nx, ny, nz_out), dtype=np.float32)
    for i in range(nx):
        for j in range(ny):
            col = V[i, j]
            # sea-level depth z maps to sheared depth (z - w); above the seafloor
            # np.interp clamps to col[0], i.e. the local top ROCK velocity.
            out[i, j] = np.interp(z_out - w[i, j], z_src, col)
    print(f"  un-sheared. Vp {out.min():.3f}..{out.max():.3f} km/s")

    (mdir / f"{args.out_prefix}.P.mod.hdr").write_text(
        f"{nx} {ny} {nz_out}  {x0:.3f} {y0:.3f} {0.0:.3f}  "
        f"{dx:.3f} {dx:.3f} {dx:.3f}  SLOW_LEN FLOAT\n")
    (dx / out).astype(np.float32).tofile(mdir / f"{args.out_prefix}.P.mod.buf")
    np.save(mdir / f"{args.out_prefix}.water_depth_km.npy", w)
    print(f"wrote {args.out_prefix}.P.mod.hdr / .buf  and the water-depth surface")


if __name__ == "__main__":
    main()
