"""Overlay the NLLoc grid extent and station-network polygon on the NLLoc map
to diagnose the apparent edge-pileup of events.

Projects the 60x40 km Stingray-frame grid back into lat/lon via the
TRANS SIMPLE -62.4413 -58.44 36 inverse, and the convex hull of the 15
ZX OBS station polygon. Colors events by their relation to both polygons:
  - interior to grid AND inside station hull (well-located)
  - interior to grid but outside station hull (poor azimuthal coverage)
  - on/near grid boundary (extent-limited; pyocto-only)
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from scipy.io import netcdf_file
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

LAT0, LON0, ROT_DEG = -62.4413, -58.44, 36.0


def stingray_to_latlon(x_km: np.ndarray, y_km: np.ndarray):
    """Inverse of TRANS SIMPLE lat0/lon0/rotCW: (x,y) km -> (lat,lon) deg."""
    c, s = np.cos(np.radians(ROT_DEG)), np.sin(np.radians(ROT_DEG))
    # NLLoc CW rotation: x = dE*c + dN*s ; y = -dE*s + dN*c.  Invert.
    dE = x_km * c - y_km * s
    dN = x_km * s + y_km * c
    lat = LAT0 + dN / 111.32
    lon = LON0 + dE / (np.cos(np.radians(LAT0)) * 111.32)
    return lat, lon


GRID_LAYOUTS = {
    "ORCA":    ((-29.8, 30.2), (-20.0, 20.0)),
    "ORCA_v2": ((-150.0, 80.0), (-110.0, 70.0)),
}


def grid_polygon(prefix: str = "ORCA"):
    (x0, x1), (y0, y1) = GRID_LAYOUTS[prefix]
    xs = np.array([x0, x1, x1, x0])
    ys = np.array([y0, y0, y1, y1])
    lat, lon = stingray_to_latlon(xs, ys)
    return lon, lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only_no_shots")
    ap.add_argument("--tt-prefix", default="ORCA",
                    choices=sorted(GRID_LAYOUTS.keys()))
    args = ap.parse_args()

    nl = pd.read_csv(REPO / "catalogs" / f"nlloc_{args.label}.csv")
    stations = pd.read_csv(ST)
    zx = stations[stations.network == "ZX"].copy()

    with netcdf_file(str(BATHY), "r") as ds:
        lat_b = ds.variables["latitude"][:].copy()
        lon_b = ds.variables["longitude"][:].copy()
        z_b = ds.variables["data"][:].copy()
    z_b = np.where(z_b > 2000, np.nan, z_b)
    LON, LAT = np.meshgrid(lon_b, lat_b)

    grid_lon, grid_lat = grid_polygon(args.tt_prefix)

    sta_pts = zx[["longitude", "latitude"]].to_numpy()
    hull = ConvexHull(sta_pts)
    hull_lon = sta_pts[hull.vertices, 0]
    hull_lat = sta_pts[hull.vertices, 1]
    hull_path = MplPath(np.c_[hull_lon, hull_lat])

    grid_path = MplPath(np.c_[grid_lon, grid_lat])
    inside_grid = grid_path.contains_points(np.c_[nl.lon, nl.lat])
    inside_hull = hull_path.contains_points(np.c_[nl.lon, nl.lat])

    print(f"total: {len(nl):,}")
    print(f"inside grid polygon (lat/lon): {inside_grid.sum():,} ({100*inside_grid.mean():.1f}%)")
    print(f"inside station hull:           {inside_hull.sum():,} ({100*inside_hull.mean():.1f}%)")
    print(f"in both:                       {(inside_grid & inside_hull).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ----- Wide view: grid + hull + all events -----
    ax = axes[0]
    levels = np.arange(-2400, 200, 50)
    norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    ax.contourf(LON, LAT, z_b, levels=levels, cmap=plt.cm.GnBu_r,
                norm=norm_b, extend="both")
    ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)

    # Event colors
    mask_out_grid = ~inside_grid
    mask_in_grid_out_hull = inside_grid & ~inside_hull
    mask_in_both = inside_grid & inside_hull
    ax.scatter(nl.lon[mask_out_grid], nl.lat[mask_out_grid], s=2,
               c="red", alpha=0.3, label=f"outside grid ({mask_out_grid.sum():,})")
    ax.scatter(nl.lon[mask_in_grid_out_hull], nl.lat[mask_in_grid_out_hull], s=2,
               c="goldenrod", alpha=0.4,
               label=f"in grid, outside station hull ({mask_in_grid_out_hull.sum():,})")
    ax.scatter(nl.lon[mask_in_both], nl.lat[mask_in_both], s=2,
               c="navy", alpha=0.5,
               label=f"in grid AND station hull ({mask_in_both.sum():,})")

    # Grid extent polygon (closed)
    grid_xy = np.c_[np.r_[grid_lon, grid_lon[0]], np.r_[grid_lat, grid_lat[0]]]
    ax.plot(grid_xy[:, 0], grid_xy[:, 1], color="black", linewidth=2,
            linestyle="--", zorder=12, label="NLLoc grid extent")
    # Station hull (closed)
    hull_xy = np.c_[np.r_[hull_lon, hull_lon[0]], np.r_[hull_lat, hull_lat[0]]]
    ax.plot(hull_xy[:, 0], hull_xy[:, 1], color="white", linewidth=2,
            zorder=12, label="ZX station hull")

    ax.scatter(zx.longitude, zx.latitude, marker="^", s=80, c="white",
               edgecolors="k", linewidths=1.2, zorder=13)
    ax.set_xlim(grid_lon.min() - 0.3, grid_lon.max() + 0.3)
    ax.set_ylim(grid_lat.min() - 0.2, grid_lat.max() + 0.2)
    ax.set_aspect(1.0 / np.cos(np.radians(LAT0)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("NLLoc events vs. grid extent + station hull")
    ax.legend(loc="lower left", markerscale=3, framealpha=0.9, fontsize=9)

    # ----- Zoom on Orca summit -----
    ax = axes[1]
    ax.contourf(LON, LAT, z_b, levels=levels, cmap=plt.cm.GnBu_r,
                norm=norm_b, extend="both")
    ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
    ax.scatter(nl.lon[mask_in_both], nl.lat[mask_in_both], s=4,
               c=nl.depth_km[mask_in_both], cmap="magma_r",
               vmin=0, vmax=20, alpha=0.6, zorder=7)
    ax.plot(hull_xy[:, 0], hull_xy[:, 1], color="white", linewidth=2, zorder=12)
    ax.scatter(zx.longitude, zx.latitude, marker="^", s=80, c="white",
               edgecolors="k", linewidths=1.2, zorder=13)
    ax.set_xlim(-58.7, -58.2); ax.set_ylim(-62.55, -62.35)
    ax.set_aspect(1.0 / np.cos(np.radians(LAT0)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Orca zoom — {mask_in_both.sum():,} events inside both")

    (gx0, gx1), (gy0, gy1) = GRID_LAYOUTS[args.tt_prefix]
    fig.suptitle(
        f"NLLoc grid extent vs. station coverage — {args.label}\n"
        f"{args.tt_prefix} grid {gx1-gx0:.0f}×{gy1-gy0:.0f} km @ "
        f"rot=+{ROT_DEG:.0f}° CW from east",
        fontsize=12,
    )
    plt.tight_layout()
    out = REPO / "notes" / "figures" / f"nlloc_grid_extent_{args.label}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
