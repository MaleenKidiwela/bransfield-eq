"""Plot hypoDD relocations with depth re-referenced to the local seafloor.

For each event: depth_below_seafloor = event_depth - bathymetric_depth_at(lat,lon).
Positive values = below seafloor (in rock).
Negative values = above seafloor (in water column; unphysical).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

LON_MIN, LON_MAX = -58.7, -58.2
LAT_MIN, LAT_MAX = -62.55, -62.35
DEP_BSF_MAX = 8.0   # km below seafloor for cross-section ylim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--title-extra", default="")
    args = ap.parse_args()
    csv = REPO / "catalogs" / f"hypodd_{args.label}.csv"
    out = REPO / "notes" / "figures" / f"hypodd_relocations_{args.label}_belowfloor.png"

    df = pd.read_csv(csv)
    print(f"loaded {len(df):,} events from {csv}")

    stations = pd.read_csv(ST)
    ds = Dataset(BATHY)
    lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
    z_b = ds.variables["data"][:]; ds.close()
    z_b_plot = np.where(z_b > 2000, np.nan, z_b)

    # Compute depth-below-seafloor per event
    lat_idx = np.clip(np.searchsorted(lat_b, df.lat.values), 0, len(lat_b) - 1)
    lon_idx = np.clip(np.searchsorted(lon_b, df.lon.values), 0, len(lon_b) - 1)
    sf_elev_m = z_b[lat_idx, lon_idx]
    sf_depth_km = np.where(sf_elev_m < 0, -sf_elev_m / 1000.0, 0.0)
    df["depth_below_seafloor_km"] = df.dep - sf_depth_km
    df["seafloor_depth_km"] = sf_depth_km

    def in_zoom(la, lo):
        return (lo >= LON_MIN) & (lo <= LON_MAX) & (la >= LAT_MIN) & (la <= LAT_MAX)
    dz = df[in_zoom(df.lat, df.lon)].reset_index(drop=True)
    print(f"in zoom: {len(dz):,}")
    print(f"depth-below-seafloor stats: median {dz.depth_below_seafloor_km.median():.2f} km, "
          f"p5 {dz.depth_below_seafloor_km.quantile(.05):.2f}, "
          f"p95 {dz.depth_below_seafloor_km.quantile(.95):.2f}")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.18)

    def plot_bathy(ax):
        levels = np.arange(-2400, 200, 50)
        norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
        LON, LAT = np.meshgrid(lon_b, lat_b)
        ax.contourf(LON, LAT, z_b_plot, levels=levels,
                    cmap=plt.cm.GnBu_r, norm=norm_b, extend="both")
        ax.contour(LON, LAT, z_b_plot, levels=[0], colors="k", linewidths=0.6)
        ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect(1.0 / np.cos(np.radians(np.mean([LAT_MIN, LAT_MAX]))))

    ob = stations[stations.network == "ZX"]

    # (a) map colored by depth-below-seafloor (the geological depth)
    ax = fig.add_subplot(gs[0, 0])
    plot_bathy(ax)
    ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
               edgecolors="k", linewidths=1.0, zorder=8)
    sc1 = ax.scatter(dz.lon, dz.lat, c=dz.depth_below_seafloor_km, s=4,
                     cmap="magma_r", vmin=0, vmax=DEP_BSF_MAX,
                     edgecolors="none", alpha=0.7, zorder=7)
    ax.set_title(f"hypoDD — depth BELOW SEAFLOOR ({len(dz):,} in zoom)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.colorbar(sc1, ax=ax, label="depth below seafloor (km)", shrink=0.7, pad=0.02)

    # (b) Same map but color by days since start
    ax = fig.add_subplot(gs[0, 1])
    plot_bathy(ax)
    ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
               edgecolors="k", linewidths=1.0, zorder=8)
    t = pd.to_datetime(dict(year=dz.yr, month=dz.mo, day=dz.dy,
                            hour=dz.hr.clip(0, 23), minute=dz.mi.clip(0, 59)),
                       utc=True, errors="coerce") + pd.to_timedelta(dz.sc, unit="s")
    days = (t - t.min()).dt.total_seconds() / 86400.0
    sc2 = ax.scatter(dz.lon, dz.lat, c=days, s=4, cmap="viridis",
                     edgecolors="none", alpha=0.7, zorder=7)
    ax.set_title("colored by days since start")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.colorbar(sc2, ax=ax, label="days since start", shrink=0.7, pad=0.02)

    # (c) Cross-section in below-seafloor coordinates (seafloor = horizontal line at 0)
    ax = fig.add_subplot(gs[1, 0])
    sc3 = ax.scatter(dz.lon, dz.depth_below_seafloor_km, c=days, s=4,
                     cmap="viridis", alpha=0.7, zorder=5)
    ax.axhline(0, color="0.25", linewidth=1.4, label="seafloor", zorder=4)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(DEP_BSF_MAX, -1.5)   # invert; allow some "above seafloor" airquakes
    ax.set_xlabel("longitude"); ax.set_ylabel("depth below seafloor (km)")
    ax.set_title("Longitude cross-section, depth-below-seafloor frame")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    plt.colorbar(sc3, ax=ax, label="days since start", shrink=0.7, pad=0.02)

    # (d) Histogram of depth-below-seafloor
    ax = fig.add_subplot(gs[1, 1])
    bins = np.arange(-1.5, DEP_BSF_MAX + 0.25, 0.25)
    ax.hist(dz.depth_below_seafloor_km, bins=bins, color="steelblue",
            edgecolor="k", alpha=0.8)
    ax.axvline(0, color="red", linewidth=1.4, linestyle="--", label="seafloor")
    ax.set_xlabel("depth below seafloor (km)"); ax.set_ylabel("count")
    ax.set_title(f"Depth-below-seafloor distribution\n"
                 f"(median {dz.depth_below_seafloor_km.median():.2f} km below seafloor)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(f"hypoDD relocations (depth-below-seafloor frame) — {args.label}  {args.title_extra}\n"
                 f"{len(df):,} relocated events  ({(dz.depth_below_seafloor_km < 0).sum()} above seafloor)",
                 fontsize=13)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
