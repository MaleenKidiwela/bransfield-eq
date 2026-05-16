"""hypoDD relocation plot with seafloor profile overlaid on the longitude
cross-section. Same layout as plot_hypodd_stage_b.py but the cross-section
includes the bathymetric seafloor band. Output: hypodd_relocations_<label>_seafloor.png."""
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
DEP_MAX = 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--title-extra", default="")
    args = ap.parse_args()

    csv = REPO / "catalogs" / f"hypodd_{args.label}.csv"
    out = REPO / "notes" / "figures" / f"hypodd_relocations_{args.label}_seafloor.png"

    df = pd.read_csv(csv)
    print(f"loaded {len(df):,} events from {csv}")

    stations = pd.read_csv(ST)
    ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
    z_b = ds.variables["data"][:]; ds.close()
    z_b_plot = np.where(z_b > 2000, np.nan, z_b)

    def in_zoom(la, lo):
        return (lo >= LON_MIN) & (lo <= LON_MAX) & (la >= LAT_MIN) & (la <= LAT_MAX)
    dz = df[in_zoom(df.lat, df.lon)].reset_index(drop=True)
    print(f"in zoom: {len(dz):,}")

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

    def overlay_1000m(ax):
        """Draw the -1000 m bathymetric contour (seamount edifice) on top of
        the earthquake scatter."""
        LON, LAT = np.meshgrid(lon_b, lat_b)
        ax.contour(LON, LAT, z_b_plot, levels=[-1000], colors="k",
                   linewidths=1.2, linestyles="-", zorder=10)

    ob = stations[stations.network == "ZX"]

    # (a) map colored by depth
    ax = fig.add_subplot(gs[0, 0])
    plot_bathy(ax)
    ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
               edgecolors="k", linewidths=1.0, zorder=8)
    sc1 = ax.scatter(dz.lon, dz.lat, c=dz.dep, s=4, cmap="magma_r",
                     vmin=0, vmax=12, edgecolors="none", alpha=0.7, zorder=7)
    overlay_1000m(ax)
    ax.set_title(f"hypoDD relocated events — {len(dz):,} in zoom")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.colorbar(sc1, ax=ax, label="depth below sea level (km)", shrink=0.7, pad=0.02)

    # (b) map colored by cluster id
    ax = fig.add_subplot(gs[0, 1])
    plot_bathy(ax)
    ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
               edgecolors="k", linewidths=1.0, zorder=8)
    sc2 = ax.scatter(dz.lon, dz.lat, c=dz.cid, s=4, cmap="tab20",
                     edgecolors="none", alpha=0.7, zorder=7)
    overlay_1000m(ax)
    ax.set_title("colored by cluster id (cid; region-local)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    plt.colorbar(sc2, ax=ax, label="cid", shrink=0.7, pad=0.02)

    # (c) longitude cross-section + seafloor profile overlay
    ax = fig.add_subplot(gs[1, 0])
    t = pd.to_datetime(dict(year=dz.yr, month=dz.mo, day=dz.dy,
                            hour=dz.hr.clip(0, 23), minute=dz.mi.clip(0, 59)),
                       utc=True, errors="coerce") + pd.to_timedelta(dz.sc, unit="s")
    days = (t - t.min()).dt.total_seconds() / 86400.0
    sc3 = ax.scatter(dz.lon, dz.dep, c=days, s=4, cmap="viridis",
                     alpha=0.7, zorder=5)
    # Seafloor band: convert elevation to depth (km), take min/median/max across
    # the latitude strip at each longitude.
    in_lat_sf = (lat_b >= LAT_MIN) & (lat_b <= LAT_MAX)
    in_lon_sf = (lon_b >= LON_MIN) & (lon_b <= LON_MAX)
    z_sub = z_b[in_lat_sf][:, in_lon_sf]
    lon_sf = lon_b[in_lon_sf]
    sf_km = np.where(-z_sub / 1000.0 > 0, -z_sub / 1000.0, np.nan)
    ax.fill_between(lon_sf, np.nanmin(sf_km, axis=0), np.nanmax(sf_km, axis=0),
                    color="0.7", alpha=0.5, label="seafloor (lat-strip range)",
                    zorder=3)
    ax.plot(lon_sf, np.nanmedian(sf_km, axis=0), color="0.25", linewidth=1.2,
            label="seafloor (median)", zorder=4)
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(DEP_MAX, 0)
    ax.set_xlabel("longitude"); ax.set_ylabel("depth below sea level (km)")
    ax.set_title(f"Longitude cross-section + seafloor profile "
                 f"(events colored by days since {t.min().strftime('%Y-%m-%d')})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02),
              fontsize=8, framealpha=0.9, borderpad=0.6)
    plt.colorbar(sc3, ax=ax, label="days since start", shrink=0.7, pad=0.02)

    # (d) cluster-size histogram
    ax = fig.add_subplot(gs[1, 1])
    csize = df.groupby(["region", "cid"]).size() if "region" in df.columns else df.groupby("cid").size()
    multi = csize[csize >= 2]
    if len(multi) > 0:
        ax.hist(multi, bins=np.logspace(0, np.log10(multi.max() + 1), 30),
                edgecolor="k", alpha=0.7, color="steelblue")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"hypoDD cluster sizes — {len(multi):,} multi-event clusters")
    else:
        ax.text(0.5, 0.5, "no multi-event clusters",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("hypoDD cluster sizes")
    ax.set_xlabel("cluster size (events)"); ax.set_ylabel("count")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"hypoDD relocations + seafloor — label {args.label}  {args.title_extra}\n"
                 f"{len(df):,} relocated events",
                 fontsize=13)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
