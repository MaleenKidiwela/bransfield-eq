"""Animate hypoDD relocated events with the cross-section using
depth-below-seafloor (positive down, seafloor = horizontal line at zero)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import pandas as pd
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"


def build_catalog(label: str, z_b, lat_b, lon_b) -> pd.DataFrame:
    df = pd.read_csv(REPO / "catalogs" / f"hypodd_{label}.csv")
    df["t"] = pd.to_datetime(dict(
        year=df.yr, month=df.mo, day=df.dy,
        hour=df.hr.clip(0, 23), minute=df.mi.clip(0, 59)),
        utc=True, errors="coerce") + pd.to_timedelta(df.sc, unit="s")
    df = df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    # depth-below-seafloor
    lat_idx = np.clip(np.searchsorted(lat_b, df.lat.values), 0, len(lat_b) - 1)
    lon_idx = np.clip(np.searchsorted(lon_b, df.lon.values), 0, len(lon_b) - 1)
    sf_elev_m = z_b[lat_idx, lon_idx]
    sf_depth_km = np.where(sf_elev_m < 0, -sf_elev_m / 1000.0, 0.0)
    df["depth_below_seafloor_km"] = df.dep - sf_depth_km
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bin-days", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--keep-fresh", type=float, default=14.0)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (REPO / "notes" / "figures" /
                                                f"hypodd_animation_{args.label}_belowfloor.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv(ST)
    ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
    z_b = ds.variables["data"][:]; ds.close()
    z_b_plot = np.where(z_b > 2000, np.nan, z_b)
    LON, LAT = np.meshgrid(lon_b, lat_b)

    df = build_catalog(args.label, z_b, lat_b, lon_b)
    print(f"loaded {len(df):,} events  ({df.t.min()} -> {df.t.max()})")

    lon_min, lon_max = -58.7, -58.2
    lat_min, lat_max = -62.55, -62.35
    bsf_min, bsf_max = -1.5, 8.0

    df = df[df.lon.between(lon_min, lon_max) & df.lat.between(lat_min, lat_max)].reset_index(drop=True)
    print(f"in zoom: {len(df):,}")

    t0 = df.t.min().normalize()
    t1 = df.t.max().normalize() + pd.Timedelta(days=1)
    bin_td = pd.Timedelta(days=args.bin_days)
    n_frames = int(np.ceil((t1 - t0) / bin_td))
    print(f"frames: {n_frames} at {args.bin_days} days/frame, {args.fps} fps -> {n_frames/args.fps:.1f} s")

    fig, (ax_map, ax_xs) = plt.subplots(1, 2, figsize=(14, 6),
                                        gridspec_kw={"width_ratios": [1.2, 1]})
    levels = np.arange(-2400, 200, 50)
    norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    ax_map.contourf(LON, LAT, z_b_plot, levels=levels,
                    cmap=plt.cm.GnBu_r, norm=norm_b, extend="both")
    ax_map.contour(LON, LAT, z_b_plot, levels=[0], colors="k", linewidths=0.6)
    ob = stations[stations.network == "ZX"]
    ax_map.scatter(ob.longitude, ob.latitude, marker="^", s=50, c="white",
                   edgecolors="k", linewidths=1.0, zorder=8)
    ax_map.set_xlim(lon_min, lon_max); ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
    ax_map.set_xlabel("longitude"); ax_map.set_ylabel("latitude")

    ax_xs.axhline(0, color="0.25", linewidth=1.4, label="seafloor", zorder=4)
    ax_xs.set_xlim(lon_min, lon_max); ax_xs.set_ylim(bsf_max, bsf_min)
    ax_xs.set_xlabel("longitude"); ax_xs.set_ylabel("depth below seafloor (km)")
    ax_xs.grid(alpha=0.3); ax_xs.legend(loc="upper right", fontsize=8)

    map_fade = ax_map.scatter([], [], s=4, c="0.35", alpha=0.35, zorder=7)
    map_fresh = ax_map.scatter([], [], s=12, c="red", alpha=0.9, edgecolors="k",
                               linewidths=0.2, zorder=9)
    xs_fade = ax_xs.scatter([], [], s=4, c="0.35", alpha=0.35)
    xs_fresh = ax_xs.scatter([], [], s=12, c="red", alpha=0.9,
                             edgecolors="k", linewidths=0.2)
    title_txt = fig.suptitle("", fontsize=13)

    times_np = df.t.values
    lon_np = df.lon.values
    lat_np = df.lat.values
    bsf_np = df.depth_below_seafloor_km.values

    def update(i):
        frame_t = t0 + (i + 1) * bin_td
        mask_all = times_np <= np.datetime64(frame_t)
        fresh_cut = frame_t - pd.Timedelta(days=args.keep_fresh)
        mask_fresh = mask_all & (times_np > np.datetime64(fresh_cut))
        mask_old = mask_all & ~mask_fresh
        map_fade.set_offsets(np.c_[lon_np[mask_old], lat_np[mask_old]])
        map_fresh.set_offsets(np.c_[lon_np[mask_fresh], lat_np[mask_fresh]])
        xs_fade.set_offsets(np.c_[lon_np[mask_old], bsf_np[mask_old]])
        xs_fresh.set_offsets(np.c_[lon_np[mask_fresh], bsf_np[mask_fresh]])
        title_txt.set_text(
            f"hypoDD {args.label} (depth-below-seafloor)  —  "
            f"{pd.Timestamp(frame_t).strftime('%Y-%m-%d')}  "
            f"({int(mask_all.sum()):,} events)"
        )
        return map_fade, map_fresh, xs_fade, xs_fresh, title_txt

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    print(f"rendering to {out_path} ...")
    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    anim.save(str(out_path), writer=writer, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
