"""Animate the HQ NLLoc catalog through time: map view + W-E depth cross-section.

Same visual style as animate_hypodd.py.
"""
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
from scipy.io import netcdf_file
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

GX_MIN, GX_MAX = -29.8, 30.2
GY_MIN, GY_MAX = -20.0, 20.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only_no_shots")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bin-days", type=float, default=2.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--keep-fresh", type=float, default=14.0)
    ap.add_argument("--hq-only", action="store_true", default=False,
                    help="Restrict to interior, gap<180, RMS<0.5, Nphs>=6 (default: show ALL events).")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        REPO / "notes" / "figures" / f"nlloc_animation_{args.label}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPO / "catalogs" / f"nlloc_{args.label}.csv")
    df["t"] = pd.to_datetime(df.origin_time, utc=True)
    on_boundary = ((df.nlloc_x_km - GX_MIN < 0.5) | (GX_MAX - df.nlloc_x_km < 0.5) |
                   (df.nlloc_y_km - GY_MIN < 0.5) | (GY_MAX - df.nlloc_y_km < 0.5))
    if args.hq_only:
        df = df[~on_boundary & (df.gap_deg < 180) &
                (df.rms_s < 0.5) & (df.n_phases >= 6)].copy()
    df = df.sort_values("t").reset_index(drop=True)
    print(f"events to animate: {len(df):,}  ({df.t.min()} -> {df.t.max()})")

    t0 = df.t.min().normalize()
    t1 = df.t.max().normalize() + pd.Timedelta(days=1)
    bin_td = pd.Timedelta(days=args.bin_days)
    n_frames = int(np.ceil((t1 - t0) / bin_td))
    print(f"frames: {n_frames} @ {args.bin_days} d/frame -> "
          f"{n_frames/args.fps:.1f} s @ {args.fps} fps")

    stations = pd.read_csv(ST)
    with netcdf_file(str(BATHY), "r") as ds:
        lat_b = ds.variables["latitude"][:].copy()
        lon_b = ds.variables["longitude"][:].copy()
        z_b = ds.variables["data"][:].copy()
    z_b = np.where(z_b > 2000, np.nan, z_b)
    LON, LAT = np.meshgrid(lon_b, lat_b)

    # NLLoc depths are depth-BELOW-SEAFLOOR (srModel parameterization).
    # For the cross-section we add the local bathymetric water depth to
    # show events on a sea-level-referenced axis with the bathymetry visible.
    sf_interp = RegularGridInterpolator(
        (lat_b, lon_b), -z_b / 1000.0,
        bounds_error=False, fill_value=np.nan,
    )
    df["seafloor_km"] = sf_interp(np.c_[df.lat, df.lon])
    df["depth_bsl_km"] = df.depth_km + df.seafloor_km

    lon_min, lon_max = -58.7, -58.2
    lat_min, lat_max = -62.55, -62.35
    dep_min, dep_max = 0.0, 20.0
    in_zoom = (df.lon.between(lon_min, lon_max) &
               df.lat.between(lat_min, lat_max))
    df = df[in_zoom].reset_index(drop=True)
    print(f"in Orca zoom: {len(df):,}")

    fig, (ax_map, ax_xs) = plt.subplots(1, 2, figsize=(14, 6),
                                        gridspec_kw={"width_ratios": [1.2, 1]})

    levels = np.arange(-2400, 200, 50)
    norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    ax_map.contourf(LON, LAT, z_b, levels=levels,
                    cmap=plt.cm.GnBu_r, norm=norm_b, extend="both")
    ax_map.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
    ax_map.contour(LON, LAT, z_b, levels=[-1000], colors="k",
                   linewidths=1.2, zorder=10)
    ob = stations[stations.network == "ZX"]
    ax_map.scatter(ob.longitude, ob.latitude, marker="^", s=50, c="white",
                   edgecolors="k", linewidths=1.0, zorder=8)
    ax_map.set_xlim(lon_min, lon_max); ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
    ax_map.set_xlabel("longitude"); ax_map.set_ylabel("latitude")

    # Cross-section: depth below SEA LEVEL with bathymetric seafloor envelope.
    # Events come from NLLoc as depth-below-seafloor; their depth_bsl_km
    # (above) shifts them onto the sea-level datum so bathymetry is visible.
    ax_xs.set_xlim(lon_min, lon_max); ax_xs.set_ylim(20.0, 0.0)
    ax_xs.set_xlabel("longitude"); ax_xs.set_ylabel("depth below sea level (km)")
    in_lat_sf = (lat_b >= lat_min) & (lat_b <= lat_max)
    in_lon_sf = (lon_b >= lon_min) & (lon_b <= lon_max)
    z_sub = z_b[in_lat_sf][:, in_lon_sf]
    sf_strip = -z_sub / 1000.0   # positive down
    lon_sf = lon_b[in_lon_sf]
    sf_min = np.nanmin(sf_strip, axis=0)
    sf_max = np.nanmax(sf_strip, axis=0)
    sf_med = np.nanmedian(sf_strip, axis=0)
    ax_xs.fill_between(lon_sf, 0, sf_max, color="lightblue", alpha=0.5, zorder=2)
    ax_xs.fill_between(lon_sf, sf_min, sf_max, color="0.75", alpha=0.6, zorder=3)
    ax_xs.plot(lon_sf, sf_med, color="0.2", linewidth=1.0, zorder=4)
    ax_xs.grid(alpha=0.3)

    map_fade = ax_map.scatter([], [], s=4, c="0.35", alpha=0.35, zorder=7)
    map_fresh = ax_map.scatter([], [], s=12, c="red", alpha=0.9,
                               edgecolors="k", linewidths=0.2, zorder=9)
    xs_fade = ax_xs.scatter([], [], s=4, c="0.35", alpha=0.35)
    xs_fresh = ax_xs.scatter([], [], s=12, c="red", alpha=0.9,
                             edgecolors="k", linewidths=0.2)
    title_txt = fig.suptitle("", fontsize=13)

    times_np = df.t.values
    lon_np = df.lon.values
    lat_np = df.lat.values
    dep_np = df.depth_bsl_km.values   # sea-level datum for cross-section

    def update(i):
        frame_t = t0 + (i + 1) * bin_td
        mask_all = times_np <= np.datetime64(frame_t)
        fresh_cut = frame_t - pd.Timedelta(days=args.keep_fresh)
        mask_fresh = mask_all & (times_np > np.datetime64(fresh_cut))
        mask_old = mask_all & ~mask_fresh
        map_fade.set_offsets(np.c_[lon_np[mask_old], lat_np[mask_old]])
        map_fresh.set_offsets(np.c_[lon_np[mask_fresh], lat_np[mask_fresh]])
        xs_fade.set_offsets(np.c_[lon_np[mask_old], dep_np[mask_old]])
        xs_fresh.set_offsets(np.c_[lon_np[mask_fresh], dep_np[mask_fresh]])
        title_txt.set_text(
            f"NLLoc {args.label} (HQ)  —  "
            f"{pd.Timestamp(frame_t).strftime('%Y-%m-%d')}  "
            f"({int(mask_all.sum()):,} events; "
            f"{int(mask_fresh.sum())} in last {args.keep_fresh:.0f} d)"
        )
        return map_fade, map_fresh, xs_fade, xs_fresh, title_txt

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    print(f"rendering -> {out_path} ...")
    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    anim.save(str(out_path), writer=writer, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
