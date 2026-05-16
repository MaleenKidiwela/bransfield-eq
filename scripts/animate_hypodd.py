"""Animate a hypoDD relocated catalog: events appear through time, shown on a
side-by-side map view and W-E depth cross-section. Outputs an MP4."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import pandas as pd
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"


def build_catalog(label: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / "catalogs" / f"hypodd_{label}.csv")
    df["t"] = pd.to_datetime(dict(
        year=df.yr, month=df.mo, day=df.dy,
        hour=df.hr.clip(0, 23), minute=df.mi.clip(0, 59)),
        utc=True, errors="coerce") + pd.to_timedelta(df.sc, unit="s")
    return df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="Reads catalogs/hypodd_<label>.csv.")
    ap.add_argument("--out", default=None,
                    help="Output MP4 path (default: notes/figures/hypodd_animation_<label>.mp4).")
    ap.add_argument("--bin-days", type=float, default=2.0,
                    help="Time bin per frame in days. Smaller -> more frames, smoother.")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--keep-fresh", type=float, default=14.0,
                    help="Within this many days of frame-time, events draw bright; older fade.")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (REPO / "notes" / "figures" /
                                                f"hypodd_animation_{args.label}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_catalog(args.label)
    print(f"loaded {len(df):,} events  ({df.t.min()} -> {df.t.max()})")

    t0 = df.t.min().normalize()
    t1 = df.t.max().normalize() + pd.Timedelta(days=1)
    bin_td = pd.Timedelta(days=args.bin_days)
    n_frames = int(np.ceil((t1 - t0) / bin_td))
    print(f"frames: {n_frames} at {args.bin_days} days/frame, {args.fps} fps -> "
          f"{n_frames/args.fps:.1f} s video")

    # Static background data
    stations = pd.read_csv(ST)
    ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
    z_b = ds.variables["data"][:]; ds.close()
    z_b = np.where(z_b > 2000, np.nan, z_b)
    LON, LAT = np.meshgrid(lon_b, lat_b)

    lon_min, lon_max = -58.7, -58.2
    lat_min, lat_max = -62.55, -62.35
    dep_min, dep_max = 0.0, 12.0   # depth axis for cross-section

    in_zoom = (df.lon.between(lon_min, lon_max) &
               df.lat.between(lat_min, lat_max))
    df = df[in_zoom].reset_index(drop=True)
    print(f"in zoom: {len(df):,}")

    fig, (ax_map, ax_xs) = plt.subplots(1, 2, figsize=(14, 6),
                                        gridspec_kw={"width_ratios": [1.2, 1]})

    # Map background
    levels = np.arange(-2400, 200, 50)
    norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    ax_map.contourf(LON, LAT, z_b, levels=levels,
                    cmap=plt.cm.GnBu_r, norm=norm_b, extend="both")
    ax_map.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
    # -1000 m bathymetric contour (seamount edifice). Drawn at zorder 10 so
    # it sits on TOP of the event scatter dots (zorder 7-9) in every frame.
    ax_map.contour(LON, LAT, z_b, levels=[-1000], colors="k",
                   linewidths=1.2, linestyles="-", zorder=10)
    ob = stations[stations.network == "ZX"]
    ax_map.scatter(ob.longitude, ob.latitude, marker="^", s=50, c="white",
                   edgecolors="k", linewidths=1.0, zorder=8)
    ax_map.set_xlim(lon_min, lon_max); ax_map.set_ylim(lat_min, lat_max)
    ax_map.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
    ax_map.set_xlabel("longitude"); ax_map.set_ylabel("latitude")

    # Cross-section background -- include seafloor profile across the lat strip
    ax_xs.set_xlim(lon_min, lon_max); ax_xs.set_ylim(dep_max, dep_min)
    ax_xs.set_xlabel("longitude"); ax_xs.set_ylabel("depth below sea level (km)")
    in_lat_sf = (lat_b >= lat_min) & (lat_b <= lat_max)
    in_lon_sf = (lon_b >= lon_min) & (lon_b <= lon_max)
    z_sub = z_b[in_lat_sf][:, in_lon_sf]
    lon_sf = lon_b[in_lon_sf]
    sf_km = np.where(-z_sub / 1000.0 > 0, -z_sub / 1000.0, np.nan)
    ax_xs.fill_between(lon_sf, np.nanmin(sf_km, axis=0), np.nanmax(sf_km, axis=0),
                       color="0.75", alpha=0.5, zorder=3)
    ax_xs.plot(lon_sf, np.nanmedian(sf_km, axis=0), color="0.25", linewidth=1.0, zorder=4)
    ax_xs.grid(alpha=0.3)

    # Two scatter layers per axes: "fade" (older events, dim) + "fresh" (recent, bright)
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
    dep_np = df.dep.values

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
            f"hypoDD {args.label}  —  {pd.Timestamp(frame_t).strftime('%Y-%m-%d')}  "
            f"({int(mask_all.sum()):,} events; {int(mask_fresh.sum())} in last {args.keep_fresh:.0f} d)"
        )
        return map_fade, map_fresh, xs_fade, xs_fresh, title_txt

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    print(f"rendering to {out_path} ...")
    writer = FFMpegWriter(fps=args.fps, bitrate=4000)
    anim.save(str(out_path), writer=writer, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
