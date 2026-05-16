"""Compare pyocto events flagged as shots vs. real earthquakes to confirm
the discriminator picks out the NW-SE and SW-NE survey tracks."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent

ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag.csv")
ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True, format="mixed", errors="coerce")
ev = ev.dropna(subset=["origin_time"])

# Bathy
ds = Dataset(REPO / "notes" / "figures" / "Orca_bathymetry.nc")
lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b_plot = np.where(z_b > 2000, np.nan, z_b)
LON, LAT = np.meshgrid(lon_b, lat_b)

stations = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
ob = stations[stations.network == "ZX"]

lon_min, lon_max = -59.5, -57.5
lat_min, lat_max = -63.0, -62.1

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, mask, title in [
    (axes[0], ev.flag_shot.astype(bool), f"flagged as SHOTS ({ev.flag_shot.sum():,})"),
    (axes[1], ~ev.flag_shot.astype(bool),
     f"NOT flagged ({(~ev.flag_shot).sum():,} real EQs)"),
    (axes[2], pd.Series(True, index=ev.index),
     f"BEFORE discrimination ({len(ev):,})"),
]:
    levels = np.arange(-2400, 200, 50)
    norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    ax.contourf(LON, LAT, z_b_plot, levels=levels,
                cmap=plt.cm.GnBu_r, norm=norm_b, extend="both")
    ax.contour(LON, LAT, z_b_plot, levels=[0], colors="k", linewidths=0.5)
    ax.contour(LON, LAT, z_b_plot, levels=[-1000], colors="k",
               linewidths=1.0, zorder=9)
    sub = ev[mask]
    color = "red" if "SHOTS" in title else "blue" if "NOT" in title else "k"
    ax.scatter(sub.longitude, sub.latitude, s=2, c=color, alpha=0.4, zorder=7)
    ax.scatter(ob.longitude, ob.latitude, marker="^", s=60, c="white",
               edgecolors="k", linewidths=0.8, zorder=8)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title)

fig.suptitle("Pyocto catalog: shot-discrimination by ±1 s temporal match to BRAVOSEIS shotfiles",
             fontsize=13)
plt.tight_layout()
out = REPO / "notes" / "figures" / "shot_discrimination.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
