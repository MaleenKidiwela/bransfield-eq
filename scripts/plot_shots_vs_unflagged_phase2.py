"""Plot Phase-2 (Jan 31 - Feb 4) shot tracks overlaid with pyocto events
that are still NOT flagged as shots, to see what residual contamination
remains and whether the unflagged events trace survey tracks."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent

# Load shotfiles
shots = []
for p in sorted(Path("shotfiles").glob("*_shotfile_final.txt")):
    df = pd.read_csv(p, comment="#", sep=r"\s+", engine="python",
                     names=["shotnum","date","time","sl","sln","shl","shln","wd","tag"])
    df["survey"] = p.stem.replace("_shotfile_final","")
    df["dt"] = pd.to_datetime(df.date + " " + df.time, utc=True, format="mixed", errors="coerce")
    shots.append(df[["dt","sl","sln","survey"]].dropna(subset=["dt"]))
shots = pd.concat(shots).sort_values("dt").reset_index(drop=True)

# Bathy
ds = Dataset(REPO / "notes" / "figures" / "Orca_bathymetry.nc")
lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b_plot = np.where(z_b > 2000, np.nan, z_b)
LON, LAT = np.meshgrid(lon_b, lat_b)
stations = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
ob = stations[stations.network == "ZX"]

# Plot one panel per day (Jan 31 - Feb 4), each showing:
#   - bathymetry background
#   - shot tracks for that day (small black dots)
#   - pyocto events that day, colored by flag_shot status
ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag.csv")
ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True, format="mixed", errors="coerce")
ev["date"] = ev.origin_time.dt.date

dates = pd.to_datetime(["2019-01-31", "2019-02-01", "2019-02-02", "2019-02-03", "2019-02-04"]).date

fig, axes = plt.subplots(1, 5, figsize=(28, 6))
lon_min, lon_max = -60.0, -57.5
lat_min, lat_max = -63.0, -62.0
for ax, d in zip(axes, dates):
    levels = np.arange(-2400, 200, 50)
    norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    ax.contourf(LON, LAT, z_b_plot, levels=levels, cmap=plt.cm.GnBu_r,
                norm=norm_b, extend="both")
    ax.contour(LON, LAT, z_b_plot, levels=[-1000], colors="k", linewidths=1.0, zorder=9)
    ax.scatter(ob.longitude, ob.latitude, marker="^", s=40, c="white",
               edgecolors="k", linewidths=0.7, zorder=8)

    shots_today = shots[shots.dt.dt.date == d]
    ax.scatter(shots_today.sln, shots_today.sl, s=3, c="black", alpha=0.4,
               zorder=6, label=f"shots ({len(shots_today)})")

    ev_today = ev[ev.date == d]
    flagged = ev_today[ev_today.flag_shot]
    unflagged = ev_today[~ev_today.flag_shot]
    ax.scatter(unflagged.longitude, unflagged.latitude, s=10, c="red",
               edgecolors="k", linewidths=0.2, zorder=7,
               label=f"NOT flagged ({len(unflagged)})")
    ax.scatter(flagged.longitude, flagged.latitude, s=8, c="lime",
               edgecolors="k", linewidths=0.2, zorder=7,
               label=f"flagged ({len(flagged)})")

    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
    ax.set_title(d.strftime("%Y-%m-%d"))
    ax.legend(loc="lower left", fontsize=8)

fig.suptitle("Phase-2 shot tracks (black) vs pyocto events (red = NOT flagged, lime = flagged shot)\n"
             "Red dots aligned with black tracks → missed shots",
             fontsize=13)
plt.tight_layout()
out = REPO / "notes" / "figures" / "phase2_shot_residuals.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
