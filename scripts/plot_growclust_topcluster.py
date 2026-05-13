"""Zoom in on the largest GrowClust cluster: map, depth section, time evolution."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
GROW = REPO / "catalogs" / "growclust_partial30days.csv"
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
OUT = REPO / "notes" / "figures" / "growclust_topcluster.png"

COLS = ['year','mo','dy','hr','mn','sec','evid','lat_gc','lon_gc','dep_gc',
        'mag','evid2','cid','nbranch','qID','qNN','qNX','rmsP','rmsS','eh','ez','et',
        'lat_py','lon_py','dep_py']
df = pd.read_csv(GROW, names=COLS, skiprows=1)
df = df[(df.lat_gc != df.lat_py) | (df.lon_gc != df.lon_py) |
        (df.dep_gc != df.dep_py)].copy()
def _to_dt(r):
    return pd.Timestamp(year=int(r.year), month=int(r.mo), day=int(r.dy),
                        hour=int(r.hr), minute=int(r.mn),
                        second=int(r.sec), microsecond=int((r.sec % 1) * 1e6),
                        tz="UTC")
df["dt"] = df.apply(_to_dt, axis=1)

# Identify the largest cluster by member count
top_cid = df.groupby("cid").size().idxmax()
clu = df[df.cid == top_cid].copy().reset_index(drop=True)
print(f"top cluster cid={top_cid}: {len(clu)} events")
print(f"time span: {clu.dt.min()} -> {clu.dt.max()}")

# Auto-zoom around cluster
lat_pad = max(0.02, (clu.lat_gc.max() - clu.lat_gc.min()) * 0.15)
lon_pad = max(0.04, (clu.lon_gc.max() - clu.lon_gc.min()) * 0.15)
lon_min, lon_max = clu.lon_gc.min() - lon_pad, clu.lon_gc.max() + lon_pad
lat_min, lat_max = clu.lat_gc.min() - lat_pad, clu.lat_gc.max() + lat_pad

stations = pd.read_csv(ST)

ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)

# normalize event time to days from start for color
t0 = clu.dt.min()
clu["days"] = (clu.dt - t0).dt.total_seconds() / 86400

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], width_ratios=[1.4, 1], wspace=0.22, hspace=0.28)

# (a) map view, color = time
ax = fig.add_subplot(gs[0, 0])
levels = np.arange(-2400, 200, 50)
norm = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
LON, LAT = np.meshgrid(lon_b, lat_b)
ax.contourf(LON, LAT, z_b, levels=levels, cmap=plt.cm.GnBu_r, norm=norm, extend="both")
ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
ob = stations[stations.network == "ZX"]
ax.scatter(ob.longitude, ob.latitude, marker="^", s=90, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sc = ax.scatter(clu.lon_gc, clu.lat_gc, c=clu.days, s=8, cmap="plasma",
                edgecolors="k", linewidths=0.2, alpha=0.8, zorder=7)
plt.colorbar(sc, ax=ax, label=f"days since {t0.strftime('%Y-%m-%d %H:%M')}",
             shrink=0.7, pad=0.02)
ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"Largest GrowClust cluster (cid={top_cid}) — {len(clu)} events\n"
             f"colored by time since first event")

# (b) depth vs longitude (W-E cross-section)
ax = fig.add_subplot(gs[0, 1])
ax.scatter(clu.lon_gc, clu.dep_gc, c=clu.days, cmap="plasma",
           s=8, edgecolors="k", linewidths=0.2, alpha=0.8)
ax.set_xlim(lon_min, lon_max)
ax.invert_yaxis()
ax.set_xlabel("longitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Longitude cross-section")
ax.grid(alpha=0.3)

# (c) depth vs latitude (N-S cross-section)
ax = fig.add_subplot(gs[1, 0])
ax.scatter(clu.lat_gc, clu.dep_gc, c=clu.days, cmap="plasma",
           s=8, edgecolors="k", linewidths=0.2, alpha=0.8)
ax.set_xlim(lat_min, lat_max)
ax.invert_yaxis()
ax.set_xlabel("latitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Latitude cross-section")
ax.grid(alpha=0.3)

# (d) event time series
ax = fig.add_subplot(gs[1, 1])
hours = (clu.dt - clu.dt.min()).dt.total_seconds() / 3600
ax.hist(hours, bins=60, color="steelblue", edgecolor="k", linewidth=0.3)
ax.set_xlabel(f"hours since {t0.strftime('%m-%d %H:%M UTC')}")
ax.set_ylabel("events / bin")
ax.set_title("Cluster event rate")
ax.grid(alpha=0.3)

fig.suptitle(f"Largest GrowClust cluster — {len(clu)} relocated events, "
             f"{(clu.dt.max() - clu.dt.min()).days+1}-day span",
             fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
