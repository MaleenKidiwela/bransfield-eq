"""First-30-days slice of the largest GrowClust cluster from the YEAR catalog.
Lets you compare against the 30-day partial catalog topcluster figure on equal
time footing. Writes a new filename so the other topcluster plots are preserved."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
GROW = REPO / "catalogs" / "growclust_picker_only.csv"
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
OUT = REPO / "notes" / "figures" / "growclust_topcluster_year_first30days.png"

COLS = ['year','mo','dy','hr','mn','sec','evid','lat_gc','lon_gc','dep_gc',
        'mag','evid2','cid','nbranch','qID','qNN','qNX','rmsP','rmsS','eh','ez','et',
        'lat_py','lon_py','dep_py']
df = pd.read_csv(GROW, names=COLS, skiprows=1)
df = df[(df.lat_gc != df.lat_py) | (df.lon_gc != df.lon_py) |
        (df.dep_gc != df.dep_py)].copy()

def _to_dt(r):
    return pd.Timestamp(year=int(r.year), month=int(r.mo), day=int(r.dy),
                        hour=int(r.hr), minute=int(r.mn),
                        second=int(r.sec) % 60,
                        microsecond=int((r.sec % 1) * 1e6),
                        tz="UTC") + pd.Timedelta(seconds=int(r.sec) // 60 * 60)
df["dt"] = df.apply(_to_dt, axis=1)

top_cid = df.groupby("cid").size().idxmax()
clu_all = df[df.cid == top_cid].copy().reset_index(drop=True)

# Filter to the same wall-clock window as the 30-day catalog's top cluster
# so the two figures are directly comparable. The 30-day cluster spans
# 2019-01-15 -> 2019-01-31 (16 days), so we use that exact interval here.
T_START = pd.Timestamp("2019-01-15", tz="UTC")
T_END   = pd.Timestamp("2019-02-01", tz="UTC")  # inclusive through Jan 31
clu = clu_all[(clu_all.dt >= T_START) & (clu_all.dt < T_END)].copy().reset_index(drop=True)
t0 = clu.dt.min()
print(f"top cluster cid={top_cid}: {len(clu_all)} events total in cluster")
print(f"window {T_START} -> {T_END}: {len(clu)} events  "
      f"(matches 30-day catalog time range)")

lat_pad = max(0.02, (clu.lat_gc.max() - clu.lat_gc.min()) * 0.15)
lon_pad = max(0.04, (clu.lon_gc.max() - clu.lon_gc.min()) * 0.15)
lon_min, lon_max = clu.lon_gc.min() - lon_pad, clu.lon_gc.max() + lon_pad
lat_min, lat_max = clu.lat_gc.min() - lat_pad, clu.lat_gc.max() + lat_pad

stations = pd.read_csv(ST)
ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)

clu["days"] = (clu.dt - t0).dt.total_seconds() / 86400

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], width_ratios=[1.4, 1],
                      wspace=0.22, hspace=0.28)

ax = fig.add_subplot(gs[0, 0])
levels = np.arange(-2400, 200, 50)
norm = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
LON, LAT = np.meshgrid(lon_b, lat_b)
ax.contourf(LON, LAT, z_b, levels=levels, cmap=plt.cm.GnBu_r, norm=norm, extend="both")
ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
ob = stations[stations.network == "ZX"]
ax.scatter(ob.longitude, ob.latitude, marker="^", s=90, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
d_max = float(np.ceil(clu.days.max()))
sc = ax.scatter(clu.lon_gc, clu.lat_gc, c=clu.days, s=8, cmap="plasma",
                edgecolors="k", linewidths=0.2, alpha=0.8, zorder=7,
                vmin=0, vmax=d_max)
cb = plt.colorbar(sc, ax=ax, label=f"days since {t0.strftime('%Y-%m-%d %H:%M')}",
                  shrink=0.7, pad=0.02)
cb.set_ticks(np.arange(0, d_max + 1, 5))
ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"Largest GrowClust cluster (cid={top_cid}) — Jan 15 → Jan 31, 2019\n"
             f"{len(clu)} events  (matched window of 30-day catalog)")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(clu.lon_gc, clu.dep_gc, c=clu.days, cmap="plasma",
           s=8, edgecolors="k", linewidths=0.2, alpha=0.8, vmin=0, vmax=d_max)
ax.set_xlim(lon_min, lon_max)
ax.invert_yaxis()
ax.set_xlabel("longitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Longitude cross-section")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(clu.lat_gc, clu.dep_gc, c=clu.days, cmap="plasma",
           s=8, edgecolors="k", linewidths=0.2, alpha=0.8, vmin=0, vmax=d_max)
ax.set_xlim(lat_min, lat_max)
ax.invert_yaxis()
ax.set_xlabel("latitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Latitude cross-section")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.hist(clu.dt, bins=60, color="steelblue", edgecolor="k", linewidth=0.3)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
for lbl in ax.get_xticklabels():
    lbl.set_rotation(30); lbl.set_ha("right")
ax.set_ylabel("events / bin")
ax.set_title("Cluster event rate (first 30 days)")
ax.grid(alpha=0.3)

fig.suptitle(f"Largest GrowClust cluster — Jan 15 → Jan 31 2019 slice from year-long run "
             f"({len(clu)} events; cluster has {len(clu_all):,} events total over 399 days)",
             fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
