"""Plot the Jan 1 - Mar 15 2019 GrowClust relocations (front-loaded run).
Companion to plot_growclust_relocations*.py — writes to a separate filename so
neither the 30-day nor the year-long figure is overwritten."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
GROW_CSV = REPO / "catalogs" / "growclust_jan2019.csv"
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
OUT = REPO / "notes" / "figures" / "growclust_relocations_jan2019_prewindow.png"

COLS = ['year','mo','dy','hr','mn','sec','evid','lat_gc','lon_gc','dep_gc',
        'mag','evid2','cid','nbranch','qID','qNN','qNX','rmsP','rmsS','eh','ez','et',
        'lat_py','lon_py','dep_py']
df = pd.read_csv(GROW_CSV, names=COLS, skiprows=1)
moved = (df.lat_gc != df.lat_py) | (df.lon_gc != df.lon_py) | (df.dep_gc != df.dep_py)
print(f"total: {len(df):,}, relocated: {moved.sum():,}")

stations = pd.read_csv(ST)
ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)

lon_min, lon_max = -58.7, -58.2
lat_min, lat_max = -62.55, -62.35

def in_zoom(la, lo):
    return (lo >= lon_min) & (lo <= lon_max) & (la >= lat_min) & (la <= lat_max)

dz = df[moved & in_zoom(df.lat_gc, df.lon_gc)].reset_index(drop=True)
print(f"relocated in zoom: {len(dz):,}")

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.18)

def plot_bathy(ax):
    levels = np.arange(-2400, 200, 50)
    cmap = plt.cm.GnBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    LON, LAT = np.meshgrid(lon_b, lat_b)
    ax.contourf(LON, LAT, z_b, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))

ob = stations[stations.network == "ZX"]

ax = fig.add_subplot(gs[0, 0])
plot_bathy(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sc1 = ax.scatter(dz.lon_py, dz.lat_py, c=dz.dep_py, s=4, cmap="magma_r",
                 vmin=0, vmax=20, edgecolors="none", alpha=0.6, zorder=7)
ax.set_title(f"Pyocto locations (1.5 km octree grid) — {len(dz):,} events")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc1, ax=ax, label="depth (km)", shrink=0.7, pad=0.02)

ax = fig.add_subplot(gs[0, 1])
plot_bathy(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sc2 = ax.scatter(dz.lon_gc, dz.lat_gc, c=dz.dep_gc, s=4, cmap="magma_r",
                 vmin=0, vmax=20, edgecolors="none", alpha=0.6, zorder=7)
ax.set_title(f"GrowClust relocated (unlocked) — same {len(dz):,} events")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc2, ax=ax, label="depth (km)", shrink=0.7, pad=0.02)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(dz.lon_py, dz.dep_py, c="lightcoral", s=4, alpha=0.4,
           label="pyocto (gridded)")
ax.scatter(dz.lon_gc, dz.dep_gc, c="steelblue", s=4, alpha=0.5,
           label="GrowClust (relocated)")
ax.set_xlim(lon_min, lon_max); ax.set_ylim(20, 0)
ax.set_xlabel("longitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Longitude cross-section (depth view)")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
clust = df[df.nbranch >= 2].nbranch
ax.hist(clust, bins=np.logspace(0, np.log10(clust.max()+1), 30),
        edgecolor="k", alpha=0.7, color="steelblue")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("cluster size (events)")
ax.set_ylabel("count")
ax.set_title(f"GrowClust cluster sizes — {(df.nbranch>=2).sum():,} events in multi-event clusters")
ax.grid(alpha=0.3, which="both")

fig.suptitle(f"GrowClust relocations — Jan 1 → Jan 30, 2019 (pre-windowed run)\n"
             f"{moved.sum():,} of {len(df):,} events relocated; un-gridded relative positions",
             fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
