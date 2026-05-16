"""High-certainty subset of year-long GrowClust relocations.

Note: GrowClust was run with nboot=0 so eh/ez/et columns are all -1. Quality is
filtered by the cross-correlation count (qID), nearest-neighbour count (nbranch),
and pick-residual RMS (rmsP, rmsS) -- standard practice when bootstrap errors
are unavailable.

Filter (default): qID >= 10  AND  nbranch >= 5  AND  rmsP < 0.3  AND  rmsS < 0.3
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
GROW = REPO / "catalogs" / "growclust_picker_only.csv"
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
OUT = REPO / "notes" / "figures" / "growclust_relocations_year_highQ.png"

Q_MIN = 10        # min cross-correlation observations
NB_MIN = 5        # min cluster size
RMS_MAX = 0.30    # max P and S RMS residual (s)

COLS = ['year','mo','dy','hr','mn','sec','evid','lat_gc','lon_gc','dep_gc',
        'mag','evid2','cid','nbranch','qID','qNN','qNX','rmsP','rmsS','eh','ez','et',
        'lat_py','lon_py','dep_py']
df = pd.read_csv(GROW, names=COLS, skiprows=1)
moved = (df.lat_gc != df.lat_py) | (df.lon_gc != df.lon_py) | (df.dep_gc != df.dep_py)
r = df[moved].copy()
hq = r[(r.qID >= Q_MIN) & (r.nbranch >= NB_MIN) &
       (r.rmsP < RMS_MAX) & (r.rmsS < RMS_MAX)].reset_index(drop=True)
print(f"total relocated: {len(r):,}")
print(f"high-certainty (qID>={Q_MIN}, nbranch>={NB_MIN}, rmsP/S<{RMS_MAX}): "
      f"{len(hq):,} ({len(hq)/len(r)*100:.1f}%)")

lon_min, lon_max = -58.7, -58.2
lat_min, lat_max = -62.55, -62.35

def in_zoom(la, lo):
    return (lo >= lon_min) & (lo <= lon_max) & (la >= lat_min) & (la <= lat_max)

dz = hq[in_zoom(hq.lat_gc, hq.lon_gc)].reset_index(drop=True)
print(f"high-certainty in zoom: {len(dz):,}")

stations = pd.read_csv(ST)
ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)

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
sc1 = ax.scatter(dz.lon_py, dz.lat_py, c=dz.dep_py, s=6, cmap="magma_r",
                 vmin=0, vmax=20, edgecolors="none", alpha=0.7, zorder=7)
ax.set_title(f"Pyocto locations (gridded) — high-Q subset: {len(dz):,} events")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc1, ax=ax, label="depth (km)", shrink=0.7, pad=0.02)

ax = fig.add_subplot(gs[0, 1])
plot_bathy(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sc2 = ax.scatter(dz.lon_gc, dz.lat_gc, c=dz.dep_gc, s=6, cmap="magma_r",
                 vmin=0, vmax=20, edgecolors="none", alpha=0.7, zorder=7)
ax.set_title(f"GrowClust relocated — same {len(dz):,} high-Q events")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc2, ax=ax, label="depth (km)", shrink=0.7, pad=0.02)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(dz.lon_py, dz.dep_py, c="lightcoral", s=6, alpha=0.5,
           label="pyocto (gridded)")
ax.scatter(dz.lon_gc, dz.dep_gc, c="steelblue", s=6, alpha=0.7,
           label="GrowClust (relocated)")
ax.set_xlim(lon_min, lon_max); ax.set_ylim(20, 0)
ax.set_xlabel("longitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Longitude cross-section")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
clust = hq[hq.nbranch >= 2].nbranch
ax.hist(clust, bins=np.logspace(0, np.log10(clust.max()+1), 30),
        edgecolor="k", alpha=0.7, color="steelblue")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("cluster size (events)")
ax.set_ylabel("count")
ax.set_title(f"GrowClust cluster sizes (high-Q only)")
ax.grid(alpha=0.3, which="both")

fig.suptitle(
    f"GrowClust year-long: high-certainty subset  "
    f"(qID≥{Q_MIN}, nbranch≥{NB_MIN}, rmsP/S<{RMS_MAX} s)\n"
    f"{len(hq):,} of {len(r):,} relocated events kept "
    f"({len(hq)/len(r)*100:.1f}%); {len(dz):,} in zoom",
    fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
