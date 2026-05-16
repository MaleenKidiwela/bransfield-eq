"""Plot hypoDD relocations on Orca bathymetry. Companion to
plot_growclust_relocations*.py — same panel layout for direct visual comparison.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="jan2019")
ap.add_argument("--title-extra", default="")
args = ap.parse_args()
HD_CSV = REPO / "catalogs" / f"hypodd_{args.label}.csv"
OUT = REPO / "notes" / "figures" / f"hypodd_relocations_{args.label}.png"

df = pd.read_csv(HD_CSV)
print(f"loaded {len(df):,} hypoDD events")

stations = pd.read_csv(ST)
ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)

lon_min, lon_max = -58.7, -58.2
lat_min, lat_max = -62.55, -62.35

def in_zoom(la, lo):
    return (lo >= lon_min) & (lo <= lon_max) & (la >= lat_min) & (la <= lat_max)

dz = df[in_zoom(df.lat, df.lon)].reset_index(drop=True)
print(f"in zoom: {len(dz):,}")

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
sc1 = ax.scatter(dz.lon, dz.lat, c=dz.dep, s=4, cmap="magma_r",
                 vmin=0, vmax=20, edgecolors="none", alpha=0.6, zorder=7)
ax.set_title(f"hypoDD locations — {len(dz):,} events in zoom")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc1, ax=ax, label="depth (km)", shrink=0.7, pad=0.02)

# Color by cluster id
ax = fig.add_subplot(gs[0, 1])
plot_bathy(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sc2 = ax.scatter(dz.lon, dz.lat, c=dz.cid, s=4, cmap="tab20",
                 edgecolors="none", alpha=0.7, zorder=7)
ax.set_title(f"colored by cluster id (cid)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc2, ax=ax, label="cid", shrink=0.7, pad=0.02)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(dz.lon, dz.dep, c="steelblue", s=4, alpha=0.5)
ax.set_xlim(lon_min, lon_max); ax.set_ylim(20, 0)
ax.set_xlabel("longitude"); ax.set_ylabel("depth (km)")
ax.set_title("Longitude cross-section")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
csize = df.groupby("cid").size()
multi = csize[csize >= 2]
if len(multi) > 0:
    ax.hist(multi, bins=np.logspace(0, np.log10(multi.max()+1), 30),
            edgecolor="k", alpha=0.7, color="steelblue")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"hypoDD cluster sizes — {len(multi):,} multi-event clusters")
else:
    ax.text(0.5, 0.5, "no multi-event clusters", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("hypoDD cluster sizes")
ax.set_xlabel("cluster size (events)")
ax.set_ylabel("count")
ax.grid(alpha=0.3, which="both")

fig.suptitle(f"hypoDD relocations — label {args.label}  {args.title_extra}\n"
             f"{len(df):,} relocated events",
             fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
