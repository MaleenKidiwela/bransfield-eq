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
ap.add_argument("--label", default="stage_b",
                help="Reads catalogs/hypodd_<label>.csv -> writes notes/figures/hypodd_relocations_<label>.png")
ap.add_argument("--title-extra", default="")
ap.add_argument("--out-subdir", default="",
                help="Subdirectory under notes/figures/ to write into.")
args = ap.parse_args()
HD_CSV = REPO / "catalogs" / f"hypodd_{args.label}.csv"
out_dir = REPO / "notes" / "figures" / args.out_subdir if args.out_subdir else REPO / "notes" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
OUT = out_dir / f"hypodd_relocations_{args.label}.png"

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


def overlay_1000m_contour(ax, z_b, lat_b, lon_b):
    """Draw the -1000 m bathymetric contour (seamount edifice) ON TOP of
    the event scatter (zorder 10 vs event-scatter zorder 7)."""
    LON, LAT = np.meshgrid(lon_b, lat_b)
    ax.contour(LON, LAT, z_b, levels=[-1000], colors="k",
               linewidths=1.2, linestyles="-", zorder=10)

ob = stations[stations.network == "ZX"]

ax = fig.add_subplot(gs[0, 0])
plot_bathy(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sc1 = ax.scatter(dz.lon, dz.lat, c=dz.dep, s=4, cmap="magma_r",
                 vmin=0, vmax=20, edgecolors="none", alpha=0.6, zorder=7)
overlay_1000m_contour(ax, z_b, lat_b, lon_b)
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
overlay_1000m_contour(ax, z_b, lat_b, lon_b)
ax.set_title(f"colored by cluster id (cid)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc2, ax=ax, label="cid", shrink=0.7, pad=0.02)

# Cross-section colored by event origin time, with seafloor profile overlaid
ax = fig.add_subplot(gs[1, 0])
t = pd.to_datetime(dict(year=dz.yr, month=dz.mo, day=dz.dy,
                        hour=dz.hr.clip(0,23), minute=dz.mi.clip(0,59)),
                   utc=True, errors="coerce") + pd.to_timedelta(dz.sc, unit="s")
days = (t - t.min()).dt.total_seconds() / 86400.0
sc3 = ax.scatter(dz.lon, dz.dep, c=days, s=4, cmap="viridis", alpha=0.7, zorder=5)
# Seafloor profile across the zoom latitude strip
_in_lat = (lat_b >= lat_min) & (lat_b <= lat_max)
_in_lon = (lon_b >= lon_min) & (lon_b <= lon_max)
_sub = z_b[_in_lat][:, _in_lon]
_lon_sf = lon_b[_in_lon]
_sf_km = np.where(-_sub / 1000.0 > 0, -_sub / 1000.0, np.nan)
_sf_min = np.nanmin(_sf_km, axis=0)
_sf_max = np.nanmax(_sf_km, axis=0)
_sf_med = np.nanmedian(_sf_km, axis=0)
ax.fill_between(_lon_sf, _sf_min, _sf_max, color="0.7", alpha=0.5,
                label="seafloor (lat strip range)", zorder=3)
ax.plot(_lon_sf, _sf_med, color="0.25", linewidth=1.2,
        label="seafloor (median)", zorder=4)
ax.set_xlim(lon_min, lon_max); ax.set_ylim(20, 0)
ax.set_xlabel("longitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title(f"Longitude cross-section + seafloor (colored by days since {t.min().strftime('%Y-%m-%d')})")
ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
plt.colorbar(sc3, ax=ax, label="days since start", shrink=0.7, pad=0.02)

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
