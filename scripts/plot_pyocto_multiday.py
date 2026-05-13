"""Plot all pyocto daily-chunk events accumulated so far over Orca bathymetry."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
from pyproj import CRS, Transformer

REPO = Path(__file__).resolve().parent.parent
DAILY = REPO / "catalogs" / "pyocto_daily"
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
OUT = REPO / "notes" / "figures" / "pyocto_multiday.png"

# Load all daily events
files = sorted(glob.glob(str(DAILY / "events_*.csv")))
print(f"loading {len(files)} day files ...")
dfs = []
for f in files:
    tag = Path(f).stem.replace("events_", "")
    if Path(f).stat().st_size < 10:   # empty/header-only file from low-station days
        continue
    try:
        d = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        continue
    if d.empty: continue
    d["day"] = tag
    dfs.append(d)
e = pd.concat(dfs, ignore_index=True)
e["t"] = pd.to_datetime(e.time, unit="s", utc=True)
print(f"total events: {len(e):,}  from {len(files)} days  ({e.day.nunique()} days with events)")

# Project to lat/lon
stations = pd.read_csv(ST)
crs = CRS.from_proj4(f"+proj=tmerc +lat_0={stations.latitude.mean()} "
                     f"+lon_0={stations.longitude.mean()} +ellps=WGS84")
inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
elons, elats = inv.transform(e.x.values * 1000, e.y.values * 1000)
e["lon"] = elons; e["lat"] = elats

# Bathymetry
ds = Dataset(BATHY)
lat_b = ds.variables["latitude"][:]
lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]
ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)

# Zoom out — full Orca / Bransfield basin region
lon_min, lon_max = -59.19, -57.69
lat_min, lat_max = -62.79, -62.13
in_zoom = (e.lon >= lon_min) & (e.lon <= lon_max) & (e.lat >= lat_min) & (e.lat <= lat_max)
ez = e[in_zoom].reset_index(drop=True)
print(f"in zoom: {len(ez):,} / {len(e):,}")

fig, axes = plt.subplots(2, 2, figsize=(14, 11),
                         gridspec_kw={"width_ratios": [1.4, 1], "height_ratios": [1.4, 1]})

# (a) map view with bathymetry
ax = axes[0, 0]
levels = np.arange(-2400, 200, 50)
cmap_b = plt.cm.GnBu_r
norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
LON, LAT = np.meshgrid(lon_b, lat_b)
cf = ax.contourf(LON, LAT, z_b, levels=levels, cmap=cmap_b, norm=norm_b, extend="both")
ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
ob = stations[stations.network == "ZX"]
land = stations[stations.network != "ZX"]
ax.scatter(ob.longitude, ob.latitude, marker="^", s=80, c="white",
           edgecolors="k", linewidths=1.0, zorder=8, label=f"OBS ({len(ob)})")
ax.scatter(land.longitude, land.latitude, marker="^", s=70, c="red",
           edgecolors="k", linewidths=0.6, zorder=8, label=f"Land ({len(land)})")
sc = ax.scatter(ez.lon, ez.lat, c=ez.z, s=3, cmap="magma_r",
                vmin=0, vmax=20, edgecolors="none",
                alpha=0.7, zorder=7,
                label=f"events ({len(ez)})")
plt.colorbar(cf, ax=ax, label="seafloor elev. (m)", shrink=0.65, pad=0.02,
             ticks=np.arange(-2400, 200, 400))
cax = fig.add_axes([0.45, 0.55, 0.012, 0.25])
plt.colorbar(sc, cax=cax, label="event depth (km)")
ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"pyocto events around Orca — {e.day.nunique()} days "
             f"({e.day.min()} → {e.day.max()})")
ax.legend(loc="lower left", fontsize=9)

# (b) events per day
ax = axes[0, 1]
per_day = e.groupby("day").size().sort_index()
ax.bar(range(len(per_day)), per_day.values, color="steelblue", edgecolor="k", linewidth=0.4)
ax.set_xticks(range(len(per_day)))
ax.set_xticklabels(per_day.index, rotation=70, fontsize=7)
ax.set_ylabel("events per day")
ax.set_title(f"daily counts ({per_day.sum():,} total)")
ax.grid(alpha=0.3, axis="y")
for i, v in enumerate(per_day.values):
    ax.text(i, v + per_day.max()*0.01, str(v), ha="center", fontsize=6.5)

# (c) depth histogram
ax = axes[1, 0]
in_net = (e.x.abs() < 30) & (e.y.abs() < 30)
ax.hist(e.z, bins=np.arange(0, 41, 1.25), edgecolor="k", alpha=0.5,
        label=f"all ({len(e)})")
ax.hist(e.z[in_net], bins=np.arange(0, 41, 1.25), edgecolor="k", alpha=0.7,
        color="C2", label=f"in-network ({in_net.sum()})")
ax.axvspan(35, 40, alpha=0.15, color="red", label="model floor")
ax.set_xlabel("depth below sea level (km)")
ax.set_ylabel("count")
ax.set_title(f"depth distribution (in-net median {e.z[in_net].median():.1f} km)")
ax.legend(); ax.grid(alpha=0.3)

# (d) events per hour-of-day (all 21 days)
ax = axes[1, 1]
hod = e.t.dt.hour
ax.hist(hod, bins=np.arange(0, 25, 1), edgecolor="k", color="C3", alpha=0.7)
ax.set_xlabel("hour of day (UTC)")
ax.set_ylabel("events")
ax.set_title(f"hour-of-day distribution")
ax.grid(alpha=0.3); ax.set_xticks(range(0, 25, 4))

plt.tight_layout()
plt.subplots_adjust(right=0.97)
plt.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
