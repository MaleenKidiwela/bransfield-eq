"""Zoomed-in map of 2019-01-17 pyocto events over GEBCO bathymetry, Orca region."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
from pyproj import CRS, Transformer

REPO = Path(__file__).resolve().parent.parent
EV = REPO / "catalogs" / "pyocto_events_1day_2019-01-17.csv"
ST = REPO / "catalogs" / "station_geometry.csv"
MAN = REPO / "catalogs" / "manual_picks.csv"
BATHY_LOCAL = REPO / "notes" / "figures" / "Orca_bathymetry.nc"  # MGDS Orca high-res
BATHY_FALLBACK = Path("/home/jovyan/ooi/rsn_cabled/SummerSchool2025/global_ocean_data/GEBCO_2023.nc")
OUT = REPO / "notes" / "figures" / "pyocto_1day_2019-01-17_orca.png"

# Zoom window — tight on Orca caldera (~-58.43°, -62.45°)
lon_min, lon_max = -58.7, -58.2
lat_min, lat_max = -62.55, -62.35

# --- load bathymetry subset ---
if BATHY_LOCAL.exists():
    ds = Dataset(BATHY_LOCAL)
    lat2 = ds.variables["latitude"][:]
    lon2 = ds.variables["longitude"][:]
    z = ds.variables["data"][:]
    ds.close()
    # mask spurious extreme values (data file has a few glitch cells > 2000 m)
    z = np.where(z > 2000, np.nan, z)
    bathy_source = "MGDS Orca high-res"
else:
    ds = Dataset(BATHY_FALLBACK)
    lat_v = ds.variables.get("lat") or ds.variables.get("latitude")
    lon_v = ds.variables.get("lon") or ds.variables.get("longitude")
    z_v = ds.variables.get("elevation") or ds.variables.get("z")
    lat_all = lat_v[:]; lon_all = lon_v[:]
    jmin, jmax = np.searchsorted(lat_all, [lat_min, lat_max])
    imin, imax = np.searchsorted(lon_all, [lon_min, lon_max])
    z = z_v[jmin:jmax, imin:imax]
    lon2 = lon_all[imin:imax]; lat2 = lat_all[jmin:jmax]
    ds.close()
    bathy_source = "GEBCO 2023"
print(f"bathymetry source: {bathy_source}, shape {z.shape}, "
      f"depth {np.nanmin(z):.0f} to {np.nanmax(z):.0f} m")

# --- load events ---
stations = pd.read_csv(ST)
crs = CRS.from_proj4(f"+proj=tmerc +lat_0={stations.latitude.mean()} "
                     f"+lon_0={stations.longitude.mean()} +ellps=WGS84")
inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
e = pd.read_csv(EV)
elons, elats = inv.transform(e.x.values * 1000, e.y.values * 1000)
e["lon"] = elons; e["lat"] = elats
e["t"] = pd.to_datetime(e.time, unit="s", utc=True)
in_zoom = (e.lon >= lon_min) & (e.lon <= lon_max) & (e.lat >= lat_min) & (e.lat <= lat_max)
ez = e[in_zoom].reset_index(drop=True)
print(f"events in zoom window: {len(ez)} / {len(e)}")

# manual catalog for the day
man = pd.read_csv(MAN)
man = man[(man.source_file == "nllmaleen_mag07_202210.out")].copy()
man["t"] = pd.to_datetime(man.origin_time, utc=True)
man_day = man[man.t.dt.date == pd.Timestamp("2019-01-17").date()]
man_ev = man_day.drop_duplicates(subset=["event_id"])
# manual events don't have lat/lon -- skip plotting them on the map but show count

# --- plot ---
fig, ax = plt.subplots(figsize=(11, 8))

# bathymetry: blues below 0, greens above
levels = np.arange(-2400, 200, 50)
cmap = plt.cm.GnBu_r
norm = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
LON, LAT = np.meshgrid(lon2, lat2)
cf = ax.contourf(LON, LAT, z, levels=levels, cmap=cmap, norm=norm, extend="both")
# contour lines at 500 m
ax.contour(LON, LAT, z, levels=np.arange(-2400, 0, 250),
           colors="0.3", linewidths=0.3, alpha=0.5)
# coastline (0 m)
ax.contour(LON, LAT, z, levels=[0], colors="k", linewidths=0.8)

# stations
ob = stations[stations.network == "ZX"]
land = stations[stations.network != "ZX"]
ax.scatter(ob.longitude, ob.latitude, marker="^", s=110, c="white",
           edgecolors="k", linewidths=1.2, zorder=8, label=f"OBS ({len(ob)})")
ax.scatter(land.longitude, land.latitude, marker="^", s=110, c="red",
           edgecolors="k", linewidths=0.8, zorder=8, label=f"Land ({len(land)})")

# events colored by depth
sc = ax.scatter(ez.lon, ez.lat, c=ez.z, s=28, cmap="magma_r",
                vmin=0, vmax=20, edgecolors="k", linewidths=0.4,
                alpha=0.85, zorder=7,
                label=f"pyocto events ({len(ez)})")

cb_bathy = plt.colorbar(cf, ax=ax, label="elevation (m)",
                        ticks=np.arange(-2400, 200, 400), shrink=0.7, pad=0.02)
cax2 = fig.add_axes([0.86, 0.15, 0.022, 0.3])
cb_ev = plt.colorbar(sc, cax=cax2, label="event depth (km)")

ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"Orca Volcano region — pyocto events 2019-01-17  ({bathy_source})\n"
             f"{len(ez)} pyocto events in zoom; manual mag07: {len(man_ev)} total that day "
             f"(13.4 min wall, 8 threads)")
ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.25)

plt.subplots_adjust(right=0.84)
plt.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
