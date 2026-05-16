"""Date-bounded comparison: only events from Jan 15 -> Jan 31 2019 (the top-cluster
swarm time window) are compared between the 30-day and year-long GrowClust runs.
Writes to a new figure so the wider-scope compare plot is preserved."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
GROW_30 = REPO / "catalogs" / "growclust_partial30days.csv"
GROW_YR = REPO / "catalogs" / "growclust_picker_only.csv"
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
OUT = REPO / "notes" / "figures" / "growclust_30day_vs_year_compare_swarm.png"

T_START = pd.Timestamp("2019-01-15", tz="UTC")
T_END   = pd.Timestamp("2019-02-01", tz="UTC")

COLS = ['year','mo','dy','hr','mn','sec','evid','lat_gc','lon_gc','dep_gc',
        'mag','evid2','cid','nbranch','qID','qNN','qNX','rmsP','rmsS','eh','ez','et',
        'lat_py','lon_py','dep_py']

def load(path):
    df = pd.read_csv(path, names=COLS, skiprows=1)
    base = pd.to_datetime(dict(
        year=df.year, month=df.mo, day=df.dy,
        hour=df.hr.clip(0, 23), minute=df.mn.clip(0, 59)),
        utc=True, errors="coerce")
    df["t"] = base + pd.to_timedelta(df.sec, unit="s")
    return df.dropna(subset=["t"]).reset_index(drop=True)

print(f"Restricting both catalogs to {T_START.date()} -> {T_END.date()} ...")
d30 = load(GROW_30)
dyr = load(GROW_YR)
d30 = d30[(d30.t >= T_START) & (d30.t < T_END)].reset_index(drop=True)
dyr = dyr[(dyr.t >= T_START) & (dyr.t < T_END)].reset_index(drop=True)
print(f"  30-day: {len(d30):,} events in window")
print(f"  yearly: {len(dyr):,} events in window")

d30s = d30.sort_values("t").reset_index(drop=True)
dyrs = dyr.sort_values("t").reset_index(drop=True)
matched = pd.merge_asof(d30s[["t","lat_gc","lon_gc","dep_gc","nbranch"]]
                        .rename(columns={"lat_gc":"lat_30","lon_gc":"lon_30",
                                         "dep_gc":"dep_30","nbranch":"nb_30"}),
                        dyrs[["t","lat_gc","lon_gc","dep_gc","nbranch"]]
                        .rename(columns={"lat_gc":"lat_yr","lon_gc":"lon_yr",
                                         "dep_gc":"dep_yr","nbranch":"nb_yr"}),
                        on="t", direction="nearest",
                        tolerance=pd.Timedelta("500ms")).dropna()
print(f"  matched: {len(matched):,} events (within 0.5 s)")

lat0 = matched.lat_30.mean()
R = 6371.0
dlat_km = np.radians(matched.lat_yr - matched.lat_30) * R
dlon_km = np.radians(matched.lon_yr - matched.lon_30) * R * np.cos(np.radians(lat0))
matched["dh_km"] = np.hypot(dlat_km, dlon_km)
matched["dz_km"] = matched.dep_yr - matched.dep_30

def pct(arr, qs=(50, 75, 90, 95)):
    return ", ".join(f"p{q}={np.percentile(arr, q):.2f}" for q in qs)

print("\nHorizontal shift |Δh| (km):", pct(matched.dh_km))
print("Vertical |Δz| (km):", pct(np.abs(matched.dz_km)))
print(f"Events moved >5 km horiz: {(matched.dh_km > 5).sum():,} "
      f"({(matched.dh_km > 5).mean()*100:.1f}%)")

stations = pd.read_csv(ST)
ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)

# Zoom tight on the swarm core (ignore far-flung outliers that auto-zoom would
# include). Box matches the 30-day topcluster footprint.
lon_min, lon_max = -58.65, -58.15
lat_min, lat_max = -62.55, -62.30
ob = stations[stations.network == "ZX"]

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.28, wspace=0.22)

def bathy_axes(ax):
    levels = np.arange(-2400, 200, 50)
    cmap = plt.cm.GnBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
    LON, LAT = np.meshgrid(lon_b, lat_b)
    ax.contourf(LON, LAT, z_b, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
    ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean([lat_min, lat_max]))))

ax = fig.add_subplot(gs[0, 0])
bathy_axes(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=60, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
ax.quiver(matched.lon_30, matched.lat_30,
          matched.lon_yr - matched.lon_30, matched.lat_yr - matched.lat_30,
          matched.dh_km, cmap="viridis",
          angles="xy", scale_units="xy", scale=1, width=0.0015, zorder=7,
          clim=(0, np.percentile(matched.dh_km, 95)))
ax.scatter(matched.lon_30, matched.lat_30, s=3, c="red", alpha=0.4, zorder=6,
           label=f"30-day position ({len(matched):,})")
ax.scatter(matched.lon_yr, matched.lat_yr, s=3, c="blue", alpha=0.4, zorder=6,
           label="yearly position")
ax.set_title("Swarm-only shift: 30-day -> yearly (arrows colored by |Δh| km)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.legend(loc="lower left", fontsize=8)

ax = fig.add_subplot(gs[0, 1])
hi = np.percentile(matched.dh_km, 99)
bins = np.linspace(0, hi, 50)
ax.hist(matched.dh_km, bins=bins, color="steelblue", edgecolor="k", alpha=0.8)
ax.axvline(np.median(matched.dh_km), color="red", linewidth=1.5,
           label=f"median {np.median(matched.dh_km):.2f} km")
ax.axvline(np.percentile(matched.dh_km, 95), color="orange", linewidth=1.5,
           linestyle="--", label=f"p95 {np.percentile(matched.dh_km, 95):.2f} km")
ax.set_xlabel("|Δh| horizontal shift (km)"); ax.set_ylabel("count")
ax.set_title(f"Horizontal shift distribution  (n={len(matched):,})")
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 0])
lim = np.percentile(np.abs(matched.dz_km), 99)
bins = np.linspace(-lim, lim, 50)
ax.hist(matched.dz_km, bins=bins, color="darkorange", edgecolor="k", alpha=0.8)
ax.axvline(0, color="k", linewidth=0.6)
ax.axvline(np.median(matched.dz_km), color="red", linewidth=1.5,
           label=f"median {np.median(matched.dz_km):.2f} km")
ax.set_xlabel("Δdepth (yearly − 30-day) (km)"); ax.set_ylabel("count")
ax.set_title("Depth shift distribution")
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(matched.dh_km, matched.dz_km, c=matched.nb_yr, s=3,
                cmap="magma_r", alpha=0.6,
                norm=mcolors.LogNorm(vmin=1, vmax=max(2, matched.nb_yr.max())))
ax.set_xlabel("|Δh| (km)"); ax.set_ylabel("Δz (km)")
ax.set_title("Shift correlation (color = yearly cluster size)")
ax.grid(alpha=0.3); ax.axhline(0, color="k", linewidth=0.4)
plt.colorbar(sc, ax=ax, label="nbranch (yearly)")

fig.suptitle(
    f"GrowClust: 30-day vs year-long relocation — swarm window "
    f"{T_START.date()} → {T_END.date()}\n"
    f"{len(matched):,} events matched (≤0.5 s).  "
    f"Median |Δh| = {np.median(matched.dh_km):.2f} km,  "
    f"p95 |Δh| = {np.percentile(matched.dh_km, 95):.2f} km",
    fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"\nwrote {OUT}")
