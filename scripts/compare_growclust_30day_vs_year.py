"""Compare GrowClust relocations from the 30-day partial catalog vs the full-year
catalog. Matches events by origin time (yr/mo/dy/hr/mn/sec within 0.5 s) and
reports the spatial shift between the two relocations for each matched event."""
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
OUT = REPO / "notes" / "figures" / "growclust_30day_vs_year_compare.png"

COLS = ['year','mo','dy','hr','mn','sec','evid','lat_gc','lon_gc','dep_gc',
        'mag','evid2','cid','nbranch','qID','qNN','qNX','rmsP','rmsS','eh','ez','et',
        'lat_py','lon_py','dep_py']

def load(path):
    df = pd.read_csv(path, names=COLS)
    # Build origin time defensively: GrowClust occasionally emits sec >= 60 from
    # rounding, which breaks naive pd.to_datetime. Construct from yr/mo/dy and
    # add seconds-of-day as a Timedelta.
    base = pd.to_datetime(dict(
        year=df.year, month=df.mo, day=df.dy,
        hour=df.hr.clip(0, 23), minute=df.mn.clip(0, 59)),
        utc=True, errors="coerce")
    df["t"] = base + pd.to_timedelta(df.sec, unit="s")
    return df.dropna(subset=["t"]).reset_index(drop=True)

print("Loading catalogs ...")
d30 = load(GROW_30)
dyr = load(GROW_YR)
print(f"  30-day: {len(d30):,} events")
print(f"  yearly: {len(dyr):,} events")

# Match by origin time within 0.5 s. Use merge_asof for a fast nearest-time join.
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
print(f"  matched: {len(matched):,} events (within 0.5 s origin-time tolerance)")

# Horizontal shift via flat-Earth at the matched-event centroid
lat0 = matched.lat_30.mean()
R = 6371.0
dlat_km = np.radians(matched.lat_yr - matched.lat_30) * R
dlon_km = np.radians(matched.lon_yr - matched.lon_30) * R * np.cos(np.radians(lat0))
matched["dh_km"] = np.hypot(dlat_km, dlon_km)
matched["dz_km"] = matched.dep_yr - matched.dep_30

def pct(arr, qs=(50, 75, 90, 95)):
    return ", ".join(f"p{q}={np.percentile(arr, q):.2f}" for q in qs)

print("\nHorizontal shift |Δh| (km):", pct(matched.dh_km))
print("Vertical shift Δz (km, signed):",
      f"median={np.median(matched.dz_km):.2f}, "
      f"|Δz| p75={np.percentile(np.abs(matched.dz_km), 75):.2f}, "
      f"|Δz| p95={np.percentile(np.abs(matched.dz_km), 95):.2f}")
print(f"Events that moved >5 km horiz: {(matched.dh_km > 5).sum():,} "
      f"({(matched.dh_km > 5).mean()*100:.1f}%)")

# ----- figure -----
stations = pd.read_csv(ST)
ds = Dataset(BATHY); lat_b = ds.variables["latitude"][:]; lon_b = ds.variables["longitude"][:]
z_b = ds.variables["data"][:]; ds.close()
z_b = np.where(z_b > 2000, np.nan, z_b)
lon_min, lon_max = -58.7, -58.2
lat_min, lat_max = -62.55, -62.35
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

# (a) arrow map: 30-day -> yearly for each matched event in zoom
ax = fig.add_subplot(gs[0, 0])
bathy_axes(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=60, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
m = matched[(matched.lon_30 >= lon_min) & (matched.lon_30 <= lon_max) &
            (matched.lat_30 >= lat_min) & (matched.lat_30 <= lat_max)]
ax.quiver(m.lon_30, m.lat_30, m.lon_yr - m.lon_30, m.lat_yr - m.lat_30,
          m.dh_km, cmap="viridis",
          angles="xy", scale_units="xy", scale=1, width=0.0015, zorder=7,
          clim=(0, np.percentile(m.dh_km, 95)))
ax.scatter(m.lon_30, m.lat_30, s=2, c="red", alpha=0.4, zorder=6,
           label=f"30-day position ({len(m):,})")
ax.scatter(m.lon_yr, m.lat_yr, s=2, c="blue", alpha=0.4, zorder=6,
           label="yearly position")
ax.set_title("Position shift: 30-day -> yearly (arrows colored by |Δh| km)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.legend(loc="lower left", fontsize=8)

# (b) histogram of horizontal shift
ax = fig.add_subplot(gs[0, 1])
bins = np.linspace(0, np.percentile(matched.dh_km, 99), 60)
ax.hist(matched.dh_km, bins=bins, color="steelblue", edgecolor="k", alpha=0.8)
ax.axvline(np.median(matched.dh_km), color="red", linewidth=1.5,
           label=f"median {np.median(matched.dh_km):.2f} km")
ax.axvline(np.percentile(matched.dh_km, 95), color="orange", linewidth=1.5,
           linestyle="--", label=f"p95 {np.percentile(matched.dh_km, 95):.2f} km")
ax.set_xlabel("|Δh| horizontal shift (km)"); ax.set_ylabel("count")
ax.set_title(f"Horizontal shift distribution  (n={len(matched):,} matched)")
ax.legend(); ax.grid(alpha=0.3)

# (c) depth shift histogram (signed)
ax = fig.add_subplot(gs[1, 0])
lim = np.percentile(np.abs(matched.dz_km), 99)
bins = np.linspace(-lim, lim, 60)
ax.hist(matched.dz_km, bins=bins, color="darkorange", edgecolor="k", alpha=0.8)
ax.axvline(0, color="k", linewidth=0.6)
ax.axvline(np.median(matched.dz_km), color="red", linewidth=1.5,
           label=f"median {np.median(matched.dz_km):.2f} km")
ax.set_xlabel("Δdepth (yearly − 30-day) (km)"); ax.set_ylabel("count")
ax.set_title("Depth shift distribution")
ax.legend(); ax.grid(alpha=0.3)

# (d) horizontal vs depth scatter, colored by cluster size in the yearly run
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(matched.dh_km, matched.dz_km, c=matched.nb_yr, s=3,
                cmap="magma_r", alpha=0.6,
                norm=mcolors.LogNorm(vmin=1, vmax=max(2, matched.nb_yr.max())))
ax.set_xlabel("|Δh| (km)"); ax.set_ylabel("Δz (km)")
ax.set_title("Shift correlation (color = yearly cluster size)")
ax.grid(alpha=0.3)
ax.axhline(0, color="k", linewidth=0.4)
plt.colorbar(sc, ax=ax, label="nbranch (yearly)")

fig.suptitle(
    f"GrowClust: 30-day partial vs full-year relocation comparison\n"
    f"{len(matched):,} events matched by origin time (≤0.5 s).  "
    f"Median |Δh| = {np.median(matched.dh_km):.2f} km,  "
    f"p95 |Δh| = {np.percentile(matched.dh_km, 95):.2f} km",
    fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"\nwrote {OUT}")
