"""Plot NLLoc QC diagnostics + relocation map for catalogs/nlloc_<label>.csv.

Outputs:
    notes/figures/nlloc_qc_<label>.png    QC histograms + boundary-pinning map
    notes/figures/nlloc_relocations_<label>.png    Map view + depth section
                                                     (same panel layout as
                                                      plot_hypodd_relocations.py)
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import netcdf_file
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="picker_only_no_shots")
args = ap.parse_args()

nl = pd.read_csv(REPO / "catalogs" / f"nlloc_{args.label}.csv")
print(f"loaded {len(nl):,} NLLoc events")

# Grid bounds (from nlloc/run/<label>.in: LOCGRID 301 201 126 -29.8 -20.0 0 0.2 ...)
GX_MIN, GX_MAX = -29.8, -29.8 + 0.2 * 300
GY_MIN, GY_MAX = -20.0, -20.0 + 0.2 * 200
nl["on_boundary"] = (
    (nl.nlloc_x_km - GX_MIN < 0.5) | (GX_MAX - nl.nlloc_x_km < 0.5) |
    (nl.nlloc_y_km - GY_MIN < 0.5) | (GY_MAX - nl.nlloc_y_km < 0.5)
)
hq = ~nl.on_boundary & (nl.gap_deg < 180) & (nl.rms_s < 0.5) & (nl.n_phases >= 6)
nl["hq"] = hq
print(f"on grid boundary: {nl.on_boundary.sum():,} ({100*nl.on_boundary.mean():.1f}%)")
print(f"high-quality (interior, gap<180, rms<0.5, Nphs>=6): {hq.sum():,} ({100*hq.mean():.1f}%)")


# ============================================================
# QC figure
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

def hist(ax, x, bins, xlabel, log=False):
    ax.hist(x, bins=bins, edgecolor="k", alpha=0.7, color="steelblue")
    ax.set_xlabel(xlabel); ax.set_ylabel("count"); ax.grid(alpha=0.3)
    if log:
        ax.set_yscale("log")

hist(axes[0, 0], nl.n_phases, np.arange(4, 31), "n phases")
hist(axes[0, 1], nl.gap_deg, np.linspace(0, 360, 37), "azimuthal gap (°)")
hist(axes[0, 2], nl.rms_s.clip(0, 5), np.linspace(0, 5, 51), "RMS (s)")
hist(axes[0, 3], nl.depth_km, np.linspace(0, 26, 53), "depth (km)")

# sigma distributions
for ax, col, label in zip(
    axes[1, :3],
    ["sigma_x_km", "sigma_y_km", "sigma_z_km"],
    ["σ_x (km)", "σ_y (km)", "σ_z (km)"],
):
    hist(ax, nl[col].clip(0, 10), np.linspace(0, 10, 51), label, log=True)

# boundary-pinning map
ax = axes[1, 3]
ax.scatter(nl[nl.on_boundary].lon, nl[nl.on_boundary].lat,
           s=2, c="red", alpha=0.3, label="boundary-pinned")
ax.scatter(nl[~nl.on_boundary].lon, nl[~nl.on_boundary].lat,
           s=2, c="steelblue", alpha=0.4, label="interior")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("grid-boundary pinning")
ax.legend(loc="upper right", markerscale=3, framealpha=0.9)
ax.grid(alpha=0.3)

fig.suptitle(
    f"NLLoc QC — label {args.label}  ({len(nl):,} events, "
    f"{hq.sum():,} HQ, {nl.on_boundary.sum():,} pinned)",
    fontsize=13,
)
plt.tight_layout()
out_qc = REPO / "notes" / "figures" / f"nlloc_qc_{args.label}.png"
plt.savefig(out_qc, dpi=140, bbox_inches="tight")
print(f"wrote {out_qc}")
plt.close(fig)


# ============================================================
# Relocation map (HQ subset on bathymetry)
# ============================================================
stations = pd.read_csv(ST)
ob = stations[stations.network == "ZX"]

with netcdf_file(str(BATHY), "r") as ds:
    lat_b = ds.variables["latitude"][:].copy()
    lon_b = ds.variables["longitude"][:].copy()
    z_b = ds.variables["data"][:].copy()
z_b = np.where(z_b > 2000, np.nan, z_b)

# Shift NLLoc's depth-below-seafloor onto a sea-level datum for the
# cross-section (so bathymetry shape is visible). Map view keeps the
# native depth-below-seafloor for the depth colorbar.
sf_interp = RegularGridInterpolator(
    (lat_b, lon_b), -z_b / 1000.0,
    bounds_error=False, fill_value=np.nan,
)
nl["seafloor_km"] = sf_interp(np.c_[nl.lat, nl.lon])
nl["depth_bsl_km"] = nl.depth_km + nl.seafloor_km

lon_min, lon_max = -58.7, -58.2
lat_min, lat_max = -62.55, -62.35

dz = nl[(nl.lon.between(lon_min, lon_max)) & (nl.lat.between(lat_min, lat_max))]
print(f"events in Orca zoom (all, regardless of quality): {len(dz):,}")

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

ax = fig.add_subplot(gs[0, 0])
plot_bathy(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sc1 = ax.scatter(dz.lon, dz.lat, c=dz.depth_km, s=4, cmap="magma_r",
                 vmin=0, vmax=15, edgecolors="none", alpha=0.6, zorder=7)
ax.set_title(f"NLLoc locations (all) — {len(dz):,} events in zoom")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc1, ax=ax, label="depth below seafloor (km)",
             shrink=0.7, pad=0.02)

ax = fig.add_subplot(gs[0, 1])
plot_bathy(ax)
ax.scatter(ob.longitude, ob.latitude, marker="^", s=70, c="white",
           edgecolors="k", linewidths=1.0, zorder=8)
sigma_h = np.sqrt(dz.sigma_x_km**2 + dz.sigma_y_km**2)
sc2 = ax.scatter(dz.lon, dz.lat, c=sigma_h.clip(0, 2), s=4, cmap="viridis",
                 vmin=0, vmax=2, edgecolors="none", alpha=0.7, zorder=7)
ax.set_title("colored by horizontal σ (km)")
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.colorbar(sc2, ax=ax, label="σ_h (km)", shrink=0.7, pad=0.02)

# Cross-section: depth below SEA LEVEL with bathymetric seafloor envelope.
in_lat_sf = (lat_b >= lat_min) & (lat_b <= lat_max)
in_lon_sf = (lon_b >= lon_min) & (lon_b <= lon_max)
z_sub = z_b[in_lat_sf][:, in_lon_sf]
sf_km_along_lon = -z_sub / 1000.0   # (n_lat, n_lon), positive down
lon_strip = lon_b[in_lon_sf]
lat_strip = lat_b[in_lat_sf]
sf_lon_med = np.nanmedian(sf_km_along_lon, axis=0)
sf_lon_min = np.nanmin(sf_km_along_lon, axis=0)
sf_lon_max = np.nanmax(sf_km_along_lon, axis=0)
sf_lat_med = np.nanmedian(sf_km_along_lon, axis=1)
sf_lat_min = np.nanmin(sf_km_along_lon, axis=1)
sf_lat_max = np.nanmax(sf_km_along_lon, axis=1)

ax = fig.add_subplot(gs[1, 0])
ax.fill_between(lon_strip, 0, sf_lon_max, color="lightblue", alpha=0.5, zorder=2)
ax.fill_between(lon_strip, sf_lon_min, sf_lon_max, color="0.75",
                alpha=0.6, zorder=3)
ax.plot(lon_strip, sf_lon_med, color="0.2", linewidth=1.0, zorder=4)
ax.scatter(dz.lon, dz.depth_bsl_km, c="steelblue", s=4, alpha=0.55, zorder=7)
ax.set_xlim(lon_min, lon_max); ax.set_ylim(20, 0)
ax.set_xlabel("longitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Longitude cross-section (seafloor profile shaded)")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.fill_between(lat_strip, 0, sf_lat_max, color="lightblue", alpha=0.5, zorder=2)
ax.fill_between(lat_strip, sf_lat_min, sf_lat_max, color="0.75",
                alpha=0.6, zorder=3)
ax.plot(lat_strip, sf_lat_med, color="0.2", linewidth=1.0, zorder=4)
ax.scatter(dz.lat, dz.depth_bsl_km, c="steelblue", s=4, alpha=0.55, zorder=7)
ax.set_xlim(lat_min, lat_max); ax.set_ylim(20, 0)
ax.set_xlabel("latitude"); ax.set_ylabel("depth below sea level (km)")
ax.set_title("Latitude cross-section (seafloor profile shaded)")
ax.grid(alpha=0.3)

fig.suptitle(
    f"NLLoc relocations (all events) — {args.label}\n"
    f"{len(nl):,} events total, {len(dz):,} in Orca zoom, "
    f"{hq.sum():,} HQ (interior, gap<180°, RMS<0.5s, Nphs≥6)",
    fontsize=13,
)
out_map = REPO / "notes" / "figures" / f"nlloc_relocations_{args.label}.png"
plt.savefig(out_map, dpi=140, bbox_inches="tight")
print(f"wrote {out_map}")
plt.close(fig)
