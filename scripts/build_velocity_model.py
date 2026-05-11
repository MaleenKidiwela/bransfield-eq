"""
Rebuild 1D velocity model from Orca tomography, prepending a water layer.

Orca convention (confirmed empirically): zg=0 sits at the seafloor (the top
cell averages water + sediment with median Vp~1.91 km/s, not pure water).
So to make a sea-level-referenced 1D model for pyocto, we prepend a 1.3 km
water layer (mean OBS water depth) and shift the Orca-derived rock down.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
NC = REPO / "configs" / "Pg_Orca_velocity.nc"
OUT_CSV = REPO / "configs" / "velocity_model.csv"
OUT_FIG = REPO / "notes" / "figures" / "velocity_model.png"
WATER_THICKNESS_KM = 1.3       # mean OBS water depth
WATER_VP_KMS = 1.4558          # measured Bransfield water-column Vp
VPVS = 1.78

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

ds = Dataset(NC)
V = ds.variables["Velocity"][:]      # (nz, ny, nx) km/s
zg = ds.variables["zg"][:]           # 0 -> -10 km
ds.close()
depth = -np.asarray(zg)              # 0 -> 10 km (positive down, seafloor-relative)

V2 = V.reshape(V.shape[0], -1)
p10  = np.percentile(V2, 10, axis=1)
p50  = np.percentile(V2, 50, axis=1)
p90  = np.percentile(V2, 90, axis=1)
pmin = V2.min(axis=1)
pmax = V2.max(axis=1)

# Orca's top cell (z=0) is the water/sediment smear -- discard it; use only z>=0.2 km
# as "rock below seafloor" and prepend a clean water layer + sea-level reference.
keep = depth >= 0.2
d_rock = depth[keep]
vp_rock = p50[keep]

# Below ~6 km Orca is unresolved (P10==P90); pin to its background, then add a
# mild Moho-region gradient by hand at 15/20/30 km (same as notes/13).
deep_rows = [(15.0, 6.6), (20.0, 6.8), (30.0, 7.0)]

water_n = 14   # 0, 0.1, ..., 1.3 km
water_d = np.linspace(0.0, WATER_THICKNESS_KM, water_n)
water_vp = np.full_like(water_d, WATER_VP_KMS)
# Water has no S waves physically, but pyocto's Eikonal solver crashes on Vs=0.
# Use a small positive value -- well below rock Vs so Fermat's principle routes
# S rays through rock and this number is never actually used.
water_vs = np.full_like(water_d, 0.5)

shifted_d = d_rock + WATER_THICKNESS_KM
rock_vp = vp_rock
rock_vs = rock_vp / VPVS

deep_d  = np.array([d + WATER_THICKNESS_KM for d, _ in deep_rows])
deep_vp = np.array([v for _, v in deep_rows])
deep_vs = deep_vp / VPVS

# Insert a tiny epsilon "seafloor jump" row so pyocto sees a sharp interface.
seafloor_d = WATER_THICKNESS_KM + 1e-3
seafloor_vp = rock_vp[0]
seafloor_vs = rock_vp[0] / VPVS

all_d  = np.concatenate([water_d, [seafloor_d], shifted_d, deep_d])
all_vp = np.concatenate([water_vp, [seafloor_vp], rock_vp, deep_vp])
all_vs = np.concatenate([water_vs, [seafloor_vs], rock_vs, deep_vs])

df = pd.DataFrame({"depth_km": all_d, "vp_kms": all_vp, "vs_kms": all_vs})
df = df.round({"depth_km": 4, "vp_kms": 4, "vs_kms": 4})
df.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(df)} rows)")
print(df.head(20).to_string(index=False))

# ----- plot -----
fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)

ax = axes[0]
# Orca spread (seafloor-referenced depth)
ax.fill_betweenx(depth, pmin, pmax, color="0.85", label="Orca 3D range (min–max)")
ax.fill_betweenx(depth, p10, p90, color="0.65", alpha=0.7, label="Orca P10–P90")
ax.plot(p50, depth, "k-", lw=1.5, label="Orca median Vp")
ax.set_xlabel("Vp (km/s)")
ax.set_ylabel("depth below seafloor (km)")
ax.set_ylim(10, 0)
ax.set_xlim(1.0, 7.5)
ax.set_title("Orca 3D → 1D averaging (seafloor-referenced)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

ax = axes[1]
# New 1D model (sea-level referenced) with water layer
ax.plot(all_vp, all_d, "C0-", lw=2, label="Vp (new 1D)")
ax.plot(all_vs, all_d, "C3-", lw=2, label="Vs (new 1D)")
ax.plot(all_vp, all_d, "C0o", ms=3)
ax.plot(all_vs, all_d, "C3o", ms=3)
ax.axhspan(0, WATER_THICKNESS_KM, color="C0", alpha=0.08,
           label=f"water ({WATER_THICKNESS_KM:.1f} km, Vp={WATER_VP_KMS*1000:.1f} m/s)")
ax.axhline(WATER_THICKNESS_KM, color="k", lw=0.5, ls="--")
ax.text(7.0, WATER_THICKNESS_KM - 0.05, "seafloor", ha="right", va="bottom", fontsize=8)
ax.set_xlabel("velocity (km/s)")
ax.set_ylabel("depth below sea level (km)")
ax.set_ylim(30, 0)
ax.set_xlim(0, 7.5)
ax.set_title("New 1D model (sea-level datum, water + Orca + deep)")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
print(f"wrote {OUT_FIG}")
