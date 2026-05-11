# 1D velocity model — derived from Orca Pg tomography

Source: `configs/Pg_Orca_velocity.nc` — isotropic Pg (P-wave) 3D tomography
model from the BRAVOSEIS active-source experiment (51 z × 101 y × 136 x grid,
0.2 km cubes, covering Orca Volcano and SW ridge).

Output: `configs/velocity_model.csv` — 1D depth/Vp/Vs profile (54 layers,
0 → 30 km depth), the format `scripts/17_pyocto_associate.py` expects.

## Derivation

1. Read the 3D Velocity array (km/s) and zg axis from the .nc.
2. zg is given as 0 → −10 km; flipped sign so depth_km is 0 → 10 km (down-positive).
3. **At each depth layer**, took the **median Vp across all 101 × 136 (x,y) cells**.
   Median (not mean) to be robust to a few anomalous cells; very tight P10–P90
   spread below ~3 km depth indicates the 1D summary is a good representation
   of the resolved structure there.
4. Vp profile (from tomography):
   - Surface (0 km): 1.91 km/s — water + shallow sediment dominate
   - 1–2 km: 4.0 → 5.5 km/s — sediment / upper-crust transition
   - 2–4 km: 5.5 → 6.3 km/s — upper crust
   - 4–6 km: 6.3 → 6.5 km/s — upper crust, deeper
   - 6–10 km: flat 6.51 km/s — **tomography no longer resolves**; this is the
     starting/background model the inversion didn't update.
5. **Extrapolated below 10 km** with three hand-picked layers (15, 20, 30 km)
   at 6.6 / 6.8 / 7.0 km/s — a mild gradient toward typical Moho-region values.
   Most Bransfield local seismicity is shallow (<15 km), so this just gives
   pyocto something sensible to ray-trace through for the rare deeper event.
6. **Vs derived as Vp / 1.78** (the pyocto-script's default `--vpvs`). Standard
   crystalline-crust ratio; the Orca .nc only provides Vp.

## Why this is reasonable for pyocto

- pyocto's `VelocityModel1D` only needs depth/Vp/Vs and a tolerance; it does
  not care about the 3D heterogeneity the tomography resolves.
- A 1D median profile preserves the depth-dependent gradient that matters
  most for association — the sediment-to-crust transition is what dominates
  P-S travel-time differences at the OBS network's apertures.
- Lateral variation in the tomography is small compared to vertical
  (P10–P90 spread < 0.05 km/s below 4 km).

## Where this differs from the "Orca-3D / 1D blend" called out in `SUMMARY.md` §10 step 8

The note there flags the user has a "blended Orca-3D / 1D model" in mind for
NLLoc location. That's a more careful product: typically the resolved Orca
volume is preserved as-is and only the unresolved deep / out-of-volume regions
are filled from the 1D. That's a location-time concern (NLLoc 3D travel-time
grids); for **association** (pyocto), a 1D model is the standard input and is
fine. Re-evaluate when moving to NLLoc location.

## Reproduce

```bash
.venv/bin/python -c "
import h5py, numpy as np, pandas as pd
with h5py.File('configs/Pg_Orca_velocity.nc','r') as f:
    zg = f['zg'][:]; V = f['Velocity'][:]
depth = -zg
vp = np.median(V, axis=(1,2))
extra_d = np.array([15.0, 20.0, 30.0])
extra_v = np.array([6.6, 6.8, 7.0])
depth = np.concatenate([depth, extra_d]); vp = np.concatenate([vp, extra_v])
vs = vp / 1.78
pd.DataFrame({'depth_km':depth,'vp_kms':vp,'vs_kms':vs}).round(4).to_csv('configs/velocity_model.csv', index=False)
"
```
