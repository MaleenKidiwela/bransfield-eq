"""Compare station_geometry.csv against Kidiwela+ Table S1 (inverted lon/lat
and Almendros et al. 2020 bathymetric depths).
"""
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SG = REPO / "catalogs" / "station_geometry.csv"

# Table S1 from supporting info: inverted lon, lat, bathymetric depth (km)
S1 = pd.DataFrame([
    ("BRA13", -58.59685, -62.45666, 1.447),
    ("BRA14", -58.56904, -62.48072, 1.499),
    ("BRA15", -58.54442, -62.43361, 1.323),
    ("BRA16", -58.50607, -62.45566, 1.333),
    ("BRA18", -58.48997, -62.42123, 1.245),
    ("BRA19", -58.45841, -62.43880, 1.084),
    ("BRA20", -58.45315, -62.46845, 1.363),
    ("BRA21", -58.41302, -62.42670, 1.093),
    ("BRA22", -58.39121, -62.43655, 1.023),
    ("BRA23", -58.38415, -62.46350, 1.356),
    ("BRA24", -58.39849, -62.39504, 1.160),
    ("BRA25", -58.39411, -62.41485, 0.896),   # big z diff
    ("BRA26", -58.33942, -62.42110, 1.411),
    ("BRA27", -58.33115, -62.44238, 1.460),
    ("BRA05", -58.44720, -62.41020, 1.146),
], columns=["sta", "lon_S1", "lat_S1", "bath_km_S1"])

sg = pd.read_csv(SG)
sg = sg[sg.network == "ZX"][["station", "latitude", "longitude", "water_depth_m"]].copy()
sg.columns = ["sta", "lat_repo", "lon_repo", "water_m_repo"]

m = S1.merge(sg, on="sta", how="left")
import numpy as np
# horizontal distance in metres via simple equirectangular at -62.43°
deg2km = 111.32
lat_mean = -62.43
m["dlon_m"] = (m.lon_S1 - m.lon_repo) * deg2km * np.cos(np.radians(lat_mean)) * 1000
m["dlat_m"] = (m.lat_S1 - m.lat_repo) * deg2km * 1000
m["horiz_m"] = np.hypot(m.dlon_m, m.dlat_m)
m["depth_diff_m"] = m.bath_km_S1 * 1000 - m.water_m_repo

cols = ["sta", "lon_S1", "lon_repo", "lat_S1", "lat_repo", "horiz_m",
        "bath_km_S1", "water_m_repo", "depth_diff_m"]
print(m[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}"))

print("\n--- summary ---")
print(f"horizontal offset (S1 inverted vs repo drop):  "
      f"min={m.horiz_m.min():.1f} m  median={m.horiz_m.median():.1f} m  max={m.horiz_m.max():.1f} m")
print(f"depth diff (S1 bathymetric vs repo water_depth_m):  "
      f"min={m.depth_diff_m.min():.1f} m  median={m.depth_diff_m.median():.1f} m  max={m.depth_diff_m.max():.1f} m")
print(f"|depth diff| > 100 m:")
print(m[m.depth_diff_m.abs() > 100][["sta", "bath_km_S1", "water_m_repo", "depth_diff_m"]].to_string(index=False))
