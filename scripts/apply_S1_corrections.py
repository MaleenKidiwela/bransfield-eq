"""Apply Kidiwela et al. supporting-info Table S1 corrections:

  1. ZX station horizontal coords -> inverted lon/lat
  2. ZX station depths            -> bathymetric depth (from S1)
  3. BRA05 clock offset           -> subtract 0.167 s from every BRA05 pick time

Idempotent: writes a marker file (.bra05_clock_corrected) and skips if seen.
"""
from pathlib import Path
import pandas as pd
import shutil

REPO = Path(__file__).resolve().parent.parent
SG = REPO / "catalogs" / "station_geometry.csv"

# Table S1: station, inverted lon, inverted lat, bathymetric depth (km)
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
    ("BRA25", -58.39411, -62.41485, 0.896),
    ("BRA26", -58.33942, -62.42110, 1.411),
    ("BRA27", -58.33115, -62.44238, 1.460),
    ("BRA05", -58.44720, -62.41020, 1.146),
], columns=["station", "lon_inv", "lat_inv", "bath_km"])

# ---- 1+2: patch station_geometry.csv ----
backup = SG.with_suffix(".csv.bak_pre_S1")
if not backup.exists():
    shutil.copy(SG, backup)
    print(f"backed up {SG.name} -> {backup.name}")

sg = pd.read_csv(SG)
n_lon = n_lat = n_z = 0
for _, r in S1.iterrows():
    m = (sg.network == "ZX") & (sg.station == r.station)
    if not m.any():
        print(f"  warn: {r.station} not in station_geometry.csv")
        continue
    dl = sg.loc[m, "longitude"].iloc[0] - r.lon_inv
    da = sg.loc[m, "latitude"].iloc[0]  - r.lat_inv
    dz = sg.loc[m, "water_depth_m"].iloc[0] - r.bath_km * 1000
    if abs(dl) > 1e-5: n_lon += 1
    if abs(da) > 1e-5: n_lat += 1
    if abs(dz) > 1:    n_z   += 1
    sg.loc[m, "longitude"]     = r.lon_inv
    sg.loc[m, "latitude"]      = r.lat_inv
    sg.loc[m, "water_depth_m"] = r.bath_km * 1000
    sg.loc[m, "elevation_m"]   = -r.bath_km * 1000

sg.to_csv(SG, index=False)
print(f"patched station_geometry.csv: {n_lon} lon, {n_lat} lat, {n_z} depth updates")

# ---- 3: BRA05 clock correction (-0.167 s) ----
CLOCK_OFFSET_S = 0.167
BRA05_DIRS = [
    REPO / "catalogs" / "picks_obst_01" / "ZX.BRA05",
    Path("/home/jovyan/my_data/bravoseis/picks/ZX.BRA05"),  # PN output
]
for d in BRA05_DIRS:
    if not d.exists():
        print(f"  skip: {d} not found")
        continue
    marker = d / ".bra05_clock_corrected"
    if marker.exists():
        print(f"  already corrected: {d}")
        continue
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        print(f"  no csvs in {d}")
        continue
    print(f"  correcting {len(csvs)} csvs in {d} ...")
    for c in csvs:
        df = pd.read_csv(c)
        for col in ("time", "start", "end"):
            if col in df.columns:
                df[col] = (pd.to_datetime(df[col], utc=True)
                            - pd.Timedelta(seconds=CLOCK_OFFSET_S)
                          ).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df.to_csv(c, index=False)
    marker.write_text(f"applied -{CLOCK_OFFSET_S}s offset\n")
    print(f"    done; marker written")
print("BRA05 clock correction complete.")
