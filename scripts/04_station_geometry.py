"""
Stage 1.d — extract station geometry for downstream travel-time prediction.

For each station in the inventory, write:
  - lat, lon
  - elevation (m, positive up; OBS will be negative-elevation seafloor)
  - sensor depth (m below ground/sea surface; OBS sensor depth in water)
  - water_depth (m; for OBS = -elevation, for land = 0)
  - on_seafloor (bool)

This file feeds the Stage 3 associator and Stage 4 locator: OBS travel times
need an explicit water layer in the velocity model, parameterized by water depth.

Output: catalogs/station_geometry.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from obspy import read_inventory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bransfield_eq.config import REPO, XML_DIR


def main() -> None:
    out_rows = []
    for xml in sorted(XML_DIR.glob("*.xml")):
        inv = read_inventory(str(xml))
        for net in inv:
            for sta in net:
                # Use the first channel's depth as canonical sensor depth.
                depth_m = sta[0].depth if len(sta) > 0 else 0.0
                # Elevation is the surface above the sensor (m, positive up).
                # On OBS, ObsPy reports elevation as the seafloor elevation
                # (negative number, since seafloor is below sea level).
                elev_m = sta.elevation
                on_seafloor = elev_m < -10.0   # >10 m below sea level
                water_depth_m = max(0.0, -elev_m) if on_seafloor else 0.0
                out_rows.append({
                    "network": net.code,
                    "station": sta.code,
                    "latitude": sta.latitude,
                    "longitude": sta.longitude,
                    "elevation_m": elev_m,
                    "sensor_depth_m": depth_m,
                    "water_depth_m": water_depth_m,
                    "on_seafloor": on_seafloor,
                    "n_channels": len(sta),
                })
    if not out_rows:
        print(f"No stationxml found in {XML_DIR}")
        return
    df = pd.DataFrame(out_rows).drop_duplicates(subset=["network", "station"])
    out = REPO / "catalogs" / "station_geometry.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out.relative_to(REPO)}  ({len(df)} stations)")
    print()
    print(df.to_string(index=False))
    print()
    obs = df[df.on_seafloor]
    if len(obs):
        print(f"OBS stations: {len(obs)}, water-depth range: "
              f"{obs.water_depth_m.min():.0f} to {obs.water_depth_m.max():.0f} m")


if __name__ == "__main__":
    main()
