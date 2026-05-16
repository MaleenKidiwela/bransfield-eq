"""Merge the Stage B sub-cluster hypoDD catalogs into one final catalog.

Each pyocto event may have been relocated in multiple sub-regions (overlap /
bridge events). To resolve duplicates we keep the relocation from the
sub-region whose centroid is *closest* to the event's relocated position --
i.e. the region for which the event is most interior, not a bridge.

Inputs:
    catalogs/stage_b_regions.csv          - centroid lat/lon/radius per sub
    catalogs/hypodd_sub_<i>.csv           - per-sub-region hypoDD outputs
    catalogs/pyocto_events_picker_only.csv - to map back to pyocto event_idx

Output:
    catalogs/hypodd_stage_b.csv  - final merged catalog
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def latlon_to_xy_km(lat, lon, lat0, lon0):
    R = 6371.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    return R * dlon * np.cos(np.radians(lat0)), R * dlat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="catalogs/hypodd_stage_b.csv")
    ap.add_argument("--regions-csv-name", default="stage_b_regions.csv")
    args = ap.parse_args()

    regions = pd.read_csv(REPO / "catalogs" / args.regions_csv_name)
    print(f"Loaded {len(regions)} sub-region centroids")

    # Reference frame from regions (consistent with what stage B used)
    lat0 = regions.lat.mean()
    lon0 = regions.lon.mean()

    frames = []
    for _, reg in regions.iterrows():
        path = REPO / "catalogs" / f"hypodd_{reg.label}.csv"
        if not path.exists():
            print(f"  [skip] {path.name} -- missing")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"  [skip] {path.name} -- empty")
            continue
        df["region"] = int(reg.region)
        df["region_label"] = reg.label
        df["region_x_km"] = reg.x_km
        df["region_y_km"] = reg.y_km
        # Event-to-centroid distance using stage-B reference frame.
        ex, ey = latlon_to_xy_km(df.lat.values, df.lon.values, lat0, lon0)
        df["ev_x_km"] = ex
        df["ev_y_km"] = ey
        df["dist_to_centroid_km"] = np.hypot(ex - reg.x_km, ey - reg.y_km)
        frames.append(df)
        print(f"  {reg.label}: {len(df):,} events  "
              f"(median dist-to-centroid {df['dist_to_centroid_km'].median():.2f} km)")

    cat = pd.concat(frames, ignore_index=True)
    print(f"\nConcatenated rows (with duplicates): {len(cat):,}")

    # `id` in hypoDD output is the event ID from event.sel, which we wrote as
    # pyocto_row+1 in 22_pyocto_to_hypodd_input.py. So `id` is the global key
    # that allows us to detect duplicates across sub-regions.
    cat = cat.sort_values(["id", "dist_to_centroid_km"]).reset_index(drop=True)
    n_unique = cat["id"].nunique()
    print(f"Unique pyocto event ids: {n_unique:,}")
    overlap_count = (len(cat) - n_unique)
    print(f"Duplicate (bridge) rows resolved by closest centroid: {overlap_count:,}")

    # Keep the closest one per id
    deduped = cat.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

    # Also flag events that were in multiple sub-regions (information for QC)
    counts = cat.groupby("id").size().rename("n_subregions")
    deduped = deduped.merge(counts, on="id", how="left")
    deduped["is_bridge"] = deduped["n_subregions"] > 1

    out_path = REPO / args.out
    deduped.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(deduped):,} events; "
          f"{deduped.is_bridge.sum():,} bridge events; "
          f"{deduped.physical.sum() if 'physical' in deduped.columns else 'n/a'} physical.")


if __name__ == "__main__":
    sys.exit(main() or 0)
