"""Stage B: dense sub-cluster relocation, anchored to the Stage A backbone.

Steps:
  1. Load Stage A backbone events (catalogs/hypodd_picker_only_pruned.csv).
  2. K-means partition the backbone into N regions in (x, y) km space.
  3. For each region:
       - Find centroid and a containment radius (90th-percentile distance to
         backbone members + an overlap buffer for bridge events).
       - Pull every pyocto event that falls inside the buffered radius -- these
         are the events the sub-cluster relocation will solve for.
       - Write subset phase.dat / station.dat into hypodd/sub_<i>/.
       - Run ph2dt and hypoDD on the subset.
  4. Concatenate all sub-cluster hypoDD outputs into one final CSV.

Notes:
  - Stage A's 676 backbone events sit inside the larger 42,040-event catalog
    in known positions. Any pyocto event near a backbone member is more
    densely relocatable than the strict pruning would allow because we have
    its full pick set.
  - Overlap (buffer) makes sure bridge events appear in adjacent sub-clusters
    so the relative geometry between regions stays consistent.

Usage:
    python scripts/25_stage_b_partition_and_run.py --n-regions 8 --buffer-km 1.5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

DEFAULT_BACKBONE_CSV = REPO / "catalogs" / "hypodd_picker_only_pruned.csv"
DEFAULT_EV_CSV = REPO / "catalogs" / "pyocto_events_picker_only.csv"
DEFAULT_PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only.csv"


def latlon_to_xy_km(lat, lon, lat0, lon0):
    R = 6371.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    return R * dlon * np.cos(np.radians(lat0)), R * dlat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-regions", type=int, default=16,
                    help="K-means cluster count over the Stage A backbone.")
    ap.add_argument("--max-radius-km", type=float, default=2.0,
                    help="Cap a region's radius at this value (prevents one "
                         "outlier-bloated region from swallowing the swarm).")
    ap.add_argument("--buffer-km", type=float, default=1.0,
                    help="Extra radius beyond the cap to create overlap / "
                         "bridge events between sub-regions.")
    ap.add_argument("--workers", type=int, default=4,
                    help="How many sub-region pipelines to run in parallel.")
    ap.add_argument("--backbone-csv", default=str(DEFAULT_BACKBONE_CSV),
                    help="Backbone catalog to seed sub-region centroids "
                         "(default: Stage A pruned).")
    ap.add_argument("--label-prefix", default="sub",
                    help="Prefix for per-sub-region labels (e.g. 'sub' -> "
                         "'sub_0', 'sub_1', ...).")
    ap.add_argument("--regions-csv-name", default="stage_b_regions.csv",
                    help="Filename in catalogs/ for the regions metadata.")
    ap.add_argument("--events-csv", default=str(DEFAULT_EV_CSV),
                    help="Source pyocto events catalog (full-year scope).")
    ap.add_argument("--picks-csv", default=str(DEFAULT_PK_CSV),
                    help="Source pyocto picks catalog.")
    args = ap.parse_args()

    backbone_path = Path(args.backbone_csv)
    if not backbone_path.is_absolute():
        backbone_path = REPO / backbone_path
    print(f"Loading backbone: {backbone_path}")
    bb = pd.read_csv(backbone_path)
    bb["lat0"] = bb.lat.mean()
    bb["lon0"] = bb.lon.mean()
    bb["x_km"], bb["y_km"] = latlon_to_xy_km(bb.lat, bb.lon, bb.lat.mean(), bb.lon.mean())
    print(f"  {len(bb):,} backbone events; range "
          f"lat [{bb.lat.min():.3f}, {bb.lat.max():.3f}], "
          f"lon [{bb.lon.min():.3f}, {bb.lon.max():.3f}]")

    print(f"K-means into {args.n_regions} regions ...")
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.n_regions, n_init=10, random_state=42)
    bb["region"] = km.fit_predict(bb[["x_km", "y_km"]].values)

    # Centroids and radii in km
    regions = []
    for r in range(args.n_regions):
        members = bb[bb.region == r]
        cx, cy = members.x_km.mean(), members.y_km.mean()
        clat, clon = members.lat.mean(), members.lon.mean()
        d = np.hypot(members.x_km - cx, members.y_km - cy)
        p90 = float(np.percentile(d, 90)) if len(d) > 1 else 0.0
        # Cap to max-radius so an outlier-spread backbone region doesn't
        # capture half the catalog.
        radius = min(p90, args.max_radius_km) + args.buffer_km
        regions.append({"region": r, "n_backbone": len(members),
                        "lat": clat, "lon": clon,
                        "x_km": cx, "y_km": cy, "radius_km": radius})
    regions = pd.DataFrame(regions)
    print(regions.to_string(index=False))

    # Pull every pyocto event into the (possibly multiple) regions whose
    # buffered radius contains it.
    EV_CSV = Path(args.events_csv)
    PK_CSV = Path(args.picks_csv)
    print(f"\nLoading full pyocto catalog: {EV_CSV}")
    ev = pd.read_csv(EV_CSV)
    if "origin_time" not in ev.columns:
        ev["origin_time"] = pd.to_datetime(ev.time, unit="s", utc=True)
    ev = ev.reset_index(drop=True)
    ev["pyocto_row"] = ev.index
    ev["x_km"], ev["y_km"] = latlon_to_xy_km(ev.latitude, ev.longitude,
                                              bb.lat.mean(), bb.lon.mean())
    print(f"  {len(ev):,} events total")

    # Build per-region event subsets (an event can appear in multiple regions)
    region_events = {}
    for _, reg in regions.iterrows():
        dist = np.hypot(ev.x_km - reg.x_km, ev.y_km - reg.y_km)
        sub = ev[dist <= reg.radius_km].copy()
        region_events[int(reg.region)] = sub
        print(f"  region {int(reg.region)}: centroid ({reg.lat:.3f},{reg.lon:.3f}), "
              f"r={reg.radius_km:.2f} km -> {len(sub):,} events")

    # Write per-region subsets to catalogs/ then run scripts 22+23+24
    print(f"\nLoading picks: {PK_CSV}")
    pk = pd.read_csv(PK_CSV)
    pk_by_event = {idx: g for idx, g in pk.groupby("event_idx")}

    for r, sub in region_events.items():
        label = f"{args.label_prefix}_{r}"
        ev_out = REPO / "catalogs" / f"pyocto_events_{label}.csv"
        pk_out = REPO / "catalogs" / f"pyocto_picks_{label}.csv"
        sub_drop = sub.drop(columns=["x_km", "y_km", "pyocto_row"])
        sub_drop.to_csv(ev_out, index=False)
        sub_event_idx = set(sub.event_idx if "event_idx" in sub.columns
                            else sub.idx)
        sub_pk = pk[pk.event_idx.isin(sub_event_idx)].reset_index(drop=True)
        sub_pk.to_csv(pk_out, index=False)
        print(f"  region {r}: wrote {len(sub):,} events, {len(sub_pk):,} picks "
              f"to label={label}")

    # Save region metadata for downstream merging
    regions["label"] = [f"{args.label_prefix}_{r}" for r in regions.region]
    regions.to_csv(REPO / "catalogs" / args.regions_csv_name, index=False)
    print(f"\nWrote catalogs/stage_b_regions.csv")
    print(f"\nNext: for each sub_<i> label, run scripts 22 -> 23 -> 24.")
    print(f"Suggest: bash scripts/run_stage_b_subclusters.sh --workers {args.workers}")


if __name__ == "__main__":
    sys.exit(main() or 0)
