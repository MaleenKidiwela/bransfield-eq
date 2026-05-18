"""Build a hybrid catalog: NLLoc absolutes anchoring HypoDD relative geometry.

Inputs:
    catalogs/nlloc_<label>.csv                (script 31 output)
    catalogs/hypodd_picker_only_pruned.csv    (Stage A backbone)
    catalogs/hypodd_stage_b.csv               (Stage B dense sub-clusters)

Logic:
    For each HypoDD cluster (cid for Stage A, region for Stage B), compute the
    per-cluster median shift (NLLoc - HypoDD) across events that have both.
    Add this shift to every HypoDD member of the cluster. Result is a HypoDD
    relative geometry rigidly translated so its centroid lands on NLLoc's.

    Events that don't appear in a HypoDD cluster but do have an NLLoc location
    are kept as NLLoc-only rows.

Output:
    catalogs/hybrid_<label>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def anchor_per_cluster(
    relative: pd.DataFrame, cluster_col: str,
    nlloc_by_idx: pd.DataFrame, idx_col: str,
) -> pd.DataFrame:
    """Return relative df with anchored lat/lon/depth columns."""
    merged = relative.merge(
        nlloc_by_idx.rename(columns={"lat": "nl_lat", "lon": "nl_lon",
                                     "depth_km": "nl_depth"}),
        left_on=idx_col, right_on="event_idx", how="left",
    )
    have = merged.dropna(subset=["nl_lat"])
    shifts = (have.groupby(cluster_col)
              .agg(dlat=("nl_lat", "median"), dlon=("nl_lon", "median"),
                   ddep=("nl_depth", "median"),
                   rlat=("lat", "median"), rlon=("lon", "median"),
                   rdep=("dep", "median"), n_anchor=("nl_lat", "size"))
              .reset_index())
    shifts["dlat"] = shifts["dlat"] - shifts["rlat"]
    shifts["dlon"] = shifts["dlon"] - shifts["rlon"]
    shifts["ddep"] = shifts["ddep"] - shifts["rdep"]
    shifts = shifts[[cluster_col, "dlat", "dlon", "ddep", "n_anchor"]]
    out = merged.merge(shifts, on=cluster_col, how="left")
    out["anchored_lat"] = out["lat"] + out["dlat"]
    out["anchored_lon"] = out["lon"] + out["dlon"]
    out["anchored_depth_km"] = out["dep"] + out["ddep"]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="picker_only_no_shots")
    p.add_argument("--stage-a", default="catalogs/hypodd_picker_only_pruned.csv")
    p.add_argument("--stage-b", default="catalogs/hypodd_stage_b.csv")
    args = p.parse_args()

    nl = pd.read_csv(REPO / "catalogs" / f"nlloc_{args.label}.csv")
    print(f"NLLoc events: {len(nl)}")
    nl_by_idx = nl[["event_idx", "lat", "lon", "depth_km"]]

    # HypoDD writes the pyocto event_idx as its `id` column (script 22).
    pieces = []
    sa_path = REPO / args.stage_a
    if sa_path.exists():
        sa = pd.read_csv(sa_path)
        sa_out = anchor_per_cluster(sa, "cid", nl_by_idx, "id")
        sa_out["stage"] = "A"
        pieces.append(sa_out)
        print(f"Stage A: {len(sa)} HypoDD events, {sa_out['nl_lat'].notna().sum()} with NLLoc")
    sb_path = REPO / args.stage_b
    if sb_path.exists():
        sb = pd.read_csv(sb_path)
        cluster_col = "region_label" if "region_label" in sb.columns else "region"
        sb_out = anchor_per_cluster(sb, cluster_col, nl_by_idx, "id")
        sb_out["stage"] = "B"
        sb_out = sb_out.rename(columns={cluster_col: "cluster_label"})
        pieces.append(sb_out)
        print(f"Stage B: {len(sb)} HypoDD events, {sb_out['nl_lat'].notna().sum()} with NLLoc")

    if "cid" in pieces[0].columns and "cluster_label" not in pieces[0].columns:
        pieces[0]["cluster_label"] = pieces[0]["cid"].apply(lambda c: f"cidA_{c}")

    common_cols = ["event_idx", "stage", "cluster_label",
                   "anchored_lat", "anchored_lon", "anchored_depth_km",
                   "lat", "lon", "dep",
                   "nl_lat", "nl_lon", "nl_depth", "n_anchor"]
    pieces = [d[[c for c in common_cols if c in d.columns]].copy() for d in pieces]
    hybrid = pd.concat(pieces, ignore_index=True)

    # NLLoc-only rows for events not in any HypoDD stage
    have = set(hybrid["event_idx"].dropna().astype(int))
    nl_only = nl[~nl["event_idx"].isin(have)].copy()
    nl_only["stage"] = "NLLoc-only"
    nl_only["cluster_label"] = ""
    nl_only["anchored_lat"] = nl_only["lat"]
    nl_only["anchored_lon"] = nl_only["lon"]
    nl_only["anchored_depth_km"] = nl_only["depth_km"]
    nl_only["nl_lat"] = nl_only["lat"]
    nl_only["nl_lon"] = nl_only["lon"]
    nl_only["nl_depth"] = nl_only["depth_km"]
    nl_only["dep"] = np.nan
    nl_only = nl_only.rename(columns={"lat": "lat", "lon": "lon"})  # no-op for clarity
    nl_only["lat"] = np.nan
    nl_only["lon"] = np.nan
    nl_only["n_anchor"] = np.nan
    nl_only = nl_only[[c for c in common_cols if c in nl_only.columns]]
    hybrid = pd.concat([hybrid, nl_only], ignore_index=True)

    out = REPO / "catalogs" / f"hybrid_{args.label}.csv"
    hybrid.to_csv(out, index=False)
    print(f"wrote {len(hybrid)} rows -> {out}")


if __name__ == "__main__":
    main()
