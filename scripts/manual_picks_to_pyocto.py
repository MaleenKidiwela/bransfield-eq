"""Convert catalogs/manual_picks.csv to the pyocto_events/picks_*.csv format
used by the Stage A + Stage B hypoDD pipeline.

Manual picks have no event locations (NLLoc didn't write them into the
extracted CSV). We assign every event a default initial source at the OBS
array centroid (-62.43, -58.45, 5 km depth). hypoDD's ISTART=1 will use this
as the trial location and DD will refine via differential times.

Writes:
    catalogs/pyocto_events_manual.csv
    catalogs/pyocto_picks_manual.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Default trial source — array centroid
DEFAULT_LAT = -62.43
DEFAULT_LON = -58.45
DEFAULT_DEP = 5.0


def main():
    src = REPO / "catalogs" / "manual_picks.csv"
    print(f"Loading {src} ...")
    df = pd.read_csv(src)
    print(f"  {len(df):,} pick rows, {df.event_id.nunique():,} unique event_ids")

    # Drop weird phase rows
    df = df[df.phase.isin(["P", "S"])].copy()

    df["origin_time"] = pd.to_datetime(df.origin_time, utc=True, errors="coerce")
    df["pick_time_dt"] = pd.to_datetime(df.pick_time, utc=True, errors="coerce")
    df = df.dropna(subset=["origin_time", "pick_time_dt"]).reset_index(drop=True)

    # Build events table
    ev = df.groupby("event_id").agg(
        origin_time=("origin_time", "first"),
        magnitude=("magnitude", "first"),
        picks=("phase", "size"),
    ).reset_index()
    ev = ev.sort_values("origin_time").reset_index(drop=True)
    ev["event_idx"] = ev.index.astype(int)
    ev["idx"] = ev["event_idx"]
    ev["time"] = ev["origin_time"].astype("int64") // 10**9
    ev["latitude"] = DEFAULT_LAT
    ev["longitude"] = DEFAULT_LON
    ev["depth"] = DEFAULT_DEP
    ev["x"] = 0.0
    ev["y"] = 0.0
    ev["z"] = DEFAULT_DEP

    # Map event_id (str) -> event_idx (int)
    eid_to_idx = dict(zip(ev["event_id"], ev["event_idx"]))

    # Build picks table
    pk = df.copy()
    pk["event_idx"] = pk["event_id"].map(eid_to_idx)
    pk["pick_idx"] = pk.groupby("event_idx").cumcount()
    pk["station"] = pk["network"].astype(str) + "." + pk["station"].astype(str)
    pk["time"] = pk["pick_time_dt"].astype("int64") / 10**9
    # Convert per-pick uncertainty (s) to a "prob"-like score in [0.2, 1.0]:
    # smaller uncertainty -> higher prob. Cap at ±0.5 s.
    pk["uncertainty_s"] = pd.to_numeric(pk["uncertainty_s"], errors="coerce").fillna(0.2)
    pk["prob"] = np.clip(1.0 - pk["uncertainty_s"] / 0.5, 0.2, 1.0)
    pk["residual"] = 0.0

    pk_out = pk[["event_idx", "pick_idx", "residual", "station", "time", "phase", "prob"]].copy()
    ev_out = ev[["idx", "time", "x", "y", "z", "picks", "event_idx",
                 "longitude", "latitude", "depth", "origin_time"]].copy()

    out_ev = REPO / "catalogs" / "pyocto_events_manual.csv"
    out_pk = REPO / "catalogs" / "pyocto_picks_manual.csv"
    ev_out.to_csv(out_ev, index=False)
    pk_out.to_csv(out_pk, index=False)
    print(f"wrote {out_ev}  ({len(ev_out):,} events)")
    print(f"wrote {out_pk}  ({len(pk_out):,} picks)")


if __name__ == "__main__":
    main()
