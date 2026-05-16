"""Flag pyocto "events" that are actually active-source shots from the
BRAVOSEIS 2019 cruise (Orca MCS, Orca Tomo, Edifice-A MCS, Rift MCS, Other MCS).

Method: temporal match within a ±tolerance window between each pyocto event's
origin_time and any shot time. Sub-second shot timing precision plus pyocto's
typical few-hundred-ms origin-time error means a 1 s window catches them
cleanly without false-flagging real earthquakes (which are not synchronised to
sub-second precision with the shot schedule).

Writes:
    catalogs/pyocto_events_picker_only_with_shot_flag.csv
    catalogs/pyocto_picks_picker_only_no_shots.csv  (picks with event_idx of
                                                     flagged events removed)
    catalogs/pyocto_events_picker_only_no_shots.csv (events with flag_shot==False)

Usage:
    python scripts/discriminate_shots.py --tol-sec 1.0
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SHOT_DIR = REPO / "shotfiles"


def load_shots() -> pd.DataFrame:
    rows = []
    for path in sorted(SHOT_DIR.glob("*_shotfile_final.txt")):
        # 9-column whitespace-separated: shotnumber date time srcLat srcLon
        # shipLat shipLon waterDepth sciTag
        df = pd.read_csv(path, comment="#", sep=r"\s+", engine="python",
                         names=["shotnum", "date", "time", "src_lat", "src_lon",
                                "ship_lat", "ship_lon", "water_depth", "sci_tag"])
        df["survey"] = path.stem.replace("_shotfile_final", "")
        df["dt"] = pd.to_datetime(df.date + " " + df.time, utc=True,
                                  format="mixed", errors="coerce")
        df = df.dropna(subset=["dt"])
        rows.append(df[["dt", "src_lat", "src_lon", "shotnum", "survey"]])
        print(f"  {path.name:<40s}  {len(df):>6,} shots  "
              f"{df.dt.min()} → {df.dt.max()}")
    shots = pd.concat(rows, ignore_index=True).sort_values("dt").reset_index(drop=True)
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only")
    ap.add_argument("--tol-sec", type=float, default=1.0,
                    help="Max |event_origin_time - shot_time| to flag as a shot.")
    ap.add_argument("--tol-km", type=float, default=None,
                    help="If set, also require event epicentre within this "
                         "many km of the matched shot point (joint temporal + "
                         "spatial filter). Looser --tol-sec is safe with this on.")
    args = ap.parse_args()

    print(f"Loading shotfiles from {SHOT_DIR} ...")
    shots = load_shots()
    print(f"  total shots: {len(shots):,}")
    print(f"  combined span: {shots.dt.min()} → {shots.dt.max()}")

    print(f"\nLoading events {args.label} ...")
    ev = pd.read_csv(REPO / "catalogs" / f"pyocto_events_{args.label}.csv")
    if "origin_time" in ev.columns:
        ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True,
                                           format="mixed", errors="coerce")
    else:
        ev["origin_time"] = pd.to_datetime(ev.time, unit="s", utc=True)
    ev = ev.sort_values("origin_time").reset_index(drop=True)
    print(f"  events: {len(ev):,}")

    # Temporal match via merge_asof (each event paired with its nearest shot)
    shots_t = shots[["dt"]].rename(columns={"dt": "shot_dt"}).copy()
    shots_t["shot_idx"] = shots.index
    matched = pd.merge_asof(ev[["origin_time"]].rename(columns={"origin_time": "dt"}),
                            shots_t.rename(columns={"shot_dt": "dt"}),
                            on="dt", direction="nearest",
                            tolerance=pd.Timedelta(seconds=args.tol_sec))
    has_time_match = matched["shot_idx"].notna()
    print(f"\nEvents within ±{args.tol_sec} s of a shot: "
          f"{has_time_match.sum():,} ({has_time_match.mean()*100:.1f}%)")

    # Attach matched-shot info
    matched_shots = shots.loc[matched["shot_idx"].fillna(-1).astype(int)
                              .clip(lower=0)].reset_index(drop=True)
    ev["shot_idx"] = matched["shot_idx"].values
    ev["shot_survey"] = np.where(has_time_match.values, matched_shots["survey"].values, "")

    if args.tol_km is not None:
        # Spatial check: event epicentre within tol-km of matched shot's source location.
        R = 6371.0
        ev_lat = ev["latitude"].values if "latitude" in ev.columns else ev["lat"].values
        ev_lon = ev["longitude"].values if "longitude" in ev.columns else ev["lon"].values
        sh_lat = matched_shots["src_lat"].values
        sh_lon = matched_shots["src_lon"].values
        dlat = np.radians(ev_lat - sh_lat)
        dlon = np.radians(ev_lon - sh_lon) * np.cos(np.radians(ev_lat))
        dist_km = R * np.hypot(dlat, dlon)
        ev["dist_to_shot_km"] = np.where(has_time_match.values, dist_km, np.nan)
        spatial_match = pd.Series(dist_km <= args.tol_km, index=ev.index) & has_time_match
        print(f"Events also within {args.tol_km} km of matched shot: "
              f"{spatial_match.sum():,} ({spatial_match.mean()*100:.1f}%)")
        has_match = spatial_match
    else:
        has_match = has_time_match

    ev["flag_shot"] = has_match.values

    # Breakdown by survey
    print("\nShot matches by survey:")
    for s, n in ev[ev.flag_shot]["shot_survey"].value_counts().items():
        print(f"  {s:<25s} {n:>5,}")

    out_full = REPO / "catalogs" / f"pyocto_events_{args.label}_with_shot_flag.csv"
    out_clean = REPO / "catalogs" / f"pyocto_events_{args.label}_no_shots.csv"
    ev_sorted_back = ev.sort_values("event_idx" if "event_idx" in ev.columns else "idx").reset_index(drop=True)
    ev_sorted_back.to_csv(out_full, index=False)
    ev_sorted_back[~ev_sorted_back.flag_shot].drop(columns=["flag_shot", "shot_idx", "shot_survey"]).to_csv(out_clean, index=False)
    print(f"\nwrote {out_full}")
    print(f"wrote {out_clean}  ({(~ev_sorted_back.flag_shot).sum():,} non-shot events)")

    # Picks subset for downstream pipelines
    pk_in = REPO / "catalogs" / f"pyocto_picks_{args.label}.csv"
    if pk_in.exists():
        pk = pd.read_csv(pk_in)
        keep_ev = set(ev_sorted_back[~ev_sorted_back.flag_shot]["event_idx"].astype(int))
        n_before = len(pk)
        pk_clean = pk[pk.event_idx.isin(keep_ev)].reset_index(drop=True)
        out_pk = REPO / "catalogs" / f"pyocto_picks_{args.label}_no_shots.csv"
        pk_clean.to_csv(out_pk, index=False)
        print(f"wrote {out_pk}  ({len(pk_clean):,} picks of {n_before:,})")


if __name__ == "__main__":
    main()
