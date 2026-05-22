"""Compute per-station P/S timing corrections from reliable-tier events.

For every (station, phase) pair, take the median of
    residual = (pyocto_pick_time − NLLoc_v2_origin_time) − predicted_arrival
across all events in `nlloc_picker_only_no_shots_v2_reliable.csv` that
have a pyocto pick on that station. `predicted_arrival` comes from the
NLLoc v2 travel-time grids evaluated at the event's NLLoc hypocenter.

The resulting `catalogs/station_corrections.csv` is meant to be added
to predicted arrival times when defining refinement search windows:
    search_center = tP_pred + correction_P[station]

Usage:
    python scripts/45_compute_station_corrections.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_s42", REPO / "scripts" / "42_plot_event_picker_probs.py"
)
_s42 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s42)
predict_arrivals = _s42.predict_arrivals

PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv"
NLLOC_RELIABLE = REPO / "catalogs" / "nlloc_picker_only_no_shots_v2_reliable.csv"
OUT_CSV = REPO / "catalogs" / "station_corrections.csv"


def main() -> None:
    rel = pd.read_csv(NLLOC_RELIABLE)
    pk = pd.read_csv(PK_CSV)
    rel["origin_epoch"] = pd.to_datetime(rel.origin_time, utc=True).astype("int64") / 1e9
    print(f"reliable events: {len(rel)}   pyocto picks: {len(pk):,}")

    # Cache predictions per event (most events touch ~10 stations)
    rows = []
    for i, ev in enumerate(rel.itertuples()):
        if i % 500 == 0:
            print(f"  event {i}/{len(rel)}")
        pred = predict_arrivals(float(ev.lat), float(ev.lon), float(ev.depth_km))
        if not pred:
            continue
        ev_picks = pk[pk.event_idx == ev.event_idx]
        if ev_picks.empty:
            continue
        ot = float(ev.origin_epoch)
        for _, p in ev_picks.iterrows():
            net, sta = p.station.split(".")
            if sta not in pred:
                continue
            tp, ts = pred[sta]
            obs = float(p.time) - ot
            pred_t = tp if p.phase == "P" else ts
            rows.append({
                "station": p.station, "phase": p.phase,
                "residual": obs - pred_t,
                "event_idx": int(ev.event_idx),
            })

    rec = pd.DataFrame(rows)
    print(f"\ntotal pick-event pairs used (raw): {len(rec):,}")
    # Discard physically impossible residuals (these are pyocto mis-
    # associations where the pick belongs to a different earthquake).
    n_before = len(rec)
    rec = rec[rec["residual"].abs() <= 5.0].reset_index(drop=True)
    print(f"  kept after |res| ≤ 5s filter: {len(rec):,} "
          f"({100 * len(rec) / n_before:.1f} %)")

    # Aggregate per (station, phase)
    agg = (rec.groupby(["station", "phase"])
              .agg(n=("residual", "size"),
                   median_residual_s=("residual", "median"),
                   mad_s=("residual", lambda v: float(np.median(np.abs(v - np.median(v))))),
                   p16_s=("residual", lambda v: float(np.percentile(v, 16))),
                   p84_s=("residual", lambda v: float(np.percentile(v, 84))))
              .reset_index())
    agg = agg.sort_values(["phase", "station"]).reset_index(drop=True)
    print(agg.to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
