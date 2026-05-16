"""Convert pyocto events + picks into HypoDD input format.

Produces (in hypodd/<label>/):
    phase.dat   - per-event header + pick lines for ph2dt
    station.dat - bare station codes + lat/lon/elev for ph2dt + hypoDD

phase.dat per-event format:
    # YYYY MM DD HH MM SS.SS LAT LON DEP MAG EH EZ RMS ID
    STA TT WT PHASE
    STA TT WT PHASE
    ...

The event ID written is (pyocto row index + 1), matching the IDs the
GrowClust-prep stage wrote to dt.cc -- so the existing dt.cc can be reused
directly when wiring cross-correlation data into hypoDD.

Station codes use bare (sta) form so they match the post-strip dt.cc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only")
    args = ap.parse_args()

    ev_path = REPO / "catalogs" / f"pyocto_events_{args.label}.csv"
    pk_path = REPO / "catalogs" / f"pyocto_picks_{args.label}.csv"
    st_path = REPO / "catalogs" / "station_geometry.csv"
    out_dir = REPO / "hypodd" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading events: {ev_path}")
    ev = pd.read_csv(ev_path)
    if "origin_time" not in ev.columns:
        ev["origin_time"] = pd.to_datetime(ev.time, unit="s", utc=True)
    else:
        # pyocto sometimes writes origin_time as a numeric epoch-second string;
        # try datetime parse first, then fall back to epoch-seconds. Drop NaT.
        ot = pd.to_datetime(ev.origin_time, utc=True, errors="coerce")
        if ot.isna().mean() > 0.5:
            ot = pd.to_datetime(pd.to_numeric(ev.origin_time, errors="coerce"),
                                unit="s", utc=True)
        ev["origin_time"] = ot
    n_before = len(ev)
    ev = ev.dropna(subset=["origin_time"]).reset_index(drop=True)
    if len(ev) != n_before:
        print(f"  dropped {n_before-len(ev)} events with un-parseable origin_time")
    # Use pyocto's stable event_idx as the basis for hypodd_id so the *same*
    # physical event gets the *same* hypodd_id across every sub-region subset.
    # The previous version used row_index+1, which gave the same event different
    # ids in different sub-region runs and broke the Stage B merge dedup.
    if "event_idx" not in ev.columns and "idx" in ev.columns:
        ev = ev.rename(columns={"idx": "event_idx"})
    ev["hypodd_id"] = (ev["event_idx"].astype(int) + 1)

    print(f"Loading picks: {pk_path}")
    pk = pd.read_csv(pk_path)
    if "pick_time" not in pk.columns:
        pk["pick_time"] = pd.to_datetime(pk.time, unit="s", utc=True)
    if "sta_key" not in pk.columns:
        pk["sta_key"] = pk["station"].astype(str)
    pk["sta_bare"] = pk["sta_key"].str.split(".").str[-1]

    # Map pyocto event_idx -> hypodd_id (event.row_index + 1).
    if "event_idx" not in ev.columns and "idx" in ev.columns:
        ev = ev.rename(columns={"idx": "event_idx"})
    ev_idx_to_hid = dict(zip(ev["event_idx"], ev["hypodd_id"]))
    ev_idx_to_origin = dict(zip(ev["event_idx"], ev["origin_time"]))

    print(f"  events: {len(ev):,}   picks: {len(pk):,}")

    # ----- station.dat -----
    st_full = pd.read_csv(st_path)
    used_stas = set(pk["sta_bare"].unique())
    st = st_full[st_full["station"].isin(used_stas)].copy()
    # Bransfield stations may have ZX (OBS) at negative elevation (seafloor).
    # hypoDD station.dat columns: STA LAT LON [ELEV in meters, positive up].
    # station_geometry.csv uses the column name 'elevation_m' (a previous version
    # of this script silently fell back to 0.0 for every station because it
    # looked for 'elevation' -- which biased OBS travel times by ~1 s).
    for cand in ("elevation_m", "elevation", "elev_m", "elev"):
        if cand in st.columns:
            elev_col = cand
            break
    else:
        elev_col = None
    print(f"  station elevation column: {elev_col!r}")
    lines = []
    for _, r in st.iterrows():
        elev = float(r[elev_col]) if elev_col else 0.0
        lines.append(f"{r.station:<6s} {r.latitude:9.5f} {r.longitude:10.5f} {elev:8.1f}")
    (out_dir / "station.dat").write_text("\n".join(lines) + "\n")
    print(f"  wrote station.dat with {len(lines)} stations")

    # ----- phase.dat -----
    # Group picks by event for fast per-event iteration.
    picks_by_event = {idx: g for idx, g in pk.groupby("event_idx")}
    n_used_picks = 0
    out_lines = []
    for _, e in ev.iterrows():
        t = e.origin_time
        sec = t.second + t.microsecond * 1e-6
        mag = float(e.get("magnitude", 0.0)) if not pd.isna(e.get("magnitude", np.nan)) else 0.0
        lat = float(e.get("latitude", e.get("lat", np.nan)))
        lon = float(e.get("longitude", e.get("lon", np.nan)))
        dep = float(e.get("depth", e.get("z", 0.0)))
        rms = float(e.get("rms_residual", 0.0)) if "rms_residual" in ev.columns else 0.0
        hid = int(e.hypodd_id)
        out_lines.append(
            f"# {t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} "
            f"{sec:6.3f} {lat:9.5f} {lon:10.5f} {dep:7.3f} {mag:5.2f} "
            f"0.000 0.000 {rms:5.2f} {hid:8d}"
        )
        evt_picks = picks_by_event.get(e.event_idx)
        if evt_picks is None or evt_picks.empty:
            continue
        for p in evt_picks.itertuples():
            tt = (p.pick_time - t).total_seconds()
            # MAXDIST is 100 km and min layer Vp >= 1.4 km/s; max physical
            # one-way TT < 75 s. Accept up to 60 s as a hard sanity cap.
            if tt <= 0 or tt > 60:
                continue
            # weight: PhaseNet `prob` is a detection-confidence score, NOT a
            # timing-uncertainty proxy. Piecewise-linear calibration:
            #   prob >= 0.8 -> wt = 1.0   (high-confidence pick)
            #   prob >= 0.5 -> wt linearly 0.5 .. 1.0
            #   prob >= 0.2 -> wt linearly 0.2 .. 0.5
            #   prob <  0.2 -> wt = 0.2   (still keep, but downweight)
            # Missing/NaN prob defaults to 0.5 (conservative middle), NOT 1.0.
            prob = getattr(p, "prob", None)
            if prob is None or pd.isna(prob) or prob <= 0:
                wt = 0.5
            elif prob >= 0.8:
                wt = 1.0
            elif prob >= 0.5:
                wt = 0.5 + (prob - 0.5) / 0.3 * 0.5
            elif prob >= 0.2:
                wt = 0.2 + (prob - 0.2) / 0.3 * 0.3
            else:
                wt = 0.2
            phase = str(p.phase)
            out_lines.append(f"{p.sta_bare:<6s} {tt:8.3f} {wt:5.3f} {phase}")
            n_used_picks += 1

    (out_dir / "phase.dat").write_text("\n".join(out_lines) + "\n")
    print(f"  wrote phase.dat: {len(ev):,} event headers + {n_used_picks:,} phase lines")
    print(f"  output dir: {out_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
