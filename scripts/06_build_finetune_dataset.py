"""
Build a SeisBench-format HDF5 dataset for PhaseNet fine-tuning.

For each mag07 event-station-day combination, cuts a centered waveform window
covering the manual picks, resamples to 100 Hz, and writes to a SeisBench-format
HDF5 dataset (`bransfield_train.hdf5` + `bransfield_train.csv`).

A separate noise dataset is also built from station-day windows with no manual
pick within ±N seconds, for use as a `RealNoise` source during training.

Splits are assigned by event_id deterministic hash so that all observations of a
given event land in the same split (avoids train/val leakage).

Usage:
    python scripts/06_build_finetune_dataset.py
    python scripts/06_build_finetune_dataset.py --window-seconds 60 \
        --pre-pick-seconds 15 --noise-per-station 200 --max-events 50  # pilot
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import mseed_path  # noqa: E402

import seisbench.data as sbd  # noqa: E402

OUT_DIR = REPO / "data" / "seisbench"
TARGET_RATE = 100.0


def split_for_event(event_id: str) -> str:
    """Deterministic 70/15/15 train/dev/test by event_id."""
    h = int(hashlib.md5(event_id.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "dev"
    return "test"


def cut_window(stream, t_center: UTCDateTime,
               window_pre: float, window_post: float,
               picking_glob: str = "EH?,HH?,BH?,SH?,EL?,HL?,BL?,SL?") -> tuple | None:
    """Trim, merge, drop hydrophone, resample to 100 Hz; returns (np.ndarray (3,N), start_utc)."""
    import fnmatch
    t0 = t_center - window_pre
    t1 = t_center + window_post
    st = stream.copy().trim(t0, t1)
    st.traces = [tr for tr in st
                 if any(fnmatch.fnmatchcase(tr.stats.channel, g.strip())
                        for g in picking_glob.split(","))]
    if len(st) == 0:
        return None
    st.merge(method=1, fill_value=0)
    for tr in st:
        if abs(tr.stats.sampling_rate - TARGET_RATE) > 1e-6:
            tr.resample(TARGET_RATE)
    # Pick first 3 components in Z-N-E (or 1-2-Z) order if available
    by_letter = {tr.stats.channel[-1]: tr for tr in st}
    chans = [by_letter.get(c) for c in ("Z", "N", "E")]
    if any(c is None for c in chans):
        chans = [by_letter.get(c) for c in ("Z", "1", "2")]
    if any(c is None for c in chans):
        return None
    n = min(tr.stats.npts for tr in chans)
    if n < int(TARGET_RATE * (window_pre + window_post) * 0.9):  # need ~90% of expected length
        return None
    arr = np.stack([tr.data[:n].astype(np.float32) for tr in chans], axis=0)
    return arr, chans[0].stats.starttime


def build_event_dataset(picks_df: pd.DataFrame, window_pre: float, window_post: float,
                        max_events: int | None = None) -> int:
    """Cut event windows for each (event, station-channel) and write to HDF5."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "bransfield_train.csv"
    wf_path = OUT_DIR / "bransfield_train.hdf5"

    n_written = 0
    n_skipped = 0
    events = picks_df.event_id.unique()
    if max_events:
        events = events[:max_events]
    print(f"Building event dataset over {len(events)} events ...")

    with sbd.WaveformDataWriter(meta_path, wf_path) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": TARGET_RATE,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        for ei, eid in enumerate(events):
            ev_picks = picks_df[picks_df.event_id == eid]
            origin = UTCDateTime(ev_picks.origin_time.iloc[0])
            split = split_for_event(eid)
            for (net, sta), sta_picks in ev_picks.groupby(["network", "station"]):
                # Find the day to load
                first_pick = UTCDateTime(sta_picks.pick_time.min())
                day = UTCDateTime(first_pick.date)
                mp = mseed_path(net, sta, day)
                if not mp.exists() or mp.stat().st_size == 0:
                    n_skipped += 1
                    continue
                try:
                    st = read(str(mp))
                except Exception:
                    n_skipped += 1
                    continue
                p_picks = sta_picks[sta_picks.phase == "P"]
                s_picks = sta_picks[sta_picks.phase == "S"]
                t_center = UTCDateTime(p_picks.pick_time.iloc[0]) if len(p_picks) \
                           else UTCDateTime(s_picks.pick_time.iloc[0]) if len(s_picks) \
                           else UTCDateTime(sta_picks.pick_time.iloc[0])
                cut = cut_window(st, t_center, window_pre, window_post)
                if cut is None:
                    n_skipped += 1
                    continue
                wf, t_start = cut
                # Sample indices of picks in the cut window
                p_sample = (UTCDateTime(p_picks.pick_time.iloc[0]) - t_start) * TARGET_RATE \
                           if len(p_picks) else np.nan
                s_sample = (UTCDateTime(s_picks.pick_time.iloc[0]) - t_start) * TARGET_RATE \
                           if len(s_picks) else np.nan
                meta = {
                    "source_id": eid,
                    "source_origin_time": str(origin),
                    "station_network_code": net,
                    "station_code": sta,
                    "station_location_code": "",
                    "trace_channel": sta_picks.channel.iloc[0][:2] if len(sta_picks) else "",
                    "trace_sampling_rate_hz": TARGET_RATE,
                    "trace_npts": wf.shape[1],
                    "trace_start_time": str(t_start),
                    "trace_p_arrival_sample": float(p_sample) if not np.isnan(p_sample) else np.nan,
                    "trace_s_arrival_sample": float(s_sample) if not np.isnan(s_sample) else np.nan,
                    "trace_p_status": "manual" if len(p_picks) else "",
                    "trace_s_status": "manual" if len(s_picks) else "",
                    "split": split,
                }
                writer.add_trace(meta, wf)
                n_written += 1
            if (ei + 1) % 100 == 0:
                print(f"  ... {ei+1}/{len(events)} events  written={n_written}  skipped={n_skipped}", flush=True)
    print(f"Event dataset: wrote {n_written} traces, skipped {n_skipped}")
    return n_written


def build_noise_dataset(picks_df: pd.DataFrame, window_pre: float, window_post: float,
                        per_station: int, exclusion_seconds: float = 60.0) -> int:
    """Cut quiet-window noise samples per station (no mag07 pick within ±exclusion_s)."""
    meta_path = OUT_DIR / "bransfield_noise.csv"
    wf_path = OUT_DIR / "bransfield_noise.hdf5"

    rng = np.random.default_rng(42)
    win_len = window_pre + window_post

    # Stations with full year on disk
    stations_full = []
    for sta_dir in (REPO / "data" / "waveforms").glob("*/*"):
        if sta_dir.is_dir() and len(list(sta_dir.glob("*.mseed"))) >= 420:
            stations_full.append((sta_dir.parent.name, sta_dir.name))
    print(f"Building noise dataset over {len(stations_full)} fully-downloaded stations, "
          f"target {per_station}/station ...")

    n_written = 0
    with sbd.WaveformDataWriter(meta_path, wf_path) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": TARGET_RATE,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        for net, sta in stations_full:
            sta_pick_times = picks_df[(picks_df.network == net) & (picks_df.station == sta)] \
                .pick_time.apply(lambda x: UTCDateTime(x).timestamp).values
            sta_pick_times.sort()
            mseed_files = sorted((REPO / "data" / "waveforms" / net / sta).glob("*.mseed"))
            if not mseed_files:
                continue
            written_for_station = 0
            attempts = 0
            while written_for_station < per_station and attempts < per_station * 4:
                attempts += 1
                # Random day, random hour-offset
                f = rng.choice(mseed_files)
                # Parse year/julday from filename
                parts = f.name.split(".")
                year, jday = int(parts[2]), int(parts[3])
                day = UTCDateTime(f"{year}-01-01") + (jday - 1) * 86400
                hour_offset = rng.uniform(0, 86400 - win_len)
                t_center = day + hour_offset + window_pre
                # Reject if any manual pick within ±exclusion_s
                ts = t_center.timestamp
                if len(sta_pick_times):
                    idx = np.searchsorted(sta_pick_times, ts)
                    nearest = min(
                        abs(sta_pick_times[max(idx - 1, 0)] - ts),
                        abs(sta_pick_times[min(idx, len(sta_pick_times) - 1)] - ts),
                    )
                    if nearest < exclusion_seconds:
                        continue
                try:
                    st = read(str(f))
                except Exception:
                    continue
                cut = cut_window(st, t_center, window_pre, window_post)
                if cut is None:
                    continue
                wf, t_start = cut
                meta = {
                    "source_id": f"noise_{net}.{sta}_{year}.{jday:03d}_{int(hour_offset)}",
                    "source_origin_time": "",
                    "station_network_code": net,
                    "station_code": sta,
                    "station_location_code": "",
                    "trace_channel": "",
                    "trace_sampling_rate_hz": TARGET_RATE,
                    "trace_npts": wf.shape[1],
                    "trace_start_time": str(t_start),
                    "trace_p_arrival_sample": np.nan,
                    "trace_s_arrival_sample": np.nan,
                    "trace_p_status": "",
                    "trace_s_status": "",
                    "split": "train",  # all noise goes to train pool; never use as eval
                }
                writer.add_trace(meta, wf)
                written_for_station += 1
                n_written += 1
            print(f"  {net}.{sta}: {written_for_station}/{per_station} noise windows", flush=True)
    print(f"Noise dataset: wrote {n_written} traces")
    return n_written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-pre", type=float, default=15.0,
                    help="seconds before pick (default 15)")
    ap.add_argument("--window-post", type=float, default=45.0,
                    help="seconds after pick (default 45 → 60s window total)")
    ap.add_argument("--noise-per-station", type=int, default=100)
    ap.add_argument("--max-events", type=int, default=None,
                    help="cap event count for pilot runs")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--start-date", default=None,
                    help="restrict events to pick_time >= this date (YYYY-MM-DD)")
    ap.add_argument("--end-date", default=None,
                    help="restrict events to pick_time <  this date (YYYY-MM-DD)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    m = pd.read_csv(REPO / "catalogs" / "manual_picks.csv",
                    parse_dates=["origin_time", "pick_time"])
    m07 = m[m.source_file == "nllmaleen_mag07_202210.out"].copy()
    if args.start_date:
        m07 = m07[m07.pick_time >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        m07 = m07[m07.pick_time <  pd.Timestamp(args.end_date, tz="UTC")]
    print(f"mag07 picks (after date filter): {len(m07):,}, events: {m07.event_id.nunique():,}")

    n_ev = build_event_dataset(m07, args.window_pre, args.window_post,
                               max_events=args.max_events)
    if not args.skip_noise:
        n_noise = build_noise_dataset(m07, args.window_pre, args.window_post,
                                      args.noise_per_station)
    print(f"\nDone.  events={n_ev}  noise={n_noise if not args.skip_noise else '(skipped)'}")
    print(f"  output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
