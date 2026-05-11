"""
Phase 1a: build SeisBench-format event + noise pools for OBS denoiser training.

Reads `configs/finetune_train_days.csv` (30 curated training days) and
`catalogs/manual_picks.csv` (mag07, uncertainty_s ≤ 0.1). For each curated day:

  Event windows  — cut a 30-s window around each high-confidence mag07 pick on
                   any station with mseed available. 100 Hz, 3-component.

  Noise windows  — sample candidate 30-s windows on the same station-days,
                   reject if any of:
                     - within ±60 s of any pick in EITHER mag07 OR magall
                       sources (broader exclusion than mag07-only)
                     - any sample of vertical STA/LTA (0.5/10 s) > 3.0
                     - window RMS < 1e-3× or > 10× the station-day median RMS

Outputs (SeisBench format):
  data/seisbench/bransfield_events.h5  + metadata.csv
  data/seisbench/bransfield_noise.h5   + metadata.csv
  figures/noise_qc.png                 — per-station-day acceptance rates

Parallelism: per-station-day dispatched across a multiprocessing pool to
saturate the 176-CPU machine. I/O bound (NFS reads); ~30 days × 21 stations
= ~630 station-day jobs.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import mseed_path  # noqa: E402

import seisbench.data as sbd  # noqa: E402

OUT_DIR = REPO / "data" / "seisbench"
TARGET_RATE = 100.0
WINDOW_SEC = 30.0
WINDOW_SAMPLES = int(TARGET_RATE * WINDOW_SEC)
PRE_PICK_SEC = 5.0  # pick at sample ~500 of 3000

PICKING_GLOB = "EH?,HH?,BH?,SH?,EL?,HL?,BL?,SL?"
NOISE_TARGET_PER_STN_DAY = 50  # tunable; ~30 days × 21 stns × 50 ≈ 31,500 noise


def _channel_match(code: str, glob: str) -> bool:
    import fnmatch
    return any(fnmatch.fnmatchcase(code, g.strip()) for g in glob.split(","))


def _read_and_clean(net: str, sta: str, day: UTCDateTime,
                    t_center: UTCDateTime, pre_sec: float = PRE_PICK_SEC):
    """Trim, drop hydrophone, merge, resample, return (3, N) ndarray + start UTC."""
    mp_path = mseed_path(net, sta, day)
    if not mp_path.exists() or mp_path.stat().st_size == 0:
        return None
    t0 = t_center - pre_sec
    t1 = t0 + WINDOW_SEC
    try:
        st = read(str(mp_path), starttime=t0, endtime=t1).copy()
    except Exception:
        return None
    st.traces = [tr for tr in st if _channel_match(tr.stats.channel, PICKING_GLOB)]
    if len(st) < 3:
        return None
    st.merge(method=1, fill_value=0)
    for tr in st:
        if abs(tr.stats.sampling_rate - TARGET_RATE) > 1e-6:
            tr.resample(TARGET_RATE)
    by_letter = {tr.stats.channel[-1]: tr for tr in st}
    chans = [by_letter.get(c) for c in ("Z", "N", "E")]
    if any(c is None for c in chans):
        chans = [by_letter.get(c) for c in ("Z", "1", "2")]
    if any(c is None for c in chans):
        return None
    n = min(tr.stats.npts for tr in chans)
    if n < int(0.95 * WINDOW_SAMPLES):
        return None
    arr = np.stack([tr.data[:n].astype(np.float32) for tr in chans], axis=0)
    if arr.shape[1] < WINDOW_SAMPLES:
        # right-pad with zeros to exact length
        pad = WINDOW_SAMPLES - arr.shape[1]
        arr = np.pad(arr, ((0, 0), (0, pad)), mode="constant")
    arr = arr[:, :WINDOW_SAMPLES]
    return arr, chans[0].stats.starttime


def _stalta_max(z_trace: np.ndarray, fs: float = TARGET_RATE) -> float:
    from obspy.signal.trigger import classic_sta_lta
    nsta = max(1, int(0.5 * fs))
    nlta = max(2, int(10.0 * fs))
    if len(z_trace) < nlta + 1:
        return 0.0
    try:
        cft = classic_sta_lta(z_trace, nsta, nlta)
    except Exception:
        return 0.0
    if not np.isfinite(cft).any():
        return 0.0
    return float(np.nanmax(cft))


def _process_event_picks_for_station_day(args) -> list:
    """Cut event windows for one (net, sta, day, picks-DataFrame)."""
    net, sta, day_iso, picks_records = args
    day = UTCDateTime(day_iso)
    rows = []
    for pk in picks_records:
        t_center = UTCDateTime(pk["pick_time"])
        cut = _read_and_clean(net, sta, day, t_center)
        if cut is None:
            continue
        wf, t_start = cut
        # Sample index of the center pick in this window
        pick_sample = int(round((t_center - t_start) * TARGET_RATE))
        meta = {
            "source_id": pk["event_id"],
            "source_origin_time": pk["origin_time"] or "",
            "station_network_code": net,
            "station_code": sta,
            "station_location_code": "",
            "trace_channel": pk["channel"][:2] if pk["channel"] else "",
            "trace_sampling_rate_hz": TARGET_RATE,
            "trace_npts": wf.shape[1],
            "trace_start_time": str(t_start),
            "trace_p_arrival_sample": float(pick_sample) if pk["phase"] == "P" else np.nan,
            "trace_s_arrival_sample": float(pick_sample) if pk["phase"] == "S" else np.nan,
            "trace_phase": pk["phase"],
            "trace_uncertainty_s": pk["uncertainty_s"],
        }
        rows.append((meta, wf))
    return rows


def _process_noise_for_station_day(args) -> tuple:
    """Sample candidate noise windows for one (net, sta, day), gate via filters."""
    net, sta, day_iso, all_pick_times, target = args
    day = UTCDateTime(day_iso)
    rng = np.random.default_rng(int(day.timestamp) ^ hash(f"{net}.{sta}") & 0xFFFFFFFF)

    mp_path = mseed_path(net, sta, day)
    if not mp_path.exists() or mp_path.stat().st_size == 0:
        return ([], 0, 0)  # rows, n_attempted, n_accepted
    # Compute station-day median RMS for amplitude sanity (sample 100 random windows once)
    try:
        full = read(str(mp_path)).copy()
    except Exception:
        return ([], 0, 0)
    full.traces = [tr for tr in full if _channel_match(tr.stats.channel, PICKING_GLOB)]
    if len(full) < 3:
        return ([], 0, 0)
    full.merge(method=1, fill_value=0)
    for tr in full:
        if abs(tr.stats.sampling_rate - TARGET_RATE) > 1e-6:
            tr.resample(TARGET_RATE)
    by_letter = {tr.stats.channel[-1]: tr for tr in full}
    z = by_letter.get("Z")
    if z is None:
        return ([], 0, 0)
    z_data = z.data.astype(np.float32)
    # Random RMS sample across the day for normalization baseline
    n_full = z_data.shape[0]
    rms_samples = []
    for _ in range(100):
        if n_full <= WINDOW_SAMPLES + 1:
            break
        ix = rng.integers(0, n_full - WINDOW_SAMPLES)
        seg = z_data[ix:ix + WINDOW_SAMPLES]
        rms_samples.append(np.sqrt(np.mean(seg**2) + 1e-20))
    if not rms_samples:
        return ([], 0, 0)
    rms_med = float(np.median(rms_samples))

    # Sample candidate noise windows
    rows = []
    n_attempted = 0
    n_accepted = 0
    pick_ts = all_pick_times.get((net, sta), np.array([]))
    pick_ts = pick_ts[(pick_ts >= day.timestamp - 86400) & (pick_ts <= day.timestamp + 2 * 86400)]
    pick_ts.sort()

    max_attempts = target * 6
    while n_accepted < target and n_attempted < max_attempts:
        n_attempted += 1
        offset = rng.uniform(0, 86400 - WINDOW_SEC - 1)
        t_center = day + offset + WINDOW_SEC / 2
        ts = t_center.timestamp
        # Manual-pick exclusion (any pick on this station within ±60 s)
        if len(pick_ts):
            idx = np.searchsorted(pick_ts, ts)
            near = min(abs(pick_ts[max(idx - 1, 0)] - ts),
                       abs(pick_ts[min(idx, len(pick_ts) - 1)] - ts))
            if near < 60:
                continue
        # Read the candidate window
        cut = _read_and_clean(net, sta, day, t_center, pre_sec=WINDOW_SEC / 2)
        if cut is None:
            continue
        wf, t_start = cut
        # STA/LTA reject on Z (vertical)
        z_seg = wf[0]
        if _stalta_max(z_seg) > 3.0:
            continue
        # Amplitude sanity
        rms = float(np.sqrt(np.mean(z_seg**2) + 1e-20))
        if rms < 1e-3 * rms_med or rms > 10 * rms_med:
            continue
        # Accept
        meta = {
            "source_id": f"noise_{net}.{sta}_{day.year}_{day.julday:03d}_{int(offset)}",
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
            "trace_phase": "noise",
            "trace_uncertainty_s": np.nan,
        }
        rows.append((meta, wf))
        n_accepted += 1
    return (rows, n_attempted, n_accepted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days-file", default=str(REPO / "configs" / "finetune_train_days.csv"))
    ap.add_argument("--noise-per-station-day", type=int, default=NOISE_TARGET_PER_STN_DAY)
    ap.add_argument("--workers", type=int, default=24,
                    help="Parallel processes (NFS-bound; 24 is usually sweet spot)")
    ap.add_argument("--max-uncertainty", type=float, default=0.1)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load curated training days
    days = pd.read_csv(args.days_file)["date"].tolist()
    day_set = set(pd.to_datetime(days).date)
    print(f"Curated training days: {len(days)}")

    # Load high-confidence mag07 picks restricted to those days
    m = pd.read_csv(REPO / "catalogs" / "manual_picks.csv",
                    parse_dates=["origin_time", "pick_time"])
    m07 = m[(m.source_file == "nllmaleen_mag07_202210.out") &
            (m.uncertainty_s <= args.max_uncertainty)].copy()
    m07["day"] = m07.pick_time.dt.date
    m07 = m07[m07.day.isin(day_set)].copy()
    print(f"mag07 picks (uncertainty ≤ {args.max_uncertainty}) on curated days: "
          f"{len(m07):,}  events={m07.event_id.nunique():,}")

    # All-source pick exclusion table for noise filtering (broader: mag07 + magall)
    m_all = m[m.source_file.isin(["nllmaleen_mag07_202210.out",
                                   "nllmaleen_magall_202210.out"])].copy()
    m_all["t_unix"] = m_all.pick_time.values.astype("datetime64[s]").astype(np.int64)
    pick_times_by_sta = {(net, sta): grp.t_unix.values
                          for (net, sta), grp in
                          m_all.groupby(["network", "station"])}
    print(f"Noise exclusion pool (mag07 ∪ magall): {len(m_all):,} picks across "
          f"{len(pick_times_by_sta):,} stations")

    # Stations with enough mseed data: any station present in our event picks
    event_station_days = set()
    event_jobs = []  # (net, sta, day_iso, [pick_records])
    for (net, sta, day), grp in m07.groupby(["network", "station", "day"]):
        recs = grp[["event_id", "origin_time", "pick_time", "channel", "phase",
                     "uncertainty_s"]].copy()
        # Use ISO 8601 with T separator (UTCDateTime parses this reliably)
        recs["origin_time"] = recs["origin_time"].apply(
            lambda x: x.isoformat() if pd.notna(x) else "")
        recs["pick_time"] = recs["pick_time"].apply(lambda x: x.isoformat())
        recs["channel"] = recs["channel"].fillna("")
        event_jobs.append((net, sta, str(day), recs.to_dict("records")))
        event_station_days.add((net, sta, str(day)))

    print(f"Event jobs (station-days with picks): {len(event_jobs):,}")

    # Noise jobs: every (station, day) where we have data — use the union of
    # event-bearing stations × all curated days.
    noise_stations = sorted({(n, s) for (n, s, _) in event_station_days})
    noise_jobs = [(n, s, str(d), pick_times_by_sta, args.noise_per_station_day)
                   for (n, s) in noise_stations for d in sorted(day_set)]
    print(f"Noise jobs: {len(noise_jobs):,}  ({len(noise_stations)} stations × {len(day_set)} days, target {args.noise_per_station_day}/job)")

    # === Build event dataset ===
    print(f"\n--- Cutting event windows ({args.workers} workers) ---", flush=True)
    ev_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = [pool.submit(_process_event_picks_for_station_day, j) for j in event_jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                ev_rows.extend(f.result())
            except Exception as e:
                print(f"  [event err] {e}", flush=True)
            if i % 50 == 0:
                print(f"    {i}/{len(event_jobs)} event-jobs  rows={len(ev_rows):,}", flush=True)
    print(f"  total event rows: {len(ev_rows):,}")

    ev_meta_path = OUT_DIR / "bransfield_events" / "metadata.csv"
    ev_wf_path = OUT_DIR / "bransfield_events" / "waveforms.hdf5"
    ev_meta_path.parent.mkdir(parents=True, exist_ok=True)
    if ev_meta_path.exists():
        ev_meta_path.unlink()
    if ev_wf_path.exists():
        ev_wf_path.unlink()
    with sbd.WaveformDataWriter(ev_meta_path, ev_wf_path) as wr:
        wr.data_format = {
            "dimension_order": "CW", "component_order": "ZNE",
            "sampling_rate": TARGET_RATE, "measurement": "velocity",
            "unit": "counts", "instrument_response": "not restituted",
        }
        for meta, wf in ev_rows:
            wr.add_trace(meta, wf)
    print(f"  wrote {ev_meta_path} (+ hdf5)")

    # === Build noise dataset + QC stats ===
    print(f"\n--- Cutting noise windows ({args.workers} workers) ---", flush=True)
    qc_data = []  # (net, sta, day, n_attempted, n_accepted)
    nz_rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futs = {pool.submit(_process_noise_for_station_day, j): j for j in noise_jobs}
        for i, f in enumerate(as_completed(futs), 1):
            net, sta, day_iso, _, _ = futs[f]
            try:
                rows, n_att, n_acc = f.result()
                nz_rows.extend(rows)
                qc_data.append((net, sta, day_iso, n_att, n_acc))
            except Exception as e:
                print(f"  [noise err] {net}.{sta} {day_iso}: {e}", flush=True)
            if i % 50 == 0:
                print(f"    {i}/{len(noise_jobs)} noise-jobs  rows={len(nz_rows):,}", flush=True)
    print(f"  total noise rows: {len(nz_rows):,}")

    nz_meta_path = OUT_DIR / "bransfield_noise" / "metadata.csv"
    nz_wf_path = OUT_DIR / "bransfield_noise" / "waveforms.hdf5"
    nz_meta_path.parent.mkdir(parents=True, exist_ok=True)
    if nz_meta_path.exists():
        nz_meta_path.unlink()
    if nz_wf_path.exists():
        nz_wf_path.unlink()
    with sbd.WaveformDataWriter(nz_meta_path, nz_wf_path) as wr:
        wr.data_format = {
            "dimension_order": "CW", "component_order": "ZNE",
            "sampling_rate": TARGET_RATE, "measurement": "velocity",
            "unit": "counts", "instrument_response": "not restituted",
        }
        for meta, wf in nz_rows:
            wr.add_trace(meta, wf)
    print(f"  wrote {nz_meta_path} (+ hdf5)")

    # === Noise QC plot ===
    print("\n--- Writing noise QC plot ---")
    qc = pd.DataFrame(qc_data, columns=["net", "sta", "day", "attempted", "accepted"])
    qc["accept_rate"] = qc.accepted / qc.attempted.replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(11, 7))
    sta_unique = sorted(qc.sta.unique())
    days_unique = sorted(qc.day.unique())
    pivot = qc.pivot_table(index="sta", columns="day", values="accept_rate")
    pivot = pivot.reindex(index=sta_unique, columns=days_unique)
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(days_unique)))
    ax.set_xticklabels(days_unique, rotation=70, fontsize=7)
    ax.set_yticks(range(len(sta_unique)))
    ax.set_yticklabels(sta_unique, fontsize=8)
    ax.set_xlabel("Day"); ax.set_ylabel("Station")
    ax.set_title(f"Noise window acceptance rate (target {args.noise_per_station_day}/job)\n"
                 f"low rate (<20%) = station likely contains unmarked swarm activity")
    fig.colorbar(im, ax=ax, label="accepted / attempted")
    fig.tight_layout()
    out_png = REPO / "figures" / "noise_qc.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")

    print("\n=== Summary ===")
    print(f"  Event windows: {len(ev_rows):,}")
    print(f"  Noise windows: {len(nz_rows):,}  (median accept rate: "
          f"{qc.accept_rate.median():.2f})")


if __name__ == "__main__":
    main()
