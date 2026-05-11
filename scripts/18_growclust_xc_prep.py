"""
GrowClust input prep — waveform cross-correlation differential times.

Reads pyocto association output (catalogs/pyocto_{events,picks}_<label>.csv)
and produces GrowClust3D-format input files:

    growclust/<label>/dt.cc         — differential times per event pair
    growclust/<label>/evlist.txt    — event list
    growclust/<label>/stlist.txt    — station list

For each event pair within --max-dist-km and --max-dt-sec, for every shared
(station, phase), slices a ±--win-sec window around each pick, FFT-cross-correlates,
keeps the lag if peak coherence > --cc-thresh. Sub-sample peak via parabolic interp.

Usage:
    python scripts/18_growclust_xc_prep.py --label picker_only

Pure Python — no GrowClust binary required at this stage.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
WAVE_DIR = REPO / "data" / "waveforms"

# --- defaults tuned for OBS / Bransfield ---
MAX_DIST_KM = 5.0       # event-pair max separation
MAX_DT_SEC  = 60 * 60   # event-pair max origin-time difference (1 h)
WIN_SEC     = 1.5       # ± window around each pick
CC_THRESH   = 0.6       # min normalised cross-correlation to keep
FS_TARGET   = 100.0     # resample to this rate before XC (matches PhaseNet input)
MAX_LAG_SEC = 0.5       # search window for best lag
MAX_PAIRS   = None      # cap pairs per event (None = no cap)


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------

def latlon_to_xy_km(lat, lon, lat0, lon0):
    """Flat-Earth approximation, sufficient for ~50 km network."""
    R = 6371.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = R * dlon * np.cos(np.radians(lat0))
    y = R * dlat
    return x, y


def load_pyocto(label: str):
    ev = pd.read_csv(REPO / "catalogs" / f"pyocto_events_{label}.csv")
    pk = pd.read_csv(REPO / "catalogs" / f"pyocto_picks_{label}.csv")
    # column normalisation — pyocto column names can vary by version
    rename_ev = {}
    for src, dst in [("idx", "event_idx"), ("time", "origin_time")]:
        if src in ev.columns and dst not in ev.columns:
            rename_ev[src] = dst
    ev = ev.rename(columns=rename_ev)
    rename_pk = {}
    for src, dst in [("idx", "event_idx"), ("time", "pick_time"), ("t", "pick_time")]:
        if src in pk.columns and dst not in pk.columns:
            rename_pk[src] = dst
    pk = pk.rename(columns=rename_pk)
    ev["origin_time"] = pd.to_datetime(ev["origin_time"], utc=True)
    pk["pick_time"]   = pd.to_datetime(pk["pick_time"], utc=True)
    return ev, pk


def load_stations():
    sg = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    sg["sta_key"] = sg["network"].astype(str) + "." + sg["station"].astype(str)
    return sg.set_index("sta_key")[["network", "station", "latitude", "longitude", "elevation_m"]]


# --------------------------------------------------------------------------------------
# Waveform cache + slicing
# --------------------------------------------------------------------------------------

def _waveform_path(network, station, dt):
    return WAVE_DIR / network / station / f"{network}.{station}.{dt.year}.{dt.dayofyear:03d}.mseed"


def slice_window(stream, pick_time_utc, win_sec):
    """Return a single 1-D numpy array (vertical-preferred component) windowed around pick."""
    # Prefer Z, fall back to merged horizontal RMS
    z_traces = [tr for tr in stream if tr.stats.channel.endswith("Z")]
    if z_traces:
        tr = z_traces[0].copy()
    else:
        tr = stream.copy().merge(fill_value=0)[0]
    # Trim
    from obspy import UTCDateTime
    pt = UTCDateTime(pick_time_utc.isoformat())
    tr.trim(starttime=pt - win_sec, endtime=pt + win_sec, pad=True, fill_value=0.0)
    if tr.stats.sampling_rate != FS_TARGET:
        tr.resample(FS_TARGET, no_filter=True)
    data = np.asarray(tr.data, dtype=np.float32)
    # De-mean + unit norm
    data = data - data.mean()
    n = np.linalg.norm(data)
    if n > 0:
        data /= n
    return data


# --------------------------------------------------------------------------------------
# XC kernel
# --------------------------------------------------------------------------------------

def xcorr_with_lag(a: np.ndarray, b: np.ndarray, max_lag_samples: int):
    """Return (lag_samples_subsample, peak_cc). a, b assumed unit-norm + zero-mean."""
    from scipy.signal import correlate
    if a.size != b.size:
        return None
    n = a.size
    cc = correlate(a, b, mode="full", method="fft")
    centre = n - 1
    lo = max(0, centre - max_lag_samples)
    hi = min(cc.size, centre + max_lag_samples + 1)
    cc_window = cc[lo:hi]
    k = int(np.argmax(cc_window))
    peak = float(cc_window[k])
    lag_int = (lo + k) - centre
    # Parabolic interpolation for sub-sample lag
    if 0 < k < cc_window.size - 1:
        y_m1, y_0, y_p1 = cc_window[k-1], cc_window[k], cc_window[k+1]
        denom = (y_m1 - 2*y_0 + y_p1)
        delta = 0.5 * (y_m1 - y_p1) / denom if denom != 0 else 0.0
    else:
        delta = 0.0
    return lag_int + delta, peak


# --------------------------------------------------------------------------------------
# Pair processing
# --------------------------------------------------------------------------------------

def collect_pair_dts(ev_a, ev_b, picks_a, picks_b, wf_cache):
    """For one event pair, return list of (sta_key, phase, dt_sec, cc) rows."""
    rows = []
    a_lookup = {(p.sta_key, p.phase): p for p in picks_a.itertuples()}
    for pb in picks_b.itertuples():
        key = (pb.sta_key, pb.phase)
        pa = a_lookup.get(key)
        if pa is None:
            continue
        # The differential time we want is observed (a_time - b_time) minus predicted
        # (a_origin - b_origin). GrowClust expects dt = (tt_a - tt_b) at this station:
        #   tt_a = pick_a - origin_a, tt_b = pick_b - origin_b → dt = (pick_a-pick_b) - (oa-ob)
        # We feed the *observed* differential pick time and let GrowClust handle predicted.
        # Cross-correlation refines the pick differential below.
        wa = wf_cache.get((pa.network, pa.station, pa.pick_time))
        wb = wf_cache.get((pb.network, pb.station, pb.pick_time))
        if wa is None or wb is None or wa.size != wb.size:
            continue
        result = xcorr_with_lag(wa, wb, max_lag_samples=int(MAX_LAG_SEC * FS_TARGET))
        if result is None:
            continue
        lag, cc = result
        if cc < CC_THRESH:
            continue
        # Observed pick-time difference (B relative to A), corrected by sub-sample lag
        pick_dt_obs = (pa.pick_time - pb.pick_time).total_seconds() + (lag / FS_TARGET)
        rows.append((pa.sta_key, pa.phase, pick_dt_obs, cc))
    return rows


def build_wf_cache_for_event(picks_e, stations_df):
    """Load + window waveforms for every pick of a single event. Returns dict keyed by (net, sta, pick_time)."""
    from obspy import read
    cache = {}
    for pk in picks_e.itertuples():
        try:
            row = stations_df.loc[pk.sta_key]
        except KeyError:
            continue
        path = _waveform_path(row.network, row.station, pk.pick_time)
        if not path.exists():
            continue
        try:
            st = read(str(path))
            arr = slice_window(st, pk.pick_time, WIN_SEC)
            cache[(row.network, row.station, pk.pick_time)] = arr
        except Exception:
            continue
    return cache


def attach_sta_key(picks: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Make sure picks has a 'sta_key' (net.sta) and matching network/station columns."""
    if "sta_key" in picks.columns:
        return picks
    if "network" in picks.columns and "station" in picks.columns:
        picks["sta_key"] = picks["network"].astype(str) + "." + picks["station"].astype(str)
        return picks
    # Otherwise station column may already be in net.sta form, or just sta.
    if "station" in picks.columns:
        # try matching against known station_geometry sta names
        sta_to_key = {row.station: idx for idx, row in stations_df.iterrows()}
        picks["sta_key"] = picks["station"].map(sta_to_key).fillna(picks["station"])
        picks["network"] = picks["sta_key"].str.split(".").str[0]
        picks["station"] = picks["sta_key"].str.split(".").str[1]
    return picks


# --------------------------------------------------------------------------------------
# GrowClust file writers
# --------------------------------------------------------------------------------------

def write_evlist(events: pd.DataFrame, out_path: Path):
    """GrowClust event list: yr mon day hr min sec lat lon dep mag eh ez rms id"""
    lines = []
    for i, e in enumerate(events.itertuples(), start=1):
        t = e.origin_time
        sec = t.second + t.microsecond * 1e-6
        mag = getattr(e, "magnitude", 0.0) if not np.isnan(getattr(e, "magnitude", np.nan)) else 0.0
        rms = getattr(e, "rms_residual", 0.0) if hasattr(e, "rms_residual") else 0.0
        eh = 0.0
        ez = 0.0
        depth = getattr(e, "depth", 0.0)
        lat = getattr(e, "latitude", getattr(e, "lat", np.nan))
        lon = getattr(e, "longitude", getattr(e, "lon", np.nan))
        lines.append(
            f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} "
            f"{sec:6.3f} {lat:9.5f} {lon:10.5f} {depth:7.3f} {mag:5.2f} "
            f"{eh:6.3f} {ez:6.3f} {rms:5.2f} {i:8d}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_stlist(stations_df: pd.DataFrame, used_keys: set, out_path: Path):
    """GrowClust stlist_fmt=1 expects 3 columns: sta lat lon (no elev)."""
    lines = []
    for sk in sorted(used_keys):
        if sk not in stations_df.index:
            continue
        r = stations_df.loc[sk]
        lines.append(f"{sk:12s} {r.latitude:9.5f} {r.longitude:10.5f}")
    out_path.write_text("\n".join(lines) + "\n")


def write_dtcc(pairs: dict, out_path: Path):
    """Write dt.cc. Format per GrowClust: '# id1 id2 0.0' then 'sta dt cc phase'."""
    lines = []
    for (id_a, id_b), rows in pairs.items():
        if not rows:
            continue
        lines.append(f"# {id_a:8d} {id_b:8d}   0.0")
        for sta_key, phase, dt, cc in rows:
            lines.append(f"{sta_key:12s} {dt:8.4f} {cc:6.3f} {phase}")
    out_path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    global WIN_SEC, CC_THRESH, MAX_LAG_SEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only")
    ap.add_argument("--max-dist-km", type=float, default=MAX_DIST_KM)
    ap.add_argument("--max-dt-sec", type=float, default=MAX_DT_SEC)
    ap.add_argument("--win-sec", type=float, default=WIN_SEC)
    ap.add_argument("--cc-thresh", type=float, default=CC_THRESH)
    ap.add_argument("--max-lag-sec", type=float, default=MAX_LAG_SEC)
    ap.add_argument("--max-pairs-per-event", type=int, default=80,
                    help="Cap nearest neighbours per event; keeps runtime bounded.")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # Override module globals so XC kernel sees consistent values
    WIN_SEC = args.win_sec
    CC_THRESH = args.cc_thresh
    MAX_LAG_SEC = args.max_lag_sec

    out_dir = REPO / "growclust" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pyocto output for label={args.label} ...")
    events, picks = load_pyocto(args.label)
    stations_df = load_stations()
    picks = attach_sta_key(picks, stations_df)
    print(f"  events: {len(events):,}   picks: {len(picks):,}")

    # Spatial neighbour search (flat-Earth)
    lat0 = events["latitude"].mean() if "latitude" in events.columns else events["lat"].mean()
    lon0 = events["longitude"].mean() if "longitude" in events.columns else events["lon"].mean()
    lats = events["latitude"].values if "latitude" in events.columns else events["lat"].values
    lons = events["longitude"].values if "longitude" in events.columns else events["lon"].values
    depths = events["depth"].values if "depth" in events.columns else np.zeros(len(events))
    ox, oy = latlon_to_xy_km(lats, lons, lat0, lon0)
    xyz = np.column_stack([ox, oy, depths])

    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    print(f"  building event neighbour tree on {len(events)} events (lat0={lat0:.3f}, lon0={lon0:.3f})")

    # Group picks by event_idx for fast lookup
    picks_by_event = {idx: g for idx, g in picks.groupby("event_idx")}

    # Identify pairs once
    print(f"  finding pairs within {args.max_dist_km} km (cap {args.max_pairs_per_event} per event)")
    pair_set = set()
    for i in range(len(events)):
        dists, idxs = tree.query(xyz[i], k=args.max_pairs_per_event + 1,
                                 distance_upper_bound=args.max_dist_km)
        for d, j in zip(dists[1:], idxs[1:]):
            if not np.isfinite(d) or j >= len(events) or j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            pair_set.add((a, b))
    print(f"  {len(pair_set):,} unique event pairs")

    # Process each event's waveform cache once, then iterate its pairs
    pair_results = {}
    used_keys = set()
    events_idx_col = events["event_idx"] if "event_idx" in events.columns else events.index
    event_idx_values = list(events_idx_col)

    # Group pairs by anchor event (the lower-index event) for cache reuse
    pairs_by_anchor = defaultdict(list)
    for a, b in pair_set:
        pairs_by_anchor[a].append(b)

    print(f"  cross-correlating ...")
    import time as _time
    t0 = _time.time()
    cache_a = None
    last_anchor = None
    n_done = 0
    for anchor, partners in pairs_by_anchor.items():
        ev_a_idx = event_idx_values[anchor]
        picks_a = picks_by_event.get(ev_a_idx, pd.DataFrame())
        if picks_a.empty:
            continue
        cache_a = build_wf_cache_for_event(picks_a, stations_df)
        for b in partners:
            ev_b_idx = event_idx_values[b]
            picks_b = picks_by_event.get(ev_b_idx, pd.DataFrame())
            if picks_b.empty:
                continue
            cache_b = build_wf_cache_for_event(picks_b, stations_df)
            cache = {**cache_a, **cache_b}
            rows = collect_pair_dts(events.iloc[anchor], events.iloc[b],
                                     picks_a, picks_b, cache)
            if rows:
                # event ids 1-based for GrowClust
                pair_results[(anchor + 1, b + 1)] = rows
                for sk, *_ in rows:
                    used_keys.add(sk)
        n_done += 1
        if n_done % 50 == 0:
            elapsed = _time.time() - t0
            kept = sum(len(r) for r in pair_results.values())
            print(f"    {n_done}/{len(pairs_by_anchor)} anchors  "
                  f"kept {len(pair_results)} pairs / {kept} obs  ({elapsed:.0f}s)")

    elapsed = _time.time() - t0
    kept = sum(len(r) for r in pair_results.values())
    print(f"  done. {len(pair_results):,} pairs, {kept:,} differential obs, "
          f"{len(used_keys)} stations  ({elapsed:.0f}s)")

    # Write GrowClust input files
    print(f"\nWriting GrowClust inputs to {out_dir}/")
    write_evlist(events, out_dir / "evlist.txt")
    write_stlist(stations_df, used_keys, out_dir / "stlist.txt")
    write_dtcc(pair_results, out_dir / "dt.cc")
    print(f"  wrote {out_dir/'evlist.txt'}")
    print(f"  wrote {out_dir/'stlist.txt'}")
    print(f"  wrote {out_dir/'dt.cc'}")

    print(f"\nDone. Next step: install GrowClust3D (Fortran or Julia) and run on these inputs.")


if __name__ == "__main__":
    sys.exit(main() or 0)
