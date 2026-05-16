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
    # pyocto writes times as float epoch-seconds; pd.to_datetime defaults to ns
    # if the input is numeric, so we have to force unit='s' to avoid all events
    # landing on 1970-01-01.
    def _to_utc(s):
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_datetime(s, unit="s", utc=True)
        return pd.to_datetime(s, utc=True)
    ev["origin_time"] = _to_utc(ev["origin_time"])
    pk["pick_time"]   = _to_utc(pk["pick_time"])
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
        # Travel-time differential: (tt_a - tt_b) = (pick_a - origin_a) - (pick_b - origin_b)
        # Earlier versions wrote raw pick_a - pick_b (dominated by origin-time spread,
        # which broke GrowClust). Now properly subtract origin-time difference.
        oa = ev_a.origin_time if hasattr(ev_a, "origin_time") else ev_a.get("origin_time")
        ob = ev_b.origin_time if hasattr(ev_b, "origin_time") else ev_b.get("origin_time")
        origin_dt = (oa - ob).total_seconds()
        pick_dt_obs = (pa.pick_time - pb.pick_time).total_seconds() + (lag / FS_TARGET)
        tt_dt = pick_dt_obs - origin_dt
        rows.append((pa.sta_key, pa.phase, tt_dt, cc))
    return rows


import threading
from collections import OrderedDict
# Bounded LRU waveform cache. Each hit value is (data_float32_array_at_FS_TARGET,
# starttime_unix). False sentinel = "tried to load, failed". When the entry count
# exceeds _WF_CACHE_MAX, the oldest entry is evicted (the file is just re-read on
# demand later). Set _WF_CACHE_MAX via set_wf_cache_max() before workers start.
_WF_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_WF_CACHE_LOCK = threading.Lock()
_WF_CACHE_MAX = 1500  # ~35 MB per entry at 1 day × 100 Hz × float32 -> ~52 GiB cap

def set_wf_cache_max(n: int):
    global _WF_CACHE_MAX
    _WF_CACHE_MAX = int(n)

def _load_station_day_array(network, station, year, doy):
    """Read mseed for one station-day, return (data_array_float32_at_FS_TARGET, starttime_unix).
    Picks a Z (vertical) channel preferentially. Returns None if file missing/unreadable."""
    from obspy import read
    import numpy as _np
    # Build path
    class _DT:
        def __init__(s, y, d): s.year = y; s.dayofyear = d
    path = _waveform_path(network, station, _DT(year, doy))
    if not path.exists():
        return None
    try:
        st = read(str(path))
    except Exception:
        return None
    if len(st) == 0:
        return None
    # Prefer Z, else first available channel merged
    z_traces = [tr for tr in st if tr.stats.channel.endswith("Z")]
    if z_traces:
        tr = z_traces[0]
    else:
        try:
            tr = st.merge(fill_value=0)[0]
        except Exception:
            tr = st[0]
    # Resample once to FS_TARGET, then collapse to a plain numpy array
    if tr.stats.sampling_rate != FS_TARGET:
        try:
            tr.resample(FS_TARGET, no_filter=True)
        except Exception:
            return None
    data = _np.ascontiguousarray(tr.data, dtype=_np.float32)
    starttime_unix = float(tr.stats.starttime.timestamp)
    return data, starttime_unix


def _get_wf(network, station, dt):
    """Return (data_array, starttime_unix) for the station-day containing dt, or None.
    Bounded LRU: on hit, mark recently used; on miss, load then evict oldest if over cap."""
    key = (network, station, dt.year, dt.dayofyear)
    with _WF_CACHE_LOCK:
        val = _WF_CACHE.get(key)
        if val is not None:
            _WF_CACHE.move_to_end(key)
            return val if val is not False else None
    # Slow path: load outside the lock so other threads can still hit the cache.
    loaded = _load_station_day_array(network, station, dt.year, dt.dayofyear)
    with _WF_CACHE_LOCK:
        # Re-check in case another thread loaded the same key.
        existing = _WF_CACHE.get(key)
        if existing is not None:
            _WF_CACHE.move_to_end(key)
            return existing if existing is not False else None
        _WF_CACHE[key] = loaded if loaded is not None else False
        _WF_CACHE.move_to_end(key)
        # Evict oldest entries until under the cap.
        while len(_WF_CACHE) > _WF_CACHE_MAX:
            _WF_CACHE.popitem(last=False)
    return loaded


def slice_window_from_array(data, starttime_unix, pick_time_unix, win_sec):
    """Slice ±win_sec around a pick from a preloaded float32 array. Returns
    de-meaned, unit-norm numpy array. Pure numpy; no obspy."""
    import numpy as _np
    n_each = int(round(win_sec * FS_TARGET))
    n_total = 2 * n_each
    center_sample = int(round((pick_time_unix - starttime_unix) * FS_TARGET))
    s0 = center_sample - n_each
    s1 = center_sample + n_each
    if s0 < 0 or s1 > len(data):
        # Pad with zeros at boundaries
        out = _np.zeros(n_total, dtype=_np.float32)
        a, b = max(0, s0), min(len(data), s1)
        oa, ob = a - s0, b - s0
        if b > a:
            out[oa:ob] = data[a:b]
    else:
        out = _np.ascontiguousarray(data[s0:s1], dtype=_np.float32)
    out = out - out.mean()
    nm = _np.linalg.norm(out)
    if nm > 0:
        out = out / nm
    return out


def preload_all_station_days(picks, stations_df, workers):
    """Read+parse every station-day waveform file referenced by picks, once, in
    parallel. Populates _WF_CACHE so subsequent slice_window calls are pure numpy."""
    needed = set()
    for pk in picks.itertuples():
        try:
            row = stations_df.loc[pk.sta_key]
        except KeyError:
            continue
        dt = pk.pick_time
        needed.add((row.network, row.station, dt.year, dt.dayofyear))
    print(f"  preloading {len(needed):,} station-day waveform files "
          f"({workers} parallel readers) ...", flush=True)
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = _time.time()
    def _load_one(args):
        net, sta, year, doy = args
        key = (net, sta, year, doy)
        if key in _WF_CACHE:
            return None
        loaded = _load_station_day_array(net, sta, year, doy)
        _WF_CACHE[key] = loaded if loaded is not None else False
        return None
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in as_completed([ex.submit(_load_one, a) for a in needed]):
            done += 1
            if done % 200 == 0:
                el = _time.time() - t0
                print(f"    preloaded {done}/{len(needed)} ({el:.0f}s, "
                      f"{done/max(el,1):.1f}/s)", flush=True)
    el = _time.time() - t0
    n_ok = sum(1 for v in _WF_CACHE.values() if v is not False and v is not None)
    n_miss = sum(1 for v in _WF_CACHE.values() if v is False)
    print(f"  preload done: {n_ok} loaded, {n_miss} missing/failed ({el:.0f}s)", flush=True)


def build_wf_cache_for_event(picks_e, stations_df):
    """Build per-pick window arrays for one event. Pre-windowing mode (the default)
    looks each pick's window up by `pick_id` in the global PICK_WINDOWS memmap that
    18b_prewindow_picks.py wrote -- pure numpy, no mseed I/O. Legacy mode (when
    PICK_WINDOWS is None) falls back to the on-the-fly slice path."""
    cache = {}
    if PICK_WINDOWS is not None:
        for pk in picks_e.itertuples():
            try:
                row = stations_df.loc[pk.sta_key]
            except KeyError:
                continue
            pid = getattr(pk, "pick_id", None)
            if pid is None or pid < 0 or pid >= PICK_WINDOWS.shape[0]:
                continue
            arr = PICK_WINDOWS[pid]
            # Pre-window writes zeros when the window couldn't be cut (file missing /
            # out-of-bounds). Skip those — they have zero norm and would just produce
            # nonsense cross-correlations.
            if not arr.any():
                continue
            cache[(row.network, row.station, pk.pick_time)] = np.ascontiguousarray(arr)
        return cache
    # Legacy path: on-the-fly slice from per-station-day cache.
    for pk in picks_e.itertuples():
        try:
            row = stations_df.loc[pk.sta_key]
        except KeyError:
            continue
        val = _get_wf(row.network, row.station, pk.pick_time)
        if val is None:
            continue
        data_arr, starttime_unix = val
        try:
            pick_time_unix = pk.pick_time.timestamp()
            arr = slice_window_from_array(data_arr, starttime_unix, pick_time_unix, WIN_SEC)
            cache[(row.network, row.station, pk.pick_time)] = arr
        except Exception:
            continue
    return cache


# Module-level pre-windowed memmap, populated in main() when --prewindow path exists.
PICK_WINDOWS = None


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
    global WIN_SEC, CC_THRESH, MAX_LAG_SEC, PICK_WINDOWS
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
    ap.add_argument("--cache-size", type=int, default=1500,
                    help="Max station-days held in the in-memory LRU waveform cache. "
                         "Each entry ~35 MB at 1 day × 100 Hz × float32; 1500 -> ~52 GiB. "
                         "Bounded to keep the JupyterHub pod under its ~187 GiB cgroup cap.")
    ap.add_argument("--preload", action="store_true",
                    help="Eagerly preload every needed station-day into the LRU "
                         "cache before XC begins (legacy 'front-load' behavior). "
                         "Only sensible when --cache-size >= number of unique "
                         "station-days; otherwise eviction defeats the point.")
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

    # Pre-windowed snippet memmap (written by scripts/18b_prewindow_picks.py).
    # Strongly preferred over on-the-fly mseed reads — at 42k events the legacy
    # path projected ~12-42 days even with LRU + sort. See notes/17_session_2026-05-13.md.
    mm_path = out_dir / "pick_windows.npy"
    idx_path = out_dir / "pick_index.parquet"
    if mm_path.exists() and idx_path.exists():
        print(f"  loading pre-windowed snippets from {mm_path.name} ...")
        PICK_WINDOWS = np.load(mm_path, mmap_mode="r")
        idx = pd.read_parquet(idx_path)
        # 18b writes one row per pick in the same order as the input picks CSV,
        # so pick_id == row index. load_pyocto() doesn't reorder, so we can just
        # take the index. Sanity-check length matches.
        if len(picks) != len(idx):
            sys.exit(f"pick_index.parquet has {len(idx)} rows but loaded picks "
                     f"has {len(picks)} -- regenerate Stage 2.5.")
        picks = picks.reset_index(drop=True)
        picks["pick_id"] = picks.index.astype(np.int64)
        n_valid = int(idx["valid"].sum()) if "valid" in idx.columns else len(idx)
        print(f"  PICK_WINDOWS shape {PICK_WINDOWS.shape}  "
              f"({n_valid:,} valid / {len(idx):,} picks)")
    else:
        print(f"  [warn] no pre-windowed snippets found at {mm_path} -- "
              f"falling back to legacy on-the-fly mseed reads (very slow).")
        PICK_WINDOWS = None

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

    # Group pairs by anchor event (the lower-index event) for cache reuse.
    # We sort anchors AND each anchor's partner list by origin time below so the
    # LRU waveform cache sees temporal locality (adjacent anchors share most
    # station-days). Without this sort, 32 threads working anchors in arbitrary
    # order touch thousands of unique station-days per second and the cache
    # thrashes — observed 2026-05-13 with ETA ~42 days.
    pairs_by_anchor = defaultdict(list)
    for a, b in pair_set:
        pairs_by_anchor[a].append(b)

    # Compute event origin-time-in-seconds (cheap key) once, for sorting.
    if "origin_time" in events.columns:
        _ev_ts = events["origin_time"].astype("int64").values // 10**9  # ns -> s
    else:
        _ev_ts = events["time"].astype("int64").values
    # Sort partners within each anchor by their event time.
    for a in pairs_by_anchor:
        pairs_by_anchor[a].sort(key=lambda i: _ev_ts[i])

    if PICK_WINDOWS is None:
        # Legacy fallback: bounded LRU of on-the-fly mseed reads.
        set_wf_cache_max(args.cache_size)
        print(f"  lazy waveform cache enabled (LRU cap = {args.cache_size} station-days, "
              f"~{args.cache_size * 35 / 1024:.1f} GiB)", flush=True)
        if args.preload:
            preload_all_station_days(picks, stations_df,
                                     workers=min(args.workers * 2, 64))
    else:
        print(f"  using pre-windowed snippets in memmap (no station-day cache needed)",
              flush=True)

    print(f"  cross-correlating with {args.workers} threads ...")
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    t0 = _time.time()
    n_done = 0
    n_done_lock = Lock()

    def _process_anchor(anchor):
        """Build cache for this anchor, process all its partners. Returns dict of pair->rows."""
        local_results = {}
        ev_a_idx = event_idx_values[anchor]
        picks_a = picks_by_event.get(ev_a_idx, pd.DataFrame())
        if picks_a.empty:
            return local_results
        cache_a = build_wf_cache_for_event(picks_a, stations_df)
        for b in pairs_by_anchor[anchor]:
            ev_b_idx = event_idx_values[b]
            picks_b = picks_by_event.get(ev_b_idx, pd.DataFrame())
            if picks_b.empty:
                continue
            cache_b = build_wf_cache_for_event(picks_b, stations_df)
            cache = {**cache_a, **cache_b}
            rows = collect_pair_dts(events.iloc[anchor], events.iloc[b],
                                     picks_a, picks_b, cache)
            if rows:
                local_results[(anchor + 1, b + 1)] = rows
        return local_results

    # Sort anchors by origin time so ThreadPoolExecutor processes near-in-time
    # events together -- they share most station-days, giving the LRU real locality.
    anchors = sorted(pairs_by_anchor.keys(), key=lambda i: _ev_ts[i])
    total_anchors = len(anchors)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_anchor, a): a for a in anchors}
        for fut in as_completed(futures):
            try:
                lr = fut.result()
            except Exception as e:
                print(f"    !!! anchor {futures[fut]} raised: {type(e).__name__}: {e}", flush=True)
                continue
            for k, v in lr.items():
                pair_results[k] = v
                for sk, *_ in v:
                    used_keys.add(sk)
            with n_done_lock:
                n_done += 1
                if n_done % 50 == 0:
                    elapsed = _time.time() - t0
                    kept = sum(len(r) for r in pair_results.values())
                    rate = n_done / max(elapsed, 1)
                    eta = (total_anchors - n_done) / max(rate, 1e-6)
                    print(f"    {n_done}/{total_anchors} anchors  "
                          f"kept {len(pair_results)} pairs / {kept} obs  "
                          f"({elapsed:.0f}s, {rate:.1f}/s, ETA {eta/60:.1f}min)",
                          flush=True)

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
