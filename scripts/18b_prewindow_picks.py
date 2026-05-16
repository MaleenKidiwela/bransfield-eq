"""
Pre-window every (event, station, phase) pick into a flat float32 memmap.

This collapses the XC-prep working set from ~420 GB (cached station-day arrays)
to ~580 MB (one ±WIN_SEC window per pick), eliminating the mseed-decode-in-pair-loop
bottleneck that was projecting >12 days for Stage 3.

Reads:
    catalogs/pyocto_picks_<label>.csv

Writes:
    growclust/<label>/pick_windows.npy        float32 [n_picks, 300]
    growclust/<label>/pick_index.parquet      columns: pick_id, event_idx,
                                              network, station, sta_key, phase,
                                              pick_time, valid

Parallelised by station-day (each worker reads one mseed file). Workers write to
disjoint rows of the memmap, so no locking is needed.

Usage:
    python scripts/18b_prewindow_picks.py --label picker_only --workers 32
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
WAVE_DIR = REPO / "data" / "waveforms"

# Keep these constants in sync with scripts/18_growclust_xc_prep.py.
WIN_SEC = 1.5
FS_TARGET = 100.0
N_SAMPLES = int(2 * WIN_SEC * FS_TARGET)   # 300


def _waveform_path(network: str, station: str, year: int, doy: int) -> Path:
    """Match the convention used by 18_growclust_xc_prep.py:
    data/waveforms/{net}/{sta}/{net}.{sta}.{year}.{doy:03d}.mseed"""
    return WAVE_DIR / network / station / f"{network}.{station}.{year}.{doy:03d}.mseed"


def load_stations() -> pd.DataFrame:
    df = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    df["sta_key"] = df["network"].astype(str) + "." + df["station"].astype(str)
    return df.set_index("sta_key")


def _load_station_day_array(network, station, year, doy):
    """Read mseed, return (data_float32_at_FS_TARGET, starttime_unix) or None."""
    from obspy import read
    path = _waveform_path(network, station, year, doy)
    if not path.exists():
        return None
    try:
        st = read(str(path))
    except Exception:
        return None
    if len(st) == 0:
        return None
    z = [tr for tr in st if tr.stats.channel.endswith("Z")]
    if z:
        tr = z[0]
    else:
        try:
            tr = st.merge(fill_value=0)[0]
        except Exception:
            tr = st[0]
    if tr.stats.sampling_rate != FS_TARGET:
        try:
            tr.resample(FS_TARGET, no_filter=True)
        except Exception:
            return None
    data = np.ascontiguousarray(tr.data, dtype=np.float32)
    return data, float(tr.stats.starttime.timestamp)


def _slice_window(data, starttime_unix, pick_time_unix):
    """Cut a 2*WIN_SEC window, de-mean and L2-normalize. Same logic as XC script."""
    n_each = int(round(WIN_SEC * FS_TARGET))
    n_total = 2 * n_each
    centre = int(round((pick_time_unix - starttime_unix) * FS_TARGET))
    s0, s1 = centre - n_each, centre + n_each
    if s0 < 0 or s1 > len(data):
        out = np.zeros(n_total, dtype=np.float32)
        a, b = max(0, s0), min(len(data), s1)
        if b > a:
            out[a - s0 : b - s0] = data[a:b]
    else:
        out = np.ascontiguousarray(data[s0:s1], dtype=np.float32)
    out = out - out.mean()
    nm = np.linalg.norm(out)
    if nm > 0:
        out = out / nm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    out_dir = REPO / "growclust" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading picks for label={args.label} ...")
    picks = pd.read_csv(REPO / "catalogs" / f"pyocto_picks_{args.label}.csv")
    if pd.api.types.is_numeric_dtype(picks["time"] if "time" in picks.columns else picks["pick_time"]):
        time_col = "time" if "time" in picks.columns else "pick_time"
        picks["pick_time"] = pd.to_datetime(picks[time_col], unit="s", utc=True)
    elif "pick_time" not in picks.columns:
        picks["pick_time"] = pd.to_datetime(picks["time"], utc=True)
    if "event_idx" not in picks.columns and "idx" in picks.columns:
        picks = picks.rename(columns={"idx": "event_idx"})

    stations_df = load_stations()
    if "sta_key" not in picks.columns:
        if {"network", "station"}.issubset(picks.columns):
            picks["sta_key"] = picks["network"].astype(str) + "." + picks["station"].astype(str)
        else:
            picks["sta_key"] = picks["station"].astype(str)
            picks["network"] = picks["sta_key"].str.split(".").str[0]
            picks["station"] = picks["sta_key"].str.split(".").str[-1]
    else:
        if "network" not in picks.columns:
            picks["network"] = picks["sta_key"].str.split(".").str[0]
        if "station" not in picks.columns:
            picks["station"] = picks["sta_key"].str.split(".").str[-1]

    picks = picks.reset_index(drop=True)
    picks["pick_id"] = picks.index.astype(np.int64)
    n_picks = len(picks)
    print(f"  {n_picks:,} picks across {picks.sta_key.nunique()} stations")

    # Pre-allocate memmap (one row per pick, zero-initialised).
    mm_path = out_dir / "pick_windows.npy"
    print(f"  allocating memmap {mm_path}  ({n_picks * N_SAMPLES * 4 / 1e9:.2f} GB)")
    mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32,
                                   shape=(n_picks, N_SAMPLES))
    valid = np.zeros(n_picks, dtype=bool)

    # Group picks by station-day so each worker handles one mseed file.
    picks["year"] = picks.pick_time.dt.year
    picks["doy"] = picks.pick_time.dt.dayofyear
    groups = defaultdict(list)
    for row in picks.itertuples():
        groups[(row.network, row.station, row.year, row.doy)].append(
            (row.pick_id, row.pick_time.timestamp())
        )
    print(f"  {len(groups):,} station-day files to read")

    def _process(key):
        net, sta, year, doy = key
        loaded = _load_station_day_array(net, sta, year, doy)
        if loaded is None:
            return 0
        data, t0 = loaded
        n_ok = 0
        for pid, ptime_unix in groups[key]:
            mm[pid] = _slice_window(data, t0, ptime_unix)
            valid[pid] = True
            n_ok += 1
        return n_ok

    t_start = time.time()
    n_done_files = 0
    n_ok_total = 0
    n_miss = 0
    print(f"  pre-windowing with {args.workers} threads ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process, k): k for k in groups}
        for fut in as_completed(futures):
            try:
                got = fut.result()
            except Exception as e:
                k = futures[fut]
                print(f"    !!! {k} raised {type(e).__name__}: {e}", flush=True)
                got = 0
            n_done_files += 1
            if got == 0:
                n_miss += 1
            n_ok_total += got
            if n_done_files % 500 == 0:
                el = time.time() - t_start
                rate = n_done_files / max(el, 1)
                eta = (len(groups) - n_done_files) / max(rate, 1e-6)
                print(f"    {n_done_files}/{len(groups)} files  "
                      f"({n_ok_total:,} pick-windows)  "
                      f"({el:.0f}s, {rate:.1f} files/s, ETA {eta/60:.1f} min)",
                      flush=True)

    el = time.time() - t_start
    print(f"  done. wrote {valid.sum():,} / {n_picks:,} pick windows  "
          f"({n_miss} files missing/failed)  in {el:.0f}s")

    # Flush memmap to disk.
    mm.flush()
    del mm

    # Write pick index sidecar.
    idx = picks[["pick_id", "event_idx", "network", "station",
                 "sta_key", "phase", "pick_time"]].copy()
    idx["valid"] = valid
    idx_path = out_dir / "pick_index.parquet"
    idx.to_parquet(idx_path, index=False)
    print(f"  wrote {idx_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
