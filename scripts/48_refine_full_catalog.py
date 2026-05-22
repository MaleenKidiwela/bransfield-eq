"""Apply the pick-refinement workflow to a full event pool (not a 50-event
sample). Re-uses refine_event() from script 46 but wraps load_picker_picks
in an LRU cache so each (picker, network, station, day) CSV is read only
once across the whole catalog.

Output:
    nlloc/obs/<label>.obs
    nlloc/obs/<label>.event_order.csv
    nlloc/run/<label>.in
    catalogs/<label>_refinement_summary.csv

Then run:  NLLoc nlloc/run/<label>.in   (or use --shards via script 30)
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_s42_spec = importlib.util.spec_from_file_location(
    "_s42", REPO / "scripts" / "42_plot_event_picker_probs.py"
)
_s42 = importlib.util.module_from_spec(_s42_spec)
_s42_spec.loader.exec_module(_s42)

# --- Cache picker-CSV loads -------------------------------------------------
# load_picker_picks normally re-reads per-station-per-day CSVs every event,
# which is the dominant cost across thousands of events. Cache by
# (picker_dir, network, station, julian-day).
_cache_hits = 0
_cache_misses = 0
_orig_load = _s42.load_picker_picks


def _picks_for_day(picker_dir: Path, net: str, sta: str,
                   year: int, doy: int) -> pd.DataFrame:
    pdir = picker_dir / f"{net}.{sta}"
    fp = pdir / f"{year}-{doy:03d}.csv"
    if not fp.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(fp)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["time_obj"] = pd.to_datetime(df["time"], utc=True)
    df["start_obj"] = pd.to_datetime(df["start"], utc=True)
    df["end_obj"] = pd.to_datetime(df["end"], utc=True)
    return df


@lru_cache(maxsize=100_000)
def _picks_for_day_cached(picker_dir: str, net: str, sta: str,
                          year: int, doy: int) -> pd.DataFrame:
    global _cache_misses
    _cache_misses += 1
    return _picks_for_day(Path(picker_dir), net, sta, year, doy)


def cached_load_picker_picks(picker_dir: Path, net: str, sta: str,
                             t0: UTCDateTime, t1: UTCDateTime) -> pd.DataFrame:
    global _cache_hits
    days = sorted({(t0.year, t0.julday), (t1.year, t1.julday)})
    frames = []
    for yr, doy in days:
        df = _picks_for_day_cached(str(picker_dir), net, sta, yr, doy)
        if df is None or df.empty:
            continue
        frames.append(df)
    _cache_hits += 1
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ts0 = pd.Timestamp(t0.datetime, tz="UTC")
    ts1 = pd.Timestamp(t1.datetime, tz="UTC")
    mask = (df["time_obj"] >= ts0) & (df["time_obj"] <= ts1)
    return df.loc[mask].reset_index(drop=True)


_s42.load_picker_picks = cached_load_picker_picks

_s46_spec = importlib.util.spec_from_file_location(
    "_s46", REPO / "scripts" / "46_ablation_pick_refinement.py"
)
_s46 = importlib.util.module_from_spec(_s46_spec)
_s46_spec.loader.exec_module(_s46)
# Script 46 imports its OWN _s42 module instance -- patch that too,
# otherwise refine_event() bypasses our cache entirely.
_s46._s42.load_picker_picks = cached_load_picker_picks
refine_event = _s46.refine_event
fmt_obs_line = _s46.fmt_obs_line
P_ERR_S = _s46.P_ERR_S
S_ERR_S = _s46.S_ERR_S

_s44_spec = importlib.util.spec_from_file_location(
    "_s44", REPO / "scripts" / "44_refine_picks_event.py"
)
_s44 = importlib.util.module_from_spec(_s44_spec)
_s44_spec.loader.exec_module(_s44)

PYO_PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv"
ST_CSV     = REPO / "catalogs" / "station_geometry.csv"
BASE_CTRL  = REPO / "nlloc" / "run" / "picker_only_no_shots_v2.in"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-csv", required=True,
                    help="CSV of events to refine (must have event_idx, lat, "
                         "lon, depth_km, origin_time)")
    ap.add_argument("--label", required=True,
                    help="output label (used for obs file, run, output dir)")
    ap.add_argument("--min-phases", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on events processed (for quick tests)")
    args = ap.parse_args()

    pool = pd.read_csv(args.pool_csv).sort_values("origin_time").reset_index(drop=True)
    if args.limit:
        pool = pool.iloc[:args.limit].reset_index(drop=True)
    print(f"event pool: {len(pool)} from {args.pool_csv}")

    pyo_pk_all = pd.read_csv(PYO_PK_CSV)
    pyo_by_ev = pyo_pk_all.groupby("event_idx", sort=False)
    st_df = pd.read_csv(ST_CSV)
    corrections = _s44.load_station_corrections()

    obs_dir = REPO / "nlloc" / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    obs_path = obs_dir / f"{args.label}.obs"
    order_path = obs_dir / f"{args.label}.event_order.csv"

    summary = []
    t_start = time.time()
    with obs_path.open("w") as fh, order_path.open("w") as ofh:
        ofh.write("obs_order,event_idx\n")
        order_idx = 0
        for i, ev in pool.iterrows():
            eidx = int(ev.event_idx)
            ot = UTCDateTime(pd.Timestamp(ev.origin_time).to_pydatetime())
            try:
                pk_ev = pyo_by_ev.get_group(eidx)
            except KeyError:
                continue
            rows, n_repl, n_add, n_keep = refine_event(
                eidx, float(ev.lat), float(ev.lon), float(ev.depth_km),
                ot, st_df, pk_ev, corrections)
            if len(rows) < args.min_phases:
                continue
            rows_sorted = sorted(rows, key=lambda r: r[2])
            for full, phase, abs_time in rows_sorted:
                bare = full.split(".", 1)[1]
                dt = pd.Timestamp(abs_time, unit="s", tz="UTC")
                err = P_ERR_S if phase == "P" else S_ERR_S
                fh.write(fmt_obs_line(bare, dt, phase, err) + "\n")
            fh.write("\n")
            ofh.write(f"{order_idx},{eidx}\n")
            order_idx += 1
            summary.append({"event_idx": eidx,
                            "n_pyocto": len(pk_ev),
                            "n_refined": len(rows),
                            "n_replaced": n_repl,
                            "n_added": n_add,
                            "n_kept": n_keep})
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(pool) - i - 1) / rate / 60
                print(f"  {i+1:5d}/{len(pool):5d}  "
                      f"rate {rate:.1f} ev/s  ETA {eta:.1f} min  "
                      f"cache_misses {_cache_misses}", flush=True)

    sdf = pd.DataFrame(summary)
    print(f"\nwrote {obs_path}  ({len(sdf)} events refined)")
    print(f"      {order_path}")
    print(f"cache hits {_cache_hits:,}  misses {_cache_misses:,}")
    sdf_path = REPO / "catalogs" / f"{args.label}_refinement_summary.csv"
    sdf.to_csv(sdf_path, index=False)
    print(f"      {sdf_path}")

    # Build the NLLoc control file by template-replacing the v2 control.
    base_text = BASE_CTRL.read_text()
    out_dir = REPO / "nlloc" / "output" / args.label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = base_text.replace(
        "nlloc/obs/picker_only_no_shots_v2.obs",
        f"nlloc/obs/{args.label}.obs",
    ).replace(
        "nlloc/output/picker_only_no_shots_v2/loc",
        f"nlloc/output/{args.label}/loc",
    ).replace(
        "LOCCOM picker_only_no_shots_v2",
        f"LOCCOM {args.label}",
    )
    ctrl_path = REPO / "nlloc" / "run" / f"{args.label}.in"
    ctrl_path.write_text(text)
    print(f"      {ctrl_path}")
    print(f"\nNow run:  python scripts/30_run_nlloc.py --label {args.label} --shards 8")


if __name__ == "__main__":
    main()
