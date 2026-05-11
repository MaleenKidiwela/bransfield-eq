"""
Run STA/LTA on DD-denoised waveforms for Feb 4-13, output triggers, and
report which manual mag07 picks were caught vs missed.

Pipeline per station-day:
  1. Load raw mseed (Z component)
  2. Apply OBS DeepDenoiser via model.annotate(stream)
  3. Compute STA/LTA on denoised Z
  4. trigger_onset to get start/end of each candidate event
  5. Output one "pick" per trigger onset (no P/S distinction)

Then: match each manual mag07 pick against STA/LTA triggers within tolerance.
- "Caught": any trigger within ±2 s of the manual pick
- "Missed": no trigger within ±2 s

Outputs:
  catalogs/picks_stalta_dd/<NET>.<STA>/<YYYY-DDD>.csv
  catalogs/stalta_dd_recall.csv  (per-pick caught/missed)
  figures/stalta_dd_summary.png
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
import torch
from obspy import UTCDateTime, read
from obspy.signal.trigger import classic_sta_lta, trigger_onset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import mseed_path, daterange, Config  # noqa: E402

import seisbench.models as sbm  # noqa: E402

DENOISER_CKPT = REPO / "models" / "deepdenoiser_obs" / "best.pt"
TARGET_RATE = 100.0
STA_SEC = 0.5
LTA_SEC = 10.0
TRIG_ON = 3.0
TRIG_OFF = 1.5
MATCH_TOL_S = 2.0

_W: dict = {}


def _ch_match(code, glob):
    import fnmatch
    return any(fnmatch.fnmatchcase(code, g.strip()) for g in glob.split(","))


def _init_worker(out_root_str, sta_sec, lta_sec, trig_on, trig_off,
                 picking_glob, device):
    model = sbm.DeepDenoiser.from_pretrained("original")
    ck = torch.load(DENOISER_CKPT, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    _W.update(model=model, out_root=Path(out_root_str),
              sta_sec=sta_sec, lta_sec=lta_sec,
              trig_on=trig_on, trig_off=trig_off,
              picking_glob=picking_glob, device=device)


def _process_one(task):
    net, sta, day_iso, mp_str = task
    s = _W
    day = UTCDateTime(day_iso)
    out_csv = s["out_root"] / f"{net}.{sta}" / f"{day.year}-{day.julday:03d}.csv"
    if out_csv.exists():
        return ("skip", net, sta, str(day.date), 0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        st = read(mp_str).copy()
        st.traces = [tr for tr in st if _ch_match(tr.stats.channel, s["picking_glob"])]
        if len(st) == 0:
            pd.DataFrame().to_csv(out_csv, index=False)
            return ("done", net, sta, str(day.date), 0)
        st.merge(method=1, fill_value=0)
        for tr in st:
            if abs(tr.stats.sampling_rate - TARGET_RATE) > 1e-6:
                tr.resample(TARGET_RATE)
        denoised = s["model"].annotate(st)
        z = next((t for t in denoised if t.stats.channel.endswith("Z")), None)
        if z is None:
            pd.DataFrame().to_csv(out_csv, index=False)
            return ("done", net, sta, str(day.date), 0)
        nsta = int(s["sta_sec"] * TARGET_RATE)
        nlta = int(s["lta_sec"] * TARGET_RATE)
        cft = classic_sta_lta(z.data.astype(np.float32), nsta, nlta)
        triggers = trigger_onset(cft, s["trig_on"], s["trig_off"])
        rows = []
        for on, off in triggers:
            t_on = z.stats.starttime + on / TARGET_RATE
            t_off = z.stats.starttime + off / TARGET_RATE
            cft_max = float(cft[on:off+1].max()) if off >= on else 0.0
            rows.append({
                "time": str(t_on),
                "trace_id": f"{net}.{sta}..{z.stats.channel}",
                "phase": "P",
                "prob": cft_max,
                "start": str(t_on),
                "end": str(t_off),
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return ("done", net, sta, str(day.date), len(rows))
    except Exception as e:
        return ("err", net, sta, str(day.date), str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-02-04")
    ap.add_argument("--end", default="2019-02-14")
    ap.add_argument("--out-subdir", default="picks_stalta_dd")
    ap.add_argument("--sta-sec", type=float, default=STA_SEC)
    ap.add_argument("--lta-sec", type=float, default=LTA_SEC)
    ap.add_argument("--trig-on", type=float, default=TRIG_ON)
    ap.add_argument("--trig-off", type=float, default=TRIG_OFF)
    ap.add_argument("--match-tol", type=float, default=MATCH_TOL_S)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    out_root = REPO / "catalogs" / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"STA/LTA on DD-denoised  {args.start} → {args.end}")
    print(f"  STA={args.sta_sec}s, LTA={args.lta_sec}s, trig_on={args.trig_on}, trig_off={args.trig_off}")
    print(f"  workers={args.workers}  device={device}")
    print(f"  Output: {out_root}\n")

    cfg = Config.load(None)
    start, end = UTCDateTime(args.start), UTCDateTime(args.end)

    # List all station-days with mseed
    pairs = []
    wave_dir = REPO / "data" / "waveforms"
    for net_dir in sorted(wave_dir.iterdir()):
        if not net_dir.is_dir(): continue
        for sta_dir in sorted(net_dir.iterdir()):
            if not sta_dir.is_dir(): continue
            for day in daterange(start, end):
                p = mseed_path(net_dir.name, sta_dir.name, day)
                if p.exists() and p.stat().st_size > 0:
                    pairs.append((net_dir.name, sta_dir.name, day, p))
    print(f"  {len(pairs)} station-days to process\n")

    tasks = [(n, s, d.isoformat(), str(p)) for (n, s, d, p) in pairs]
    n_done = n_skip = n_err = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(out_root), args.sta_sec, args.lta_sec, args.trig_on,
                  args.trig_off, cfg.picking_channels, device),
    ) as pool:
        futs = [pool.submit(_process_one, t) for t in tasks]
        for fut in as_completed(futs):
            status, net, sta, day, npks = fut.result()
            if status == "skip":
                n_skip += 1
            elif status == "err":
                n_err += 1
                print(f"  ERR {net}.{sta} {day}: {npks}", flush=True)
            else:
                n_done += 1
                if (n_done + n_skip) % 25 == 0:
                    print(f"  {n_done+n_skip}/{len(tasks)} done  picks_so_far={npks}",
                          flush=True)
    print(f"\nPicker done. done={n_done} skip={n_skip} err={n_err}\n")

    # Now check coverage of manual picks
    print("=== Matching manual mag07 picks vs STA/LTA triggers ===\n")
    m = pd.read_csv(REPO/"catalogs/manual_picks.csv", parse_dates=["pick_time"])
    m07 = m[m.source_file=="nllmaleen_mag07_202210.out"].copy()
    m07["t"] = pd.to_datetime(m07.pick_time, utc=True)
    m07 = m07[(m07.t >= args.start) & (m07.t < args.end)].copy()
    m07["phase"] = m07.phase.str.upper().str[0]
    m07["day"] = m07.pick_time.dt.date

    # Load all STA/LTA triggers, group by station-day
    trigger_times = {}  # (net, sta, day) -> sorted np.array of trigger start times (unix)
    for sta_dir in sorted(out_root.iterdir()):
        if not sta_dir.is_dir(): continue
        try:
            net, sta = sta_dir.name.split(".")
        except ValueError: continue
        for csv in sorted(sta_dir.glob("*.csv")):
            year, jday = csv.stem.split("-")
            day = (UTCDateTime(f"{year}-01-01") + (int(jday)-1)*86400).date
            try:
                df = pd.read_csv(csv)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if df.empty: continue
            ts = pd.to_datetime(df["time"], utc=True).astype("int64") // 10**9
            trigger_times[(net, sta, day)] = np.array(sorted(ts.values))

    # Match
    m07["caught"] = False
    m07["nearest_trigger_dt"] = np.nan
    tol = args.match_tol
    for i, r in m07.iterrows():
        key = (r.network, r.station, r.day)
        trigs = trigger_times.get(key)
        if trigs is None or len(trigs) == 0:
            continue
        ts = int(pd.Timestamp(r.t).timestamp())
        idx = np.searchsorted(trigs, ts)
        cand = []
        if idx > 0: cand.append(abs(trigs[idx-1] - ts))
        if idx < len(trigs): cand.append(abs(trigs[idx] - ts))
        nearest = min(cand) if cand else np.nan
        m07.at[i, "nearest_trigger_dt"] = nearest
        m07.at[i, "caught"] = (nearest <= tol) if not np.isnan(nearest) else False

    out_match = REPO/"catalogs/stalta_dd_recall.csv"
    m07[["event_id","network","station","phase","pick_time","uncertainty_s",
         "caught","nearest_trigger_dt"]].to_csv(out_match, index=False)
    print(f"Wrote {out_match}")
    print()
    # Summary
    n_total = len(m07)
    n_caught = int(m07.caught.sum())
    print(f"Manual picks total:  {n_total}")
    print(f"Caught (≤±{tol}s):   {n_caught}  ({n_caught/n_total*100:.1f}%)")
    print(f"Missed:              {n_total-n_caught}  ({(n_total-n_caught)/n_total*100:.1f}%)")
    print()
    print("By phase:")
    for ph in ('P','S'):
        sub = m07[m07.phase==ph]
        if sub.empty: continue
        c = sub.caught.sum()
        print(f"  {ph}: {c}/{len(sub)} caught  ({c/len(sub)*100:.1f}%)")
    print()
    print("By confidence:")
    for label, mask in [("high (≤0.1s)", m07.uncertainty_s<=0.1),
                        ("low (>0.1s)",  m07.uncertainty_s>0.1)]:
        sub = m07[mask]
        if sub.empty: continue
        c = sub.caught.sum()
        print(f"  {label}: {c}/{len(sub)} caught ({c/len(sub)*100:.1f}%)")
    print()
    n_trig_total = sum(len(v) for v in trigger_times.values())
    print(f"Total STA/LTA triggers across all station-days: {n_trig_total:,}")


if __name__ == "__main__":
    main()
