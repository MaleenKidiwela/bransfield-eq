"""
Stage 1.c — run SeisBench PhaseNet on downloaded waveforms.

Reads per-station-day MSEED files written by 02_download_waveforms.py and
writes per-station-day pick CSVs. Idempotent: skips days that already have
a pick CSV on disk.

Usage:
    python scripts/03_run_phasenet.py
    python scripts/03_run_phasenet.py --shard 3 --of 16
    python scripts/03_run_phasenet.py --weights instance --p-thresh 0.4 --s-thresh 0.4
    python scripts/03_run_phasenet.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime, read

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bransfield_eq.config import (
    Config, WAVE_DIR, daterange, mseed_path, pick_csv_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--network", default=None)
    p.add_argument("--station", default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--of", type=int, default=1)
    p.add_argument("--weights", default="instance",
                   help="SeisBench pretrained tag (default 'instance' — "
                        "validated on Bransfield Dec 2019 to outperform "
                        "'stead' on OBS data; recall ~45% vs 9% on P)")
    p.add_argument("--p-thresh", type=float, default=0.2)
    p.add_argument("--s-thresh", type=float, default=0.2)
    p.add_argument("--target-rate", type=float, default=100.0,
                   help="Resample to this rate before picking (PhaseNet default 100 Hz)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0 | mps")
    return p.parse_args()


def list_station_days(start: UTCDateTime, end: UTCDateTime,
                      network: str | None, station: str | None
                      ) -> list[tuple[str, str, UTCDateTime, Path]]:
    """Find all (net, sta, day, mseed_path) tuples that have data on disk."""
    out = []
    if not WAVE_DIR.exists():
        return out
    for net_dir in sorted(WAVE_DIR.iterdir()):
        if not net_dir.is_dir():
            continue
        if network and net_dir.name != network:
            continue
        for sta_dir in sorted(net_dir.iterdir()):
            if not sta_dir.is_dir():
                continue
            if station and sta_dir.name != station:
                continue
            for day in daterange(start, end):
                p = mseed_path(net_dir.name, sta_dir.name, day)
                if p.exists() and p.stat().st_size > 0:
                    out.append((net_dir.name, sta_dir.name, day, p))
    return out


def _channel_match(code: str, glob: str) -> bool:
    """Match a channel code against a comma-separated FDSN-style glob (?, *)."""
    import fnmatch
    return any(fnmatch.fnmatchcase(code, g.strip()) for g in glob.split(","))


def pick_one_day(model, mseed: Path, target_rate: float,
                 p_thresh: float, s_thresh: float,
                 picking_glob: str) -> pd.DataFrame:
    """Read one station-day, run PhaseNet, return picks DataFrame."""
    st = read(str(mseed))
    # Drop hydrophone / non-seismic channels before PhaseNet sees them.
    st = st.select(channel="*")  # copy
    st.traces = [tr for tr in st if _channel_match(tr.stats.channel, picking_glob)]
    if len(st) == 0:
        return pd.DataFrame()
    st.merge(method=1, fill_value=0)
    # Resample to a uniform target rate so 3C are aligned (e.g. ZX has 200/100 mix).
    for tr in st:
        if tr.stats.sampling_rate != target_rate:
            tr.resample(target_rate)

    picks = model.classify(
        st,
        P_threshold=p_thresh,
        S_threshold=s_thresh,
    ).picks

    rows = []
    for pk in picks:
        rows.append({
            "time": str(pk.peak_time),
            "trace_id": pk.trace_id,
            "phase": pk.phase,
            "prob": float(pk.peak_value),
            "start": str(pk.start_time),
            "end": str(pk.end_time),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.start:
        cfg.start = UTCDateTime(args.start)
    if args.end:
        cfg.end = UTCDateTime(args.end)

    # Lazy import — torch + seisbench are heavy and not needed for --help.
    import seisbench.models as sbm
    print(f"Loading PhaseNet weights: {args.weights} on {args.device}", flush=True)
    model = sbm.PhaseNet.from_pretrained(args.weights)
    model.to(args.device)

    pairs = list_station_days(cfg.start, cfg.end, args.network, args.station)
    if args.of > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.of == args.shard]
    print(f"  shard {args.shard}/{args.of}: {len(pairs)} station-days to pick.")

    n_done = n_skip = n_err = n_picks = 0
    for net, sta, day, mseed in pairs:
        out_csv = pick_csv_path(net, sta, day)
        if out_csv.exists():
            n_skip += 1
            continue
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            df = pick_one_day(model, mseed, args.target_rate,
                              args.p_thresh, args.s_thresh,
                              cfg.picking_channels)
        except Exception as e:
            n_err += 1
            print(f"  ERR {net}.{sta} {day.date}: {e}", flush=True)
            continue
        df.to_csv(out_csv, index=False)
        n_done += 1
        n_picks += len(df)
        if n_done % 25 == 0:
            print(f"  ... {n_done} days picked, {n_picks} picks so far", flush=True)
    print(f"\nDone. picked={n_done}  skipped={n_skip}  errors={n_err}  total_picks={n_picks}")


if __name__ == "__main__":
    main()
