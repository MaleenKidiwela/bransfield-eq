"""
Stage 1.b — download continuous waveforms day-by-day.

Idempotent: skips any day already on disk. Designed for SLURM job arrays —
pass --shard I/N to have N parallel workers each take a disjoint slice of
(station, day) pairs.

Usage:
    python scripts/02_download_waveforms.py
    python scripts/02_download_waveforms.py --shard 3 --of 16
    python scripts/02_download_waveforms.py --network ZX --station BRA13
    python scripts/02_download_waveforms.py --start 2019-06-01 --end 2019-07-01
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from obspy import UTCDateTime

# repo-root on path so we can import the package without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bransfield_eq.config import (
    Config, daterange, get_client, mseed_path,
)
from bransfield_eq.stations import StationRef, resolve_stations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--start", default=None, help="override window start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="override window end (YYYY-MM-DD)")
    p.add_argument("--network", default=None, help="restrict to one network code")
    p.add_argument("--station", default=None, help="restrict to one station code")
    p.add_argument("--shard", type=int, default=0, help="this worker index (0-based)")
    p.add_argument("--of", type=int, default=1, help="total worker count")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-sleep", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def fetch_day(client, ref: StationRef, day: UTCDateTime, channels: str,
              retries: int, retry_sleep: float) -> int:
    """Download one station-day. Returns bytes written (0 if no data)."""
    out = mseed_path(ref.network, ref.station, day)
    if out.exists():
        return -1  # sentinel for "already on disk"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Per-channel bulk request — let FDSN expand the channel glob.
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            st = client.get_waveforms(
                network=ref.network, station=ref.station,
                location="*", channel=channels,
                starttime=day, endtime=day + 86400,
            )
            if len(st) == 0:
                # write a zero-byte sentinel so we don't retry empty days forever
                out.write_bytes(b"")
                return 0
            st.write(str(out), format="MSEED")
            return out.stat().st_size
        except Exception as e:
            msg = str(e)
            # 204 = no data for this day; treat as a real (empty) result.
            if "204" in msg or "No data available" in msg:
                out.write_bytes(b"")
                return 0
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"{ref.network}.{ref.station} {day.date}: {last_err}")


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.start:
        cfg.start = UTCDateTime(args.start)
    if args.end:
        cfg.end = UTCDateTime(args.end)

    print(f"Resolving stations from FDSN ...", flush=True)
    refs = resolve_stations(cfg)
    if args.network:
        refs = [r for r in refs if r.network == args.network]
    if args.station:
        refs = [r for r in refs if r.station == args.station]
    print(f"  {len(refs)} stations to download.")

    days = list(daterange(cfg.start, cfg.end))
    pairs = [(r, d) for r in refs for d in days]
    if args.of > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.of == args.shard]
    print(f"  shard {args.shard}/{args.of}: {len(pairs)} station-days "
          f"over {len(days)} days × {len(refs)} stations.")

    if args.dry_run:
        for r, d in pairs[:20]:
            print(f"    would fetch {r.network}.{r.station} {d.date}")
        if len(pairs) > 20:
            print(f"    ... +{len(pairs) - 20} more")
        return

    clients: dict[str, object] = {}
    n_done = n_skip = n_empty = n_err = 0
    bytes_total = 0
    for r, d in pairs:
        if r.data_center not in clients:
            clients[r.data_center] = get_client(r.data_center)
        try:
            n = fetch_day(clients[r.data_center], r, d, cfg.channels,
                          args.retries, args.retry_sleep)
        except Exception as e:
            n_err += 1
            print(f"  ERR {r.network}.{r.station} {d.date}: {e}", flush=True)
            continue
        if n == -1:
            n_skip += 1
        elif n == 0:
            n_empty += 1
        else:
            n_done += 1
            bytes_total += n
    print(f"\nDone. fetched={n_done}  skipped={n_skip}  empty={n_empty}  errors={n_err}  "
          f"bytes={bytes_total/1e9:.2f} GB")


if __name__ == "__main__":
    main()
