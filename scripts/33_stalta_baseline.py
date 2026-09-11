"""Classical STA/LTA phase picker — the non-ML reference point for the picker benchmark.

P from the vertical, S from the horizontal energy envelope, both on a 3-20 Hz
bandpass (the band the OBS local events live in). Emits the same CSV schema as
scripts/03_run_phasenet.py so the same evaluation code reads both.

Usage:
    python scripts/33_stalta_baseline.py --days 2019-277 2019-279 --network ZX \
        --out-subdir picks_bench_stalta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read, UTCDateTime
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import WAVE_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--days", nargs="+", required=True, help="YYYY-DDD julian days")
    p.add_argument("--network", default="ZX")
    p.add_argument("--out-subdir", default="picks_bench_stalta")
    p.add_argument("--rate", type=float, default=100.0)
    p.add_argument("--freqmin", type=float, default=3.0)
    p.add_argument("--freqmax", type=float, default=20.0)
    p.add_argument("--p-sta", type=float, default=0.5)
    p.add_argument("--p-lta", type=float, default=10.0)
    p.add_argument("--s-sta", type=float, default=1.0)
    p.add_argument("--s-lta", type=float, default=10.0)
    p.add_argument("--on", type=float, default=3.5, help="trigger-on STA/LTA ratio")
    p.add_argument("--off", type=float, default=1.5, help="trigger-off STA/LTA ratio")
    return p.parse_args()


def _picks_from(data: np.ndarray, rate: float, sta: float, lta: float,
                on: float, off: float, t0: UTCDateTime, phase: str,
                trace_id: str) -> list[dict]:
    if len(data) < int(lta * rate) * 2:
        return []
    cft = recursive_sta_lta(data, int(sta * rate), int(lta * rate))
    rows = []
    for i0, i1 in trigger_onset(cft, on, off):
        rows.append({
            "time": str(t0 + i0 / rate),
            "trace_id": trace_id,
            "phase": phase,
            # map peak ratio onto (0,1] so it is usable like a model probability
            "prob": float(min(cft[i0:i1 + 1].max() / (on * 4.0), 1.0)),
            "start": str(t0 + i0 / rate),
            "end": str(t0 + i1 / rate),
        })
    return rows


def pick_one_day(mseed: Path, a: argparse.Namespace) -> pd.DataFrame:
    st = read(str(mseed))
    st = st.select(channel="[EHBS][HLP]?")  # seismometer channels only
    if len(st) == 0:
        return pd.DataFrame()
    st.merge(method=1, fill_value=0)
    for tr in st:
        if tr.stats.sampling_rate != a.rate:
            tr.resample(a.rate)
    st.detrend("demean")
    st.filter("bandpass", freqmin=a.freqmin, freqmax=a.freqmax, corners=4, zerophase=True)

    net = st[0].stats.network
    sta_code = st[0].stats.station
    trace_id = f"{net}.{sta_code}."
    rows: list[dict] = []

    vert = [tr for tr in st if tr.stats.channel.endswith("Z")]
    if vert:
        tr = vert[0]
        rows += _picks_from(tr.data.astype(float), a.rate, a.p_sta, a.p_lta,
                            a.on, a.off, tr.stats.starttime, "P", trace_id)

    hors = [tr for tr in st if not tr.stats.channel.endswith("Z")]
    if len(hors) >= 2:
        n = min(len(hors[0].data), len(hors[1].data))
        env = np.sqrt(hors[0].data[:n].astype(float) ** 2
                      + hors[1].data[:n].astype(float) ** 2)
        rows += _picks_from(env, a.rate, a.s_sta, a.s_lta, a.on, a.off,
                            hors[0].stats.starttime, "S", trace_id)

    return pd.DataFrame(rows)


def main() -> None:
    a = parse_args()
    out_root = REPO / "catalogs" / a.out_subdir
    n_days = n_picks = 0
    for day in a.days:
        year, jday = day.split("-")
        for sta_dir in sorted((WAVE_DIR / a.network).iterdir()):
            if not sta_dir.is_dir():
                continue
            mseed = sta_dir / f"{a.network}.{sta_dir.name}.{year}.{jday}.mseed"
            if not mseed.exists() or mseed.stat().st_size == 0:
                continue
            out_csv = out_root / f"{a.network}.{sta_dir.name}" / f"{year}-{jday}.csv"
            if out_csv.exists():
                continue
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            try:
                df = pick_one_day(mseed, a)
            except Exception as e:
                print(f"  ERR {sta_dir.name} {day}: {e}", flush=True)
                continue
            df.to_csv(out_csv, index=False)
            n_days += 1
            n_picks += len(df)
    print(f"Done. station-days={n_days}  total_picks={n_picks}")


if __name__ == "__main__":
    main()
