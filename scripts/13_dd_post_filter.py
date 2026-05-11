"""
DD-residual post-filter for picker output.

For a given baseline picker output dir (e.g. catalogs/picks/), compute the
OBS-DeepDenoiser residual energy at each pick time and at noise-baseline
windows. Output an augmented CSV with `dd_anomaly_ratio` per pick.

Then sweep ratio thresholds and report precision/recall vs the manual
catalog at each threshold — to test whether the DD residual is a useful
post-hoc anomaly score for separating real picks from noise picks.

Usage:
    python scripts/13_dd_post_filter.py \
        --picks-subdir picks --start 2019-02-04 --end 2019-02-14 \
        --out-subdir picks_dd_filtered

The output is written to catalogs/<--out-subdir>/<NET>.<STA>/<YYYY-DDD>.csv
preserving the schema of the input plus a `dd_anomaly_ratio` column.

Validation: separately, run scripts/05_validate_picks.py against the input
subdir (existing) and the output subdir at each threshold (filtering rows
with ratio < threshold) to compare TP/FP/FN trade-offs.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obspy import Stream, Trace, UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import (Config, daterange, mseed_path)  # noqa: E402

import seisbench.models as sbm  # noqa: E402

DENOISER_CKPT = REPO / "models" / "deepdenoiser_obs" / "best.pt"
TARGET_RATE = 100.0
WINDOW_HALF_SEC = 2.0  # ±2 s window around each pick for residual energy
N_BASELINE_WINDOWS = 100  # sample N random windows on the day for noise baseline


def _channel_match(code: str, glob: str) -> bool:
    import fnmatch
    return any(fnmatch.fnmatchcase(code, g.strip()) for g in glob.split(","))


def denoise_full_day(model, st: Stream, device: str) -> Stream:
    """Apply DeepDenoiser to a full-day stream via annotate (handles STFT/iSTFT)."""
    return model.annotate(st)


def signal_energy_at(times_utc: list[UTCDateTime], denoised: Stream,
                     half_sec: float) -> np.ndarray:
    """For each requested time, compute RMS of the DENOISED (signal-only)
    waveform on Z component in a ±half_sec window. Returns 1-D array.

    Rationale: DeepDenoiser is a signal extractor — its output IS the signal
    estimate. High amplitude in the denoised stream means the model identified
    real signal there. So denoised-energy is a direct "is there a real arrival?"
    anomaly score. Higher = more event-like."""
    z_den = next((tr for tr in denoised if tr.stats.channel.endswith("Z")), None)
    if z_den is None:
        return np.zeros(len(times_utc), dtype=np.float32)
    energies = np.zeros(len(times_utc), dtype=np.float32)
    fs = z_den.stats.sampling_rate
    half = int(round(half_sec * fs))
    den_data = z_den.data.astype(np.float32)
    n = len(den_data)
    t0 = z_den.stats.starttime
    for i, t in enumerate(times_utc):
        idx = int(round((t - t0) * fs))
        a = max(0, idx - half)
        b = min(n, idx + half + 1)
        if b <= a:
            continue
        seg = den_data[a:b]
        energies[i] = float(np.sqrt(np.mean(seg ** 2) + 1e-20))
    return energies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks-subdir", default="picks",
                    help="Input picker output dir under catalogs/")
    ap.add_argument("--out-subdir", default="picks_dd_filtered",
                    help="Output dir for augmented CSVs")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--half-sec", type=float, default=WINDOW_HALF_SEC)
    ap.add_argument("--n-baseline", type=int, default=N_BASELINE_WINDOWS)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    cfg = Config.load(None)
    pick_root_in = REPO / "catalogs" / args.picks_subdir
    pick_root_out = REPO / "catalogs" / args.out_subdir
    pick_root_out.mkdir(parents=True, exist_ok=True)

    print(f"DD post-filter on {args.picks_subdir} → {args.out_subdir}")
    print(f"  device={device}  half_sec={args.half_sec}  n_baseline={args.n_baseline}")

    # Load denoiser
    model = sbm.DeepDenoiser.from_pretrained("original")
    ck = torch.load(DENOISER_CKPT, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    start, end = UTCDateTime(args.start), UTCDateTime(args.end)
    rng = np.random.default_rng(0)

    # Iterate (station-day) combos that have BOTH input picks and waveforms.
    n_done = n_skip = 0
    for sta_dir_in in sorted(pick_root_in.iterdir()):
        if not sta_dir_in.is_dir():
            continue
        try:
            net, sta = sta_dir_in.name.split(".")
        except ValueError:
            continue
        for csv_in in sorted(sta_dir_in.glob("*.csv")):
            try:
                year, jday = csv_in.stem.split("-")
                year, jday = int(year), int(jday)
            except ValueError:
                continue
            day = UTCDateTime(f"{year}-01-01") + (jday - 1) * 86400
            if day < start or day >= end:
                continue
            csv_out = pick_root_out / sta_dir_in.name / csv_in.name
            if csv_out.exists():
                n_skip += 1
                continue
            csv_out.parent.mkdir(parents=True, exist_ok=True)

            # Load picks
            try:
                picks = pd.read_csv(csv_in)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pd.DataFrame().to_csv(csv_out, index=False)
                n_done += 1
                continue
            if picks.empty:
                picks.to_csv(csv_out, index=False)
                n_done += 1
                continue

            # Load and denoise mseed for this station-day
            mp = mseed_path(net, sta, day)
            if not mp.exists() or mp.stat().st_size == 0:
                continue
            try:
                st = read(str(mp)).copy()
            except Exception:
                continue
            st.traces = [tr for tr in st
                         if _channel_match(tr.stats.channel, cfg.picking_channels)]
            if len(st) == 0:
                continue
            st.merge(method=1, fill_value=0)
            for tr in st:
                if abs(tr.stats.sampling_rate - TARGET_RATE) > 1e-6:
                    tr.resample(TARGET_RATE)
            try:
                denoised = denoise_full_day(model, st, device)
            except Exception as e:
                print(f"  [skip] {net}.{sta} {day.date}: denoise failed: {e}", flush=True)
                continue

            # Pick times → DENOISED signal energy at each
            try:
                pick_times = [UTCDateTime(t) for t in picks["time"]]
            except Exception:
                pick_times = [UTCDateTime(pd.Timestamp(t).to_pydatetime())
                              for t in picks["time"]]
            pick_energies = signal_energy_at(pick_times, denoised, args.half_sec)
            # Baseline: sample N random windows on the day (most are noise-only)
            z_den = next((tr for tr in denoised if tr.stats.channel.endswith("Z")), None)
            if z_den is None:
                continue
            day_secs = z_den.stats.endtime - z_den.stats.starttime
            base_times = [day + rng.uniform(args.half_sec, day_secs - args.half_sec)
                          for _ in range(args.n_baseline)]
            base_energies = signal_energy_at(base_times, denoised, args.half_sec)
            baseline_median = float(np.median(base_energies)) + 1e-20

            picks = picks.copy()
            picks["dd_signal_rms"] = pick_energies
            picks["dd_baseline_median"] = baseline_median
            picks["dd_anomaly_ratio"] = pick_energies / baseline_median
            picks.to_csv(csv_out, index=False)
            n_done += 1
            if n_done % 20 == 0:
                print(f"  ... {n_done} station-days done", flush=True)
    print(f"\nDone. processed={n_done}  skipped={n_skip}  → {pick_root_out}")


if __name__ == "__main__":
    main()
