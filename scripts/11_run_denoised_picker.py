"""
Picker E: pretrained PhaseNet on OBS-DeepDenoiser-cleaned waveforms.

For each station-day in the requested range:
  1. Read raw mseed
  2. Apply the OBS-fine-tuned DeepDenoiser (Phase 1b checkpoint) via
     model.annotate(stream) — handles STFT internally, returns denoised stream
  3. Run pretrained PhaseNet on the denoised stream
  4. Save picks CSV in the same format as scripts/03_run_phasenet.py output

Usage:
    python scripts/11_run_denoised_picker.py --start 2019-02-04 --end 2019-02-14
    python scripts/11_run_denoised_picker.py --start 2019-08-01 --end 2019-09-01

Output:
    catalogs/<--out-subdir>/<NET>.<STA>/<YYYY-DDD>.csv  (default subdir: picks_pn_dd)

Idempotent — skips station-days whose CSV already exists.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import (Config, WAVE_DIR, daterange, mseed_path)  # noqa: E402

import seisbench.models as sbm  # noqa: E402

DENOISER_CKPT = REPO / "models" / "deepdenoiser_obs" / "best.pt"


def _channel_match(code: str, glob: str) -> bool:
    import fnmatch
    return any(fnmatch.fnmatchcase(code, g.strip()) for g in glob.split(","))


def list_station_days(start: UTCDateTime, end: UTCDateTime,
                      network: str | None, station: str | None
                      ) -> list[tuple]:
    out = []
    if not WAVE_DIR.exists():
        return out
    for net_dir in sorted(WAVE_DIR.iterdir()):
        if not net_dir.is_dir(): continue
        if network and net_dir.name != network: continue
        for sta_dir in sorted(net_dir.iterdir()):
            if not sta_dir.is_dir(): continue
            if station and sta_dir.name != station: continue
            for day in daterange(start, end):
                p = mseed_path(net_dir.name, sta_dir.name, day)
                if p.exists() and p.stat().st_size > 0:
                    out.append((net_dir.name, sta_dir.name, day, p))
    return out


# ---- worker globals ----
_S: dict = {}


def _init_worker(p_thresh: float, s_thresh: float, target_rate: float,
                 picking_glob: str, batch_size: int, out_root_str: str,
                 device: str, picker_weights: str = "instance") -> None:
    """Load both models once per worker process."""
    # Denoiser
    denoiser = sbm.DeepDenoiser.from_pretrained("original")
    ck = torch.load(DENOISER_CKPT, weights_only=False)
    denoiser.load_state_dict(ck["model"])
    denoiser.to(device).eval()
    # Picker — accept "instance" / "obs" / "ckpt:<path>"
    if picker_weights.startswith("ckpt:") or picker_weights.startswith("local:"):
        path = picker_weights.split(":", 1)[1]
        raw = torch.load(path, weights_only=False, map_location="cpu")
        sd = raw.get("state_dict", raw.get("model", raw))
        sd = {(k[len("model."):] if k.startswith("model.") else k): v
              for k, v in sd.items()}
        base_tag = raw.get("pretrained_base", "instance") if isinstance(raw, dict) else "instance"
        picker = sbm.PhaseNet.from_pretrained(base_tag)
        picker.load_state_dict(sd, strict=False)
    else:
        picker = sbm.PhaseNet.from_pretrained(picker_weights)
    picker.to(device).eval()
    _S.update(
        denoiser=denoiser,
        picker=picker,
        p_thresh=p_thresh,
        s_thresh=s_thresh,
        target_rate=target_rate,
        picking_glob=picking_glob,
        batch_size=batch_size,
        out_root=Path(out_root_str),
    )


def _pick_one(task: tuple) -> tuple:
    """Read mseed, denoise, pick, write CSV. Returns (status, n_picks, msg)."""
    net, sta, day_iso, mseed_str = task
    s = _S
    day = UTCDateTime(day_iso)
    out_csv = s["out_root"] / f"{net}.{sta}" / f"{day.year}-{day.julday:03d}.csv"
    if out_csv.exists():
        return ("skip", 0, "")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        st = read(mseed_str)
        # Drop hydrophone before denoising
        st.traces = [tr for tr in st if _channel_match(tr.stats.channel, s["picking_glob"])]
        if len(st) == 0:
            df = pd.DataFrame()
            df.to_csv(out_csv, index=False)
            return ("done", 0, f"{net}.{sta} {day.date}")
        st.merge(method=1, fill_value=0)
        for tr in st:
            if abs(tr.stats.sampling_rate - s["target_rate"]) > 1e-6:
                tr.resample(s["target_rate"])
        # Denoise
        denoised = s["denoiser"].annotate(st)
        if denoised is None or len(denoised) == 0:
            df = pd.DataFrame()
            df.to_csv(out_csv, index=False)
            return ("done", 0, f"{net}.{sta} {day.date} (denoise empty)")
        # Pick on denoised stream
        picks = s["picker"].classify(
            denoised,
            P_threshold=s["p_thresh"],
            S_threshold=s["s_thresh"],
            batch_size=s["batch_size"],
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
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        return ("done", len(df), f"{net}.{sta} {day.date}")
    except Exception as e:
        return ("err", 0, f"{net}.{sta} {day.date}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--network", default=None)
    ap.add_argument("--station", default=None)
    ap.add_argument("--out-subdir", default="picks_pn_dd")
    ap.add_argument("--picker-weights", default="instance",
                    help="Picker weights tag or 'ckpt:<path>' / 'local:<path>'")
    ap.add_argument("--p-thresh", type=float, default=0.1)
    ap.add_argument("--s-thresh", type=float, default=0.1)
    ap.add_argument("--target-rate", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = Config.load(None)

    out_root = REPO / "catalogs" / args.out_subdir
    pairs = list_station_days(UTCDateTime(args.start), UTCDateTime(args.end),
                              args.network, args.station)
    print(f"Picker E (DeepDenoiser → PhaseNet) on {args.device}  "
          f"workers={args.workers} batch_size={args.batch_size}", flush=True)
    print(f"  {len(pairs)} station-days  → catalogs/{args.out_subdir}/", flush=True)
    print(f"  denoiser ckpt: {DENOISER_CKPT}", flush=True)

    n_done = n_skip = n_err = n_picks = 0
    if args.workers <= 1:
        # Serial path — load models in-process
        _init_worker(args.p_thresh, args.s_thresh, args.target_rate,
                     cfg.picking_channels, args.batch_size, str(out_root),
                     args.device, args.picker_weights)
        for task in pairs:
            status, npk, msg = _pick_one((task[0], task[1], task[2].isoformat(),
                                           str(task[3])))
            if status == "skip":
                n_skip += 1
            elif status == "err":
                n_err += 1
                print(f"  ERR {msg}", flush=True)
            else:
                n_done += 1
                n_picks += npk
                if n_done % 20 == 0:
                    print(f"  ... {n_done}/{len(pairs)} done  picks={n_picks:,}",
                          flush=True)
    else:
        ctx = mp.get_context("spawn")
        tasks = [(n, s_, d.isoformat(), str(p)) for (n, s_, d, p) in pairs]
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.p_thresh, args.s_thresh, args.target_rate,
                       cfg.picking_channels, args.batch_size, str(out_root),
                       args.device, args.picker_weights),
        ) as pool:
            futs = [pool.submit(_pick_one, t) for t in tasks]
            for fut in as_completed(futs):
                status, npk, msg = fut.result()
                if status == "skip":
                    n_skip += 1
                elif status == "err":
                    n_err += 1
                    print(f"  ERR {msg}", flush=True)
                else:
                    n_done += 1
                    n_picks += npk
                    if n_done % 20 == 0:
                        print(f"  ... {n_done}/{len(pairs)} done  picks={n_picks:,}",
                              flush=True)

    print(f"\nDone. picked={n_done}  skipped={n_skip}  errors={n_err}  "
          f"total_picks={n_picks}")


if __name__ == "__main__":
    main()
