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
                        "'stead' on OBS data; recall ~45%% vs 9%% on P)")
    p.add_argument("--p-thresh", type=float, default=0.2)
    p.add_argument("--s-thresh", type=float, default=0.2)
    p.add_argument("--model", default="PhaseNet",
                   choices=["PhaseNet", "EQTransformer", "OBSTransformer"],
                   help="SeisBench picking model class")
    p.add_argument("--out-subdir", default="picks",
                   help="catalogs/<out-subdir>/ — separate models to compare")
    p.add_argument("--target-rate", type=float, default=100.0,
                   help="Resample to this rate before picking (PhaseNet default 100 Hz)")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:0 | mps  "
                        "(auto picks cuda when available)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="GPU batch size passed to model.classify()")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel pick processes sharing the GPU "
                        "(default 1 = serial; 4-8 useful on a 46 GB GPU)")
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
                 picking_glob: str,
                 batch_size: int = 256) -> pd.DataFrame:
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
        batch_size=batch_size,
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


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# Per-process state for the worker pool. Each child process loads its own
# model copy on the GPU (PhaseNet is ~few hundred MB; 4-8 fit in 46 GB).
_WORKER_STATE: dict = {}


def _load_model(model_name: str, weights: str):
    """Load a SeisBench picker by pretrained tag, or from a local Lightning .ckpt /
    raw-state .pt via 'ckpt:<path>' / 'local:<path>' syntax."""
    import seisbench.models as sbm
    import torch
    ModelCls = getattr(sbm, model_name)
    if weights.startswith("ckpt:") or weights.startswith("local:"):
        path = weights.split(":", 1)[1]
        raw = torch.load(path, weights_only=False, map_location="cpu")
        sd = raw.get("state_dict", raw.get("model", raw))
        sd = {(k[len("model."):] if k.startswith("model.") else k): v
              for k, v in sd.items()}
        base_tag = raw.get("pretrained_base", "instance") if isinstance(raw, dict) else "instance"
        model = ModelCls.from_pretrained(base_tag)
        model.load_state_dict(sd, strict=False)
        return model
    return ModelCls.from_pretrained(weights)


def _init_worker(model_name: str, weights: str, device: str,
                 target_rate: float, p_thresh: float, s_thresh: float,
                 picking_glob: str, batch_size: int,
                 out_root_str: str) -> None:
    model = _load_model(model_name, weights)
    model.to(device)
    _WORKER_STATE.update(
        model=model,
        target_rate=target_rate,
        p_thresh=p_thresh,
        s_thresh=s_thresh,
        picking_glob=picking_glob,
        batch_size=batch_size,
        out_root=Path(out_root_str),
    )


def _pick_one_task(task: tuple) -> tuple:
    """Worker entry: pick one station-day, write CSV. Returns (status, n_picks, msg)."""
    net, sta, day_iso, mseed_str = task
    s = _WORKER_STATE
    day = UTCDateTime(day_iso)
    out_csv = s["out_root"] / f"{net}.{sta}" / f"{day.year}-{day.julday:03d}.csv"
    if out_csv.exists():
        return ("skip", 0, "")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = pick_one_day(
            s["model"], Path(mseed_str), s["target_rate"],
            s["p_thresh"], s["s_thresh"], s["picking_glob"],
            batch_size=s["batch_size"],
        )
    except Exception as e:
        return ("err", 0, f"{net}.{sta} {day.date}: {e}")
    df.to_csv(out_csv, index=False)
    return ("done", len(df), f"{net}.{sta} {day.date}")


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    if args.start:
        cfg.start = UTCDateTime(args.start)
    if args.end:
        cfg.end = UTCDateTime(args.end)

    args.device = _resolve_device(args.device)

    # Allow alternate output subdir so different models can be compared without
    # overwriting each other (e.g. catalogs/picks/ for PhaseNet, catalogs/picks_eqt/ for EQT).
    from bransfield_eq.config import REPO as _REPO
    out_root = _REPO / "catalogs" / args.out_subdir

    pairs = list_station_days(cfg.start, cfg.end, args.network, args.station)
    if args.of > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.of == args.shard]
    print(f"Picking with {args.model} weights={args.weights} on {args.device}  "
          f"workers={args.workers} batch_size={args.batch_size}",
          flush=True)
    print(f"  shard {args.shard}/{args.of}: {len(pairs)} station-days "
          f"→ catalogs/{args.out_subdir}/", flush=True)

    if args.workers <= 1:
        # Serial path — load once in this process.
        model = _load_model(args.model, args.weights)
        model.to(args.device)

        n_done = n_skip = n_err = n_picks = 0
        for net, sta, day, mseed in pairs:
            out_csv = out_root / f"{net}.{sta}" / f"{day.year}-{day.julday:03d}.csv"
            if out_csv.exists():
                n_skip += 1
                continue
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            try:
                df = pick_one_day(model, mseed, args.target_rate,
                                  args.p_thresh, args.s_thresh,
                                  cfg.picking_channels,
                                  batch_size=args.batch_size)
            except Exception as e:
                n_err += 1
                print(f"  ERR {net}.{sta} {day.date}: {e}", flush=True)
                continue
            df.to_csv(out_csv, index=False)
            n_done += 1
            n_picks += len(df)
            if n_done % 25 == 0:
                print(f"  ... {n_done} days picked, {n_picks} picks so far", flush=True)
    else:
        # Multi-process path — spawn N workers, each with its own model on GPU.
        # CUDA contexts can't be forked; spawn is required.
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [(net, sta, day.isoformat(), str(mseed))
                 for (net, sta, day, mseed) in pairs]
        ctx = mp.get_context("spawn")
        n_done = n_skip = n_err = n_picks = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.model, args.weights, args.device,
                      args.target_rate, args.p_thresh, args.s_thresh,
                      cfg.picking_channels, args.batch_size,
                      str(out_root)),
        ) as pool:
            futures = [pool.submit(_pick_one_task, t) for t in tasks]
            for fut in as_completed(futures):
                status, npk, msg = fut.result()
                if status == "skip":
                    n_skip += 1
                elif status == "err":
                    n_err += 1
                    print(f"  ERR {msg}", flush=True)
                else:
                    n_done += 1
                    n_picks += npk
                    if n_done % 25 == 0:
                        print(f"  ... {n_done} days picked, {n_picks} picks so far",
                              flush=True)

    print(f"\nDone. picked={n_done}  skipped={n_skip}  errors={n_err}  "
          f"total_picks={n_picks}")


if __name__ == "__main__":
    main()
