"""
v3 augmented training set: denoise event windows + random pick position.

Reads data/seisbench/bransfield_events (3,537 high-conf mag07 windows from
Phase 1a). For each event:
  - Apply the OBS-fine-tuned DeepDenoiser (via model.annotate on a single-trace
    stream) to produce a denoised version of the window.
  - Generate N=30 variants by shifting the pick to a random position in
    [200, 2799] (zero-pad shift, same as v2's shift_pick_position).
  - No noise mixing — data is already denoised, mixing raw noise back in
    would defeat the purpose.

Output: data/seisbench/bransfield_aug_dd/{metadata.csv, waveforms.hdf5}
        ~109k windows.

Inference: must use the same DD pipeline. Use scripts/11_run_denoised_picker.py
with --weights ckpt:<v3 ckpt> on the new fine-tuned PhaseNet.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obspy import Stream, Trace, UTCDateTime

import seisbench.data as sbd
import seisbench.models as sbm

REPO = Path(__file__).resolve().parent.parent
EVENTS_DIR = REPO / "data" / "seisbench" / "bransfield_events"
OUT_DIR = REPO / "data" / "seisbench" / "bransfield_aug_dd"
DENOISER_CKPT = REPO / "models" / "deepdenoiser_obs" / "best.pt"

WINDOW_SAMPLES = 3000
TARGET_RATE = 100.0
N_VARIANTS = 30


def _shape3(wf: np.ndarray) -> np.ndarray | None:
    if wf.ndim != 2:
        return None
    if wf.shape[0] != 3 and wf.shape[-1] == 3:
        wf = wf.T
    if wf.shape[0] != 3:
        return None
    if wf.shape[1] < WINDOW_SAMPLES:
        wf = np.pad(wf, ((0, 0), (0, WINDOW_SAMPLES - wf.shape[1])))
    return wf[:, :WINDOW_SAMPLES].astype(np.float32)


def shift_pick_position(event_wf: np.ndarray, orig: int, new: int) -> np.ndarray:
    """Zero-pad shift the waveform so the pick lands at sample `new`."""
    delta = new - orig
    n = event_wf.shape[1]
    out = np.zeros_like(event_wf)
    if delta >= 0:
        if delta < n:
            out[:, delta:] = event_wf[:, :n - delta]
    else:
        if -delta < n:
            out[:, :n + delta] = event_wf[:, -delta:]
    return out


def denoise_window(model, wf: np.ndarray, t_start: UTCDateTime,
                   net: str, sta: str) -> np.ndarray | None:
    """Wrap (3, N) array in an obspy Stream, denoise, return (3, N) array."""
    chans = ["BHZ", "BHN", "BHE"]
    st = Stream()
    for i, ch in enumerate(chans):
        tr = Trace(data=wf[i].astype(np.float32))
        tr.stats.network = net
        tr.stats.station = sta
        tr.stats.channel = ch
        tr.stats.sampling_rate = TARGET_RATE
        tr.stats.starttime = t_start
        st += tr
    try:
        denoised = model.annotate(st)
    except Exception as e:
        return None
    if denoised is None or len(denoised) < 3:
        return None
    by_letter = {tr.stats.channel[-1]: tr for tr in denoised}
    out_chans = [by_letter.get(c) for c in ("Z", "N", "E")]
    if any(c is None for c in out_chans):
        return None
    n = min(tr.stats.npts for tr in out_chans)
    if n < int(0.9 * WINDOW_SAMPLES):
        return None
    arr = np.stack([tr.data[:n].astype(np.float32) for tr in out_chans], axis=0)
    if arr.shape[1] < WINDOW_SAMPLES:
        arr = np.pad(arr, ((0, 0), (0, WINDOW_SAMPLES - arr.shape[1])))
    return arr[:, :WINDOW_SAMPLES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-variants", type=int, default=N_VARIANTS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_meta = OUT_DIR / "metadata.csv"
    out_wf = OUT_DIR / "waveforms.hdf5"
    if out_meta.exists(): out_meta.unlink()
    if out_wf.exists(): out_wf.unlink()

    print(f"Loading event dataset from {EVENTS_DIR} ...")
    events = sbd.WaveformDataset(path=str(EVENTS_DIR), sampling_rate=TARGET_RATE,
                                  component_order="ZNE")
    n_events = len(events)
    print(f"  events: {n_events}")

    print(f"Loading OBS DeepDenoiser from {DENOISER_CKPT} ...")
    model = sbm.DeepDenoiser.from_pretrained("original")
    ck = torch.load(DENOISER_CKPT, weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"  device: {device}")
    print()

    print(f"Generating {args.n_variants} variants per event "
          f"({n_events * args.n_variants:,} total) ...", flush=True)
    rng = np.random.default_rng(args.seed)

    out_rows = []
    t0 = time.time()
    n_skipped = 0
    for i in range(n_events):
        try:
            wf, meta_row = events.get_sample(i)
        except Exception:
            n_skipped += 1
            continue
        wf = _shape3(wf)
        if wf is None:
            n_skipped += 1
            continue

        # Denoise the event window once
        t_start = pd.to_datetime(meta_row.get("trace_start_time", None))
        if pd.isna(t_start):
            t_start_utc = UTCDateTime(0)
        else:
            t_start_utc = UTCDateTime(t_start.to_pydatetime())
        with torch.no_grad():
            den_wf = denoise_window(model, wf, t_start_utc,
                                     meta_row["station_network_code"],
                                     meta_row["station_code"])
        if den_wf is None:
            n_skipped += 1
            continue

        orig_p = int(meta_row["trace_p_arrival_sample"]) if not pd.isna(meta_row["trace_p_arrival_sample"]) else None
        orig_s = int(meta_row["trace_s_arrival_sample"]) if not pd.isna(meta_row["trace_s_arrival_sample"]) else None
        anchor = orig_p if orig_p is not None else orig_s
        if anchor is None:
            n_skipped += 1
            continue

        base_meta = {
            "source_id": meta_row["source_id"],
            "source_origin_time": meta_row.get("source_origin_time", ""),
            "station_network_code": meta_row["station_network_code"],
            "station_code": meta_row["station_code"],
            "station_location_code": "",
            "trace_channel": meta_row.get("trace_channel", ""),
            "trace_sampling_rate_hz": TARGET_RATE,
            "trace_npts": WINDOW_SAMPLES,
            "trace_start_time": meta_row.get("trace_start_time", ""),
            "trace_p_arrival_sample": orig_p if orig_p is not None else np.nan,
            "trace_s_arrival_sample": orig_s if orig_s is not None else np.nan,
            "trace_phase": meta_row.get("trace_phase", ""),
            "trace_uncertainty_s": meta_row.get("trace_uncertainty_s", np.nan),
        }

        # Clean (denoised, original position)
        m = dict(base_meta); m["variant"] = "clean_dd"
        m["source_id"] = f"{base_meta['source_id']}__clean_dd"
        out_rows.append((m, den_wf.copy()))

        # N position-shifted variants
        for v in range(args.n_variants - 1):  # -1 because clean counts as one
            new_pick = int(rng.integers(200, WINDOW_SAMPLES - 200))
            shifted = shift_pick_position(den_wf, anchor, new_pick)
            shift = new_pick - anchor
            m = dict(base_meta); m["variant"] = "shift_dd"
            m["source_id"] = f"{base_meta['source_id']}__shift_dd_{v}"
            m["trace_p_arrival_sample"] = (orig_p + shift) if orig_p is not None else np.nan
            m["trace_s_arrival_sample"] = (orig_s + shift) if orig_s is not None else np.nan
            for k in ("trace_p_arrival_sample", "trace_s_arrival_sample"):
                vk = m[k]
                if not pd.isna(vk) and (vk < 0 or vk >= WINDOW_SAMPLES):
                    m[k] = np.nan
            out_rows.append((m, shifted))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_events - i - 1) / rate
            print(f"  events {i+1}/{n_events}  rows={len(out_rows):,}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  skipped={n_skipped}",
                  flush=True)

    print(f"\nDone generating  rows={len(out_rows):,}  skipped={n_skipped}")
    print(f"Writing to {OUT_DIR} ...", flush=True)

    def _norm_dt(s):
        if pd.isna(s) or s == "" or s is None:
            return pd.NA
        try:
            return pd.to_datetime(s, utc=True).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        except Exception:
            return pd.NA

    t0 = time.time()
    with sbd.WaveformDataWriter(out_meta, out_wf) as wr:
        wr.data_format = {
            "dimension_order": "CW", "component_order": "ZNE",
            "sampling_rate": TARGET_RATE, "measurement": "velocity",
            "unit": "counts", "instrument_response": "not restituted",
        }
        for meta, wf in out_rows:
            meta["source_origin_time"] = _norm_dt(meta.get("source_origin_time"))
            meta["trace_start_time"] = _norm_dt(meta.get("trace_start_time"))
            wr.add_trace(meta, wf)
    print(f"Done writing.  {len(out_rows):,} rows in {OUT_DIR}  "
          f"(write wall: {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
