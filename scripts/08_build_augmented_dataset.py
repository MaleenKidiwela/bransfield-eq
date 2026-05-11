"""
Phase 2: build the augmented PhaseNet training set.

For each event window in data/seisbench/bransfield_events:
  - 30 NOISE VARIANTS  — pre-mix the event with random OBS noise samples at
                         variable SNR ∈ [0.5, 10]
  - 1 DENOISED VARIANT — pass the original event through the OBS-DeepDenoiser
                         (Phase 1b checkpoint), saved as the "denoised" channel.
  → ~31 × 3,537 ≈ ~110k windows in data/seisbench/bransfield_aug/.

Reads waveforms via SeisBench's WaveformDataset API (handles bucketing).
Noise variants run in a single process (loading all noise into memory once is
fast — 24k traces × 3 × 3000 × float32 ≈ 800 MB). Denoised variant is GPU-bound.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import seisbench.data as sbd
import seisbench.models as sbm

REPO = Path(__file__).resolve().parent.parent
EVENTS_DIR = REPO / "data" / "seisbench" / "bransfield_events"
NOISE_DIR = REPO / "data" / "seisbench" / "bransfield_noise"
OUT_DIR = REPO / "data" / "seisbench" / "bransfield_aug"
DENOISER_CKPT = REPO / "models" / "deepdenoiser_obs" / "best.pt"

WINDOW_SAMPLES = 3000
TARGET_RATE = 100.0
N_NOISE_VARIANTS = 30
SNR_RANGE = (0.5, 10.0)


def mix_event_noise(event: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
    """Mix event + α·noise so peak(event)/peak(α·noise) on Z = snr."""
    n = min(event.shape[1], noise.shape[1])
    event = event[:, :n].astype(np.float32)
    noise = noise[:, :n].astype(np.float32)
    e_amp = float(np.abs(event[0]).max())
    n_amp = float(np.abs(noise[0]).max())
    if n_amp < 1e-12:
        return event
    alpha = (e_amp / n_amp) / max(snr, 1e-6)
    return event + alpha * noise


def shift_pick_position(event_wf: np.ndarray, orig_pick_sample: int,
                        new_pick_sample: int) -> np.ndarray:
    """Shift the event window so the pick lands at new_pick_sample.

    Uses zero-padding (not cyclic): part of the original waveform may fall off
    one edge, replaced with zeros on the other. This is physically reasonable —
    the result is still a valid waveform with the arrival at a new position
    relative to the window start.
    """
    delta = new_pick_sample - orig_pick_sample
    n = event_wf.shape[1]
    out = np.zeros_like(event_wf)
    if delta >= 0:
        # Shift right: out[delta:] = event_wf[:n-delta]
        if delta < n:
            out[:, delta:] = event_wf[:, :n - delta]
    else:
        # Shift left: out[:n+delta] = event_wf[-delta:]
        if -delta < n:
            out[:, :n + delta] = event_wf[:, -delta:]
    return out


def _shape3(wf: np.ndarray) -> np.ndarray:
    """Force (3, T) shape."""
    if wf.ndim != 2:
        return None
    if wf.shape[0] != 3 and wf.shape[-1] == 3:
        wf = wf.T
    if wf.shape[0] != 3:
        return None
    if wf.shape[1] < WINDOW_SAMPLES:
        wf = np.pad(wf, ((0, 0), (0, WINDOW_SAMPLES - wf.shape[1])))
    return wf[:, :WINDOW_SAMPLES].astype(np.float32)


def main():
    global N_NOISE_VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-noise-variants", type=int, default=N_NOISE_VARIANTS)
    ap.add_argument("--skip-denoised", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    N_NOISE_VARIANTS = args.n_noise_variants

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_meta = OUT_DIR / "metadata.csv"
    out_wf = OUT_DIR / "waveforms.hdf5"
    if out_meta.exists():
        out_meta.unlink()
    if out_wf.exists():
        out_wf.unlink()

    print("Loading event dataset ...")
    events = sbd.WaveformDataset(path=str(EVENTS_DIR), sampling_rate=TARGET_RATE,
                                  component_order="ZNE")
    print(f"  events: {len(events)} traces")

    print("Loading noise dataset (loaded into memory once) ...")
    noise = sbd.WaveformDataset(path=str(NOISE_DIR), sampling_rate=TARGET_RATE,
                                 component_order="ZNE")
    n_noise = len(noise)
    print(f"  noise: {n_noise} traces; preloading ...")
    noise_pool = np.zeros((n_noise, 3, WINDOW_SAMPLES), dtype=np.float32)
    for i in range(n_noise):
        wf, _ = noise.get_sample(i)
        wf = _shape3(wf)
        if wf is None:
            continue
        noise_pool[i] = wf
    print(f"  noise pool tensor: {noise_pool.shape}, {noise_pool.nbytes/1e6:.1f} MB")
    print()

    n_events = len(events)
    rng = np.random.default_rng(args.seed)

    expected_total = n_events * (1 + args.n_noise_variants + (0 if args.skip_denoised else 1))
    print(f"Variants per event: 1 clean + {args.n_noise_variants} noise"
          f"{' + 1 denoised' if not args.skip_denoised else ''}")
    print(f"Expected total rows: {expected_total:,}")
    print()

    # === Pass 1: clean + noise variants ===
    print("--- Pass 1: clean + noise variants ---", flush=True)
    t0 = time.time()
    out_rows = []
    for i in range(n_events):
        try:
            wf, meta_row = events.get_sample(i)
        except Exception as e:
            print(f"  [skip] event {i}: {e}", flush=True)
            continue
        wf = _shape3(wf)
        if wf is None:
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
            "trace_p_arrival_sample": meta_row["trace_p_arrival_sample"],
            "trace_s_arrival_sample": meta_row["trace_s_arrival_sample"],
            "trace_phase": meta_row.get("trace_phase", ""),
            "trace_uncertainty_s": meta_row.get("trace_uncertainty_s", np.nan),
        }
        # Original pick positions (P at 500 from extractor PRE_PICK_SEC=5s × 100Hz)
        orig_p = int(meta_row["trace_p_arrival_sample"]) if not pd.isna(meta_row["trace_p_arrival_sample"]) else None
        orig_s = int(meta_row["trace_s_arrival_sample"]) if not pd.isna(meta_row["trace_s_arrival_sample"]) else None

        # Clean (keep at original position)
        meta = dict(base_meta); meta["variant"] = "clean"; meta["snr"] = np.nan
        meta["source_id"] = f"{base_meta['source_id']}__clean"
        out_rows.append((meta, wf.copy()))

        # N noise variants — randomize pick position to break "always at sample 500"
        # memorization. Pick position uniform in [200, WINDOW_SAMPLES-200] so the
        # arrival never falls off the window edge.
        noise_idxs = rng.integers(0, n_noise, args.n_noise_variants)
        snrs = rng.uniform(*SNR_RANGE, args.n_noise_variants)
        for v, (nidx, snr) in enumerate(zip(noise_idxs, snrs)):
            new_pick = int(rng.integers(200, WINDOW_SAMPLES - 200))
            anchor = orig_p if orig_p is not None else orig_s
            if anchor is None:
                continue  # event with no pick at all; skip
            shifted_wf = shift_pick_position(wf, anchor, new_pick)
            mixed = mix_event_noise(shifted_wf, noise_pool[nidx], float(snr))
            new_meta = dict(base_meta)
            new_meta["variant"] = "noise"; new_meta["snr"] = float(snr)
            new_meta["source_id"] = f"{base_meta['source_id']}__noise{v}"
            shift = new_pick - anchor
            new_meta["trace_p_arrival_sample"] = (orig_p + shift) if orig_p is not None else np.nan
            new_meta["trace_s_arrival_sample"] = (orig_s + shift) if orig_s is not None else np.nan
            # Sanity-clip pick samples to window
            for k in ("trace_p_arrival_sample", "trace_s_arrival_sample"):
                v_ = new_meta[k]
                if not pd.isna(v_) and (v_ < 0 or v_ >= WINDOW_SAMPLES):
                    new_meta[k] = np.nan
            out_rows.append((new_meta, mixed))
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"  events {i+1}/{n_events}  rows={len(out_rows):,}  "
                  f"({(i+1)/dt:.1f}/s)", flush=True)
    print(f"  Pass 1 rows: {len(out_rows):,}  (wall: {time.time()-t0:.1f}s)")

    # === Pass 2: denoised variants (GPU) ===
    if not args.skip_denoised:
        print()
        if not DENOISER_CKPT.exists():
            print(f"  WARNING: {DENOISER_CKPT} not found; skipping denoised variants")
        else:
            device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
            print(f"--- Pass 2: denoised variants on {device} ---", flush=True)
            model = sbm.DeepDenoiser.from_pretrained("original")
            ck = torch.load(DENOISER_CKPT, weights_only=False)
            model.load_state_dict(ck["model"])
            model.to(device); model.eval()
            t0 = time.time()
            BATCH = 32
            n_added = 0
            with torch.no_grad():
                for i0 in range(0, n_events, BATCH):
                    indices = list(range(i0, min(i0 + BATCH, n_events)))
                    wfs, metas = [], []
                    for idx in indices:
                        try:
                            wf, mrow = events.get_sample(idx)
                        except Exception:
                            continue
                        wf = _shape3(wf)
                        if wf is None:
                            continue
                        wfs.append(wf); metas.append(mrow)
                    if not wfs:
                        continue
                    X = torch.from_numpy(np.stack(wfs)).to(device).float()
                    peak = X.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                    Xn = X / peak
                    pred = model(Xn)
                    pred_np = (pred * peak).cpu().numpy()
                    if pred_np.ndim != 3 or pred_np.shape[1] != 3:
                        # Mask format — fallback: use original waveform
                        for j, mrow in enumerate(metas):
                            wfs[j] = wfs[j]
                    else:
                        wfs = [pred_np[j] for j in range(len(metas))]
                    for wf, mrow in zip(wfs, metas):
                        meta = {
                            "source_id": f"{mrow['source_id']}__denoised",
                            "source_origin_time": mrow.get("source_origin_time", ""),
                            "station_network_code": mrow["station_network_code"],
                            "station_code": mrow["station_code"],
                            "station_location_code": "",
                            "trace_channel": mrow.get("trace_channel", ""),
                            "trace_sampling_rate_hz": TARGET_RATE,
                            "trace_npts": WINDOW_SAMPLES,
                            "trace_start_time": mrow.get("trace_start_time", ""),
                            "trace_p_arrival_sample": mrow["trace_p_arrival_sample"],
                            "trace_s_arrival_sample": mrow["trace_s_arrival_sample"],
                            "trace_phase": mrow.get("trace_phase", ""),
                            "trace_uncertainty_s": mrow.get("trace_uncertainty_s", np.nan),
                            "variant": "denoised", "snr": np.nan,
                        }
                        wf2 = _shape3(np.asarray(wf, dtype=np.float32))
                        if wf2 is None:
                            continue
                        out_rows.append((meta, wf2))
                        n_added += 1
                    if (i0 // BATCH) % 20 == 0:
                        print(f"  denoised {i0+len(indices)}/{n_events}  "
                              f"+{n_added} so far  ({(i0+len(indices))/(time.time()-t0+1e-9):.1f}/s)",
                              flush=True)
            print(f"  Pass 2 added: {n_added:,}  (wall: {time.time()-t0:.1f}s)")

    print(f"\nTotal rows to write: {len(out_rows):,}")
    print(f"Writing to {OUT_DIR} ...")
    t0 = time.time()
    # Normalize datetime fields so SeisBench will accept the metadata
    def _norm_dt(s):
        if pd.isna(s) or s == "" or s is None:
            return pd.NA
        try:
            return pd.to_datetime(s, utc=True).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        except Exception:
            return pd.NA
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
    print(f"Done. {len(out_rows):,} rows in {OUT_DIR}  (write wall: {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
