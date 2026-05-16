"""Compute a 151-D log-power-spectrum fingerprint per pyocto event by averaging
the per-pick spectra across all valid picks for that event.

Reuses:
    growclust/picker_only/pick_windows.npy   (483,516 x 300 float32, ±1.5 s @ 100 Hz)
    growclust/picker_only/pick_index.parquet (pick_id, event_idx, ..., valid)

Writes:
    catalogs/event_spectra.npy           (n_events x 151 float32)
    catalogs/event_spectra_meta.parquet  (event_idx, n_picks_used)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
WIN = REPO / "growclust" / "picker_only" / "pick_windows.npy"
IDX = REPO / "growclust" / "picker_only" / "pick_index.parquet"
OUT_SPEC = REPO / "catalogs" / "event_spectra.npy"
OUT_META = REPO / "catalogs" / "event_spectra_meta.parquet"

# Window: 300 samples at 100 Hz => 150 positive freq bins + DC = 151 bins
# Nyquist 50 Hz, resolution 0.333 Hz.


def main():
    print(f"Loading pick metadata from {IDX} ...")
    idx = pd.read_parquet(IDX)
    print(f"  {len(idx):,} picks across {idx.event_idx.nunique():,} events")

    print(f"Memmapping waveforms from {WIN} ...")
    pw = np.load(WIN, mmap_mode="r")
    print(f"  shape {pw.shape}, dtype {pw.dtype}")

    # Process per event for memory efficiency. Sort by event_idx so we can
    # slice contiguous pick ranges.
    idx = idx.sort_values(["event_idx", "pick_id"]).reset_index(drop=True)
    n_events = idx.event_idx.nunique()
    n_freq = pw.shape[1] // 2 + 1   # 151 for 300 samples

    spectra = np.zeros((n_events, n_freq), dtype=np.float32)
    n_used = np.zeros(n_events, dtype=np.int32)
    event_idxs = np.zeros(n_events, dtype=np.int64)

    print(f"Computing per-event log-power spectra ({n_events:,} events) ...")
    for i, (eid, grp) in enumerate(idx.groupby("event_idx", sort=True)):
        valid_pids = grp[grp["valid"]]["pick_id"].values if "valid" in grp.columns else grp["pick_id"].values
        if len(valid_pids) == 0:
            event_idxs[i] = eid
            continue
        # Pull rows; shape (n_valid, 300)
        wf = np.asarray(pw[valid_pids], dtype=np.float32)
        # rfft per row -> (n_valid, 151) complex
        F = np.fft.rfft(wf, axis=1)
        power = (F.real ** 2 + F.imag ** 2).astype(np.float32)
        # log1p to compress dynamic range
        logp = np.log1p(power)
        # Normalize each pick's spectrum to unit area, then average across picks
        norms = logp.sum(axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        logp_norm = logp / norms
        spectra[i] = logp_norm.mean(axis=0)
        n_used[i] = len(valid_pids)
        event_idxs[i] = eid
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{n_events:,}")

    np.save(OUT_SPEC, spectra)
    meta = pd.DataFrame({"event_idx": event_idxs, "n_picks_used": n_used})
    meta.to_parquet(OUT_META, index=False)
    print(f"\nwrote {OUT_SPEC}  ({spectra.shape}, {spectra.nbytes/1e6:.1f} MB)")
    print(f"wrote {OUT_META}  ({len(meta):,} events)")


if __name__ == "__main__":
    main()
