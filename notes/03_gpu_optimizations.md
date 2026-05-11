# GPU/CPU optimizations to `scripts/03_run_phasenet.py`

## Changes

| Change | Why |
|---|---|
| `--device auto` (was `cpu`) | Auto-select `cuda` when available — biggest single win, GPU was previously off by default |
| `--batch-size N` (default 256), passed through to `model.classify()` | Saturates the L40S; matches PhaseNet's internal default |
| `--workers N` (default 1) — process-pool with spawn context | Multiple PhaseNet copies share the GPU. Each worker independently reads + obspy-resamples on its own CPU cores → drives both **CPU saturation** and **better GPU utilization** (multi-stream overlap). Recommended: 4–8 on a 46 GB GPU |
| Per-process model init via `ProcessPoolExecutor(initializer=...)` | Each worker loads PhaseNet once at startup, reuses for the whole run |
| Removed `--parallelism` arg | SeisBench 0.11 deprecated it — was only spamming warnings |
| Argparse `%` escape in `--weights` help | Pre-existing bug — `--help` crashed without it |

## Headroom on the L40S
- PhaseNet ≈ 500 MB resident per process (model + CUDA context)
- 8 workers × 500 MB ≈ 4 GB out of 46 GB VRAM — plenty of margin

## Recommended invocation for the year run
```bash
python scripts/03_run_phasenet.py \
    --workers 8 --batch-size 256 \
    --p-thresh 0.1 --s-thresh 0.1
```

## Why `--workers 1` for the verification day
To keep the count comparison apples-to-apples with the laptop run. Multi-worker batching can shift per-day pick counts at the margins (different ordering through the SeisBench Generator); we wanted the 6,902-pick check to match cleanly. For production runs there's no reason not to use `--workers 8`.

## Idempotency preserved
Both serial and parallel paths skip station-days whose pick CSV already exists. Safe to interrupt and restart.
