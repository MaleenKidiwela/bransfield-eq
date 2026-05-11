# Year run status

Window: **2019-01-01 → 2020-03-01** (full BRAVOSEIS deployment, 14 months).

## Plan (sequential)
1. Download all 14 months → ~16,000 station-days, ~560 GB into `my_data/bravoseis/waveforms/`
2. PhaseNet pick (`--workers 8 --batch-size 256`) → `catalogs/picks/`
3. EQTransformer pick (separate output) → `catalogs/picks_eqt/`
4. Validation against mag07 for both, then ensemble analysis

## Current status (as of 2026-05-09 ~07:58 UTC, ~3h elapsed)

- **Download running**: PID 9415, log at `logs/download_full.log`
- **141 GB on disk** of ~560 GB expected (~25 %)
- **7 of 38 stations complete** (all year days present): `ZX.BRA13, BRA14, BRA15, BRA16, BRA18, BRA19, BRA20`
- BRA21 currently in progress (~58/425 days)
- Throughput ~47 GB/h → **ETA full completion: ~9 more hours from now** (~17:00 UTC)
- Persistent monitor armed (id `bfbu0ocjc`) — pings on `Done.` or any error
- Note: download script iterates station-by-station, so progress shows up as more *stations* complete, not more days

## Pickup plan for next session

### When the year download is done
1. Verify completion: `tail logs/download_full.log` should show `Done. fetched=… bytes=… GB` and 38 stations should each have ≥420 mseed files.
2. Pick top 10 mag07 days for the multi-day picker comparison. Suggested by mag07 pick count (computed today):

| Day | Manual picks (mag07) | Events |
|---|---:|---:|
| 2019-01-17 | 1,259 | 118 (swarm) |
| 2019-02-13 | 442 | 29 |
| 2019-02-05 | 176 | 16 |
| 2019-02-11 | 159 | 16 |
| 2019-02-16 | 77 | 4 |
| 2019-09-01 | 58 | 4 |
| 2019-05-10 | 62 | 4 |
| 2019-02-12 | 93 | 7 |
| 2019-12-26 | 23 | 2 (already done — apples-to-apples baseline) |
| 2019-01-13 | 91 | 7 |

(Spread across multiple months, mix of swarm + isolated events.)

3. For each of those 10 days, run the **recommended hybrid + comparison set**:
   - PhaseNet `instance` @ thresh 0.1 (best P)
   - OBSTransformer `obst2024` @ thresh 0.5 (best S, recommended sweet spot per day-26 sweep)
   - Optionally: PickBlue EQT @ 0.1 as an alternative S baseline

4. Aggregate: per-picker P/S recall and FP across the 10 days, with confidence intervals on the recall.

### Picking commands (when ready)
```bash
source .venv/bin/activate
# Hybrid recommendation: PhaseNet for P, OBSTransformer for S
python scripts/03_run_phasenet.py \
    --workers 8 --batch-size 256 \
    --p-thresh 0.1 --s-thresh 0.1 \
    > logs/pick_full_phasenet.log 2>&1 &

python scripts/03_run_phasenet.py \
    --model OBSTransformer --weights obst2024 --out-subdir picks_obst_05 \
    --workers 8 --batch-size 256 \
    --p-thresh 0.5 --s-thresh 0.5 \
    > logs/pick_full_obst.log 2>&1 &

# Optional: EQTransformer instance + PickBlue EQT for ensembling/comparison
python scripts/03_run_phasenet.py \
    --model EQTransformer --out-subdir picks_eqt \
    --workers 8 --batch-size 256 \
    --p-thresh 0.1 --s-thresh 0.1 \
    > logs/pick_full_eqt.log 2>&1 &
```

## Recovery / restart notes
- Download script is idempotent — re-running skips per-station-day mseed files that already exist on disk
- Pick scripts are idempotent the same way
- If the NFS PVC fills up unexpectedly, `df -h /home/jovyan` and check: was 90 TB free at start, ~600 GB total expected
- If PID 9415 dies: `tail logs/download_full.log` for cause, then re-launch with the same command (it picks up where it left off)
- Background process check: `ps -p 9415 -o pid,etime,stat`
