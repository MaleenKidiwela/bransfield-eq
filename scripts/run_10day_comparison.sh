#!/usr/bin/env bash
# Multi-day picker comparison: 2019-02-04 → 2019-02-13 (contiguous 10 days,
# active swarm-preceding period).
#
# Two pickers — the recommended hybrid validated on 2019-12-26:
#   1. PhaseNet `instance`   @ p/s 0.1  (best P)
#   2. OBSTransformer        @ p/s 0.5  (perfect S, threshold sweet spot)
#
# Idempotent: re-running skips per-station-day CSVs that already exist.

set -e
cd /home/jovyan/bransfield-eq
source .venv/bin/activate
mkdir -p logs

START=2019-02-04
END=2019-02-14   # exclusive

echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ)  range $START → $END ==="

echo ">>> 1/2  PhaseNet instance @ 0.1"
time python scripts/03_run_phasenet.py \
    --model PhaseNet --weights instance --out-subdir picks \
    --start $START --end $END \
    --p-thresh 0.1 --s-thresh 0.1 --workers 8 \
    > logs/multiday_picks.log 2>&1

echo ">>> 2/2  OBSTransformer obst2024 @ 0.5"
time python scripts/03_run_phasenet.py \
    --model OBSTransformer --weights obst2024 --out-subdir picks_obst_05 \
    --start $START --end $END \
    --p-thresh 0.5 --s-thresh 0.5 --workers 8 \
    > logs/multiday_picks_obst_05.log 2>&1

echo "=== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
