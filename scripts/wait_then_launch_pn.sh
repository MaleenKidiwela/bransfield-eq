#!/bin/bash
# Wait for OBST picker to exit, then launch PN picker.
OBST_PID="$1"
cd /home/jovyan/bransfield-eq || exit 1

while kill -0 "$OBST_PID" 2>/dev/null; do
    sleep 60
done

printf '\n=== AUTO-LAUNCH PN at %s (OBST PID %s exited) ===\n' \
    "$(date -u +%Y-%m-%dT%H:%MZ)" "$OBST_PID" >> logs/full_year_pn.log

exec .venv/bin/python scripts/03_run_phasenet.py \
    --start 2019-01-01 --end 2020-03-01 \
    --weights instance --p-thresh 0.1 --s-thresh 0.1 \
    --workers 8 --batch-size 256 --device cuda \
    --out-subdir picks \
    >> logs/full_year_pn.log 2>&1 < /dev/null
