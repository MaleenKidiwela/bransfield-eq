#!/bin/bash
# Wait for GrowClust XC prep (Stage 3) to start, then exit, then launch
# the GrowClust binary (Stage 4). Triggered by the Stage 2->3 watcher.
cd /home/jovyan/bransfield-eq || exit 1

DT_FILE="growclust/picker_only/dt.cc"
LOG="logs/growclust_run.log"

# Wait for Stage 3 to start (so we don't fire before it has begun).
seen=0
for _ in $(seq 1 2880); do         # up to ~48 hours of waiting to start
    if pgrep -f 18_growclust_xc_prep >/dev/null 2>&1; then
        seen=1
        break
    fi
    sleep 60
done

if [ "$seen" -ne 1 ]; then
    printf '\n=== ABORT: Stage 3 never appeared within 12h at %s ===\n' \
        "$(date -u +%Y-%m-%dT%H:%MZ)" >> "$LOG"
    exit 1
fi

# Wait for it to finish.
while pgrep -f 18_growclust_xc_prep >/dev/null 2>&1; do
    sleep 60
done

# Verify dt.cc exists and is non-trivial before launching Stage 4.
if [ ! -s "$DT_FILE" ]; then
    printf '\n=== ABORT: Stage 3 exited but %s missing/empty at %s ===\n' \
        "$DT_FILE" "$(date -u +%Y-%m-%dT%H:%MZ)" >> "$LOG"
    exit 1
fi

printf '\n=== AUTO-LAUNCH GrowClust (Stage 4) at %s ===\n' \
    "$(date -u +%Y-%m-%dT%H:%MZ)" >> "$LOG"

exec .venv/bin/python scripts/19_run_growclust.py \
    --label picker_only \
    >> "$LOG" 2>&1 < /dev/null
