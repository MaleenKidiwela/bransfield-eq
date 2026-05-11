#!/bin/bash
# Wait for pyocto to exit, then launch GrowClust XC prep.
PYOCTO_PID="$1"
cd /home/jovyan/bransfield-eq || exit 1

while kill -0 "$PYOCTO_PID" 2>/dev/null; do
    sleep 60
done

# Only advance if pyocto produced the expected catalog.
if [ ! -s catalogs/pyocto_events_picker_only.csv ]; then
    printf '\n=== ABORT: pyocto exited but catalogs/pyocto_events_picker_only.csv missing/empty at %s ===\n' \
        "$(date -u +%Y-%m-%dT%H:%MZ)" >> logs/growclust_xc_prep.log
    exit 1
fi

printf '\n=== AUTO-LAUNCH GrowClust XC prep at %s (pyocto PID %s exited) ===\n' \
    "$(date -u +%Y-%m-%dT%H:%MZ)" "$PYOCTO_PID" >> logs/growclust_xc_prep.log

exec .venv/bin/python scripts/18_growclust_xc_prep.py \
    --label picker_only --workers 8 \
    >> logs/growclust_xc_prep.log 2>&1 < /dev/null
