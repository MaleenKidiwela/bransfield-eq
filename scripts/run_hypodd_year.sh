#!/bin/bash
# Chained year-long hypoDD pipeline:
#   1. (assumes 22_pyocto_to_hypodd_input.py already ran)
#   2. waits for ph2dt to finish (PID 18404 or completion of dt.ct file)
#   3. runs hypoDD on the resulting event.sel + dt.ct
#   4. parses hypoDD.reloc into catalogs/hypodd_picker_only.csv
#   5. plots
set -e
cd /home/jovyan/bransfield-eq

LABEL="picker_only"
LOG="logs/run_hypodd_year.log"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

DT_CT="hypodd/${LABEL}/dt.ct"

log "=== waiting for ph2dt to produce $DT_CT (current bytes: $(stat -c %s $DT_CT 2>/dev/null || echo 0)) ==="
# Wait until ph2dt's parent python wrapper exits AND dt.ct is non-empty.
while pgrep -f "23_run_ph2dt.py --label ${LABEL}\b" >/dev/null; do
    sleep 30
done
if [ ! -s "$DT_CT" ]; then
    log "!!! ph2dt finished but dt.ct is empty"
    exit 1
fi
log "ph2dt done: dt.ct = $(stat -c %s "$DT_CT") bytes"

log "=== launching hypoDD (CT-only) ==="
.venv/bin/python -u scripts/24_run_hypodd.py --label "$LABEL" >> "$LOG" 2>&1

OUT_CSV="catalogs/hypodd_${LABEL}.csv"
if [ ! -s "$OUT_CSV" ]; then
    log "!!! hypoDD failed: $OUT_CSV missing or empty"
    exit 1
fi
log "hypoDD done: $(wc -l < "$OUT_CSV") rows"

log "=== plotting ==="
.venv/bin/python scripts/plot_hypodd_relocations.py --label "$LABEL" \
    --title-extra "(year-long, CT-only)" >> "$LOG" 2>&1
log "=== pipeline complete ==="
