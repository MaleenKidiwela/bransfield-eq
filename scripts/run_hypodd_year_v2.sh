#!/bin/bash
# Chained year-long hypoDD pipeline with v2 fixes already in scripts 22/23/24.
set -e
cd /home/jovyan/bransfield-eq

LABEL="picker_only"
LOG="logs/run_hypodd_year_v2.log"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== Stage A: ph2dt (CT pair generation) ==="
.venv/bin/python -u scripts/23_run_ph2dt.py --label "$LABEL" >> "$LOG" 2>&1

DT_CT="hypodd/${LABEL}/dt.ct"
if [ ! -s "$DT_CT" ]; then
    log "!!! ph2dt failed: $DT_CT missing or empty"
    exit 1
fi
log "ph2dt done: dt.ct = $(stat -c %s "$DT_CT") bytes"

log "=== Stage B: hypoDD inversion ==="
.venv/bin/python -u scripts/24_run_hypodd.py --label "$LABEL" >> "$LOG" 2>&1

OUT_CSV="catalogs/hypodd_${LABEL}.csv"
if [ ! -s "$OUT_CSV" ]; then
    log "!!! hypoDD failed: $OUT_CSV missing or empty"
    exit 1
fi
log "hypoDD done: $(wc -l < "$OUT_CSV") rows"

log "=== Stage C: plot ==="
.venv/bin/python scripts/plot_hypodd_relocations.py --label "$LABEL" \
    --title-extra "(year-long, v2 fixes, IMOD=1, CT-only)" >> "$LOG" 2>&1
log "=== pipeline complete ==="
