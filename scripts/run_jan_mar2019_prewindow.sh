#!/bin/bash
# Re-run the Jan 1 - Mar 15 2019 subset using pre-windowing (Stage 2.5).
# Output is bit-identical math to the front-load path; comparing the two
# is a sanity check that pre-windowing didn't break accuracy.
set -e
cd /home/jovyan/bransfield-eq

LABEL="jan_mar2019"
LOG="logs/run_${LABEL}_prewindow.log"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Wipe any leftover memmap so 18b regenerates fresh.
rm -f "growclust/${LABEL}/pick_windows.npy" "growclust/${LABEL}/pick_index.parquet"

log "=== Stage 2.5: pre-window picks for $LABEL ==="
.venv/bin/python -u scripts/18b_prewindow_picks.py \
    --label "$LABEL" --workers 32 >> "$LOG" 2>&1

log "=== Stage 3: XC prep (memmap path) ==="
.venv/bin/python -u scripts/18_growclust_xc_prep.py \
    --label "$LABEL" --workers 32 --max-pairs-per-event 80 >> "$LOG" 2>&1

DT_CC="growclust/${LABEL}/dt.cc"
ST_LIST="growclust/${LABEL}/stlist.txt"
log "XC prep done: $(wc -l < "$DT_CC") dt.cc lines"

log "stripping network prefixes for GrowClust ..."
.venv/bin/python - "$DT_CC" "$ST_LIST" <<'PY'
import sys
from pathlib import Path
dtcc = Path(sys.argv[1]); stlist = Path(sys.argv[2])
src = stlist.read_text().splitlines()
out, seen = [], set()
for ln in src:
    parts = ln.split()
    if len(parts) < 3: continue
    sta = parts[0]
    if len(parts) == 4:
        bare = parts[1]; lat, lon = parts[2], parts[3]
    else:
        bare = sta.split(".")[-1]; lat, lon = parts[1], parts[2]
    if bare in seen: continue
    seen.add(bare)
    out.append(f"{bare:<8s}{float(lat):10.4f}{float(lon):12.4f}")
stlist.write_text("\n".join(out) + "\n")

out_lines, n_obs = [], 0
for ln in dtcc.read_text().splitlines():
    if ln.startswith("#") or not ln.strip():
        out_lines.append(ln); continue
    parts = ln.split()
    if len(parts) < 4: continue
    sta_bare = parts[0].split(".")[-1]
    out_lines.append(f"{sta_bare:<8s} {parts[1]:>8s} {parts[2]:>6s} {parts[3]}")
    n_obs += 1
dtcc.write_text("\n".join(out_lines) + "\n")
print(f"  rewrote dt.cc with {n_obs} obs lines")
PY

log "=== Stage 4: GrowClust ==="
.venv/bin/python -u scripts/19_run_growclust.py --label "$LABEL" >> "$LOG" 2>&1

# Stash result under a distinct name for diffing.
cp "catalogs/growclust_${LABEL}.csv" "catalogs/growclust_${LABEL}_prewindow.csv"
log "=== done: catalogs/growclust_${LABEL}_prewindow.csv "
log "($(wc -l < catalogs/growclust_${LABEL}_prewindow.csv) rows)"
