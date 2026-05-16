#!/bin/bash
# Auto-trigger Stage 3 + Stage 4 once the pyocto daily runner has finished.
# Polls for the full-year merged catalog, then runs XC prep + GrowClust binary.
set -e
cd /home/jovyan/bransfield-eq

LABEL="picker_only"
FINAL_LOG="logs/pipeline_finale.log"
EVENTS_CSV="catalogs/pyocto_events_${LABEL}.csv"
PICKS_CSV="catalogs/pyocto_picks_${LABEL}.csv"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$FINAL_LOG"; }

log "=== pipeline_finale started ==="
log "waiting for daily pyocto to finish ..."

# Wait until the daily runner exits (PPID=1 process), polling every 5 min.
# Detected via either: (a) no `17e_pyocto_daily_year` shell anywhere, OR
# (b) the merged events file exists (runner writes it as last step).
while true; do
    runner_alive=$(pgrep -f "17e_pyocto_daily_year" | wc -l)
    if [ "$runner_alive" -eq 0 ] && [ -f "$EVENTS_CSV" ]; then
        break
    fi
    sleep 300
done

n_events=$(wc -l < "$EVENTS_CSV")
log "pyocto done: $EVENTS_CSV has $n_events lines"

# ---- Stage 3 prep: add lat/lon/depth columns to events file ----
log "augmenting events file with lat/lon/depth columns ..."
.venv/bin/python - "$EVENTS_CSV" <<'PY'
import sys, pandas as pd
from pyproj import CRS, Transformer
fpath = sys.argv[1]
stations = pd.read_csv("catalogs/station_geometry.csv")
crs = CRS.from_proj4(f"+proj=tmerc +lat_0={stations.latitude.mean()} "
                     f"+lon_0={stations.longitude.mean()} +ellps=WGS84")
inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
e = pd.read_csv(fpath)
if "latitude" not in e.columns:
    lons, lats = inv.transform(e.x.values * 1000, e.y.values * 1000)
    e["longitude"] = lons
    e["latitude"] = lats
    e["depth"] = e.z
    e.to_csv(fpath, index=False)
    print(f"  added lat/lon/depth to {len(e)} events")
else:
    print(f"  events already has lat/lon/depth; skipping")
PY

# Pin BLAS / OMP threads to 1 so each worker doesn't oversubscribe the 32 CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Stage 2.5: Pre-window every pick into a memmap ----
# Collapses XC working set from ~420 GB (station-day cache) to ~580 MB. Must run
# before Stage 3. See notes/17_session_2026-05-13.md for rationale.
PICK_MM="growclust/${LABEL}/pick_windows.npy"
if [ ! -s "$PICK_MM" ]; then
    log "launching Stage 2.5 (pre-window picks) ..."
    .venv/bin/python -u scripts/18b_prewindow_picks.py \
        --label "$LABEL" \
        --workers 32 \
        >> "$FINAL_LOG" 2>&1
    if [ ! -s "$PICK_MM" ]; then
        log "!!! Pre-window failed: $PICK_MM missing or empty"
        exit 1
    fi
    log "Stage 2.5 done: $(du -h "$PICK_MM" | cut -f1) pre-windowed snippet array"
else
    log "Stage 2.5 already done: $PICK_MM present, skipping"
fi

# ---- Stage 3: XC prep ----
log "launching Stage 3 (XC prep) ..."
.venv/bin/python -u scripts/18_growclust_xc_prep.py \
    --label "$LABEL" \
    --workers 32 \
    --max-pairs-per-event 80 \
    >> "$FINAL_LOG" 2>&1

DT_CC="growclust/${LABEL}/dt.cc"
ST_LIST="growclust/${LABEL}/stlist.txt"
if [ ! -s "$DT_CC" ]; then
    log "!!! XC prep failed: $DT_CC missing or empty"
    exit 1
fi
log "Stage 3 done: $(wc -l < "$DT_CC") dt.cc lines"

# ---- Strip network prefix from stlist.txt and dt.cc (GrowClust parser needs bare codes)
log "stripping network prefixes for GrowClust compatibility ..."
.venv/bin/python - "$DT_CC" "$ST_LIST" <<'PY'
import sys
from pathlib import Path
dtcc = Path(sys.argv[1]); stlist = Path(sys.argv[2])

# stlist
src = stlist.read_text().splitlines()
out = []
seen = set()
for ln in src:
    parts = ln.split()
    if len(parts) < 3: continue
    sta = parts[0]
    if len(parts) == 4:           # already "NW STN LAT LON" format
        bare = parts[1]
        lat, lon = parts[2], parts[3]
    else:                          # "NW.STN LAT LON" or "STN LAT LON"
        bare = sta.split(".")[-1]
        lat, lon = parts[1], parts[2]
    if bare in seen: continue
    seen.add(bare)
    out.append(f"{bare:<8s}{float(lat):10.4f}{float(lon):12.4f}")
stlist.write_text("\n".join(out) + "\n")
print(f"  rewrote stlist with {len(out)} bare station codes")

# dt.cc
out_lines = []
n_obs = 0
for ln in dtcc.read_text().splitlines():
    if ln.startswith("#") or not ln.strip():
        out_lines.append(ln)
        continue
    parts = ln.split()
    if len(parts) < 4: continue
    sta_bare = parts[0].split(".")[-1]
    out_lines.append(f"{sta_bare:<8s} {parts[1]:>8s} {parts[2]:>6s} {parts[3]}")
    n_obs += 1
dtcc.write_text("\n".join(out_lines) + "\n")
print(f"  rewrote dt.cc with {n_obs} obs lines")
PY

# ---- Stage 4: GrowClust binary ----
log "launching Stage 4 (GrowClust relocation) ..."
.venv/bin/python -u scripts/19_run_growclust.py \
    --label "$LABEL" \
    >> "$FINAL_LOG" 2>&1

OUT_CSV="catalogs/growclust_${LABEL}.csv"
if [ ! -s "$OUT_CSV" ]; then
    log "!!! Stage 4 failed: $OUT_CSV missing or empty"
    exit 1
fi
log "=== pipeline_finale done: relocated catalog at $OUT_CSV ($(wc -l < "$OUT_CSV") rows) ==="
