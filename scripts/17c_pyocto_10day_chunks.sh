#!/bin/bash
# Run pyocto association in parallel 10-day chunks across the full year.
# 22 parallel processes × 8 threads each = 176 cores fully utilized.
set -e
cd /home/jovyan/bransfield-eq

START="2019-01-01"
END="2020-03-01"
CHUNK_DAYS=10
THREADS=8
PARALLEL=22
MARGIN=120

mkdir -p catalogs/pyocto_chunks logs/pyocto_chunks

# Build the list of [start, end) chunk pairs
PYBUILD='
from datetime import date, timedelta
import sys
d  = date.fromisoformat(sys.argv[1])
e  = date.fromisoformat(sys.argv[2])
n  = int(sys.argv[3])
while d < e:
    nxt = min(d + timedelta(days=n), e)
    print(f"{d.isoformat()} {nxt.isoformat()}")
    d = nxt
'
mapfile -t CHUNKS < <(.venv/bin/python -c "$PYBUILD" "$START" "$END" "$CHUNK_DAYS")
echo "scheduled ${#CHUNKS[@]} chunks of $CHUNK_DAYS days each"

run_one() {
  local start=$1 end=$2
  local tag="${start}_${end}"
  local log="logs/pyocto_chunks/${tag}.log"
  echo "[$(date -u +%H:%M:%S)] launching ${tag} ..."
  .venv/bin/python -u scripts/17_pyocto_associate.py \
    --start "$start" --end "$end" \
    --velocity-model configs/velocity_model.csv \
    --label "chunk_${tag}" \
    --n-threads $THREADS \
    --margin-seconds $MARGIN \
    > "$log" 2>&1 || echo "[$(date -u +%H:%M:%S)] !!! ${tag} FAILED (see $log)"
  # consolidate output into pyocto_chunks/ and free pyocto_*.csv top names
  mv "catalogs/pyocto_events_chunk_${tag}.csv" "catalogs/pyocto_chunks/events_${tag}.csv" 2>/dev/null || true
  mv "catalogs/pyocto_picks_chunk_${tag}.csv"  "catalogs/pyocto_chunks/picks_${tag}.csv"  2>/dev/null || true
  echo "[$(date -u +%H:%M:%S)] finished ${tag}"
}
export -f run_one
export THREADS MARGIN

printf '%s\n' "${CHUNKS[@]}" | xargs -P $PARALLEL -I {} bash -c 'run_one $(echo {})'

echo "=== merging all chunk outputs ==="
.venv/bin/python <<'PY'
import pandas as pd, glob, os
ev = sorted(glob.glob('catalogs/pyocto_chunks/events_*.csv'))
pk = sorted(glob.glob('catalogs/pyocto_chunks/picks_*.csv'))
print(f"event files: {len(ev)}")
print(f"pick  files: {len(pk)}")
if ev:
    big = pd.concat([pd.read_csv(f) for f in ev], ignore_index=True)
    big.to_csv('catalogs/pyocto_events_picker_only.csv', index=False)
    print(f"merged events -> catalogs/pyocto_events_picker_only.csv ({len(big):,} events)")
if pk:
    big = pd.concat([pd.read_csv(f) for f in pk], ignore_index=True)
    big.to_csv('catalogs/pyocto_picks_picker_only.csv', index=False)
    print(f"merged picks  -> catalogs/pyocto_picks_picker_only.csv ({len(big):,} pick rows)")
PY
echo "=== full-year pyocto (10-day chunks) complete ==="
