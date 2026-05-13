#!/bin/bash
# 10-day pyocto test, run as 10 daily chunks in parallel.
# 10 chunks × 10 threads = 100 cores.
set -e
cd /home/jovyan/bransfield-eq

START="2019-01-01"
N_DAYS=10
THREADS=10
PARALLEL=10
MARGIN=120

mkdir -p catalogs/pyocto_daily logs/pyocto_daily

# Build 10 daily [start, end) pairs starting from START
mapfile -t DAYS < <(.venv/bin/python -c "
from datetime import date, timedelta
import sys
d = date.fromisoformat(sys.argv[1])
for i in range(int(sys.argv[2])):
    s = d + timedelta(days=i)
    e = s + timedelta(days=1)
    print(f'{s.isoformat()} {e.isoformat()}')
" "$START" "$N_DAYS")

echo "scheduled ${#DAYS[@]} daily chunks starting from $START ($THREADS threads each, $PARALLEL parallel)"

run_one() {
  local start=$1 end=$2
  local tag="${start}"
  local log="logs/pyocto_daily/${tag}.log"
  echo "[$(date -u +%H:%M:%S)] launching ${tag} ..."
  .venv/bin/python -u scripts/17_pyocto_associate.py \
    --start "$start" --end "$end" \
    --velocity-model configs/velocity_model.csv \
    --label "day_${tag}" \
    --n-threads $THREADS \
    --margin-seconds $MARGIN \
    > "$log" 2>&1 || echo "[$(date -u +%H:%M:%S)] !!! ${tag} FAILED (see $log)"
  mv "catalogs/pyocto_events_day_${tag}.csv" "catalogs/pyocto_daily/events_${tag}.csv" 2>/dev/null || true
  mv "catalogs/pyocto_picks_day_${tag}.csv"  "catalogs/pyocto_daily/picks_${tag}.csv"  2>/dev/null || true
  # quick line count summary
  if [ -f "catalogs/pyocto_daily/events_${tag}.csv" ]; then
    n=$(($(wc -l < "catalogs/pyocto_daily/events_${tag}.csv") - 1))
    echo "[$(date -u +%H:%M:%S)] finished ${tag}: ${n} events"
  else
    echo "[$(date -u +%H:%M:%S)] finished ${tag}: NO EVENTS FILE"
  fi
}
export -f run_one
export THREADS MARGIN

printf '%s\n' "${DAYS[@]}" | xargs -P $PARALLEL -I {} bash -c 'run_one $(echo {})'

echo ""
echo "=== 10-day daily-chunk test complete ==="
ls -la catalogs/pyocto_daily/events_*.csv 2>/dev/null
echo ""
.venv/bin/python <<'PY'
import pandas as pd, glob
files = sorted(glob.glob('catalogs/pyocto_daily/events_*.csv'))
total = 0
for f in files:
    df = pd.read_csv(f)
    tag = f.split('events_')[1].replace('.csv','')
    print(f"  {tag}: {len(df)} events")
    total += len(df)
print(f"  TOTAL: {total} events across {len(files)} days")
PY
