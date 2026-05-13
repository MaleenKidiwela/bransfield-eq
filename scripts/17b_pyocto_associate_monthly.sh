#!/bin/bash
# Run pyocto association in parallel month-chunks across the full year.
# Each chunk writes its own pyocto_events_YYYY-MM.csv; merged at the end.
set -e
cd /home/jovyan/bransfield-eq

MONTHS=(
  "2019-01-01 2019-02-01"
  "2019-02-01 2019-03-01"
  "2019-03-01 2019-04-01"
  "2019-04-01 2019-05-01"
  "2019-05-01 2019-06-01"
  "2019-06-01 2019-07-01"
  "2019-07-01 2019-08-01"
  "2019-08-01 2019-09-01"
  "2019-09-01 2019-10-01"
  "2019-10-01 2019-11-01"
  "2019-11-01 2019-12-01"
  "2019-12-01 2020-01-01"
  "2020-01-01 2020-02-01"
  "2020-02-01 2020-03-01"
)

PARALLEL=8        # 8 concurrent chunks × 8 threads each = 64 cores
THREADS=8

mkdir -p catalogs/pyocto_monthly logs/pyocto_monthly

run_one() {
  local start=$1 end=$2
  local tag="${start:0:7}"
  local log="logs/pyocto_monthly/${tag}.log"
  echo "[$(date -u +%H:%M:%S)] launching ${tag} ..."
  .venv/bin/python -u scripts/17_pyocto_associate.py \
    --start "$start" --end "$end" \
    --velocity-model configs/velocity_model.csv \
    --label "monthly_${tag}" \
    --n-threads $THREADS \
    --margin-seconds 120 \
    > "$log" 2>&1
  # rename outputs so they don't collide
  mv "catalogs/pyocto_events_monthly_${tag}.csv" "catalogs/pyocto_monthly/events_${tag}.csv" 2>/dev/null || true
  mv "catalogs/pyocto_picks_monthly_${tag}.csv"  "catalogs/pyocto_monthly/picks_${tag}.csv"  2>/dev/null || true
  echo "[$(date -u +%H:%M:%S)] finished ${tag}"
}
export -f run_one
export THREADS

printf '%s\n' "${MONTHS[@]}" | xargs -P $PARALLEL -I {} bash -c 'run_one $(echo {})'

echo "=== merging all monthly outputs ==="
.venv/bin/python -c "
import pandas as pd, glob
ev = sorted(glob.glob('catalogs/pyocto_monthly/events_*.csv'))
pk = sorted(glob.glob('catalogs/pyocto_monthly/picks_*.csv'))
if ev:
    pd.concat([pd.read_csv(f) for f in ev], ignore_index=True).to_csv(
        'catalogs/pyocto_events_picker_only.csv', index=False)
    print(f'merged {len(ev)} months -> catalogs/pyocto_events_picker_only.csv')
if pk:
    pd.concat([pd.read_csv(f) for f in pk], ignore_index=True).to_csv(
        'catalogs/pyocto_picks_picker_only.csv', index=False)
    print(f'merged {len(pk)} months -> catalogs/pyocto_picks_picker_only.csv')
import os
e = pd.read_csv('catalogs/pyocto_events_picker_only.csv')
print(f'total events: {len(e):,}')
"
echo "=== full-year pyocto association complete ==="
