#!/bin/bash
# Full-year pyocto: daily chunks, run in batches of 10 days in parallel.
# 10 chunks × 10 threads = 100 cores per batch; batches run sequentially.
set -e
cd /home/jovyan/bransfield-eq

START="2019-01-01"
END="2020-03-01"
BATCH_DAYS=10
THREADS=10
PARALLEL=10
MARGIN=120

mkdir -p catalogs/pyocto_daily logs/pyocto_daily

run_one() {
  local start=$1 end=$2
  local tag="${start}"
  local log="logs/pyocto_daily/${tag}.log"
  echo "    [$(date -u +%H:%M:%S)] launch  ${tag}"
  .venv/bin/python -u scripts/17_pyocto_associate.py \
    --start "$start" --end "$end" \
    --velocity-model configs/velocity_model.csv \
    --label "day_${tag}" \
    --n-threads $THREADS \
    --margin-seconds $MARGIN \
    > "$log" 2>&1 || echo "    [$(date -u +%H:%M:%S)] !!! ${tag} FAILED"
  mv "catalogs/pyocto_events_day_${tag}.csv" "catalogs/pyocto_daily/events_${tag}.csv" 2>/dev/null || true
  mv "catalogs/pyocto_picks_day_${tag}.csv"  "catalogs/pyocto_daily/picks_${tag}.csv"  2>/dev/null || true
  local n
  if [ -f "catalogs/pyocto_daily/events_${tag}.csv" ]; then
    n=$(($(wc -l < "catalogs/pyocto_daily/events_${tag}.csv") - 1))
    echo "    [$(date -u +%H:%M:%S)] finish  ${tag}: ${n} events"
  else
    echo "    [$(date -u +%H:%M:%S)] finish  ${tag}: NO FILE"
  fi
}
export -f run_one
export THREADS MARGIN

# Build list of all daily (start,end) pairs
mapfile -t ALL_DAYS < <(.venv/bin/python -c "
from datetime import date, timedelta
import sys
d = date.fromisoformat(sys.argv[1])
e = date.fromisoformat(sys.argv[2])
while d < e:
    nxt = d + timedelta(days=1)
    print(f'{d.isoformat()} {nxt.isoformat()}')
    d = nxt
" "$START" "$END")
TOTAL_DAYS=${#ALL_DAYS[@]}
echo "=== full year: $TOTAL_DAYS days, batches of $BATCH_DAYS ($PARALLEL parallel, $THREADS threads each) ==="

# Iterate in chunks of BATCH_DAYS
i=0
batch_num=0
while [ $i -lt $TOTAL_DAYS ]; do
    batch_num=$((batch_num + 1))
    batch_start=$i
    batch_end=$((i + BATCH_DAYS))
    [ $batch_end -gt $TOTAL_DAYS ] && batch_end=$TOTAL_DAYS

    # Skip days already done (resume support)
    todo=()
    for ((j=batch_start; j<batch_end; j++)); do
        d=$(echo "${ALL_DAYS[$j]}" | awk '{print $1}')
        if [ -f "catalogs/pyocto_daily/events_${d}.csv" ]; then
            continue
        fi
        todo+=("${ALL_DAYS[$j]}")
    done

    if [ ${#todo[@]} -eq 0 ]; then
        echo "[$(date -u +%H:%M:%S)] batch $batch_num: all $BATCH_DAYS days already done, skipping"
        i=$batch_end
        continue
    fi

    echo ""
    echo "[$(date -u +%H:%M:%S)] === batch $batch_num: ${#todo[@]} days, ${ALL_DAYS[$batch_start]:0:10} -> ${ALL_DAYS[$((batch_end-1))]:0:10} ==="
    printf '%s\n' "${todo[@]}" | xargs -P $PARALLEL -I {} bash -c 'run_one $(echo {})'
    echo "[$(date -u +%H:%M:%S)] === batch $batch_num done ==="

    i=$batch_end
done

echo ""
echo "=== all batches complete, merging ==="
.venv/bin/python <<'PY'
import pandas as pd, glob
ev = sorted(glob.glob('catalogs/pyocto_daily/events_*.csv'))
pk = sorted(glob.glob('catalogs/pyocto_daily/picks_*.csv'))
print(f"  event files: {len(ev)}")
print(f"  pick  files: {len(pk)}")
if ev:
    big = pd.concat([pd.read_csv(f) for f in ev], ignore_index=True)
    big.to_csv('catalogs/pyocto_events_picker_only.csv', index=False)
    print(f"  merged events ({len(big):,}) -> catalogs/pyocto_events_picker_only.csv")
if pk:
    big = pd.concat([pd.read_csv(f) for f in pk], ignore_index=True)
    big.to_csv('catalogs/pyocto_picks_picker_only.csv', index=False)
    print(f"  merged picks ({len(big):,}) -> catalogs/pyocto_picks_picker_only.csv")
PY
echo "=== full-year pyocto (daily chunks) complete ==="
