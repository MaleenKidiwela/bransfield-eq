#!/bin/bash
# Full-year pyocto association on the BENCHMARKED pick pool (2026-09-12).
#
# Replaces 17e_pyocto_daily_year.sh, which had two faults:
#   1. It never passed --pick-sources, so it silently fell through to the
#      default (picks:P,picks_obst_01:PS) -- the OLD production pool. It would
#      have run for ~12 h and produced a year of the wrong catalogue, with no
#      error at any point.
#   2. It assumed 10 parallel x 10 threads = 100 cores. This pod's cgroup quota
#      is 32 CPUs (/sys/fs/cgroup/cpu.max), so that is a 3x oversubscription.
#
# Daily chunks with a 120 s margin so events near midnight keep their late
# picks; the strict [start, end) filter stops adjacent chunks double-counting.
# Resumable: a day whose output already exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

START=${START:-2019-01-01}
END=${END:-2020-03-01}
PARALLEL=${PARALLEL:-6}       # 6 x 5 threads = 30 of the 32-core quota
THREADS=${THREADS:-5}
MARGIN=${MARGIN:-120}
POOL=${POOL:-'picks_pn_diting:PS,picks_pnlight_obs:PS'}
EXCLUDE=${EXCLUDE:-}

OUT=catalogs/pyocto_daily_newpool
mkdir -p "$OUT" logs/pyocto_newpool

# Fail loudly rather than silently associating the wrong picks.
for spec in ${POOL//,/ }; do
  d="catalogs/${spec%%:*}"
  [ -d "$d" ] || { echo "FATAL: pick dir $d missing -- refusing to fall back to defaults" >&2; exit 1; }
done
echo "pool:      $POOL"
echo "window:    $START .. $END"
echo "parallel:  $PARALLEL x $THREADS threads (cgroup quota: $(awk '{print $1/$2}' /sys/fs/cgroup/cpu.max) cores)"
[ -n "$EXCLUDE" ] && echo "excluding: $EXCLUDE"

run_one() {
  local start=$1 end=$2 tag=$1
  [ -f "$OUT/events_${tag}.csv" ] && { echo "  skip   $tag (done)"; return 0; }
  local extra=()
  [ -n "$EXCLUDE" ] && extra=(--exclude-stations "$EXCLUDE")
  PYTHONPATH=src python3 -u scripts/17_pyocto_associate.py \
      --start "$start" --end "$end" \
      --velocity-model configs/velocity_model.csv \
      --label "nd_${tag}" --pick-sources "$POOL" \
      --n-threads "$THREADS" --margin-seconds "$MARGIN" "${extra[@]}" \
      > "logs/pyocto_newpool/${tag}.log" 2>&1
  local rc=$?
  mv "catalogs/pyocto_events_nd_${tag}.csv" "$OUT/events_${tag}.csv" 2>/dev/null
  mv "catalogs/pyocto_picks_nd_${tag}.csv"  "$OUT/picks_${tag}.csv"  2>/dev/null
  rm -f "catalogs/pyocto_origin_nd_${tag}.json"
  if [ -f "$OUT/events_${tag}.csv" ]; then
    echo "  done   $tag: $(($(wc -l < "$OUT/events_${tag}.csv") - 1)) events"
  else
    echo "  FAILED $tag (rc=$rc) -- see logs/pyocto_newpool/${tag}.log"
  fi
}
export -f run_one; export OUT THREADS MARGIN POOL EXCLUDE

mapfile -t DAYS < <(python3 -c "
from datetime import date, timedelta
import sys
d=date.fromisoformat(sys.argv[1]); e=date.fromisoformat(sys.argv[2])
while d<e: print(d.isoformat(), (d+timedelta(days=1)).isoformat()); d+=timedelta(days=1)
" "$START" "$END")
echo "days to process: ${#DAYS[@]}"
printf '%s\n' "${DAYS[@]}" | xargs -P "$PARALLEL" -L1 bash -c 'run_one $0 $1'

echo
echo "=== merging ==="
PYTHONPATH=src python3 -u - <<'PYEOF'
import glob, pandas as pd, pathlib
out = pathlib.Path("catalogs/pyocto_daily_newpool")
ev = sorted(out.glob("events_*.csv")); pk = sorted(out.glob("picks_*.csv"))
print(f"  {len(ev)} daily event files, {len(pk)} pick files")
E=[]; P=[]; off=0
for e,p in zip(ev,pk):
    de=pd.read_csv(e); dp=pd.read_csv(p)
    if de.empty: continue
    # Daily 'idx' restarts at 0 every chunk. Make a GLOBALLY unique event key
    # before merging -- joining on a per-chunk idx matched 2.5% of rows the
    # last time this was done wrong.
    de["event_uid"]=de["idx"]+off
    kc="event_idx" if "event_idx" in dp.columns else "idx"
    dp["event_uid"]=dp[kc]+off
    off += int(de["idx"].max())+1
    E.append(de); P.append(dp)
ev_all=pd.concat(E,ignore_index=True); pk_all=pd.concat(P,ignore_index=True)
assert ev_all.event_uid.is_unique, "event_uid not unique after merge"
ev_all.to_csv("catalogs/pyocto_events_year_newpool.csv", index=False)
pk_all.to_csv("catalogs/pyocto_picks_year_newpool.csv", index=False)
print(f"  wrote {len(ev_all):,} events / {len(pk_all):,} picks")
print(f"  lat {ev_all.latitude.min():.3f}..{ev_all.latitude.max():.3f}  "
      f"lon {ev_all.longitude.min():.3f}..{ev_all.longitude.max():.3f}")
PYEOF
echo "=== done $(date -u) ==="
