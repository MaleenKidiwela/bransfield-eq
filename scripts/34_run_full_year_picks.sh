#!/usr/bin/env bash
# Full-year re-pick with the benchmarked pool (session 2026-09-11).
#
#   diting   PhaseNet/diting        3C, ALL networks  -> catalogs/picks_pn_diting/
#   pnlight  PhaseNetLight/obs      4C, ZX only       -> catalogs/picks_pnlight_obs/
#            (PickBlue; the 4th channel is the hydrophone, which only ZX has)
#
# Both at P/S threshold 0.1 so downstream can filter by probability instead of
# re-picking. Resumable: scripts/03_run_phasenet.py skips station-days whose
# CSV already exists, so re-running after a crash or a pod restart continues.
#
# Run the preflight first:
#   PYTHONPATH=src python scripts/35_preflight_gpu_run.py
#
# Usage:
#   bash scripts/34_run_full_year_picks.sh                  # auto device, auto workers
#   bash scripts/34_run_full_year_picks.sh --workers 6
#   bash scripts/34_run_full_year_picks.sh --only diting
#   bash scripts/34_run_full_year_picks.sh --shard 0 --of 4  # split across 4 processes/pods
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

START=${START:-2019-01-01}
END=${END:-2020-03-01}
P_THRESH=${P_THRESH:-0.1}
S_THRESH=${S_THRESH:-0.1}
WORKERS=""
ONLY=""
SHARD=0
OF=1
SEED_FROM_BENCH=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers) WORKERS="$2"; shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;
    --shard)   SHARD="$2"; shift 2 ;;
    --of)      OF="$2"; shift 2 ;;
    --no-seed) SEED_FROM_BENCH=0; shift ;;
    --start)   START="$2"; shift 2 ;;
    --end)     END="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

CH3="EH?,HH?,BH?,SH?,EL?,HL?,BL?,SL?"
CH4="${CH3},EDH,HDH,BDH,SDH"

DEVICE=$(PYTHONPATH=src python3 -c "import torch;print('cuda' if torch.cuda.is_available() else 'cpu')")
if [[ -z "$WORKERS" ]]; then
  if [[ "$DEVICE" == "cuda" ]]; then WORKERS=6; else WORKERS=8; fi
fi
if [[ "$DEVICE" == "cuda" ]]; then BATCH=512; THREADS=2; else BATCH=256; THREADS=4; fi

mkdir -p logs
LOG="logs/full_year_picks_$(date -u +%Y%m%dT%H%M%SZ)_shard${SHARD}of${OF}.log"

{
echo "=== full-year re-pick ==="
echo "device=$DEVICE workers=$WORKERS batch=$BATCH omp_threads=$THREADS"
echo "window=$START .. $END  thresholds P=$P_THRESH S=$S_THRESH"
echo "shard $SHARD of $OF"
echo "log=$LOG"
echo

# Seed the output dirs with the benchmark's own CSVs where they exist: identical
# model, weights, channels and thresholds, so those station-days need not re-run.
if [[ "$SEED_FROM_BENCH" == "1" ]]; then
  for pair in "picks_bench_pn_diting:picks_pn_diting" "picks_bench_pnlight_obs:picks_pnlight_obs"; do
    src="catalogs/${pair%%:*}"; dst="catalogs/${pair##*:}"
    if [[ -d "$src" ]]; then
      mkdir -p "$dst"
      before=$(find "$dst" -name '*.csv' 2>/dev/null | wc -l)
      cp -rn "$src"/* "$dst"/ 2>/dev/null
      after=$(find "$dst" -name '*.csv' 2>/dev/null | wc -l)
      echo "seeded $dst from $src: $before -> $after station-days"
    fi
  done
  echo
fi

run_model () {  # tag model weights glob out_subdir [network]
  local tag=$1 model=$2 weights=$3 chans=$4 out=$5 net=${6:-}
  if [[ -n "$ONLY" && "$ONLY" != "$tag" ]]; then
    echo "--- skipping $tag (--only $ONLY) ---"; return 0
  fi
  local netarg=()
  [[ -n "$net" ]] && netarg=(--network "$net")
  echo "########## $tag : $model/$weights -> catalogs/$out ##########"
  echo "start $(date -u +%H:%M:%S)"
  OMP_NUM_THREADS=$THREADS PYTHONPATH=src python3 scripts/03_run_phasenet.py \
    --model "$model" --weights "$weights" \
    --picking-channels "$chans" \
    --start "$START" --end "$END" "${netarg[@]}" \
    --out-subdir "$out" \
    --p-thresh "$P_THRESH" --s-thresh "$S_THRESH" \
    --device "$DEVICE" --workers "$WORKERS" --batch-size "$BATCH" \
    --shard "$SHARD" --of "$OF"
  local rc=$?
  echo "end   $(date -u +%H:%M:%S)  exit=$rc"
  echo "csv count: $(find "catalogs/$out" -name '*.csv' 2>/dev/null | wc -l)"
  echo
  return $rc
}

run_model diting  PhaseNet      diting "$CH3" picks_pn_diting
run_model pnlight PhaseNetLight obs    "$CH4" picks_pnlight_obs ZX

echo "=== picking complete $(date -u) ==="
echo
echo "Next: associate with the new pool (one test month FIRST, then the year):"
echo "  PYTHONPATH=src python3 scripts/17_pyocto_associate.py \\"
echo "    --start 2020-01-01 --end 2020-02-01 --label picker_only \\"
echo "    --pick-sources 'picks_pn_diting:PS,picks_pnlight_obs:PS'"
echo
echo "Compare against the current catalogue before committing to the full year:"
echo "  event count, picks/event, post-fit RMS, NLLoc sigma."
} 2>&1 | tee "$LOG"
