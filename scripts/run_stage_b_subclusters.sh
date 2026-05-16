#!/bin/bash
# Stage B driver. For every sub_<i> label produced by 25_stage_b_partition...,
# run scripts 22 -> 23 -> 24 in parallel. Then merge the per-sub-region
# hypoDD outputs into a single catalog with conflicts resolved by
# preferring the relocation from the sub-region whose centroid is closest.
set -e
cd /home/jovyan/bransfield-eq

REGIONS_CSV="catalogs/stage_b_regions.csv"
if [ ! -s "$REGIONS_CSV" ]; then
    echo "Run scripts/25_stage_b_partition_and_run.py first."
    exit 1
fi

# Extract labels
LABELS=$(.venv/bin/python -c "
import pandas as pd
r = pd.read_csv('$REGIONS_CSV')
print(' '.join(r.label.tolist()))
")
echo "Sub-region labels: $LABELS"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_one () {
    local LABEL="$1"
    local LOG="logs/stage_b_${LABEL}.log"
    {
        echo "=== $LABEL :: phase.dat / station.dat ==="
        .venv/bin/python scripts/22_pyocto_to_hypodd_input.py --label "$LABEL"
        echo "=== $LABEL :: ph2dt ==="
        .venv/bin/python scripts/23_run_ph2dt.py --label "$LABEL" \
            --minlnk 8 --minobs 8 --maxobs 80 --maxsep 5
        echo "=== $LABEL :: hypoDD ==="
        .venv/bin/python scripts/24_run_hypodd.py --label "$LABEL"
        echo "=== $LABEL :: done ==="
    } > "$LOG" 2>&1
}

# Parallel-ish: launch in background, wait for all
PIDS=()
for L in $LABELS; do
    run_one "$L" &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} parallel sub-region pipelines."
for p in "${PIDS[@]}"; do
    wait "$p"
done
echo "All sub-region pipelines done. Catalogs in catalogs/hypodd_sub_*.csv"
