#!/bin/bash
# Stage B (jan2019-seeded variant) driver.
set -e
cd /home/jovyan/bransfield-eq

REGIONS_CSV="catalogs/stage_bj_regions.csv"
LABELS=$(.venv/bin/python -c "
import pandas as pd
r = pd.read_csv('$REGIONS_CSV')
print(' '.join(r.label.tolist()))
")
echo "Sub-region labels: $LABELS"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run_one () {
    local LABEL="$1"
    local LOG="logs/stage_bj_${LABEL}.log"
    {
        .venv/bin/python scripts/22_pyocto_to_hypodd_input.py --label "$LABEL"
        .venv/bin/python scripts/23_run_ph2dt.py --label "$LABEL" \
            --minlnk 8 --minobs 8 --maxobs 80 --maxsep 5
        .venv/bin/python scripts/24_run_hypodd.py --label "$LABEL"
        echo "=== $LABEL :: done ==="
    } > "$LOG" 2>&1
}

PIDS=()
for L in $LABELS; do
    run_one "$L" &
    PIDS+=($!)
done
echo "Launched ${#PIDS[@]} parallel pipelines."
for p in "${PIDS[@]}"; do wait "$p"; done
echo "All sub-region pipelines done."
