#!/bin/bash
#SBATCH --job-name=bransfield-pn
#SBATCH --output=logs/pn_%A_%a.out
#SBATCH --error=logs/pn_%A_%a.err
#SBATCH --array=0-31           # 32-way: matches your 32-CPU node
#SBATCH --cpus-per-task=1      # 1 CPU per shard, 32 shards = 32 CPUs
#SBATCH --mem=4G               # measured peak ~3.1 GB; headroom
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Pin threading so each shard uses exactly 1 CPU — prevents 32 shards
# from each spawning 32 OpenMP threads and thrashing the box.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1

source .venv/bin/activate

python scripts/03_run_phasenet.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --of    "${SLURM_ARRAY_TASK_COUNT}" \
    --model PhaseNet \
    --weights instance \
    --p-thresh 0.1 \
    --s-thresh 0.1 \
    --device cpu
