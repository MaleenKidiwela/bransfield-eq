#!/bin/bash
#SBATCH --job-name=bransfield-pn
#SBATCH --output=logs/pn_%A_%a.out
#SBATCH --error=logs/pn_%A_%a.err
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source .venv/bin/activate

python scripts/03_run_phasenet.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --of    "${SLURM_ARRAY_TASK_COUNT}" \
    --weights stead \
    --device cuda
