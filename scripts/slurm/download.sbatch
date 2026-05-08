#!/bin/bash
#SBATCH --job-name=bransfield-dl
#SBATCH --output=logs/dl_%A_%a.out
#SBATCH --error=logs/dl_%A_%a.err
#SBATCH --array=0-15
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00

# Edit paths for your cluster.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source .venv/bin/activate

python scripts/02_download_waveforms.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --of    "${SLURM_ARRAY_TASK_COUNT}"
