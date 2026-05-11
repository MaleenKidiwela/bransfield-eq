#!/usr/bin/env bash
# Build full Bransfield fine-tune dataset, then run real fine-tune.
set -e
cd /home/jovyan/bransfield-eq
source .venv/bin/activate
mkdir -p logs

echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo ">>> A: Build full dataset (all 1,124 mag07 events, 100 noise/station)"
time python scripts/06_build_finetune_dataset.py \
    --noise-per-station 100 \
    > logs/finetune_dataset_build.log 2>&1
echo "    [done A]"

# Re-link the canonical dataset dir so it points at fresh files
ln -sf /home/jovyan/bransfield-eq/data/seisbench/bransfield_train.csv  /home/jovyan/bransfield-eq/data/seisbench/bransfield/metadata.csv
ln -sf /home/jovyan/bransfield-eq/data/seisbench/bransfield_train.hdf5 /home/jovyan/bransfield-eq/data/seisbench/bransfield/waveforms.hdf5

echo ">>> B: Fine-tune PhaseNet 'instance' (30 epochs, batch 64, lr 1e-4)"
time python scripts/07_finetune_phasenet.py \
    --epochs 30 --batch-size 64 --workers 4 \
    > logs/finetune_train.log 2>&1
echo "    [done B]"

echo "=== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
