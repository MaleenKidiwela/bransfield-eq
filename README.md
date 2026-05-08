# bransfield-eq

ML-assisted earthquake workflow for the Bransfield Basin.

Methodology lives in the Obsidian vault under
`Obsidian Vault/BransfieldEQ/` — see `PROJECT_PLAN.md` there.

## Stages
1. ML picking (SeisBench / PhaseNet)
2. Event discrimination
3. Association
4. Location
5. ML polarity picking
6. Focal mechanisms

## Data
EarthScope FDSN, networks `ZX`, `5M`, plus stations `AI.JUBA`, `AM.R4DE2`.
Run `scripts/01_data_inventory.py` first to size the download.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Stage 1 — picking pipeline

```bash
# (1) station inventory + size estimate (laptop-OK, ~MB)
python scripts/01_data_inventory.py

# (2) waveform download — day-by-day, idempotent, shardable
python scripts/02_download_waveforms.py                 # serial
python scripts/02_download_waveforms.py --shard 3 --of 16   # one worker

# (3) PhaseNet picking — per station-day
python scripts/03_run_phasenet.py --device cuda

# SLURM job arrays
sbatch scripts/slurm/download.sbatch
sbatch scripts/slurm/pick.sbatch
```

All targets/window/channels live in `configs/targets.yaml`.
See the vault note `BransfieldEQ/stages/01-ml-picking.md` for design rationale.
