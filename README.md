# bransfield-eq

ML-assisted earthquake catalog for the Bransfield Basin, Antarctica, using the
BRAVOSEIS ocean-bottom seismometer array (`ZX`) and surrounding land stations
(`5M`). End-to-end pipeline from raw FDSN waveforms to a relatively-located
event catalog.

```
   waveforms      picks            events           relocated events
  ──────────► ─────────────► ────────────────► ──────────────────────►
  (FDSN dl)   (PhaseNet,      (pyocto           (GrowClust XC
              EQT, OBST)       associator)       relocations)
```

## Pipeline stages

| # | Stage | Script(s) | Output |
|---|---|---|---|
| 0 | Inventory + waveform download | `01_data_inventory.py`, `02_download_waveforms.py` | `data/waveforms/`, `data/stationxml/` |
| 1 | ML picking | `03_run_phasenet.py` (supports PhaseNet / EQTransformer / OBSTransformer) | `catalogs/picks*/` |
| 1b | (optional) OBS denoising + finetuned PhaseNet | `07_train_obs_denoiser.py`, `11_run_denoised_picker.py`, `09_finetune_phasenet.py` | denoised pick sets |
| 2 | Association | `17_pyocto_associate.py` | `catalogs/pyocto_events_*.csv` |
| 3 | Differential-time prep (waveform XC) | `18_growclust_xc_prep.py` | `growclust/<label>/dt.cc` |
| 4 | Relative relocation | `19_run_growclust.py` | `catalogs/growclust_*.csv` |

Stages 2–4 require a 1D velocity model (`configs/velocity_model.csv`,
sea-level-referenced with a 1.3 km water layer prepended to a 1D average of
the Orca 3D tomography — see "Velocity model" below).

## Quick start

```bash
# environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Stage 0: inventory + download (day-by-day, idempotent, shardable)
python scripts/01_data_inventory.py
python scripts/02_download_waveforms.py            # serial, or:
python scripts/02_download_waveforms.py --shard 3 --of 16

# Stage 1: pick a full year with PhaseNet + OBSTransformer
python scripts/03_run_phasenet.py \
    --start 2019-01-01 --end 2020-03-01 \
    --weights instance --p-thresh 0.1 --s-thresh 0.1 \
    --device cuda --workers 8 --batch-size 256 \
    --out-subdir picks

python scripts/03_run_phasenet.py \
    --start 2019-01-01 --end 2020-03-01 \
    --model OBSTransformer --weights obst2024 \
    --p-thresh 0.1 --s-thresh 0.1 \
    --device cuda --workers 8 --batch-size 256 \
    --out-subdir picks_obst_01

# One-time corrections from Kidiwela+ supporting info (see "Station geometry" below)
python scripts/apply_S1_corrections.py

# Build the velocity model from Pg_Orca_velocity.nc (one-time)
python scripts/build_velocity_model.py            # writes configs/velocity_model.csv

# Stage 2: associate picks into events
python scripts/17_pyocto_associate.py \
    --start 2019-01-01 --end 2020-03-01 \
    --velocity-model configs/velocity_model.csv \
    --label picker_only

# Stage 3: cross-correlation differential times
python scripts/18_growclust_xc_prep.py --label picker_only --workers 8

# Stage 4: GrowClust relocations
python scripts/19_run_growclust.py --label picker_only
```

The four pipeline stages are chained via `scripts/wait_then_launch_*.sh`
watchers — `setsid nohup`-detached so they survive terminal disconnect and
JupyterHub container restarts. Each stage is idempotent (skips work already
on disk), so a relaunch after any crash just picks up where it left off.

## Repository layout

```
configs/
  Pg_Orca_velocity.nc         3D Pg tomography from Kidiwela+ (source for 1D model)
  velocity_model.csv          1D pyocto/GrowClust input (built by build_velocity_model.py)
  README_velocity.pdf         original Orca model documentation

catalogs/
  station_geometry.csv        per-station lat/lon/elevation/water_depth (patched from SI Table S1)
  manual_picks.csv            Almendros et al. manual picks for evaluation
  picks*/                     gitignored — picker outputs

scripts/
  01_*, 02_*                  data inventory + download
  03_run_phasenet.py          PN / EQT / OBST picker harness
  04_station_geometry.py      build station_geometry.csv from StationXML
  05_validate_picks.py        compare picks against manual catalog
  06_*, 07_*, 08_*, 09_*      finetuning + denoiser pipeline
  10_*, 11_*                  augmented-dataset evaluation, denoised picking
  12_*, 13_*, 14_*            dd-filter post-processing
  15_anomaly_score*           anomaly detection at manual picks
  16_stalta_on_dd.py          STA/LTA baseline on denoised waveforms
  17_pyocto_associate.py      Stage 2 associator
  18_growclust_xc_prep.py     Stage 3 differential-time prep
  19_run_growclust.py         Stage 4 relative relocation
  apply_S1_corrections.py     patch station_geometry.csv + BRA05 clock from Kidiwela+ SI
  build_velocity_model.py     Orca .nc → 1D CSV + plot
  compare_station_geometry_S1.py  cross-check helper
  finetune/                   PyTorch Lightning training modules
  wait_then_launch_*.sh       detached pipeline-stage watchers
  slurm/                      job arrays for CPU cluster

src/bransfield_eq/            shared utilities (config, station resolution, manual-pick parsing)

notes/                        design notes and session logs (06–15 chronological)
```

## Velocity model

`configs/velocity_model.csv` is sea-level-referenced with three regions:

| Depth below sea level | Source | Notes |
|---|---|---|
| 0 → 1.3 km | water layer | Vp = 1.4558 km/s (measured Bransfield value, Kidiwela+ SI), Vs = 0.5 km/s* |
| 1.3 → 11.3 km | Orca Pg tomography median | per-depth median across the 3D volume; Vs derived as Vp/1.78 |
| 16.3, 21.3, 31.3 km | hand-picked Moho extension | Vp = 6.6 / 6.8 / 7.0 km/s; rarely used (Bransfield seismicity is shallow) |

\* Water Vs is set to 0.5 km/s only because pyocto's Eikonal solver crashes on
Vs = 0. The value is physically meaningless — first-arrival rays from
sub-seafloor sources to seafloor OBS never traverse water (Fermat routes them
through the faster rock layers below), so the water Vs is never used in
practice.

The water layer exists for **vertical-datum bookkeeping**, not ray tracing:
OBS sit at different seafloor depths (1019 → 1499 m), and a sea-level datum
lets each station live at its actual elevation in the model. Rebuilding:

```bash
python scripts/build_velocity_model.py
```

writes the CSV and a side-by-side plot at `notes/figures/velocity_model.png`.

## Station geometry

The 15 OBS in Kidiwela+ supporting info Table S1 are patched to:
- **inverted lon/lat** (from joint inversion of 8394 acoustic water-wave arrivals,
  horizontal uncertainty ~10 m vs ~100–250 m for raw drop coordinates)
- **bathymetric depth** (from Almendros et al. 2020) rather than drop z, because
  the inverted z is biased by the uniform 1456 m/s water-velocity assumption.

The largest correction is BRA25 (drop 1326 m → bathymetric 896 m, 430 m
discrepancy). Other diffs are within ±30 m.

**BRA05 clock offset**: the BRA05 OBS clock ran fast by +0.167 s (no drift) for
the entire deployment, per Kidiwela+ SI Text S1. `apply_S1_corrections.py`
subtracts 0.167 s from every pick time in `catalogs/picks*/ZX.BRA05/*.csv`
and writes a marker file to stay idempotent. Re-run after any new BRA05 pick
batch (e.g., a re-picking pass).

## Data dependencies

Bulk data (not in git):
- `data/waveforms/` — symlink to the BRAVOSEIS FDSN waveform archive
- `data/stationxml/` — symlink to the StationXML inventory
- `data/seisbench/` — SeisBench dataset/model cache
- `models/` — finetuned PhaseNet / DeepDenoiser weights

On the OOI/UW JupyterHub cluster these live under `/home/jovyan/my_data/bravoseis/`
on a 100 TB persistent NFS share. On a fresh checkout you'll need to either
mirror the FDSN archive (via `scripts/02_download_waveforms.py`) or symlink
your own location.

Networks downloaded by default: `ZX` (BRAVOSEIS OBS), `5M` (regional land
network), plus `AI.JUBA` and `AM.R4DE2`. Edit `configs/targets.yaml` to
change.

## Computing notes

- Picking GPU memory: PhaseNet @ batch 256 needs ~6 GB; OBSTransformer needs ~10 GB.
- Pyocto loads all picks into RAM (~6 GB for a year of 25 M picks).
- GrowClust XC prep is the slowest stage; `--workers 8` parallelizes over event pairs.
- The compiled GrowClust binary lives at `/home/jovyan/GrowClust/SRC/growclust`
  on the cluster (compile from upstream source on a fresh setup).

## References

- Almendros et al. (2020), *BRAVOSEIS*, J. South Am. Earth Sci.
- Kidiwela et al., *Late-Stage Rift Evolution at Back Arc Basins*,
  *Geochemistry, Geophysics, Geosystems* — see `notes/supporting info.pdf`
  for Tables S1/S2 (OBS relocation, Orca tomography parameters).
- Zhu & Beroza (2019), *PhaseNet*; Mousavi et al. (2020), *EQTransformer*;
  Niksejel & Zhang (2024), *OBSTransformer*; Trabattoni et al. (2023),
  *DeepDenoiser*; Trugman & Shearer (2017), *GrowClust*; Münchmeyer (2023),
  *PyOcto*.
