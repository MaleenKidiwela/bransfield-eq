# One-day verification (2019-12-26)

Goal: reproduce the laptop pipeline numbers on the cluster before committing to the full year, AND sanity-check both pickers on the same day.

## Inputs (laptop, predicted)
- 1.66 GB download, 35 of 37 stations (AM.R4DE2 in Uruguay epoch this date)
- 6,902 PhaseNet picks (5,809 P + 1,093 S) at `weights=instance`, `p=s=0.1`
- P recall 64%, S recall 42% vs manual catalog
- ~17 min wall on M-series CPU; expected ~5 min on L40S

## Step results (cluster, observed)

| Step | Predicted | Got | Notes |
|---|---|---|---|
| 1 — inventory | 560 GB year footprint | 560.5 GB | 22 ZX OBS + 14 5M + AI.JUBA + AM.R4DE2; 157 channel-epoch rows |
| 2 — download | 35 of 37 stns / 1.66 GB | **35 / 1.66 GB** | 2 empty (R4DE2 epoch + 1 ZX) — expected |
| 3 — picking (PhaseNet) | 6,902 picks, ~5 min | **6,899 picks, 3m32s** | Δ = 3 picks (~0.04 %, threshold-boundary noise) |
| 3 — picking (EQTransformer) | not predicted | **7,113 picks, 3m46s** | Run for picker-comparison ground truth |
| 4 — station geometry | OBS depths 785–1943 m | same | |
| 5 — validation (PhaseNet) | P 64%, S 42% | **63.6 % / 41.7 %** | mag07 trusted subset |
| 5 — validation (EQTransformer) | not predicted | **36.4 % / 50.0 %** | Same mag07 subset |

## Picker comparison on day-26 (mag07-trusted)

| Picker | Total | P TP/FP/FN | **P recall** | S TP/FP/FN | **S recall** |
|---|---|---|---|---|---|
| PhaseNet `instance` | 6,899 | 7 / 2014 / 4 | **63.6 %** | 5 / 271 / 7 | **41.7 %** |
| EQTransformer `instance` | 7,113 | 4 / 2096 / 7 | **36.4 %** | 6 / 294 / 6 | **50.0 %** |

**Models are complementary** — PhaseNet wins on P (64 % vs 36 %), EQTransformer wins on S (42 % → 50 %). Strong indication that an ensemble (union of triggers, or weighted vote) will beat either alone. Defer detailed ensembling analysis until the year run is in.

## Interpretation
- ✅ Pipeline is deterministic across hardware: same picks (within float noise), same recall.
- ✅ GPU gives ~5× speedup over laptop CPU for picking.
- ⚠️ Precision is very low (P 0.3 %, S 1.8 %). Expected — manual catalog labels only the largest events on this day; PhaseNet finds the broader microseismicity. Use `--event-window` or accept low precision as the cost of microseismicity recovery.

## Commands used (for reproduction)
```bash
source .venv/bin/activate
python scripts/01_data_inventory.py
python scripts/02_download_waveforms.py --start 2019-12-26 --end 2019-12-27

# PhaseNet
python scripts/03_run_phasenet.py --start 2019-12-26 --end 2019-12-27 \
    --p-thresh 0.1 --s-thresh 0.1 --workers 1
# EQTransformer
python scripts/03_run_phasenet.py --model EQTransformer --out-subdir picks_eqt \
    --start 2019-12-26 --end 2019-12-27 \
    --p-thresh 0.1 --s-thresh 0.1 --workers 1

python scripts/04_station_geometry.py
PYTHONPATH=src python -m bransfield_eq.manual_picks   # one-time

# Validate against mag07 trusted subset
python scripts/05_validate_picks.py --start 2019-12-26 --end 2019-12-27 \
    --picks-subdir picks --manual-source mag07
python scripts/05_validate_picks.py --start 2019-12-26 --end 2019-12-27 \
    --picks-subdir picks_eqt --manual-source mag07
```

## Known small issues to clean up
- `05_validate_picks.py` log line says "PhaseNet picks" even when reading `picks_eqt/` (cosmetic).
- Both validator runs write to the same `catalogs/validation_report.csv` / `validation_per_pick.csv` and overwrite each other. For year-run we'll want a `--out-suffix` flag or rename between runs.
