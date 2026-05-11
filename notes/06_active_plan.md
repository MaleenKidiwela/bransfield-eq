# Active plan — OBS DeepDenoiser + PhaseNet fine-tuning

Approved: 2026-05-09 evening. Replaces all earlier "fine-tuning plan" notes.

## Goal

Build an **expanded earthquake catalog** for the 14-month BRAVOSEIS deployment
beyond what the manual analyst caught. Baseline hybrid (PhaseNet `instance` for
P + OBSTransformer `obst2024` @ 0.5 for S) recovered only **P recall ≈ 0.42 / S
recall ≈ 0.78** on the Feb 4–13 multi-day comparison — too many easy events
slipping through.

## Approach (two-stage)

1. **Train an OBS-tuned DeepDenoiser** (SeisBench's STFT U-Net,
   `sbm.DeepDenoiser`, fine-tuned from `original` weights) — learns Bransfield
   ocean noise so it can distinguish signal from noise per-window.
2. **Fine-tune PhaseNet** (`sbm.PhaseNet.from_pretrained('instance')`) on a
   curated 30-day training set with the OBS-DeepDenoiser supplying:
   - **(A)** noise mixed into events at variable SNR (real OBS noise, ~30
     pre-materialized variants per pick)
   - **(B)** denoised event windows as a 31st variant per pick
3. **Honest evaluation** on a fully-held-out **August 2019** (1,144 picks,
   106 events, disjoint from training).

## Phases & files (in execution order)

| # | Phase | Wall | Script(s) / outputs |
|---:|---|---|---|
| 0 | Sanity-check off-the-shelf DeepDenoiser | ~15 min | `scripts/test_deepdenoiser.py` → `figures/dd_sanity/` |
| 1a | Build event + noise SeisBench HDF5 pools | ~30 min | `scripts/06_extract_event_noise.py` + `configs/finetune_train_days.csv` → `data/seisbench/bransfield_{events,noise}.h5` |
| 1b | Fine-tune OBS DeepDenoiser | ~1 h GPU | `scripts/07_train_obs_denoiser.py` → `models/deepdenoiser_obs/best.pt` |
| 2 | Build augmented PhaseNet training set (~100k windows) | ~30 min | `scripts/08_build_augmented_dataset.py` → `data/seisbench/bransfield_aug/` |
| 3 | Fine-tune PhaseNet (Lightning, 30 epochs) | ~1 h GPU | `scripts/09_finetune_phasenet.py` + `configs/phasenet_finetune.yaml` → `models/phasenet_bransfield_v1/best.ckpt` |
| 4 | Evaluate on August 2019 vs baselines | ~30 min | patch `scripts/03_run_phasenet.py` for `--weights ckpt:<path>`; `scripts/10_eval_aug2019.py` → `catalogs/aug2019_eval.csv`, `figures/aug2019_recall.png` |
| 5 | Document and decide | ~15 min | `notes/09_obs_denoiser_v1.md`, `notes/10_finetune_v1.md` |

**Total estimated wall: ~3.5 h.**

## Training day list (curated, 30 days, all non-August)

Top mag07-confident days (`uncertainty_s ≤ 0.1`) per non-August month, totaling
30 days, ~3,300 picks. Defined in `configs/finetune_train_days.csv`.

```
2019-01: 2019-01-17 (651), 2019-01-19 (110), 2019-01-18 (62)
2019-02: 2019-02-13 (233), 2019-02-05 (176), 2019-02-09 (136)
2019-03: 2019-03-12 (82),  2019-03-15 (60)
2019-04: 2019-04-04 (116), 2019-04-01 (104)
2019-05: 2019-05-13 (43),  2019-05-10 (33)
2019-06: 2019-06-09 (95),  2019-06-04 (54)
2019-07: 2019-07-08 (64),  2019-07-17 (53)
2019-09: 2019-09-18 (86),  2019-09-19 (77)
2019-10: 2019-10-04 (224), 2019-10-13 (197), 2019-10-31 (177)
2019-11: 2019-11-13 (54),  2019-11-29 (43)
2019-12: 2019-12-08 (86),  2019-12-28 (65)
2020-01: 2020-01-19 (111), 2020-01-25 (110), 2020-01-20 (98)
2020-02: 2020-02-07 (76),  2020-02-01 (61)
```

## Held-out test set: August 2019

| Stat | Value |
|---|---:|
| Days with mag07 picks | 26 / 31 |
| Picks (`uncertainty_s ≤ 0.1`) | 1,144 (544 P + 600 S) |
| Events | 106 |

3rd most active month after Jan/Feb 2019, similar conditions to the rest of
the year, no swarm peculiarities. Disjoint from training days.

## Pickers compared in Phase 4

| ID | Picker | Output dir |
|---|---|---|
| A | PhaseNet `instance` @ 0.1 (baseline) | `catalogs/picks/` |
| B | OBSTransformer `obst2024` @ 0.5 (baseline) | `catalogs/picks_obst_05/` |
| C | PhaseNet **fine-tuned** @ 0.1 | `catalogs/picks_pn_ft/` |
| D | C on DeepDenoiser-cleaned input (optional bonus) | `catalogs/picks_pn_ft_dd/` |

**Decision rule:** C aggregate P recall > A on Aug 2019 → fine-tune helped;
D > C → denoising at inference time helps too; pick volume not collapsed
(within ±20 % of baseline) → not just a precision tradeoff.

## What this plan deliberately does NOT do

- **No association/location.** Picker recall on held-out August is the proxy
  for "expanded catalog"; PyOcto/NLLoc adds days of work — separate stage.
- **No hyperparameter sweep** for either model (single LR/batch/loss).
- **Only PhaseNet is fine-tuned** (OBSTransformer is already strong on S).
- **Starts from `instance`**, not PickBlue (best P recall on day-26 baseline).
- **No third-party `phasenet-retrain` framework** — uses SeisBench's own
  training stack directly to avoid the broken `_load_custom_data()` stub.
- **Approach D (AE encoder as PhaseNet pretraining)** deferred as a v2
  research project.
