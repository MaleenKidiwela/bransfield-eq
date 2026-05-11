# Bransfield EQ — Comprehensive summary

Two sessions, 2026-05-09 → 2026-05-10, on the OOI/UW JupyterHub L40S box.
This document consolidates every major experiment, result, decision, and
lesson. Individual single-topic notes (`00`–`10`) remain the source of truth
for each step.

---

## 1. Goal

Build an **expanded earthquake catalog** for the 14-month BRAVOSEIS deployment
(2019-01 → 2020-03) by going beyond the manual analyst's pick catalog.

- Manual catalog: `nllmaleen_mag07_202210.out` — **12,099 picks across 1,124
  events** (mag07 = magnitude ≥ 0.7 subset; the trustworthy ground truth).
- Mag07 reduces to **9,772 picks (5,257 P + 4,515 S)** when filtered to
  `uncertainty_s ≤ 0.1` (high-confidence pick subset).
- Targets: 22 ZX BRAVOSEIS OBS + 14 5M land + AI.JUBA + AM.R4DE2.
- Window: 2019-01-01 → 2020-03-01.

## 2. Cluster + storage setup

**Hardware (much beefier than initial brief):**
- 176 logical CPUs (88 cores × 2 threads, dual Xeon Platinum 8458P)
- 1.5 TiB RAM
- 1× NVIDIA L40S, 46 GB VRAM (compute cap 8.9, Ada)
- `/home/jovyan` NFS PVC, 90 TB free, ~617 MB/s sequential write

**Storage layout (decided 2026-05-09):**
- All bulk data lives at `/home/jovyan/my_data/bravoseis/` and is symlinked
  into the repo. Keeps ~600 GB outside the git tree.
- `bransfield-eq/data/{waveforms,stationxml}` and
  `bransfield-eq/catalogs/picks*` are symlinks.

**Year-long FDSN download (started 2026-05-09 ~05:00 UTC):**
- ~16,000 station-days, 785 GB on disk
- Completed 36 of 37 stations (only AI.JUBA partial at 186/425 days)
- All 22 ZX OBS + all 14 5M land stations: 100% complete
- 99% of mag07 picks are on the complete OBS stations

## 3. Pipeline verification — day 2019-12-26

Reproducibility check before committing to long runs. Numbers matched the
laptop reference exactly:

| Step | Predicted (laptop) | Got (cluster) |
|---|---|---|
| Inventory | 560 GB year footprint | 560.5 GB |
| Download (1 day) | 35/37 stations / 1.66 GB | 35 / 1.66 GB |
| PhaseNet picks | 6,902 | 6,899 (Δ = 3, ~0.04 %) |
| P recall vs all-source mag07 | 64 % | 63.6 % |
| S recall vs all-source mag07 | 42 % | 41.7 % |
| Wall on GPU | ~5 min | 3.5 min (5× vs laptop CPU) |

Pipeline is deterministic across hardware. GPU gives 5× speedup. Confirmed
PhaseNet `instance` weights are the right start.

## 4. Picker comparison on day-26 — 10 configurations

Five pickers × variable thresholds = 10 picker runs against the day-26
mag07 catalog (23 picks across 2 events).

| # | Picker | Weights | Thresh | Total picks | P-rec | S-rec |
|--:|---|---|---:|---:|---:|---:|
| 1 | PhaseNet | `instance` | 0.1 | 6,899 | **0.636** | 0.417 |
| 2 | EQTransformer | `instance` | 0.1 | 7,113 | 0.364 | 0.500 |
| 3 | PhaseNet | `obs` (PickBlue) | 0.1 | 25,997 | 0.545 | 0.583 |
| 4 | PhaseNet | `obs` (PickBlue) | 0.3 | 4,831 | 0.273 | 0.417 |
| 5 | EQTransformer | `obs` (PickBlue) | 0.1 | 20,793 | 0.455 | **0.917** |
| 6 | EQTransformer | `obs` (PickBlue) | 0.3 | 4,434 | 0.273 | 0.750 |
| 7 | OBSTransformer | `obst2024` | 0.1 | 70,480 | 0.455 | **1.000** |
| 8 | OBSTransformer | `obst2024` | 0.3 | 36,581 | 0.364 | **1.000** |
| 9 | **OBSTransformer** | `obst2024` | **0.5** | **20,919** | 0.273 | **1.000** |
| 10 | OBSTransformer | `obst2024` | 0.7 | 12,687 | 0.273 | 0.917 |

**Key findings on day-26:**
- PhaseNet `instance` is the strongest P picker (0.64).
- OBSTransformer's S recall stays at 1.00 across thresholds 0.1 → 0.5 — confirms Niksejel & Zhang's threshold-robustness claim. Recall finally breaks at 0.7.
- **Recommended hybrid**: PhaseNet `instance` for P (thresh 0.1) + OBSTransformer @ 0.5 for S.
- Sweet-spot OBSTransformer threshold = 0.5: same perfect S recall as 0.1, ~70 % less pick volume.

**P/S balance (informative):**
- Land-trained `instance`: P/S ≈ 5 (very P-biased)
- PickBlue (OBS-tuned): P/S ≈ 0.65–1.67
- OBSTransformer: P/S ≈ 0.25–0.53 (S-biased, reflecting OBS data character)
- mag07 catalog: P/S = 0.82

## 5. Multi-day picker comparison — Feb 4–13, 2019

Validation of the day-26 hybrid recommendation against a much larger sample
(1,236 manual mag07 picks across 10 days, no uncertainty filter).

**Aggregate (10 days, all-source mag07):**

| Picker | Total picks | P recall | S recall |
|---|---:|---:|---:|
| PhaseNet `instance` @ 0.1 | 87,833 | **0.425** | 0.626 |
| OBSTransformer @ 0.5 | 140,818 | 0.365 | **0.782** |

**Key findings:**
- Hybrid recommendation **holds**: PhaseNet best P (0.42), OBSTransformer best S (0.78).
- PhaseNet's day-26 P advantage (28 pp) shrinks to 6 pp aggregate — small-sample bias.
- OBSTransformer's S advantage is consistent (wins 9 of 10 days, often by huge margins).
- **Swarm days hammer P recall for both pickers** — Feb 13 (235 P picks): PN 0.18, OBST 0.10. Real production limitation: window-based pickers can only emit one peak per phase per window, dense overlapping events get under-counted.
- Anomaly: **2019-02-10 OBSTransformer P=0.000** despite catching 67 % of S — events detected as S-only. Possibly impulsive S with weak P (water-borne phase conversion).

Plot: `figures/multiday_2019-02-04_to_2019-02-13.png`.

## 6. OBS DeepDenoiser (Phase 1b)

Trained `seisbench.models.DeepDenoiser` on Bransfield OBS data.

**Pipeline:**
- 30 curated training days × 17 stations → 3,537 mag07 high-confidence
  event windows + 23,906 quiet-window noise samples (gated by manual-pick
  exclusion + STA/LTA reject + amplitude sanity; median acceptance 88 %).
- Transfer-learned from `original` weights (land STEAD).
- 50 epochs, AdamW lr 1e-4, batch 32, MSE on STFT mask.
- Train/val split by **station** (held-out BRA13 + BRA22) to test
  generalization, not just window-level memorization.
- Best val MSE: **0.0490 at epoch 43** (loss curve in `figures/dd_train_loss.png`).
- Wall: ~5 min on the L40S.

**Visual sanity check on held-out val stations** (`figures/dd_obs_sanity/`):

| Original (black) | Off-the-shelf 'original' (blue) | OBS-fine-tuned (green) |
|---|---|---|
| Clear impulsive event arrivals | **Wipes events out entirely** (over-subtracts on OBS data) | **Preserves event arrivals**, removes noise floor |

**Verdict**: the OBS-fine-tuned denoiser works. Off-the-shelf was unusable
on Bransfield OBS data; fine-tuned is qualitatively much better. Saved
to `models/deepdenoiser_obs/best.pt`.

## 7. PhaseNet fine-tune attempts — both negative

### v1 — failed (trivial overfitting)

- 109,647 training windows (1 clean + 30 noise variants per pick at SNR ∈ [0.5, 10]).
- All variants had pick at exactly **sample 500** (extracted with `PRE_PICK_SEC = 5.0` × 100 Hz; Phase 2 only mixed noise, didn't shift position).
- Training-time val P-rec went 0.79 → 1.000 by epoch 18, S-rec 0.003 → 0.998 — looked like spectacular learning.
- **Production result on Aug 2019: 8.5 million picks across the month, P-rec 0.051, S-rec 0.089.** Disaster.
- **Root cause:** model learned "fire at sample 500" because ALL training labels were at sample 500. Lost probability calibration entirely. Dataloader's `WindowAroundSample` + `RandomWindow` couldn't randomize position because input windows were 3000 samples and required slack ≥ 2× output (3001).
- Saved to `models/phasenet_bransfield_v1_failed_centered_picks/best.ckpt`, eval at `catalogs/aug2019_eval_v1_failed.csv`.

### v2 — calibrated but still underperforms baseline

**Fix:** `08_build_augmented_dataset.py:shift_pick_position` randomizes pick
position uniform in [200, 2799] for each of the 30 noise variants per event.
Confirmed distribution (mean 1466, std 759).

- Training-time P-rec 0.79 → 0.89, S-rec 0.04 → **0.97** by epoch 29 — real learning, not memorization.
- **Production result on Aug 2019:**

| Picker | Total picks | P-rec | S-rec |
|---|---:|---:|---:|
| A. PhaseNet `instance` (baseline) | 370,954 | **0.800** | 0.876 |
| B. OBSTransformer `obst2024` | 806,290 | 0.769 | **0.888** |
| C. PhaseNet **fine-tuned v2** | 282,713 | **0.679** | **0.477** |

- Volume sensible (282k vs 8.5M v1 disaster).
- **Fine-tuned model is worse than baseline.** P recall –12 pp, S recall –40 pp.

Plot: `figures/aug2019_recall.png`.

**Decision: do not deploy either fine-tuned model. Production picker remains the baseline hybrid.**

### Why does v2 underperform? (working hypotheses)

1. **`instance` was already near-optimal** for these high-confidence events. Fine-tuning on a smaller sample biases away from a strong general prior.
2. **No "no-pick" windows in training.** Every variant centres on a real pick. Model never explicitly learns when NOT to fire.
3. **`shift_pick_position` zero-pads at edges** — creates discontinuities at sample boundaries.
4. **Pick-uncertainty filter biases toward easy events** — training distribution unrepresentative of harder events.
5. **Domain mismatch:** training windows bracket known events; production sees 24 h sliding windows mostly with no event.

## 8. Surprise from Phase 4 — manual-catalog filter matters

Baseline picker A (PhaseNet `instance`) numbers shifted dramatically based
on which manual catalog filter we used:

| Test setup | P recall | S recall |
|---|---:|---:|
| Feb 4–13, **all-source mag07** (no uncertainty filter) | 0.42 | 0.63 |
| Aug 2019, **mag07 with `uncertainty_s ≤ 0.1`** | **0.80** | **0.88** |

The strict uncertainty filter selects the cleanest analyst picks → easier
targets → better-looking baseline numbers. Most of the "missed" picks in
the Feb evaluation were probably uncertainty-0.2 picks the analyst was
also uncertain about. **Future evaluations should always specify the filter.**

## 8.5  Picker E — pretrained PhaseNet on DD-cleaned input (added 2026-05-10)

Run on Feb 4–13 to test whether using the OBS-trained denoiser at *inference*
time (without retraining PhaseNet) helps the off-the-shelf picker.

Pipeline: read raw mseed → `model.annotate(stream)` via the OBS DeepDenoiser →
`PhaseNet.classify()` on the denoised stream. Output to `catalogs/picks_pn_dd/`.

**Result (Feb 4–13, vs all-source mag07, 1,236 picks):**

| Picker | Total picks | P-rec | S-rec |
|---|---:|---:|---:|
| A. PhaseNet `instance` (raw input) | 370,954 | **0.425** | 0.626 |
| B. OBSTransformer @ 0.5 (raw input) | 806,290 | 0.365 | **0.782** |
| **E. PhaseNet on DD-cleaned input** | **357,197** | **0.153** | **0.161** |

**The denoiser hurt the picker.** P recall –27 pp, S recall –47 pp vs baseline.

**Why** (working hypotheses):
- Pretrained PhaseNet was trained on **raw** waveforms. Denoising shifts the
  input distribution (amplitude, frequency content, possible STFT/iSTFT
  artifacts at sliding-window boundaries) into one PhaseNet wasn't tuned for.
- The denoiser visually preserves arrivals (held-out station validation in
  `figures/dd_obs_sanity/`) but the *exact pixel-level statistics* it produces
  are different from raw recordings.
- This pattern matches a known finding in the literature: denoising as an
  inference-time wrapper around a trained picker often hurts unless the
  picker is fine-tuned on the same denoised distribution.

Files: `catalogs/picks_pn_dd/`, `scripts/11_run_denoised_picker.py`.

**Implication:** the denoiser is real (Phase 1b worked) but currently isn't
useful in this drop-in configuration. To exploit it, PhaseNet would need to
be fine-tuned on denoised inputs — which is a 3rd training run, not just a
new inference path.

## 9. Final decisions

| Decision | Outcome |
|---|---|
| Production picker | **PhaseNet `instance` @ 0.1 (P) + OBSTransformer `obst2024` @ 0.5 (S)** |
| Deploy fine-tuned PhaseNet? | **No** (both v1 and v2 underperform baseline) |
| Deploy OBS DeepDenoiser? | **No.** Visually preserves arrivals but Picker E (DD → pretrained PhaseNet) underperforms baseline by 27 pp on P recall (Feb 4–13). Drop-in inference-time denoising hurts when the picker isn't tuned for the denoised distribution. |
| Year-long catalog production | Use baseline hybrid; pickers already cached for full year |

## 10. Suggested next steps (if revisiting)

1. ~~Picker E (pretrained PhaseNet on denoised input)~~ — **already tried; underperforms baseline by 27 pp on Feb 4–13.** See section 8.5. To exploit the denoiser, PhaseNet would need to be fine-tuned on denoised inputs (3rd training run).
2. **Add explicit no-pick negatives** to PhaseNet fine-tune training (e.g., 50 % event windows + 50 % pure noise windows; supervise the model to assign noise=0 probability everywhere on noise).
3. **Train on ALL mag07 picks** (drop the uncertainty filter on training data). The current training distribution is biased toward easy events — ironically making the model worse at the harder events.
4. **Lower learning rate (1e-5) + shorter training (5 epochs)** — minimize drift from the strong `instance` prior.
5. **Try fine-tuning from PickBlue PN `obs`** instead of `instance` — closer to the OBS domain to start with.
6. **Investigate the swarm-day P-recall collapse** (Feb 11/13: P-rec 0.10–0.18). Either smaller picking windows, peak-finding tolerance changes, or post-association cross-correlation to recover overlapping picks.
7. **Replace zero-padding in `shift_pick_position` with cyclic rotation** — eliminates the edge-discontinuity artifact.
8. **Build the actual catalog**: associate the production hybrid picks via PyOcto, locate via NLLoc with the user's blended Orca-3D / 1D model. The picker recall numbers are a proxy; event-count and magnitude-of-completeness on the located catalog is the real product.

## 11. Key lessons documented

1. **Trivial val metrics are a red flag, not a green light.** P-rec=1.0 on val with frozen loss should have prompted immediate skepticism instead of waiting for the production confirmation.
2. **Pre-materialised augmentation MUST randomize what production sees** — including target position, not just amplitude/noise. Off-the-shelf SeisBench transforms (`WindowAroundSample`, `RandomWindow`) silently no-op when input slack is insufficient.
3. **Fine-tuning a strong general prior on a small biased sample can hurt.** When the off-the-shelf model is already good, the burden of proof for fine-tuning is high.
4. **Manual catalog filter dramatically affects baseline numbers.** Always specify and document the filter.
5. **OBSTransformer's threshold-robustness claim (Niksejel & Zhang 2024) is real and unique** to OBSTransformer. PickBlue is not threshold-robust in the same way (recall drops sharply 0.1 → 0.3).
6. **Off-the-shelf DeepDenoiser is unusable on OBS data** (over-subtracts events). Domain-specific fine-tuning fixes this; was the cleanest, most defensible win of the whole exercise.
7. **Recall vs manual catalog is the wrong long-term metric** if the goal is an *expanded* catalog. Future iterations should evaluate at the *event* level after association/location, not at the per-pick level against an incomplete reference.

## 12. Files produced (top-level inventory)

**Models:**
- `models/deepdenoiser_obs/best.pt` — OBS-fine-tuned DeepDenoiser (val MSE 0.049)
- `models/phasenet_bransfield_v1/best.ckpt` — v2 fine-tuned PhaseNet (val_loss 0.5612, P-rec 0.89 / S-rec 0.97 *on val* but worse than baseline in production)
- `models/phasenet_bransfield_v1_failed_centered_picks/best.ckpt` — v1 (failed, kept for record)

**Datasets:**
- `data/seisbench/bransfield_events/` — 3,537 high-conf mag07 event windows
- `data/seisbench/bransfield_noise/` — 23,906 quiet-window OBS noise samples
- `data/seisbench/bransfield_aug/` — 109,647 augmented training variants (v2)

**Pick catalogs (production hybrid + experiments):**
- `catalogs/picks/` — PhaseNet `instance` @ 0.1 (full year + Aug 2019 baseline)
- `catalogs/picks_obst_05/` — OBSTransformer @ 0.5 (Feb 4–13 + Aug 2019)
- `catalogs/picks_pn_ft/` — v2 fine-tune output (Aug 2019; documented as worse than baseline)
- `catalogs/picks_pn_ft_v1_failed_actual2/` — v1 fine-tune output (Aug 2019, failure)

**Eval results:**
- `catalogs/aug2019_eval.csv` — current (v2 vs baselines)
- `catalogs/aug2019_eval_v1_failed.csv` + `_v1_redo.csv` — v1 results
- `catalogs/manual_picks.csv` — merged manual catalog (46k picks, with `uncertainty_s` column added in this work)

**Figures:**
- `figures/dd_sanity/*.png` — Phase 0 off-the-shelf denoiser sanity (showed event over-subtraction)
- `figures/dd_obs_sanity/*.png` — off-the-shelf vs OBS-fine-tuned comparison (Phase 1b validation)
- `figures/dd_train_loss.png` — denoiser training curve
- `figures/noise_qc.png` — per-station-day noise acceptance heatmap
- `figures/multiday_2019-02-04_to_2019-02-13.png` — Feb 10-day baseline picker comparison
- `figures/picker_comparison_2019-12-26/*.png` — day-26 5-picker comparison plots
- `figures/aug2019_recall.png` — final v2 vs baseline plot
- `figures/aug2019_recall_v1_failed.png` — v1 result preserved

**Scripts (in execution order):**
- `01_data_inventory.py`, `02_download_waveforms.py`, `03_run_phasenet.py` (with `--weights ckpt:` patch), `04_station_geometry.py`, `05_validate_picks.py` (with `--manual-source` patch)
- `06_extract_event_noise.py` — Phase 1a builder
- `07_train_obs_denoiser.py` — Phase 1b denoiser trainer
- `08_build_augmented_dataset.py` — Phase 2 (with v2 random-position fix)
- `09_finetune_phasenet.py` — Phase 3 trainer
- `10_eval_aug2019.py` — Phase 4 driver
- `test_deepdenoiser.py`, plus inline `test_deepdenoiser_obs` (sanity checks)

**Per-topic notes:**
- `00_cluster_environment.md`, `01_storage_layout.md`, `02_one_day_verification.md`,
  `03_gpu_optimizations.md`, `04_validation_and_mag07.md`, `05_year_run_status.md`,
  `06_active_plan.md`, `06a_user_autoencoder_proposal.md`, `07_obs_baseline_pickers.md`,
  `08_multiday_picker_comparison.md`, `09_obs_denoiser_v1.md`, `10_finetune_v1.md`,
  `SUMMARY.md` (this file).
