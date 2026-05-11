# PhaseNet fine-tune v1 + v2 — both honest negative results

Two attempts at fine-tuning PhaseNet on the Bransfield mag07 dataset. **Both
underperform the off-the-shelf `instance` baseline on the held-out August
2019 test.** Documented here as honest negatives + what we learned.

## Summary table — August 2019, mag07 (`uncertainty_s ≤ 0.1`), 1,232 picks (546 P + 686 S), 106 events

| Picker | Total Aug picks | P recall | S recall | Notes |
|---|---:|---:|---:|---|
| A. PhaseNet `instance` (baseline) | 370,954 | **0.800** | 0.876 | off-the-shelf STEAD |
| B. OBSTransformer `obst2024` (baseline) | 806,290 | 0.769 | **0.888** | off-the-shelf OBS-tuned |
| **C. PhaseNet fine-tuned v1** (centered picks) | 8,484,352 | 0.051 | 0.089 | broken — see "v1 failure" below |
| **C. PhaseNet fine-tuned v2** (random pick positions) | 282,713 | **0.679** | 0.477 | calibrated, but *worse* than baseline |

Decision: **do not deploy either fine-tuned model.** Production picker remains
the baseline hybrid (PhaseNet `instance` for P + OBSTransformer @ 0.5 for S).

Plot: [`figures/aug2019_recall.png`](../figures/aug2019_recall.png) (v2 result;
v1 result preserved at `figures/aug2019_recall_v1_failed.png`).

## Pipeline

Same Phase 1a–4 pipeline both runs:
- Phase 1a: 3,537 high-conf mag07 events + 23,906 noise windows (17 OBS stations, 30 curated days)
- Phase 1b: OBS DeepDenoiser fine-tuned from `original` (val MSE 0.049) — see [`09_obs_denoiser_v1.md`](09_obs_denoiser_v1.md)
- Phase 2: 109,647 augmented variants (1 clean + 30 noise-mixed at SNR ∈ [0.5, 10] per event); v2 differs in pick-position randomisation
- Phase 3: PhaseNet from `instance`, AdamW lr 1e-4, batch 64, 30 epochs, KL loss
- Phase 4: held-out Aug 2019 evaluation against high-confidence mag07

## v1 failure — trivial overfitting

**Symptom during training:** init val P-rec=0.79, S-rec=0.003 → after epoch 1
P-rec=0.999, S-rec=0.01 → by epoch 18 (best) P-rec=1.000, S-rec=0.998. Train
and val loss flat-lined at 0.5584 from epoch 5 onward.

**Production result:** 8.5 million picks across August 2019 (~one every 10 s
across all stations), P-recall 0.051, S-recall 0.089.

**Root cause:** every Phase 2 variant had its pick at exactly sample 500 (Phase 1a
extracted with `PRE_PICK_SEC = 5.0` × 100 Hz, then Phase 2 just mixed noise
without changing position). The dataloader's `WindowAroundSample` +
`RandomWindow` augmentation expects input ≥ 2× output length to randomize
the pick position; our 3000-sample input couldn't supply slack for a
3001-sample output, so those transforms effectively no-oped. Model learned
"there is always a P at sample 500" → on a continuous mseed stream that
translates to "fire P every 30 s," destroying calibration.

**Lesson:** pre-materialised augmentation MUST randomize what production sees
— including target position, not just amplitude/noise.

## v2 — fixed pick position, still doesn't help

**Fix applied** (`scripts/08_build_augmented_dataset.py:shift_pick_position`):
each of the 30 noise variants per event now has the pick at a uniform
random sample in [200, 2799] within the 3000-sample window. Confirmed
distribution: mean 1466, std 759, full coverage of [200, 2799].

**Training behaviour was healthy:** init val P=0.79, S=0.04 → epoch 1 P=0.76,
S=0.17 → epoch 29 (best) P=0.89, S=0.97. Loss moved meaningfully epoch over
epoch (0.5663 → 0.5612 val), unlike v1's frozen 0.5584.

**Production result on Aug 2019:** 282,713 picks (similar volume to baseline
A), but **P recall 0.679 vs 0.800 baseline (–12 pp), S recall 0.477 vs 0.876
baseline (–40 pp)**. The fine-tuned model is *worse* than the off-the-shelf
weights on the held-out month.

## Why does v2 underperform?

Working hypotheses (in rough order of likelihood):

1. **`instance` was already ~optimal for these events.** The high-confidence
   mag07 picks are exactly the kind of events `instance` was trained for
   (M ≥ 0.7 events with clear arrivals). Fine-tuning on a much smaller
   sample biases the model away from a strong general prior.
2. **No "no-pick" windows in training.** Every variant centres on a real
   pick. The model never sees pure noise as an explicit negative class
   during fine-tuning, so the learned probability calibration shifts. The
   noise dataset (23,906 windows) was used only by the DeepDenoiser, not by
   PhaseNet.
3. **`shift_pick_position` zero-pads at edges.** Creates hard discontinuities
   at sample boundaries — possibly real artifacts in training signal.
4. **Pick-uncertainty filter biases toward easy events.** mag07 +
   `uncertainty_s ≤ 0.1` selects the cleanest analyst picks → training set
   is unrepresentative of the harder events PhaseNet would benefit from.
5. **Domain mismatch between training and production windows.** Even after
   v2 randomization, training windows are 30 s bracketing a known event;
   production is sliding-window inference over 24 h station-days where
   most windows contain no event at all.

## What we kept from this exercise

- **OBS DeepDenoiser (Phase 1b)** is a real, working model. Visual sanity
  check on held-out stations BRA13/BRA22 confirmed it preserves event
  arrivals while removing background noise (off-the-shelf wipes events
  out). Saved to `models/deepdenoiser_obs/best.pt`.
- **Confirmed the baseline hybrid is hard to beat.** PhaseNet `instance` +
  OBSTransformer @ 0.5 is the production pick.
- **Honest evaluation framework.** Aug 2019 held-out month, mag07
  high-confidence, three pickers compared.
- **Discovered the manual-catalog filter matters.** Strict `uncertainty_s
  ≤ 0.1` filter raises baseline P recall from 0.42 (Feb 4–13 with all
  picks) → 0.80 (Aug 2019 with strict filter). Most of the "missed" picks
  in earlier evaluations were probably uncertainty-0.2 picks the analyst
  also wasn't sure about.

## Suggested next steps if revisiting

1. **Add explicit no-pick windows** to PhaseNet training (e.g., 50/50 mix of
   event windows and pure noise windows; train model to assign noise = 0
   probability everywhere on noise windows).
2. **Train on ALL mag07 picks** (not just `uncertainty_s ≤ 0.1`) so the
   training distribution includes the noisier events that the model needs
   to handle in production.
3. **Use the OBS DeepDenoiser at inference time** (Picker E that we
   sketched but didn't run). Run pretrained PhaseNet on denoised inputs.
   Cleaner separation than fine-tuning + might preserve calibration.
4. **Try a much smaller learning rate** (e.g., 1e-5) with shorter training
   (5 epochs). Less drift from the strong `instance` prior.
5. **Try fine-tuning from PickBlue PN `obs`** instead of `instance` — closer
   to the OBS domain to start with.

## Files referenced

- `scripts/06_extract_event_noise.py`, `scripts/07_train_obs_denoiser.py`
- `scripts/08_build_augmented_dataset.py` (with `shift_pick_position` for v2)
- `scripts/09_finetune_phasenet.py`
- `scripts/10_eval_aug2019.py`
- `models/phasenet_bransfield_v1/best.ckpt` — v2 model (current)
- `models/phasenet_bransfield_v1_failed_centered_picks/best.ckpt` — v1 (failed)
- `catalogs/aug2019_eval.csv` — v2 numbers (current; deploy decision = NO)
- `catalogs/aug2019_eval_v1_failed.csv` — v1 numbers (preserved)
- `figures/aug2019_recall.png` — v2 plot
- `figures/aug2019_recall_v1_failed.png` — v1 plot
