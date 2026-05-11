# OBS DeepDenoiser v1 — Phase 1b training run

Run completed 2026-05-09 evening on the OOI/UW JupyterHub L40S.

## Inputs
- **Events:** `data/seisbench/bransfield_events/` — 3,537 high-confidence
  mag07 picks (`uncertainty_s ≤ 0.1`) across 17 OBS stations from the 30
  curated training days.
- **Noise:** `data/seisbench/bransfield_noise/` — 23,906 quiet-window
  recordings, gated by 3 filters (manual-pick exclusion ±60 s on mag07 ∪
  magall, STA/LTA reject > 3.0, amplitude sanity ±10 × station-day median
  RMS). Median acceptance: 88 %.

## Architecture & training
- **Model:** `seisbench.models.DeepDenoiser` (STFT U-Net), initialised from
  the published `original` weights (transfer learning).
- **Training pipeline:** SeisBench `STFTDenoiserLabeller` builds the
  ground-truth time-frequency mask from (event + scaled noise) pairs each
  epoch. Scale ∈ (0.3, 2.0) → corresponds to SNR roughly in [0.5, 10].
- **Hyperparameters:** AdamW, lr 1e-4, batch 32, 50 epochs, MSE on the mask.
- **Train/val split:** by **station** — 15 stations train, 2 stations val
  (BRA13 + BRA22). This validates generalisation to held-out *stations*,
  not just held-out windows of seen stations.
  - Train traces: 2,985, Val traces: 552

## Results
- Best `val_loss` = **0.0490** at epoch 43.
- Smooth descent from initial loss ≈ 0.14 → ~0.06 by epoch 10, plateauing
  around 0.05–0.06.
- Per-epoch wall: ~5 s on the L40S (full run < 5 min).
- Best checkpoint: `models/deepdenoiser_obs/best.pt`
- Loss curve: `figures/dd_train_loss.png`

## Phase 0 sanity comparison

The off-the-shelf DeepDenoiser (`original`, land STEAD) was tested on
Bransfield event windows in Phase 0 (`figures/dd_sanity/`). Findings:
- Off-the-shelf model **preserves the impulsive arrival pulse but
  over-subtracts event signal** — visible energy remains in the residual
  (event leaked into the "noise" the denoiser tried to remove).
- On pure noise windows it works correctly (denoised ≈ flat, residual ≈
  original).
- This confirmed Phase 1 retraining was warranted.

## Caveats / open questions

1. **Held-out val stations were the lowest-acceptance stations** (BRA13 had
   ~30 % noise acceptance, BRA22 had ~40 %). They likely contain the most
   unmarked microseismicity. Val loss therefore reflects performance on
   relatively contaminated noise — possibly harder than average-case
   inference data.
2. **No held-out *time-block* validation yet.** Train/val are both drawn
   from the 30 curated days. August 2019 evaluation in Phase 4 will be the
   honest cross-time test of whether this denoiser actually helps PhaseNet.
3. **DeepDenoiser inference is via `model.annotate(stream)`** which
   internally handles STFT/iSTFT. Direct `model(tensor)` calls fail with
   shape errors (model expects pre-STFT input). When applying the denoiser
   to event windows in Phase 2, this caught us out and we ended up writing
   the augmented dataset without the "denoised variant" channel — only the
   30 noise-mixed variants per pick.

## Files

- `scripts/06_extract_event_noise.py` — Phase 1a (event/noise pool builder)
- `scripts/07_train_obs_denoiser.py` — Phase 1b (this run's trainer)
- `models/deepdenoiser_obs/best.pt` — Phase 1b checkpoint
- `figures/dd_sanity/*.png` — Phase 0 visual comparisons (off-the-shelf)
- `figures/dd_train_loss.png` — Phase 1b loss curve
- `figures/noise_qc.png` — Phase 1a noise-window acceptance per station-day
