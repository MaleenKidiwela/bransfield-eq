# Original user proposal — OBS noise autoencoder for better phase picking

This is the user's original proposal that led to the active plan
([`06_active_plan.md`](06_active_plan.md)). Kept verbatim for record;
the active plan is the executable version.

---

# Project Outline: Ocean-Bottom Seismic Noise Autoencoder for Better Phase Picking

## Goal

Build a workflow that learns the background noise structure of ocean-bottom seismic data, removes or reduces that noise from event windows, and then runs a phase picker such as PhaseNet/FaceNet on the denoised data.

The motivation is that many phase pickers are trained mostly on land-based seismic data. Ocean-bottom seismic data can contain strong ocean-specific noise, ship noise, instrument noise, microseisms, and other transient signals. Instead of immediately retraining the full picker, this project first tries to improve the input data by learning and subtracting the local/regional noise field.

---

## Core Idea

1. Use existing manual picks and a simple detector such as STA/LTA to identify likely event windows.
2. Remove those windows from the training set.
3. Treat the remaining quiet windows as background-noise examples.
4. Train an autoencoder/decoder model on those quiet windows.
5. Apply the trained model to full seismic windows.
6. The autoencoder reconstructs the expected noise.
7. Subtract the reconstructed noise from the original waveform.
8. Run the picker on the denoised waveform.
9. Compare picking performance before and after denoising.

---

## Data Assumptions

- Dataset contains about 15 seismic stations.
- Data span is approximately one year.
- Data are ocean-bottom seismic records.
- There is an existing manual pick catalog, but it may be incomplete or inconsistent.
- The existing picker does not perform well because the noise environment differs from the land-based data it was trained on.

---

## Workflow Overview

```text
Raw seismic data
      |
      v
Run simple event screening
manual picks + STA/LTA + amplitude thresholds
      |
      v
Mask likely event windows
      |
      v
Select quiet/noise-only windows
      |
      v
Train autoencoder on noise windows
      |
      v
Apply autoencoder to full data
      |
      v
Reconstruct expected noise
      |
      v
Subtract reconstructed noise from original waveform
      |
      v
Run PhaseNet/FaceNet picker
      |
      v
Evaluate improvement
```
