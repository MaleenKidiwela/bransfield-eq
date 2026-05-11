# DD pickers — full attempt log (Pickers E / F / G + post-filter)

Three approaches to using the OBS DeepDenoiser to improve picking, plus
an in-flight 4th. **All three model-based approaches underperform baseline.**
The DD-anomaly-score post-filter (4th approach, in flight) remains untested.

## Summary table — Feb 4-13, all-source mag07 (1,236 picks)

| Picker | Method | Total picks | P-rec | S-rec |
|---|---|---:|---:|---:|
| **A**. PhaseNet `instance` | raw input, off-the-shelf | 370,954 | **0.425** | 0.626 |
| **B**. OBSTransformer `obst2024` @ 0.5 | raw input, off-the-shelf | 806,290 | 0.365 | **0.782** |
| **E**. PhaseNet `instance` on DD-cleaned | inference-only, no retrain | 357,197 | 0.153 | 0.161 |
| **F (= v2)**. PhaseNet retrained on raw + position-shifted | model: phasenet_bransfield_v2/best.ckpt | 58,452 | 0.300 | 0.300 |
| **G (= v3)**. PhaseNet retrained on DD-cleaned + shifted | model: phasenet_bransfield_v3/best.ckpt | 343,309 | **0.000** | 0.097 |

Same Aug 2019 numbers (where available — for v3, see future eval):
| Picker | Aug 2019 P-rec | Aug 2019 S-rec |
|---|---:|---:|
| A. PhaseNet `instance` | 0.800 | 0.876 |
| B. OBSTransformer @ 0.5 | 0.769 | 0.888 |
| F (v2) on raw input | 0.679 | 0.477 |

**Decision: do not deploy any of E / F / G. Production picker remains the baseline hybrid (A for P + B for S).**

## Picker E details (PhaseNet on DD-cleaned, no retrain)

- **Pipeline**: raw mseed → `DeepDenoiser.annotate()` (OBS-fine-tuned weights from Phase 1b) → `PhaseNet('instance').classify()` at thresh 0.1
- **Script**: `scripts/11_run_denoised_picker.py --picker-weights instance`
- **Output**: `catalogs/picks_pn_dd/`
- **Result**: P recall –27pp, S recall –47pp vs baseline A
- **Why it fails**: pretrained PhaseNet expects raw seismic statistics. Denoising shifts amplitude / frequency content / introduces STFT-iSTFT stitching artifacts at sliding-window boundaries. Off-the-shelf model isn't tuned for any of that.

## Picker F = v2 details (retrained on raw + shifted, raw inference)

- **Training data**: 109,647 windows = 3,537 mag07 high-conf events × (1 clean + 30 noise variants per pick). Pick position **randomized** per variant in [200, 2799] within 3000-sample window (the v1 fix — was at sample 500 always).
- **Training**: AdamW lr 1e-4, batch 64, 30 epochs, KL loss. Init from PhaseNet `instance`.
- **Best val**: epoch 18, val_loss 0.5612, val P-rec 0.89 / S-rec 0.97. (Train-time recall on centered-easy distribution; not the production test.)
- **Inference**: `03_run_phasenet.py --weights ckpt:models/phasenet_bransfield_v2/best.ckpt` on raw mseed
- **Output**: `catalogs/picks_pn_v2/` (Feb 4-13), `catalogs/picks_pn_ft/` (Aug 2019)
- **Why it fails**: same pattern as v1's softer failure mode — small-data fine-tuning of an already-strong general prior makes it worse. Possibly: no "no-pick" negatives in training, training distribution biased to easy events.

## Picker G = v3 details (retrained on DD-cleaned + shifted, DD inference)

- **Training data**: 106,110 windows = 3,537 mag07 events × (1 clean denoised + 29 position-shifted denoised) per event. **Source waveforms first denoised** by Phase 1b OBS DeepDenoiser, then shifted to randomize pick position. No raw-noise mixing (the source was already denoised).
- **Training**: same hyperparameters as v2 (init from `instance`).
- **Best val (epoch 14)**: val_loss 0.5641, P-rec 0.984, S-rec 1.000.
  - Training trajectory: init P=0.28 S=0.14 → ep1 P=0.27 S=0.05 → ep2 P=0.92 S=0.04 → ep8 S jumps 0.22→0.56 → ep11 S=0.95 → ep14 S=1.000
  - Both P and S recall saturate by epoch 14 on val.
- **Inference**: `11_run_denoised_picker.py --picker-weights ckpt:models/phasenet_bransfield_v3/best.ckpt` (DD pipeline + retrained PhaseNet) — matches train and inference distributions.
- **Output**: `catalogs/picks_pn_v3_dd/`
- **Why it fails — degenerate output**: ALL 343,309 picks are labeled S, ZERO P. The model converged to "everything is S" in production despite high val P-recall. Suggests the model relies on something present in the training distribution but not at inference (likely an artifact of how DD output is normalised vs how PhaseNet's input layer expects it).

## Pattern across all 3

Same root cause family: **the off-the-shelf `instance` model is strong on
the high-confidence mag07 picks**. Each adaptation perturbs the input or
weights and the perturbation hurts more than the adaptation helps. Three
different mechanisms confirm this:
- E: input perturbed, weights unchanged → distribution mismatch
- F: weights perturbed, input raw → small-data overfit to easy training task
- G: input AND weights perturbed → catastrophic degenerate solution (P=0)

## DD-anomaly post-filter results (4th approach — DONE)

Hypothesis: the OBS DD output is a **signal extractor**. RMS of denoised
output at a pick time → high if model "saw signal" there; low if it treated
the window as pure noise. So `anomaly_ratio = denoised_RMS_at_pick /
median(denoised_RMS_at_random_baseline_windows)` should separate real picks
from noise picks.

(My first implementation used residual energy = raw − denoised, which is
the OPPOSITE: residual is high where DD removed lots of noise, doesn't
indicate signal presence. Killed that, fixed to use denoised signal energy.)

**Tested on Feb 4-13** by post-filtering both the PhaseNet `instance` @ 0.1
baseline picks and OBSTransformer `obst2024` @ 0.1 picks:

### Threshold sweep on PhaseNet `instance` @ 0.1 + DD filter

| anomaly ratio thresh | total | P-rec | P-prec | S-rec | S-prec |
|---:|---:|---:|---:|---:|---:|
| 0.0 (no filter) | 87,833 | **0.425** | 0.0040 | **0.626** | 0.0330 |
| 1.0 | 64,577 | 0.240 | 0.0030 | 0.376 | 0.0270 |
| 1.5 | 45,816 | 0.218 | 0.0040 | 0.360 | 0.0310 |
| 2.0 | 33,435 | 0.193 | 0.0050 | 0.332 | 0.0330 |
| 3.0 | 20,177 | 0.174 | 0.0080 | 0.295 | 0.0380 |
| 5.0 | 10,304 | 0.123 | 0.0130 | 0.237 | 0.0440 |
| 10.0 | 4,467 | 0.081 | 0.0220 | 0.134 | 0.0460 |
| 20.0 | 2,024 | 0.049 | 0.0310 | 0.053 | 0.0360 |

→ Recall drops fast, precision barely improves. No useful filtering.

### Threshold sweep on OBSTransformer @ 0.1 + DD filter

| anomaly ratio thresh | total | P-rec | P-prec | S-rec | S-prec |
|---:|---:|---:|---:|---:|---:|
| 0.0 (no filter) | 595,645 | **0.453** | 0.0020 | **0.903** | 0.0020 |
| 1.0 | 394,827 | 0.279 | 0.0020 | 0.600 | 0.0020 |
| 1.5 | 248,085 | 0.250 | 0.0030 | 0.563 | 0.0030 |
| 2.0 | 174,814 | 0.229 | 0.0040 | 0.526 | 0.0040 |
| 3.0 | 105,005 | 0.190 | 0.0070 | 0.452 | 0.0070 |
| 5.0 | 55,735 | 0.138 | 0.0120 | 0.344 | 0.0120 |
| 10.0 | 28,985 | 0.083 | 0.0190 | 0.165 | 0.0170 |
| 20.0 | 20,237 | 0.045 | 0.0210 | 0.071 | 0.0170 |

→ Same pattern. Precision improves slightly, recall drops faster.

### TP-vs-FP anomaly score distribution (the real story)

| Picker | Phase | TP median | FP median | TP/FP ratio | % FP below TP median |
|---|---|---:|---:|---:|---:|
| PhaseNet `instance` @ 0.1 | P | 1.60 | 1.51 | 1.05 | 53% |
| PhaseNet `instance` @ 0.1 | S | 2.40 | 2.28 | 1.05 | 52% |
| OBSTransformer @ 0.1 | P | 2.09 | 1.21 | **1.74** | **76%** |
| OBSTransformer @ 0.1 | S | 2.97 | 1.34 | **2.22** | **81%** |

**For PhaseNet `instance`**: TP and FP distributions essentially overlap
(ratio 1.05). PhaseNet's own softmax already filters real signal; the DD
score adds no new information. ~50% of "FPs" sit above the TP median.

**For OBSTransformer @ 0.1**: there IS measurable separation (ratio 1.74-2.22).
~76-81% of "FPs" sit below the TP median. So the DD score does flag some
genuine noise picks vs real arrivals — useful when the picker is run
permissively.

**But: even for OBSTransformer, absolute precision improvement is small.**
Most "FPs" are real microseismic events the analyst didn't mark. The DD
score correctly identifies them as signal-bearing — they aren't noise to
be filtered out. The "true noise pick" pool is small relative to the
"unmarked real event" pool, so filtering by DD score doesn't dramatically
shift the precision/recall trade-off.

### Conclusion

**The DD post-filter is not a useful production tool for either picker.** The
underlying assumption — that "FP picks against the manual catalog" are mostly
noise — is wrong for this dataset. They're mostly real events the analyst
didn't have time to mark. Filtering with any anomaly score will throw out
the very microseismic events we're trying to *expand the catalog with*.

This is consistent with the broader theme: **recall vs an incomplete manual
catalog is the wrong target for "expanded catalog"**. The right target is
an event-level metric (PyOcto association → NLLoc location → event count).

Files:
- `catalogs/dd_filter_sweep_picks_dd_filtered.csv`
- `catalogs/dd_filter_sweep_picks_obst_01_dd_filtered.csv`
- `figures/dd_filter_sweep_picks_dd_filtered.png`
- `figures/dd_filter_sweep_picks_obst_01_dd_filtered.png`
- `figures/anomaly_score_tp_vs_fp_picks_dd_filtered.png`
- `figures/anomaly_score_tp_vs_fp_picks_obst_01_dd_filtered.png`

Hypothesis: the OBS DD output is a **signal extractor**. RMS of denoised
output at a pick time → high if the model "saw signal" there; low if it
treated the window as pure noise. So:

```
anomaly_ratio_at_pick = denoised_RMS_at_pick / median(denoised_RMS_at_random_windows)
```

Use this as a post-filter on existing baseline (PhaseNet `instance` @ 0.1)
picks. Threshold sweep should reveal whether real picks have systematically
higher anomaly ratio than spurious ones.

(My first implementation used residual energy = raw − denoised, which is
the OPPOSITE: residual is high where DD removed lots of noise, doesn't
indicate signal presence. Killed that, fixed to use denoised signal energy.)

- **Scripts**: `13_dd_post_filter.py` (compute anomaly per pick),
  `14_dd_filter_sweep.py` (sweep threshold, validate),
  `15_anomaly_score_at_manual_picks.py` (sanity-check distribution at TPs vs FPs)
- **Output dir**: `catalogs/picks_dd_filtered/`
- **Status**: running on Feb 4-13. Will update with results.

## Files

- `models/phasenet_bransfield_v2/best.ckpt` — Picker F
- `models/phasenet_bransfield_v3/best.ckpt` — Picker G
- `data/seisbench/bransfield_aug/` — v2 training set (raw + shifted)
- `data/seisbench/bransfield_aug_dd/` — v3 training set (denoised + shifted)
- `catalogs/picks_pn_dd/` — Picker E output (Feb 4-13)
- `catalogs/picks_pn_v2/` — Picker F Feb output
- `catalogs/picks_pn_ft/` — Picker F Aug output
- `catalogs/picks_pn_v3_dd/` — Picker G Feb output
- `catalogs/picks_dd_filtered/` — DD post-filter output (in flight)
- `scripts/11_run_denoised_picker.py` — DD + PhaseNet pipeline (used by E and G)
- `scripts/12_build_dd_aug_dataset.py` — denoised augmented dataset builder (v3 source)
- `scripts/13_dd_post_filter.py` — DD anomaly score per pick
- `scripts/14_dd_filter_sweep.py` — threshold sweep + plot
- `scripts/15_anomaly_score_at_manual_picks.py` — TP vs FP distribution analysis
