# notes/ — Bransfield EQ working notes

Append-only working notes for the cluster pipeline runs.
Files are roughly chronological; each is single-topic.

| File | Topic |
|---|---|
| [`00_cluster_environment.md`](00_cluster_environment.md) | Hardware, venv, deps |
| [`01_storage_layout.md`](01_storage_layout.md) | Where data lives + symlink strategy |
| [`02_one_day_verification.md`](02_one_day_verification.md) | Pipeline reproducibility check on 2019-12-26 |
| [`03_gpu_optimizations.md`](03_gpu_optimizations.md) | Changes to `03_run_phasenet.py` |
| [`04_validation_and_mag07.md`](04_validation_and_mag07.md) | mag07 as trusted ground truth + `--manual-source` flag |
| [`05_year_run_status.md`](05_year_run_status.md) | Full-year download/pick/EQT status |
| [`06_active_plan.md`](06_active_plan.md) | **Active plan: OBS DeepDenoiser + PhaseNet fine-tuning** |
| [`06a_user_autoencoder_proposal.md`](06a_user_autoencoder_proposal.md) | Original user proposal that led to the active plan (kept for record) |
| [`07_obs_baseline_pickers.md`](07_obs_baseline_pickers.md) | PickBlue + OBSTransformer baselines on day-26 |
| [`08_multiday_picker_comparison.md`](08_multiday_picker_comparison.md) | 10-day validation (2019-02-04 → 13) confirming the hybrid baseline |
| [`09_obs_denoiser_v1.md`](09_obs_denoiser_v1.md) | OBS DeepDenoiser v1 training run + Phase 0 sanity comparison |
| [`10_finetune_v1.md`](10_finetune_v1.md) | PhaseNet fine-tune v1+v2 attempts — **both underperform baseline; do not deploy** |
| [`SUMMARY.md`](SUMMARY.md) | **Comprehensive summary** of all results, findings, decisions, lessons across both sessions |
| [`11_dd_pickers_full_results.md`](11_dd_pickers_full_results.md) | Pickers E/F/G results + DD-anomaly post-filter approach |

Conventions: dates absolute (`2026-05-09`), commands quoted exactly, observed numbers shown alongside predicted ones.
