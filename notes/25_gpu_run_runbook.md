# Runbook — full-year re-pick with the benchmarked pool

Prepared 2026-09-11, to be executed when a GPU pod is available. Everything
below is committed and tested on CPU; nothing is left to improvise.

Benchmark that chose this pool: [24_session_2026-09-11.md](24_session_2026-09-11.md)
· report `notes/picker_benchmark.html`

## The pool

| model | weights | channels | networks | output |
|---|---|---|---|---|
| PhaseNet | `diting` | 3C (`ZNE`, 50 Hz internal) | **all** (ZX + 5M + AI + AM) | `catalogs/picks_pn_diting/` |
| PhaseNetLight | `obs` (PickBlue) | 4C (`Z12H`, hydrophone) | **ZX only** | `catalogs/picks_pnlight_obs/` |

Both at P/S threshold **0.1**, so downstream filters by probability instead of
re-picking. Expected on the benchmark set: **P 89.2% / S 96.0%** against the
current pipeline's 77.8% / 92.4%.

**Why PhaseNetLight is ZX-only.** PickBlue's 4th channel is the hydrophone, and
only ZX has one (`EDH` on the dense Orca instruments, `HDH` on the broadbands).
The 15 land stations (5M/AI/AM) are 3-component; SeisBench would zero-fill the
4th channel there. `diting` is 3C and covers every network, so land stations are
picked by `diting` alone.

Optional, free: append `,picks_obst_01:P` to the association pool to bring
OBSTransformer's P back for the land stations — those picks already exist for
the whole year. Deliberately **P only**: OBST's S picks run +70 ms late, and
keeping them would put that bias back into S–P times.

## Before you start — the GPU caveat

`torch 2.12.0+cu130` in this image is **built for CUDA 13.0**. The driver seen
from this pod reported CUDA 12.2, so on a GPU pod torch may refuse to
initialise. The preflight reports this explicitly. If it says
"GPU attached but torch cannot initialise it":

```bash
pip install --user --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
# then re-run the preflight; it must say "GPU usable by torch"
```

Do this **before** launching — a failed `--device cuda` falls back to nothing
useful, and finding out three hours in wastes the GPU slot.

## Step 1 — preflight (changes nothing)

```bash
cd ~/bransfield-eq
PYTHONPATH=src python3 scripts/35_preflight_gpu_run.py --timing-days 3
```

Checks epoch conversion, that no unguarded `datetime -> int64` casts remain,
that the associator takes `--pick-sources`, whether torch can drive the GPU,
that both weight sets resolve, the channel mapping per network, the station-day
inventory and resume state, disk space — then **times a few station-days on the
actual device and projects the wall time**. Exits non-zero on any failure.

Measured on CPU, 2026-09-11: `diting` 5.6 s/station-day, `pnlight`
9.0 s/station-day serial → **5.4 h at `--workers 8`** for 22,741 station-days.
A GPU should bring that under an hour; the preflight will tell you.

## Step 2 — pick the full year

```bash
bash scripts/34_run_full_year_picks.sh                 # auto device + workers
bash scripts/34_run_full_year_picks.sh --workers 6     # override
bash scripts/34_run_full_year_picks.sh --only diting   # one model
bash scripts/34_run_full_year_picks.sh --shard 0 --of 4  # split across pods
```

- **Resumable.** `03_run_phasenet.py` skips station-days whose CSV exists, so a
  pod restart or OOM costs only the day in flight. Re-run the same command.
- **Seeds itself** from `catalogs/picks_bench_*` on first run: the 220
  benchmark station-days per model used identical weights, channels and
  thresholds, so they are copied rather than recomputed. `--no-seed` to skip.
- Logs to `logs/full_year_picks_<timestamp>_shard<N>of<M>.log`.
- Device auto-detects; GPU gets `--batch-size 512`, CPU `256`.

Work: **13,955** station-days for `diting`, **8,786** for `pnlight` (the 2,195
zero-byte placeholders in `data/waveforms/` are skipped automatically).
Output: ~1.5 GB + ~1.9 GB.

## Step 3 — validation gate. Do not skip this.

Higher pick recall is **not** automatically a better catalogue. More picks can
mean more false associations, not more events. Associate one month both ways
and compare before committing the year:

```bash
# new pool
PYTHONPATH=src python3 scripts/17_pyocto_associate.py \
  --start 2020-01-01 --end 2020-02-01 --label newpool_2020_01 \
  --pick-sources 'picks_pn_diting:PS,picks_pnlight_obs:PS'

# current production pool, same month, for the baseline
PYTHONPATH=src python3 scripts/17_pyocto_associate.py \
  --start 2020-01-01 --end 2020-02-01 --label prodpool_2020_01
```

**Budget for it.** Association is CPU-bound and does *not* speed up with a GPU.
Measured 2026-09-11 on one day (2020-01-25): 54k input picks took **520 s** of
associator wall time for 187 events; the production pool's 73k picks on the same
day took longer still. A 31-day test month is therefore roughly **5-8 h** per
pool. Raise `--n-threads` if the box is quiet, and run the two pools
concurrently — they touch different inputs and different output labels.

Compare `catalogs/pyocto_events_{newpool,prodpool}_2020_01.csv` on:

| metric | what a good result looks like |
|---|---|
| event count | up, or flat with better-constrained events |
| picks per event | up (more stations per event, not more marginal events) |
| post-fit RMS | flat or lower — a jump means noise is being associated |
| station count per event | up |
| NLLoc σ (after stage 5) | down; this is the metric that matters for the paper |

If RMS rises sharply while the event count balloons, raise `--pick-prob-min`
(0.2 or 0.3) and re-associate rather than abandoning the pool — the benchmark
swept thresholds and recall degrades gracefully.

## Step 4 — rebuild downstream, in order

A changed pick pool invalidates everything derived from the old one. Sequence,
do not patch in place:

1. `17_pyocto_associate.py` — full year with the chosen pool
2. `28`–`31` — NLLoc `.obs`, control files, run, hyp → catalogue
3. pick refinement + NLLoc re-run (the σ_z −32% workflow from session 23)
4. `18`–`19` — GrowClust XC prep + relocation
5. `22`–`26` — hypoDD Stage A/B/C
6. `32_hybrid_catalog.py` — hybrid assembly

Watch pod memory at step 4: the XC prep preload has OOM-killed the pod twice
(~187 GiB cap, see the memory note).

## Verified end-to-end on 2026-09-11

Not just syntax-checked — the whole chain was run on one day (2020-01-25) with
the new pool:

- `03_run_phasenet.py` picks with both models and the 4-channel glob; the
  hydrophone ablation (4C vs 3C) confirms the 4th channel changes the output.
- `17_pyocto_associate.py --pick-sources 'picks_pn_diting:PS,picks_pnlight_obs:PS'`
  loaded 53,857 picks and produced **187 events / 1,961 associated picks**
  → `catalogs/pyocto_events_smoketest_newpool.csv`.
- The default `--pick-sources` still loads exactly 82,324 picks for 2019-07-11,
  identical to before the refactor, so the published catalogue remains
  reproducible.

### ...and the one-day comparison already complicates the recommendation

Both pools associated on 2020-01-25:

| | current production | diting + PB-PNL |
|---|---|---|
| input picks | 73,131 | 53,857 (−26%) |
| events | 117 | **187 (+60%)** |
| picks / event | 11.9 | 10.5 |
| stations / event (median) | 8 | 6 |
| **events with ≥8 stations** | **60** | **59** |

Read the last row before the second. The new pool finds 60% more events from a
quarter fewer picks — but the number of *well-constrained* events is
unchanged (59 vs 60). Every one of the ~70 extra events sits in the 3-7 station
range. At the pick level the pool is clearly better; at the catalogue level, on
this one day, it buys marginal detections rather than better locations.

That does not sink the pool — more small events is a legitimate result for a
detection paper, and σ from NLLoc is the metric that decides it — but it does
mean **the decision cannot be made on picker recall alone**, and two things
should happen during the test month:

1. **Check NLLoc σ on the well-constrained subset**, not just the event count.
   If σ is flat and the count of σ-passing events is flat, the pool change buys
   nothing for the paper's catalogue and only helps a detection-rate argument.
2. **Re-tune the associator for the new pick density.** `--min-p 3 --min-s 2
   --min-total 6` and `--pick-tol` were chosen against OBSTransformer's pick
   density. `diting` emits about half the picks with higher precision, so the
   old thresholds may be the reason the extra events come out thin. Sweep
   `--min-total` and `--pick-prob-min` before concluding.

Delete the `smoketest_*` catalogues when you no longer need them.

## Also ready

- **pandas-3 unit bug fixed** across the pipeline. `src/bransfield_eq/timeutil.py`
  provides `epoch_ns` / `epoch_seconds` / `epoch_ms`; scripts 05, 16, 17, 18, 45,
  `manual_picks_to_pyocto.py` and `manual_picks.py` now use them. Script 17
  calls `assert_nanosecond_sanity()` before association, so a future unit
  regression fails loudly instead of silently. Verified: pick times for
  2019-07-11 span 86,389 s across the day (the bug gave 86.4 s).
- **`03_run_phasenet.py`** takes `--picking-channels` (lets the hydrophone
  through) and any SeisBench picker class, not just three hard-coded ones.
- **`17_pyocto_associate.py`** takes `--pick-sources` and `--pick-prob-min`.
  The default reproduces the published catalogue exactly — verified at 82,324
  picks for 2019-07-11, unchanged by the refactor.

## Not done

- **`diting` runs at 50 Hz internally** (every other model here is 100 Hz). Its
  timing was the *best* in the benchmark — P bias −10 ms, MAD 25 ms — so this
  is not a known problem, but it is worth a look at sub-sample precision before
  the picks feed cross-correlation in GrowClust, where 10 ms matters.
- **GPD `stead`** was dropped from the benchmark; 16/220 station-days sit in
  `catalogs/picks_bench_gpd_stead/` and will resume rather than restart.
- **The benchmark was ZX-only.** The land stations are picked by `diting` on
  the strength of its ZX performance; there are only 330 land analyst picks, so
  no separate validation was possible.
- **Re-doing the fine-tune** from the `diting` base, trained on the late period
  only (scripts `06`–`09`). The existing fine-tune lost 12.9 points of S
  against its own base, most likely from training on the unreliable early
  picks.
