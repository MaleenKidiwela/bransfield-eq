# Full-year picking — production run

Resume of the killed run that crashed when JupyterHub restarted at 2026-05-11 ~06:52 UTC.

## Configuration

| Parameter | Value |
|---|---|
| Window | 2019-01-01 → 2020-03-01 (425 days, 13,955 station-days) |
| Picker A | PhaseNet `instance` @ 0.1 → `catalogs/picks/` |
| Picker B | OBSTransformer `obst2024` @ 0.1 → `catalogs/picks_obst_01/` |
| Workers per job | 8 |
| Batch size | 256 |
| Device | cuda (single L40S, 46 GB) |
| Parallelism | Both pickers run concurrently sharing the GPU |
| Idempotent | Yes — skips station-days with an existing CSV (`03_run_phasenet.py:185-186`) |
| Detachment | `setsid nohup ... < /dev/null &` so it survives JupyterHub restarts (parent PID = 1) |

Both runs use `scripts/03_run_phasenet.py` (the same script handles both models via `--model`).

## Launch commands (resumable as-is)

Run from `/home/jovyan/bransfield-eq`.

```bash
setsid nohup .venv/bin/python scripts/03_run_phasenet.py \
  --start 2019-01-01 --end 2020-03-01 \
  --weights instance --p-thresh 0.1 --s-thresh 0.1 \
  --workers 8 --batch-size 256 --device cuda \
  --out-subdir picks \
  >> logs/full_year_pn.log 2>&1 < /dev/null &

setsid nohup .venv/bin/python scripts/03_run_phasenet.py \
  --start 2019-01-01 --end 2020-03-01 \
  --model OBSTransformer --weights obst2024 \
  --p-thresh 0.1 --s-thresh 0.1 \
  --workers 8 --batch-size 256 --device cuda \
  --out-subdir picks_obst_01 \
  >> logs/full_year_obst.log 2>&1 < /dev/null &
```

Append a `=== RESUME <UTC> ===` marker to each log before relaunching so future
spelunkers can find session boundaries.

## Current resume (2026-05-11T07:12:41Z)

- PN process: PID 807 (under setsid wrapper 801), parent=1
- OBST process: PID 805 (under setsid wrapper 802), parent=1
- Prior progress before crash: PN ~1850 station-days / 287k picks; OBST ~1800 / 4.1M picks (~13 % of 13,955)
- Existing output preserved; newest CSVs were verified to parse cleanly before resume

## Output layout

- `catalogs/picks/` → `/home/jovyan/my_data/bravoseis/picks/` (symlink)
- `catalogs/picks_obst_01/` → `/home/jovyan/my_data/bravoseis/picks_obst_01/` (symlink)
- One subdir per station (e.g. `5M.GUR/`), one CSV per Julian day (e.g. `2019-008.csv`)
- CSV columns: `time, trace_id, phase, prob, start, end, ...`

## Recovery procedure if the server kills it again

1. `nvidia-smi` and `pgrep -af 03_run_phasenet` to confirm processes are gone.
2. Check newest CSV in each output dir: `find catalogs/picks{,_obst_01} -name "*.csv" -printf "%T@ %p\n" | sort -rn | head`.
3. Spot-parse the newest CSV with pandas — delete it if it's truncated or fails to parse (the idempotent check is `exists()` only, not size/validity, so a partial file is silently skipped on resume).
4. Re-run the two launch commands above. They'll skip every completed station-day.

## Why OBST @ 0.1 (not 0.5 per old `SUMMARY.md`)

The earlier "production hybrid" (PN @ 0.1 + OBST @ 0.5 → `picks_obst_05/`) is
documented in `SUMMARY.md` §9. This run deliberately uses OBST @ 0.1 instead:

- Maximises recall before association. OBST @ 0.1 had S-recall 0.90 on Feb 4–13
  vs 0.78 at 0.5 (see `notes/11_dd_pickers_full_results.md`).
- Pick filtering is deferred to PyOcto association (next pipeline stage,
  `scripts/17_pyocto_associate.py` — same date range already configured).
- Consistent with the lesson in §10 of SUMMARY: per-pick recall vs an
  incomplete manual catalog is the wrong target; event-level metrics after
  association are.

Cost: ~3× more raw picks for OBST (~4M vs ~1M at threshold 0.5), more I/O and
slightly larger association workload, but storage is not the constraint.

## Expected wall time

Pre-crash rate was ~1850 station-days in ~1 h elapsed for both jobs combined,
so ~13,955 / 1850 ≈ 7.5 × current → another ~5–6 h to finish from where it
stopped, assuming similar throughput. Sharing the GPU between two jobs slows
both vs sequential, but total wall time is shorter than running them back to
back.

## Next stage (after both jobs finish)

`scripts/17_pyocto_associate.py --start 2019-01-01 --end 2020-03-01` — already
configured for this date range. That's the first event-level output (and the
real test of whether OBST @ 0.1 was the right call vs 0.5).
