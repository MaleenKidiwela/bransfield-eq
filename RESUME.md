# Resume guide — overnight pipeline

If this terminal session is lost, here's how to get back to the same state.

## Latest resume — 2026-05-11 09:04 UTC

Pickers crashed at ~08:41 UTC (JupyterHub restart, same as the earlier 07:12 incident).
Relaunched at 09:04 UTC with the exact commands in §"Option B → Stage 1" below.

| Pre-crash progress | PN | OBST |
|---|---:|---:|
| Station-days picked | ~2,350 / 13,955 (~17%) | ~2,325 / 13,955 (~17%) |
| Picks so far | 273k | 4.72M |

Both runs are idempotent — relaunching skipped the ~2,300 station-days already on
disk and continues from there. Newest pre-crash CSVs were spot-parsed and clean
(no truncation).

`=== RESUME 2026-05-11T09:04Z ===` markers appended to both
`logs/full_year_pn.log` and `logs/full_year_obst.log` before relaunch.



## The good news

The picker processes are **detached** (`setsid nohup`, parent PID = 1). They
survive terminal disconnect, SSH drop, and JupyterHub restart. Closing this
Claude session does NOT kill them.

## What dies vs. what survives

| Component | Survives session loss? |
|---|---|
| Picker processes (PIDs 1288 OBST, 1289 PN; relaunched 2026-05-11 09:04 UTC after the 08:41 crash) | ✅ Yes — detached |
| Pyocto / XC prep / GrowClust processes (when launched) | ✅ Yes — same launch pattern |
| Pick CSVs already written | ✅ Yes — on disk |
| Claude `/loop` monitor + auto stage-advance | ❌ No — tied to this session |

So if Claude dies overnight, the **pickers keep going** but **stage 2+ won't
auto-launch**. You can either restart Claude or run the stages manually.

---

## Option A — restart Claude and reattach

From any terminal in `/home/jovyan`:

```
claude --continue
```

This re-opens the most recent Claude session (`a3c9f12a-...jsonl`) with full
context. Then ask me to "check the pipeline state and re-arm the monitor loop"
— I'll inspect file existence, advance to whatever stage is next, and re-arm
the wakeup.

If `--continue` doesn't find the session, start fresh and point me at
`notes/14_overnight_pipeline.md` and this file — that's enough context to
pick up where we left off.

---

## Option B — run the remaining stages manually

If you'd rather not restart Claude, here's the full pipeline as plain commands.
Run from `/home/jovyan/bransfield-eq`. Each stage's "done" condition tells
you when to advance to the next.

### Stage 1 — pickers (running now)

```
pgrep -af 03_run_phasenet              # are they still alive?
tail -f logs/full_year_pn.log          # PhaseNet progress
tail -f logs/full_year_obst.log        # OBSTransformer progress
```

**Done when:** both logs end with `Done. picked=… skipped=… …` AND `pgrep`
returns nothing.

If a process died early, relaunch with:

```
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

Both scripts are idempotent — they skip station-days that already have an
output CSV.

### Stage 2 — pyocto association

```
setsid nohup .venv/bin/python scripts/17_pyocto_associate.py \
  --start 2019-01-01 --end 2020-03-01 \
  --velocity-model configs/velocity_model.csv \
  --label picker_only \
  >> logs/pyocto_picker_only.log 2>&1 < /dev/null &
```

**Done when:** `catalogs/pyocto_events_picker_only.csv` exists.

### Stage 3 — GrowClust XC prep (waveform cross-correlation)

```
setsid nohup .venv/bin/python scripts/18_growclust_xc_prep.py \
  --label picker_only --workers 8 \
  >> logs/growclust_xc_prep.log 2>&1 < /dev/null &
```

**Done when:** `growclust/picker_only/dt.cc` exists AND log ends with `Done.
Next step:`.

### Stage 4 — GrowClust binary (1D relative relocation)

```
setsid nohup .venv/bin/python scripts/19_run_growclust.py \
  --label picker_only \
  >> logs/growclust_run.log 2>&1 < /dev/null &
```

**Done when:** `catalogs/growclust_picker_only.csv` exists.

The binary is at `/home/jovyan/GrowClust/SRC/growclust` (already compiled and
verified against the bundled Spanish-Springs example).

---

## Files to know

| Path | Purpose |
|---|---|
| `notes/12_full_year_run.md` | Stage 1 detail + PID 805/807 launch |
| `notes/13_velocity_model.md` | How `configs/velocity_model.csv` was derived from Orca .nc |
| `notes/14_overnight_pipeline.md` | Full 4-stage plan |
| `configs/velocity_model.csv` | 1D Vp/Vs profile (0 → 30 km) |
| `configs/velocity_model.pyocto` | Pre-built pyocto travel-time cache |
| `/home/jovyan/GrowClust/SRC/growclust` | Compiled GrowClust binary |

## If everything's truly stuck

The waveform data (`my_data/bravoseis/`, 804 GB) is untouched by any of
this — worst case you lose pick CSVs already on disk for a few station-days,
but the idempotent skip means a relaunch only redoes anything that was
mid-write at the crash. No data is lost.
