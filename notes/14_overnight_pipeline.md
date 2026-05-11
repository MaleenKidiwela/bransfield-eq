# Overnight pipeline (2026-05-11)

User went to sleep at ~07:30 UTC with auto-mode on. Three-stage pipeline
self-chained via the /loop dynamic-mode heartbeat.

## Stages

| # | Stage | Script | Output | Launch trigger |
|---:|---|---|---|---|
| 1 | Picking (running) | `03_run_phasenet.py` × 2 | `catalogs/picks/`, `catalogs/picks_obst_01/` | (Already running, resumed 07:12 UTC) |
| 2 | Association | `17_pyocto_associate.py` | `catalogs/pyocto_{events,picks}_picker_only.csv` | Auto: when both pickers print final `Done.` |
| 3 | GrowClust XC prep | `18_growclust_xc_prep.py` | `growclust/picker_only/{dt.cc, evlist.txt, stlist.txt}` | Auto: when pyocto prints `Done.` |
| 4 | GrowClust binary run | `19_run_growclust.py` | `catalogs/growclust_picker_only.csv` | Auto: when XC prep prints `Done.` |

## Velocity model

`configs/velocity_model.csv` — 54 layers (0 → 30 km), derived from
`Pg_Orca_velocity.nc` by taking the median Vp across (x,y) at each depth.
Vs = Vp / 1.78. See `notes/13_velocity_model.md` for full derivation.

`configs/velocity_model.pyocto` — pre-built pyocto cache file (regenerated
on demand by `load_velocity_model` if the source CSV is newer).

## What the loop will do tonight

1. Heartbeat every ~25 min.
2. While stage 1 alive: report station-days picked + picks-so-far per job.
3. When BOTH picker logs show final `Done.` line:
   - Launch stage 2 (pyocto) via `setsid nohup`.
   - Switch loop to monitor `logs/pyocto_picker_only.log`.
4. When pyocto log shows `Done.` (or its final `wrote` lines):
   - Launch stage 3 (XC prep) via `setsid nohup`.
   - Switch loop to monitor `logs/growclust_xc_prep.log`.
5. When XC prep finishes, auto-launch stage 4 (GrowClust binary).
6. When stage 4 finishes:
   - Report n events relocated and mean |Δlat|, |Δlon|, |Δdep| vs. pyocto absolute.
   - **Stop** the loop.

## GrowClust setup

- Source cloned to `/home/jovyan/GrowClust/` (Trugman & Shearer 2017, Fortran).
- Built at `/home/jovyan/GrowClust/SRC/growclust` using conda-forge gfortran 15.2.
- Verified end-to-end with the bundled Spanish-Springs example (3 relocated events).
- Tolerates the harmless non-zero exit code from the bootstrap-file writer.

## What will NOT auto-launch

- **NLLoc location** — out of scope per user direction "first do 1D velocity".
- **3D velocity model in GrowClust** — original Fortran GrowClust is 1D-only.
  For 3D, would need GrowClust3D.jl (Julia) — defer to a follow-up.

## Recovery if anything dies

Each stage's launch command is documented above. The pickers + pyocto are
detached (parent PID = init), so JupyterHub restarts can't kill them. If
something dies inside a process (OOM, etc.):

1. `pgrep -af 03_run_phasenet` (or `17_pyocto` or `18_growclust`)
2. Tail the corresponding log
3. Re-run the stage's command from `notes/12_full_year_run.md` (stage 1)
   or directly from this file (stages 2 + 3). All stages are idempotent
   or restartable.
