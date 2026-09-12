# Rebuild plan — relocation stack on the new pick pool

Written 2026-09-12, while the full-year re-pick was running. **Plan only; no
code written yet.** Decisions taken with the user this session:

- Use the new pick pool (`diting` + PickBlue PhaseNetLight). Old catalogues and
  relocations are discarded, not migrated.
- Rebuild the **relocation stack only**. Data acquisition, station geometry,
  shot discrimination, velocity-model construction and picking are kept.
- Relative relocation goes **3D via hypoDD `IMOD=9`**.

---

## 1. Why rebuild rather than patch

Every defect found in this pipeline has failed *silently* — wrong numbers, no
traceback, no warning. That is the argument for replacing the machinery rather
than fixing it in place.

| found | defect | failure mode |
|---|---|---|
| earlier | ~7 bugs in pyocto/XC scripts | silent |
| 2026-05-20 | NLLoc `event_idx` join scrambled ~half the v2 mapping | silent |
| 2026-06-29 | pandas-3 `.astype("int64")` 1000x wrong on datetimes | silent |
| 2026-09-12 | XC correlating unfiltered swell: 4.6 ms RMS leakage at cc~0.99 | silent |
| 2026-09-12 | XC kernel fixed-norm triangular taper biases lag toward 0 | silent |
| 2026-09-12 | `idx` vs `event_idx`: joins 2.5% of rows on merged catalogues | silent |
| 2026-09-12 | tmerc origin differs 19.9 km between two association runs | silent |

The common root is that **each stage invents its own conventions** for event
identity, coordinates and time. `timeutil.py` already fixed exactly this for
datetimes and has held since. Extend that pattern.

## 2. Scope

**Keep (verified, expensive to redo):**
- Stage 0 waveform / StationXML acquisition
- Station geometry, including the ZX.BRA05 +0.167 s clock correction
- Stage 2b shot discrimination (`discriminate_shots.py` + classifier)
- Velocity model construction, incl. `38_build_extended_velgrid.py`
- Stage 1 picking — the new pool, finishing now
- NLLoc stage (already 3D; it is the only 3D-aware stage today)

**Rebuild:**
- Association wiring (pyocto itself stays; its *thresholds* and *conventions* change)
- Differential-time production — drop the bespoke XC entirely
- hypoDD stage, moving 1D -> 3D
- Catalogue assembly (simplify `32_hybrid_catalog.py`)

## 3. Architectural rule

> One module owns event keys, map projection and time. Every stage imports it.
> No stage computes a projection origin from its own input data.

This kills the `idx`/`event_idx` class and the 19.9 km origin-drift class
outright. The origin becomes a **constant in config**, not a function of which
stations happened to have picks that run.

---

## 4. Stages

### R0 — shared foundation
`src/bransfield_eq/geo.py` (new), alongside existing `timeutil.py`.

- `EVENT_ID`: one globally unique, stable event key. Never a row index, never
  restarted per chunk.
- `project(lat, lon)` / `unproject(x, y)`: transverse Mercator with a **fixed
  origin from config**.
- Station geometry accessor with the BRA05 correction applied in one place.
- Assertion helpers that fail loudly (following `assert_nanosecond_sanity()`).

Testable immediately against existing catalogues.

### R1 — association
`17_pyocto_associate.py` with `--pick-sources 'picks_pn_diting:PS,picks_pnlight_obs:PS'`.

Thresholds must be **re-tuned**: `--min-p 3 --min-s 2 --min-total 6` and
`--pick-tol` were chosen against OBSTransformer's far denser pick stream.
`diting` emits ~half the picks at higher precision. Sweep in progress this
session. The one-day test (2020-01-25) showed the untuned result: +60% events
but the >=8-station count flat at 59 vs 60 — all gain in marginal detections.

Validation months: **2020-01 primary** (most analyst picks and most events
inside the trusted window, which the analyst puts at 2019-10-01 onward), plus
**2019-10 as a noise-regime check** — October is demonstrably harsher
(XC leakage 8.15 ms there vs 2.55 ms in January).

### R2 — shot removal
Reuse unchanged.

### R3a — `dt.ct`
`23_run_ph2dt.py`, reuse.

### R3b — `dt.cc` (new correlator)
Replace `18_growclust_xc_prep.py` / `18b_prewindow_picks.py`.

Build on **`obspy.signal.cross_correlation.correlate_template(...,
normalize='full')`** — proper per-lag normalisation, which is precisely the
defect in the hand-rolled kernel. obspy 1.5.0 is already installed; no new
dependency. `xcorr_pick_correction` is purpose-built for differential pick
times and has filtering built in.

Requirements carried over from this session's findings:
- **Bandpass before correlating.** Raw OBS is 99.9% power below 3 Hz (measured
  on ZX.BRA02 2019 doy 190). Sub-5 Hz holds 27.5% of signal power but 64.4% of
  noise power — SNR ~0 dB. See `src/bransfield_eq/xcfilter.py` for the full
  measurement; 3-20 Hz was best on the leakage test, but **re-validate on real
  event pairs** in the new implementation, since the band trade inverts when
  the two waveforms are genuinely different events.
- **Stamp filter state next to any cached windows** and refuse on mismatch.
  The interim fix in `18` does this; carry the idea forward.
- **Fix the S-phase component bug**: the old `slice_window` always preferred the
  vertical channel, including for S picks. S should use horizontals.

### R4 — hypoDD 3D (`IMOD=9`)
HypoDD 2.1beta at `/home/jovyan/HypoDD` already has the 3D sources compiled:
`get_vel3d.f`, `ray_3d.f`, `partials_3d.f`, `setorg_3d.f`. Currently we run
`IMOD=1` (1D layered, per-layer Vp/Vs) via `24_run_hypodd.py`.

`IMOD=9` = 3D model + pseudo-bending ray tracing (Um & Thurber; implemented by
Rietbrock). Needs:
- 3D model in **SIMUL2000 grid-node format** (see `get_vel3d.f` header)
- `lat_3d`, `lon_3d`, `rot_3d` — model origin and rotation
- pseudo-bending params: `ndip, iskip, scale1, scale2, xfac, tlim, nitpb`

See section 5 for the grid design.

### R5 — NLLoc absolutes  (run last, after hypoDD)
Reuse. Provides the absolute frame and sigma. Sigma on the well-constrained
(>=8-station) subset remains the metric that decides catalogue quality.

**Ordering note.** NLLoc runs *last in sequence* but is **not downstream of
hypoDD in data terms** — `28_pyocto_to_nlloc_obs.py` takes pyocto picks
directly. It is an independent branch, so it can run concurrently with R3b/R4
if CPU allows; running it last is a scheduling choice, not a dependency.

**Grid-consistency caveat (new, matters after R4).** Today NLLoc is the only
3D stage. After R4 both NLLoc and hypoDD are 3D *on the same Orca model but in
different representations* — NLLoc on the fine 0.4 km grid from script 38,
hypoDD on a coarse 25 x 25 x 20 SIMUL grid. Any disagreement between them
could then be a **grid artefact rather than a method difference**, which would
be easy to misread as one code validating or contradicting the other. Before
comparing them, either (a) run NLLoc once on a grid decimated to match the
hypoDD node set, to separate grid effect from method effect, or (b) state
explicitly that the two use different discretisations of the same model.

### R6 — catalogue assembly
Simplify. The old three-branch + hybrid-anchoring design is where the join bugs
bred. With R0 in place, joins are on one stable key.

---

## 5. 3D velocity grid design (evidence-based)

**hypoDD's compile-time cap is 25 x 25 x 20 nodes** (`include/vel3d.inc`:
`maxnx=25, maxny=25, maxnz=20`). The NLLoc grid from script 38 is
576 x 451 x 64 = 16.6M nodes. So the model must be re-gridded ~20x coarser per
axis — *or* `vel3d.inc` raised and hypoDD recompiled.

**Decision: re-grid, do not recompile.** Not for convenience — because the cap
is not the binding constraint on accuracy. The tomography's own resolution is.

*Evidence 1 — the model's true resolution is ~3-4 km.* Per
`configs/README_velocity.pdf`, the Pg model comes from 15-16 OBS at **3-4 km
spacing** (3 in the caldera ~1.5 km apart) with 2,426 airgun shots on 21 km
lines. The 0.2 km cubes are a parameterisation, not resolution. 25 x 25 nodes
across the 60 x 40 km resolved zone is ~2.4 x 1.6 km — already finer than the
experiment resolves. More nodes would interpolate noise.

*Evidence 2 — all lateral structure is in the top ~4 km.* Measured directly
from `Pg_Orca_velocity.nc`:

| depth | P90-P10 | lateral range |
|---|---|---|
| 0-2 km | 0.37-0.62 km/s | up to 2.0 km/s |
| 4.0 km | 0.091 | 0.41 |
| 4.8 km | 0.020 | 0.08 |
| 6.0 km | 0.009 | 0.03 |
| **>=6.4 km** | **0.000** | **0.00** |

Below 6.4 km the model is *literally constant* — it is the untouched starting
model the inversion never updated. RMS lateral deviation from the layer median
is 5.5% of Vp over 0-2 km, falling to 1.3% over 0-10 km.

**Therefore node placement, not node count, is the design problem:**

- **z (20 nodes):** concentrate in 0-5 km. Roughly 0, 0.25, 0.5, 0.75, 1.0,
  1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12, 18, 25 km. Spending
  uniform nodes over 25 km would waste ~14 of 20 on a constant halfspace while
  under-resolving the only part that carries structure.
- **x, y (25 each):** non-uniform. ~15-18 fine nodes (2-3 km) spanning the
  60 x 40 km srModel extent, plus a few very coarse nodes reaching out to
  +-150 km so rays to the land stations stay inside the grid. Standard SIMUL
  practice. Outside the srModel, blend to the 1D background exactly as
  script 38 already does for the NLLoc grid.
- Grid must **cover every source-receiver path**, or the tracer extrapolates.

**Caveats to state in any write-up:**
- Below ~4.5 km, 3D adds nothing over 1D — this model cannot constrain it.
  Deeper events gain only through their shallow leg.
- Double-difference is deliberately insensitive to smooth velocity error for
  *close* pairs. Expect gains in the absolute frame and for wider-aperture
  pairs, not in tight-cluster relative geometry.
- Two further models exist and are unused: `Pg_Anis_Orca_velocity.nc`
  (anisotropic) and `Pg_Pmc_Orca_velocity.nc`. Anisotropy is present in this
  dataset and we are ignoring it.

---

## 6. Validation

Not optional, given the silent-failure record.

1. **R0 against existing data** — assert the new key/projection reproduce known
   joins, and that origin drift cannot recur.
2. **New correlator vs old on the same pairs** — cc distribution, how many
   measurements survive `CC_THRESH`, and dt scatter. Include the same
   displaced-waveform leakage test (old kernel: 4.58 ms RMS, 10.2% of
   measurements >5 ms).
3. **3D vs 1D hypoDD on the same events** — if IMOD=9 does not measurably
   change relocations, say so rather than assuming 3D is better.
4. **NLLoc sigma on the >=8-station subset** — the catalogue-quality metric.
5. **Two-month association check** (2020-01 and 2019-10) before the full year.

## 7. Open questions

- Does filtering cost measurements on real event pairs? Unmeasured. Decides
  the band.
- `diting` picks are hard-quantised to 20 ms (100% on-grid; seisbench's
  `picks_from_annotations` uses a bare integer argmax with no interpolation).
  Harmless if the correlator re-measures lag from the waveforms — confirm in
  the new implementation.
- Land stations were never separately validated: the benchmark was ZX-only and
  there are just 330 land analyst picks.
- Is the hybrid NLLoc/hypoDD anchoring still needed, or does a 3D hypoDD plus
  NLLoc sigma make it redundant?
