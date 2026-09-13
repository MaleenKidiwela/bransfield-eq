# Joint 1D velocity + hypocentre + station-correction inversion (VELEST), 2026-09-13

Append-only. Every number below is pasted from program output; anything I am not
sure about is marked **UNCERTAIN**.

Question: notes/27 I13/I15/I24/I27 eliminated every *single-parameter* velocity
lever (Vp/Vs 1.78 vs 1.90, top-2-km x0.9 and x1.15, the Grid2Time -> pykonal
forward-model fix). The NLLoc depth axis stayed stretched. The remaining lever is
to stop holding the model fixed while relocating. This note is the VELEST-style
minimum-1D run.

---

## J1. VELEST obtained and built (no Python re-implementation needed)

`which velest` -> not found; a filesystem search found nothing. Source cloned from
`https://github.com/jubenjum/velest` (VELEST 3.1 of 10 April 1995, E. Kissling /
U. Kradolfer, ETH; the repo's README says it was downloaded from
`seg.ethz.ch/software/velest.html` on 2019-02-04) into `~/src/velest/jubenjum`.

- Smoke test on the bundled Calaveras example (53 events) runs clean:
  `RMS RESIDUAL 0.419818 -> 0.025453`, CPU 0.1 s.
- Production build `~/src/velest/orca/velest` from `~/src/velest/orca_src`,
  compiled with `~/.conda/envs/gf/bin/gfortran -std=legacy -w
  -fallow-argument-mismatch -fno-automatic -O2`. `vel_com.f` array sizes raised:
  `ieq 658 -> 1600`, `ist 650 -> 60`, `inltot 100 -> 40`. Single-threaded,
  ~0.2 GB resident; never more than 5 concurrent processes (quota is 8).
- Elapsed effort to get a working binary: well under the 30 min budget.

Total VELEST CPU for everything in this note is under an hour (the five
full-catalogue relocations dominate it at ~39 s per iteration each); at most five
processes ran at once.
`/sys/fs/cgroup/cpu.stat` over the session: `nr_periods` +21,832 with
`nr_throttled` +97 (0.4%) -- not throttled.

## J2. Frame: sea level, stations at true depth. The 1.0 km rigid shift is NOT used.

VELEST does **not** force stations onto a flat surface. `subr. RAYPATH`
(velest.f:8368-8388) sets `d(1)=zr` and `thk(1)=d(2)-zr`, i.e. it truncates the
top layer at the receiver depth, and only requires `zr >= d(1)`
(`stop 'RAYPATH>>> station ABOVE model!'`). So the rigid seafloor shift used for
hypoDD (`22_pyocto_to_hypodd_input.py --datum-shift-km 1.0`, notes/27 I1) is not
needed and is not used here.

- Datum = **sea level**. Station "elevations" go in as the signed numbers from
  `catalogs/station_geometry.csv`: OBS -785 m (BRA03) to -1943 m (BRA09), land
  0 to +30 m. The station file format line was widened from `i4` to `i5` so
  -1943 fits.
- Model top at **-0.05 km BSL**, 20 m above the highest land station.
- VELEST's own echo of the station table confirms the geometry is read as
  intended: `B22 62.4365S 58.3912W -1023 ... x -0.95  y -0.42  z 1.02`,
  `B09 ... z 1.94`, `FER ... +30 ... z -0.03`; SDC scale
  `one min lat 1.8575 km, one min lon 0.8609 km` at olat -62.4327 (checked by
  hand against the great-circle distance: agrees).
- The column is **rock only**. `configs/velocity_model.csv` is on a SEA-LEVEL
  datum and carries a 1.3 km water column at 1.4558 km/s (the prompt's
  "depth below seafloor" is wrong; see `24_run_hypodd.py::write_velocity`, which
  drops rows with `vp_kms <= 1.6`). OBS sit *on* the seafloor, so no ray in this
  data set crosses water; the interval above the model's rock top is filled with
  the shallowest rock velocity (2.348 km/s), the same convention as scripts/17
  (notes/27 B4). Layer 1 (-0.05..0.78 km, above every OBS, sampled only by the
  379 land-station picks) is frozen at `vdamp 999`, as are the layers below
  21 km (deeper than every event).

## J3. Inputs

`catalogs/nlloc_year_v5_strict.csv` **does not exist on disk** (only
`nlloc_year_v5.csv` and `nlloc_year_v5_standard.csv` are there). The strict tier
is a strict subset of the standard tier in `40_filter_nlloc_reliable.py`, and the
standard CSV carries every column the strict cuts need, so `62_velest_prep.py`
re-derives it:

```
catalogue: nlloc_year_v5_strict.csv ABSENT -> derived strict tier from
nlloc_year_v5_standard.csv: 1,576 of 6,902 standard events
```

1,576 matches the handoff note exactly. A copy is kept at
`velest/input/nlloc_v5_strict.csv`. No file under `catalogs/` was written.

```
picks: 23,116 -> 23,116 after dmax 150.0 km and travel-time sanity
events with >=4 usable observations: 1,576   observations 23,116 (P 10,880 / S 12,236)
reference station (P correction fixed at 0): B22  (2,653 obs)
inversion subset: 377 events (n_obs >= 12 and >= 4 stations with both P and S,
  <= 6 per 1.5 km cell); median n_obs 18, depth p10/50/90 1.37/4.07/8.18 km BSL
```

Choices made, all documented in `62_velest_prep.py`:

- Station names are `character*4` in VELEST, so ZX.BRAnn -> `Bnn` and
  AM.R4DE2 -> `R4DE`; all 38 aliases unique (asserted).
- 16 layers, tops at -0.05, 0.78, 1.30, 1.70, 2.10, 2.50, 3.00, 3.50, 4.00,
  5.00, 6.50, 8.00, 11.00, 16.00, 21.00, 31.00 km BSL. Layer velocity =
  slowness-averaged (travel-time preserving) over the interval.
- `nsp=2`: an **independent S model** and independent S station corrections,
  started at Vp/1.78. Vp/Vs(z) is therefore an *output*, which is the
  depth-dependent-Vp/Vs lever Merlin asked for (notes/27 I23).
- S weight `swtfac=0.5` (the VELEST convention). **UNCERTAIN**: not tested for
  sensitivity; with PickBlue S recall (93%) an equal weight would also be
  defensible.
- Pick weight class: `iwt=0` for `prob >= 0.30`, else 1 (weights 1.0 and 0.25).
  **UNCERTAIN**: the probabilities come from two different pickers
  (diting + PickBlue-PhaseNetLight) and are not calibrated against each other.
- `dmax=150 km`; 98.4% of the picks are ZX OBS, only 379 are land-station picks.

### Why a 377-event subset for the inversion

The first attempt inverted all 1,576 events simultaneously. VELEST solves ONE
single-precision normal system of `4*neqs + nlayers + nstacorr` unknowns
(= 6,374 here), and it failed: `runB` stopped at iteration 1 with
`4 TIMES BACKUP MADE`, `runC_slow` died with
`WARNING: error in ludecp     ier=  129`, and NaN trial steps appeared in all
four runs. The published Kissling recipe is to derive the minimum 1D model from a
well-recorded, well-distributed subset and then relocate the full catalogue with
it -- which is what is done below. The subset is depth-stratified to the strict
tier's own depth histogram, because "most observations" alone selects the deeper
events (an unstratified top-400 had depth p10/50/90 2.13/5.63/9.15 vs the strict
tier's 1.49/3.85/7.90).

### Damping

Swept on the subset (12 configurations, `velest/sw_*`). Final accepted rms:
vthet 1 -> 0.214, vthet 3 -> 0.170, vthet 10 -> 0.153, vthet 30 + lowveloclay 0
-> 0.151, vthet 100 -> diverged. Chosen: `othet 0.01, xythet 0.01, zthet 0.01,
vthet 30, stathet 0.01, lowveloclay 0, ittmax 20`. All five inversion runs then
terminate on `4 TIMES BACKUP MADE`, which is VELEST's step-halving convergence
test, not a failure. Example (invB): rms 0.866 -> 0.305 -> 0.205 -> 0.174 ->
0.158 -> 0.154 -> 0.151 -> 0.151.

`lowveloclay=1` was tried and rejected: it let the 0.78-1.30 km layer run to
Vp/Vs 1.23, which is not physical, and the runs were less stable (vthet 30 with
LVLs ended at rms 0.363).

## J4. Runs

Inversion, on the 377-event subset (`velest/inv*`):

| run | start model | free | rms before -> after |
|---|---|---|---|
| invA | start | station corrections only; hypocentres and layers FIXED | 0.8656 -> **0.7037** |
| invB | start | layers, hypocentres, station corrections | 0.8656 -> **0.1509** |
| invC_slow | start x0.90 | as invB | 1.0124 -> **0.1521** |
| invC_fast | start x1.10 | as invB | 0.8086 -> **0.1761** |
| invD_ratio | start | layers + hypocentres, S tied to P by Vp/Vs 1.78 (`nsp=3`) | 0.8659 -> **0.2271** |
| invE_188 | start | as invD but Vp/Vs 1.88 | 0.8842 -> **0.2480** |

Relocation of all 1,576 strict events with each inverted model and its station
corrections held fixed (`velest/all*`, `--freeze-model --nsinv 0 --sta-from`):

| run | rms before -> after (whole set) | per-event rms p50 | depth km BSL p10/50/90 |
|---|---|---|---|
| NLLoc v5 strict (start) | -- | 0.177 | 1.49 / 3.85 / 7.90 |
| allA start model | 0.8341 -> 0.1885 | 0.157 | 1.79 / 3.40 / 6.41 |
| **allB joint** | 0.7036 -> **0.1787** | **0.138** | **1.88 / 3.48 / 6.00** |
| allC_slow | 0.7937 -> 0.1700 | 0.137 | 2.02 / 3.63 / 6.03 |
| allC_fast | 0.6557 -> 0.2007 | 0.140 | 1.77 / 3.32 / 5.86 |
| allD Vp/Vs 1.78 | 0.8427 -> 0.2331 | 0.210 | 1.56 / 3.83 / 7.96 |

`allE_188` (Vp/Vs 1.88) did **not** converge on the full set: it ends in NaN
after `4 TIMES BACKUP MADE`, twice, with and without extra hypocentre damping.
Its best accepted state was rms 0.196. No relocated catalogue is released for it;
the subset inversion invE_188 is reported as a control only.

The iteration-0 rms of 0.70-0.88 s is NOT a statement about NLLoc. It is what a
*flat* 1D model with zero station corrections produces over an array whose
seafloor varies from 0.785 to 1.943 km: the station corrections and the
relocation together take it to 0.18.

QC of the relocated catalogues (depth vs the 30 m Orca bathymetry at the NLLoc
epicentre, `local_water_km`): events ending **above** the local seafloor:
allA 0.0%, allB 0.0%, allC_slow 0.1%, allC_fast 0.3%, allD 0.3%. For comparison
hypoDD 3D left 10-13% above the seafloor (notes/27 I25, I26). Epicentre shift
from NLLoc: median 0.39-0.43 km, p90 0.93-1.04 km. Spearman(z_NLLoc, z_VELEST)
for allB = 0.924.

## J5. THE MINIMUM 1D MODEL (invB / allB)

Sea-level datum, rock only. `velest/invB/final_model.csv`.

| top km BSL | Vp km/s | Vs km/s | Vp/Vs | Vp spread over the 3 starts | Vp/Vs spread |
|---|---|---|---|---|---|
| -0.05 | 2.35 | 1.32 | 1.780 | 20.0% (frozen layer) | 0.007 |
| 0.78 | 2.35 | 1.42 | 1.655 | 20.0% | 0.019 |
| 1.30 | 2.43 | 1.52 | 1.599 | 20.6% | 0.057 |
| 1.70 | 3.17 | 1.89 | 1.677 | 20.5% | 0.053 |
| 2.10 | 4.00 | 2.37 | 1.688 | 20.3% | 0.071 |
| 2.50 | 4.64 | 2.82 | 1.645 | 20.0% | 0.073 |
| 3.00 | 5.20 | 3.19 | 1.630 | 20.2% | 0.084 |
| 3.50 | 5.66 | 3.49 | 1.622 | 21.4% | 0.114 |
| 4.00 | 5.99 | 3.61 | 1.659 | 20.4% | 0.085 |
| 5.00 | 6.26 | 3.62 | 1.729 | 17.6% | 0.062 |
| 6.50 | 6.50 | 3.66 | 1.776 | 15.1% | 0.027 |
| 8.00 | 6.59 | 3.71 | 1.776 | 15.6% | 0.006 |
| 11.00 | 6.61 | 3.72 | 1.777 | 15.9% | 0.007 |
| 16.00 | 6.76 | 3.75 | 1.803 | 19.2% | 0.034 |
| 21.00 | 6.89 | 3.87 | 1.780 | 20.0% (frozen) | 0.004 |
| 31.00 | 7.00 | 3.93 | 1.781 | 20.0% (frozen) | 0.003 |

"Spread" = (max - min) over invB, invC_slow, invC_fast, i.e. the starts at x1.00,
x0.90 and x1.10, expressed as a percentage of invB's Vp.

**The absolute Vp is not resolved.** The +-10% perturbation survives the
inversion almost untouched (Vp at 4 km BSL: 5.39 / 5.99 / 6.61 km/s from the
slow / start / fast starts, i.e. 15-21% spread from a 20% spread of starts),
and the models trade against the station corrections
(invC_slow OBS P corrections -1.23 .. +0.15 s, invC_fast -0.05 .. +0.98 s, invB
-0.21 .. +0.19 s). The classic velocity / origin-time / station-correction
degeneracy is fully present. This is why no formal per-layer uncertainty is
quoted: the spread across starts is the honest uncertainty, and it is +-10%.

**Vp/Vs IS resolved**, and it is the interesting output: 1.60-1.70 over
1.3-5 km BSL against a start of 1.78, agreeing to within 0.02-0.11 across all
three starts. `iresolcalc` was not run (O(n^3) on 6,374 unknowns in single
precision); the cross-start spread is used instead.

**UNCERTAIN / a warning about that model:** Vp/Vs of 1.60-1.65 in the top 5 km of
a young rift basin is on the low side of anything I would publish (Poisson's
ratio ~0.20), and it *disagrees with the Wadati estimates in this project*
(global 1.795, near-path 1.86-1.90, notes/27 I11/I13). See J7 -- the inversion is
almost certainly buying part of this with a station term it cannot otherwise
represent.

## J6. Station corrections (invB, `velest/invB/station_corrections.csv`)

Reference station B22 = ZX.BRA22, P correction held at 0 by construction (VELEST
frees the reference station's S correction; it came out +0.33).

Range over all 38 stations: **P -0.54 .. +0.19 s, S -0.65 .. +1.25 s**.
Demeaned over the 22 OBS with data: P -0.266 .. +0.134 s (sd 0.100),
S -0.384 .. +0.656 s (sd 0.300).

OBS (ZX), P then S, in seconds:

```
B02 -0.03 +0.78   B03 -0.21 +0.47   B04 +0.01 +0.95   B05 -0.07 +0.22
B08 -0.09 +1.25   B09 +0.15 +1.00   B10 +0.17 +1.18   B11 +0.19 +0.36
B13 +0.09 +0.82   B14 +0.14 +0.64   B15 +0.07 +0.80   B16 +0.05 +0.47
B18 +0.06 +0.34   B19 -0.01 +0.21   B20 +0.06 +0.43   B21 +0.09 +0.58
B22 +0.00 +0.33   B23 +0.05 +0.41   B24 +0.15 +0.54   B25 +0.02 +0.28
B26 +0.17 +0.46   B27 +0.17 +0.54
```

Land, P then S: DCP -0.35/0.00, LVN -0.38/-0.38, AST -0.46/-0.31, BYE
-0.07/-0.09, ERJ -0.54/-0.10, FER -0.27/-0.30, FRE 0.00/-0.47, GUR -0.35/-0.40,
HMI -0.47/-0.65, OHI -0.50/+0.06, PEN -0.09/-0.04, ROB +0.01/+0.10, SNW
+0.06/-0.02, TOW +0.07/+0.63, JUBA -0.38/-0.45, R4DE -0.33/+0.37.

Two things stand out:

1. The OBS P corrections correlate with water depth, r = +0.71 (invA, where the
   hypocentres were held fixed, gives r = +0.76). That is exactly the signature
   expected of a flat 1D model over 0.785-1.943 km of seafloor relief: the deep
   stations (B09 1943 m +0.15, B10 1639 m +0.17) need positive corrections, the
   shallowest (B03 785 m) needs -0.21.
2. **Every single OBS has a large POSITIVE S correction (+0.21 .. +1.25 s) while
   its P correction is small (-0.21 .. +0.19 s).** There is a station-side S
   delay of roughly half a second under the whole array that P does not see.
   This is the central observation of this note.

## J7. What the data actually say about near-station S-P (bin-free, no relocation)

Regressing the observed nearest-station (<4 km) S-P on hypocentral distance:

| catalogue used for the geometry | slope d(S-P)/dR | intercept at R=0 | corr |
|---|---|---|---|
| NLLoc v5 strict | 0.0630 s/km | +0.539 s | 0.444 |
| VELEST allB | 0.1052 s/km | +0.439 s | 0.471 |

For any plausible crustal Vp/Vs the slope should be `(Vp/Vs - 1)/Vp` = 0.29 s/km
at Vp 3, 0.176 at Vp 5, 0.147 at Vp 6. The observed 0.06-0.11 s/km is far too
small, and there is a **+0.44 to +0.54 s offset at zero distance**. The
near-station S-P in this data set is dominated by a constant station term, not by
path length. That is the same number the S station corrections found
independently (J6, +0.21..+1.25 s at every OBS), and it is the classic
soft-sediment low-Vs delay under an OBS.

So the joint inversion has two ways to represent the same thing: an S station
delay, or a low Vp/Vs in the layers. It uses both, which is why the Vp/Vs in J5
is implausibly low.

## J8. THE GATE

`52_sp_depth_check.py --frame sealevel --scorer-vpvs 1.88`, driven by
`62_velest_gate.py`. Full bin tables in `velest/sp_gate_bins.csv`.

Script 52's own ratio uses the FIRST and LAST populated depth bin. On the strict
tier those are `(0,1]` with n=7 and `(12,40]` with n=2, and on the relocated
catalogues `(8,12]` falls to n=8. I therefore report three versions: 52's raw
edge-bin ratio, the `(1,2] -> (8,12]` pair the 1.25-1.3 figure in notes/27 was
built on, and a bin-free slope ratio that uses every event.

```
=== S-P spread gate (scorer Vp/Vs 1.88) ===
             catalogue      n     obs    pred   ratio  |misfit|   misfit
                                        (1,2]->(8,12]
       NLLoc_v5_strict  1,529    1.87    2.41    1.29     0.230   -0.019
     VELEST_A_startmod  1,528    3.81    2.29    0.60     0.196   -0.096
        VELEST_B_joint  1,530    4.90    2.28    0.46     0.188   -0.103
         VELEST_C_slow  1,530    5.03    2.38    0.47     0.178   -0.091
         VELEST_C_fast  1,529    3.79    2.33    0.62     0.201   -0.124
      VELEST_D_vpvs178  1,527    1.67    2.30    1.38     0.226   +0.002

  (52's own edge-bin ratio: NLLoc_v5_strict 2.20; VELEST_A_startmod 0.60;
   VELEST_B_joint 0.47; VELEST_C_slow 1.16; VELEST_C_fast 0.20;
   VELEST_D_vpvs178 2.45)

=== bin-free stretch metric: d(S-P)/dz over 1-8 km BSL (scorer Vp/Vs 1.88) ===
             catalogue      n  obs s/km  pred s/km   ratio  |misfit|   misfit
       NLLoc_v5_strict  1,402    0.0586     0.0958    1.64     0.230   -0.019
     VELEST_A_startmod  1,499    0.0902     0.0961    1.07     0.196   -0.096
        VELEST_B_joint  1,522    0.1005     0.0954    0.95     0.188   -0.103
         VELEST_C_slow  1,520    0.1058     0.0968    0.91     0.178   -0.091
         VELEST_C_fast  1,510    0.0926     0.0930    1.00     0.201   -0.124
      VELEST_D_vpvs178  1,404    0.0586     0.0930    1.59     0.226   +0.002
```

The `(1,2]->(8,12]` ratio of 1.29 for NLLoc v5 strict reproduces the
1.25-1.31 of notes/27 I24/I27 on a different population -- the metric is
consistent with the earlier work.

Depth change, allB vs NLLoc: `dz = -0.276 * z_NLLoc + 0.624 km`. By NLLoc depth
bin, median dz: 0-2 km **+0.44**, 2-4 km -0.01, 4-6 km **-0.97**,
6-9 km **-1.99**, 9-15 km -2.37, 15-40 km -0.79. Slopes for the other runs:
allA -0.227, allC_slow -0.198, allC_fast -0.363, **allD (Vp/Vs locked at 1.78)
+0.070**. For comparison, hypoDD 1D gave -0.234 and hypoDD 3D -0.171/-0.176
(notes/27 I18, I25, I26).

### VERDICT

**The gate is passed on the bin-free form and on rms; it overshoots on script
52's bin-edge form. The honest verdict is: YES, the joint inversion removes the
depth stretch -- but it does so through the S model, not through the P model, and
the fix it chooses is not unique.**

Stated precisely:

1. **rms drops, clearly.** Per-event rms p50 0.177 -> 0.138 s; whole-set rms
   0.704 -> 0.179 s; median |S-P misfit| 0.230 -> 0.188 s.
2. **The stretch metric goes to ~1.0 from all three starts.** Bin-free ratio
   1.64 (NLLoc) -> 0.95 / 0.91 / 1.00 (start / x0.90 / x1.10). It does not
   merely move toward 1.0, it lands on it, and it does so consistently from
   +-10% starting models.
3. **Script 52's own ratio overshoots** (1.29 -> 0.46) because its end bins
   collapse to n=8 after relocation. I do not think the overshoot is real; the
   bin-free number on the same events is 0.95. Reported for completeness, not
   used for the verdict.
4. **The lever is S, not P.** The control run allD -- P layers free, P station
   corrections free, but S locked to P x 1.78 -- reproduces the NLLoc depth
   distribution almost exactly (1.56/3.83/7.96 vs 1.49/3.85/7.90, dz slope
   +0.070, stretch ratio 1.59) and fits worse (rms p50 0.210). Vp/Vs 1.88
   (invE_188) fits worse still (subset rms 0.248 vs 0.227 at 1.78 vs 0.151
   free-by-layer) and would not converge on the full set. Every run that is
   allowed to decouple S from P x 1.78 -- whether through per-layer Vp/Vs (allB,
   allC) or merely through independent S *station corrections* with the layers
   frozen (allA, dz slope -0.227, ratio 1.07) -- compresses the depth axis by
   20-28%. Every run that is not, does not.
5. **The absolute velocity is not resolved** (J5): +-10% in, +-10% out, absorbed
   by the station corrections. The minimum 1D model in J5 should be used for
   *relative* structure and for relocation, not quoted as an absolute crustal
   velocity profile.

Read together with J7, the physical statement is: the excess near-station S-P
that made the NLLoc depth axis look stretched is mostly a **constant
station-side S delay of ~0.5 s under every OBS** (soft sediment, very low Vs in
the first few hundred metres below the seafloor), not depth. NLLoc, which
computes S as P x 1.78 through the LOCMETH ratio with no S station term, has no
way to represent that delay and pays for it in depth. hypoDD's 20% compression
and this inversion's 28% are the same signal seen two ways.

### What this does NOT settle

- The S-P test is not independent of VELEST: VELEST fits the same S picks that
  script 52 scores. NLLoc fits them too, so the comparison is fair, but neither
  side is an out-of-sample check. An independent check would need S arrivals not
  used in the location (or a shot/airgun calibration).
- Whether the ~0.5 s S delay is sediment Vs or a systematic S *pick* bias
  (picking a converted or later phase at the OBS) is not resolved here. Both
  produce exactly this signature. This should be settled by eye on a handful of
  near events before any of this goes in the paper -- the `seismologist-pick`
  workflow on 10-20 events with a station inside 3 km would do it.
- Vp/Vs 1.60-1.65 in the top 5 km is a fitting parameter here, not a
  measurement. Do not quote it.

## J9. Files

Scripts (new, nothing existing was edited):
- `scripts/62_velest_prep.py` -- builds `.cnv` / `.sta` / `.mod` / control
  template; derives the strict tier; selects and depth-stratifies the inversion
  subset.
- `scripts/62_run_velest.py` -- stages one run directory, writes `velest.cmn`,
  executes; `--freeze-hypo`, `--freeze-model`, `--model-from`, `--sta-from`,
  `--nsinv`, `--nsp`, `--vpvs`, `--lowveloclay`.
- `scripts/62_velest_post.py` -- parses `velest.OUT` / `velout.mod` /
  `final.STA` into `relocated.csv`, `final_model.csv`,
  `station_corrections.csv`, `rms_trajectory.csv`.
- `scripts/62_velest_gate.py` -- drives `52_sp_depth_check.py` over a set of
  catalogues and adds the bin-free slope metric.

Outputs (under `velest/`, which was added to `.gitignore` -- it was NOT ignored
before; `git check-ignore -v velest/x` now returns
`.gitignore:86:velest/	velest/x`):
- `velest/input/` -- `orca.cnv` (1,576), `orca_sub.cnv` (377), `orca.sta`,
  `orca.mod`, `orca_slow.mod`, `orca_fast.mod`, `event_order*.csv`,
  `nlloc_v5_strict.csv`, `meta.json`, `cmn_template.txt`
- `velest/invA invB invC_slow invC_fast invD_ratio invE_188/` -- inversions
- `velest/allA allB allC_slow allC_fast allD_ratio/` -- full-catalogue
  relocations; `velest/allB/relocated.csv` is the deliverable (1,576 events,
  `depth_km` below sea level, plus `lat0/lon0/depth0_km/dz_km` from NLLoc)
- `velest/sw_*/` -- the damping sweep
- `velest/sp_gate.csv`, `sp_gate_bins.csv`, `sp_gate_slope.csv`
- Binary and source: `~/src/velest/orca/velest`, `~/src/velest/orca_src/`

Nothing under `nlloc/`, `hypodd/` or `catalogs/` was written. No commit was made.
