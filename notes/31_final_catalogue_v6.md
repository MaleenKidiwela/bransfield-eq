# Validation package for the final catalogue — NLLoc v6 (2026-09-14)

Append-only. Every number below is pasted from the output of a script in this session;
nothing is carried over from an earlier note without being recomputed. Scripts written for
this package: `scripts/68_manual_sp_vs_delays.py`, `scripts/68_perturb_delays.py`,
`scripts/68_delay_depth_uncertainty.py`, `scripts/68_catalogue_accounting.py`. No existing
script, note or catalogue was modified. Catalogue release notes:
`catalogs/README_nlloc_year_v6.md` (§4 below).

Product under test: `catalogs/nlloc_year_v6.csv` (79,723 rows) and its QC tiers; ORCA_v5
pykonal travel-time grids, per-station P/S LOCDELAY terms `nlloc/delays/v6_it1B.delays`,
S-before-P guard `scripts/66_guard_obs_for_delays.py`, control `nlloc/run/year_v6.in`.

---

## 1. Do the analyst's own picks support the station S delays?

`scripts/68_manual_sp_vs_delays.py`. 46,339 manual picks → **17,470 event/station pairs with
both P and S** (17,347 of them on ZX OBS). Manual events are matched to pyocto events by
origin time ±3 s (2,622 of 5,831 matched, 45.0%, |ΔOT| p50 1.213 s), which supplies the
automatic S−P on the same event/station pair and the v6 hypocentre for the geometry check.
Manual pick times already carry the BRA05 +0.167 s clock correction, so both sides are in
the same frame as the picks NLLoc used.

`sp_delay` = (S delay − P delay) for that station from `v6_it1B.delays`; NLLoc subtracts it
from every observed S−P at that station.

| sta | n | n(S−P<0) | man min | man p1 | man p5 | man p50 | n auto | auto p5 | auto p50 | auto−man p50 | **S−P delay** | p5 − delay | % below delay | % ≤ delay+0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BRA26 | 4497 | 0 | 0.070 | 0.480 | 0.640 | 0.790 | 1603 | 0.670 | 0.830 | +0.040 | 0.289 | +0.351 | 0.33% | 0.38% |
| BRA22 | 4204 | 2 | −0.140 | 0.480 | 0.630 | 0.890 | 1690 | 0.730 | 0.940 | +0.020 | 0.292 | +0.338 | 0.19% | 0.33% |
| BRA25 | 2093 | 1 | −0.080 | 0.426 | 0.716 | 0.910 | 1048 | 0.760 | 0.960 | +0.050 | 0.259 | +0.457 | 0.48% | 0.53% |
| BRA21 | 2016 | 0 | 0.200 | 0.473 | 0.790 | 1.230 | 1041 | 0.960 | 1.280 | +0.050 | 0.422 | +0.368 | 0.74% | 1.04% |
| BRA23 | 1253 | 0 | 0.130 | 0.376 | 0.760 | 1.080 | 574 | 0.837 | 1.090 | +0.020 | 0.235 | +0.525 | 0.08% | 0.40% |
| BRA24 | 734 | 0 | 0.200 | 0.323 | 0.700 | 1.210 | 386 | 0.826 | 1.180 | −0.010 | 0.322 | +0.378 | 1.09% | 1.91% |
| BRA27 | 682 | 2 | −20.010 | 0.382 | 0.620 | 0.880 | 310 | 0.708 | 0.920 | +0.052 | 0.292 | +0.328 | 0.59% | 0.88% |
| BRA19 | 646 | 1 | −0.280 | 0.359 | 0.480 | 0.840 | 333 | 0.626 | 0.890 | +0.026 | 0.283 | +0.197 | 0.31% | 0.77% |
| BRA20 | 338 | 0 | 0.010 | 0.341 | 0.778 | 1.000 | 166 | 0.871 | 1.000 | +0.010 | 0.275 | +0.504 | 1.18% | 1.18% |
| BRA18 | 325 | 0 | 0.320 | 0.432 | 0.614 | 1.070 | 155 | 0.687 | 1.006 | +0.006 | 0.300 | +0.314 | 0.00% | 0.62% |
| BRA16 | 172 | 0 | 0.330 | 0.394 | 0.695 | 1.115 | 61 | 0.866 | 1.070 | +0.030 | 0.494 | +0.202 | 3.49% | 4.07% |
| BRA15 | 136 | 0 | 0.110 | 0.467 | 0.880 | 1.585 | 67 | 1.127 | 1.640 | +0.010 | 0.503 | +0.377 | 1.47% | 1.47% |
| BRA13 | 133 | 0 | 0.290 | 0.332 | 0.652 | 1.670 | 39 | 1.251 | 1.590 | +0.050 | 0.578 | +0.074 | 4.51% | 5.26% |
| BRA05 | 51 | 0 | 0.570 | 0.585 | 0.615 | 1.080 | 30 | 0.674 | 1.045 | +0.030 | 0.168 | +0.447 | 0.00% | 0.00% |
| BRA14 | 35 | 0 | 0.310 | 0.358 | 0.625 | 1.360 | 10 | 0.643 | 1.293 | −0.112 | 0.507 | +0.118 | 5.71% | 5.71% |

(seconds; stations with ≥ 20 manual P+S pairs. The BRA27 minimum of −20.010 s is one bad
analyst pair, not a measurement — 6 of 17,347 ZX pairs, 0.03%, have S before P.)

Automatic vs manual on the same pairs: **auto − manual S−P median +0.030 s, MAD 0.070 s**
over 7,515 ZX pairs, and per station between −0.112 and +0.052 s. The automatic picks are
therefore not inflating S−P relative to the analyst; the two agree to well under the size of
the station terms.

Aggregate: **83 of 17,347 ZX manual pairs (0.5%)** measure an S−P smaller than their own
station's S−P delay; 117 (0.7%) fall at or below the script-66 guard threshold
(delay + 0.05 s).

Geometry of those 83 (45 of them have a v6 hypocentre, so this part is a small sample):

| | below-delay pairs | all ZX pairs with a hypocentre |
|---|---|---|
| n | 45 | 9,971 |
| epicentral distance p50 | 3.57 km | 2.75 km |
| hypocentral distance p50 / p90 | 4.00 / 8.29 km | 3.87 / 7.71 km |
| source depth p50 (BSL) | 1.06 km | 3.20 km |
| manual S−P p50 | 0.250 s | 0.950 s |

60% of them are inside 4 km epicentral distance and 98% inside 10 km.

### Verdict

**The station S−P delays are supported by the analyst's picks, but not by the mechanism
previously assumed.**

- Supported: at *every* OBS the analyst's 5th-percentile S−P sits **above** the station's S−P
  delay, by +0.07 s (BRA13, the largest term) to +0.53 s (BRA23), and only 0.5% of 17,347
  hand-measured S−P values fall below their station's delay. A term that was an artefact
  would be contradicted routinely by hand picks; it is not. The margin is smallest exactly
  where the term is largest (BRA13 0.578 s delay, p5 − delay +0.074; BRA14 0.507, +0.118;
  BRA16 0.494, +0.202), which is the expected place for it to be tight.
- Not supported: the *explanation* that sub-delay measurements are near-vertical events
  directly under the station. They are **not** closer than the rest — hypocentral distance
  p50 4.00 km vs 3.87 km for the whole population — they are **shallower** (1.06 vs 3.20 km)
  with anomalously small S−P (0.250 vs 0.950 s) at ordinary distance. The honest reading is
  station/path variability that a single constant term cannot represent, which the guard then
  deletes. The residual risk is that the guard removes real, depth-informative short S−P at a
  rate of ~0.5–0.7% of near-station S pairs (the year figure is larger, §3, because the year
  pool reaches closer distances than the analyst's set).

---

## 2. Depth uncertainty contributed by the station terms

`scripts/68_perturb_delays.py` → `scripts/66_guard_obs_for_delays.py` → NLLoc → 
`scripts/68_delay_depth_uncertainty.py`. Three perturbed delay sets, each guarded with its
own delays and relocated on the same 2,000-event sample; controls
`nlloc/run/abtest_v6_{pertA,pertS,pert12}.in` differ from the reference
`nlloc/run/abtest_v6_it1Bg.in` **only** in the LOCDELAY block and the obs/output paths
(verified by diff). Events matched to the reference by hyp filename, as `67_gate_table.py` does.

| set | construction | OBS S−P delay p50 | S picks dropped by the guard |
|---|---|---|---|
| **it1B** (production) | VELEST invB + 1 NLLoc residual refinement | +0.372 s | 407 in 299 events (15.0%) |
| **pertA** | VELEST **invA** P and S corrections, mapped exactly as `65_make_locdelay.py` maps them | +0.125 s | 245 in 198 events (9.9%) |
| **pertS** | it1B P terms, S term = P + (invA S−P difference) — isolates the S−P term | +0.125 s | 243 in 196 events (9.8%) |
| **pert12** | it1B with S−P scaled **×1.2** (P terms unchanged) | +0.446 s | 463 in 330 events (16.5%) |

Where each perturbation sits against the S−P spread gate (`scripts/67_gate_table.py`,
`--ref abtest_ORCA_v5`, each run scored with its own delays):

| run | located | pinned | near-P resid (pinned) | rms p50 | S−P pred/obs | **ratio** | z p10/50/90 |
|---|---|---|---|---|---|---|---|
| abtest_ORCA_v5 (no terms) | 1994 | 28.5% | −0.160 | 0.239 | 3.51/2.80 | 1.25 | 0.00/1.20/19.22 |
| **abtest_v6_it1Bg** (production) | 1992 | 20.1% | −0.129 | 0.238 | 2.94/3.09 | **0.95** | 0.00/1.77/15.75 |
| abtest_v6_pertA | 1998 | 26.1% | −0.165 | 0.258 | 3.09/2.60 | **1.19** | 0.00/1.57/18.22 |
| abtest_v6_pertS | 1994 | 25.4% | −0.149 | 0.240 | 3.18/2.83 | **1.12** | 0.00/1.40/17.56 |
| abtest_v6_pert12 | 1993 | 19.1% | −0.132 | 0.241 | 3.08/3.35 | **0.92** | 0.00/1.93/15.44 |

So the perturbation set brackets the production choice: the invA-scale terms leave the depth
axis measurably stretched (ratio 1.12–1.19, and rms is worse for pertA) and are **rejected**
by the gate that selected it1B; ×1.2 slightly over-corrects (0.92) and is **admissible**.

### Δz vs the production run, by reference (it1Bg) depth bin — all located events

median |Δz| / p90 |Δz| / median signed Δz, km:

| ref depth bin | n | pert12 | pertA | pertS |
|---|---|---|---|---|
| 0–1 | 758 | 0.00 / 0.20 / +0.00 | 0.03 / 0.63 / +0.00 | 0.04 / 0.56 / +0.00 |
| 1–2 | 292 | 0.12 / 0.30 / +0.09 | 0.43 / 1.30 / −0.37 | 0.44 / 0.88 / −0.42 |
| 2–4 | 375 | 0.11 / 0.40 / +0.00 | 0.59 / 1.72 / −0.03 | 0.52 / 1.55 / −0.12 |
| 4–6 | 189 | 0.23 / 0.57 / −0.15 | 1.00 / 2.23 / +0.68 | 0.95 / 2.03 / +0.65 |
| 6–9 | 116 | 0.24 / 0.72 / −0.16 | 1.00 / 3.86 / +0.55 | 0.96 / 2.68 / +0.41 |
| 9–15 | 54 | 0.41 / 0.84 / −0.29 | 1.56 / 3.21 / +1.34 | 1.55 / 3.24 / +1.36 |
| 0.05–1 (unpinned only) | 358 | 0.15 / 0.28 / +0.13 | 0.34 / 0.87 / −0.25 | 0.34 / 0.65 / −0.27 |
| ALL | 1990 | 0.08 / 0.43 / +0.00 | 0.37 / 1.91 / +0.00 | 0.38 / 1.63 / +0.00 |

### Same, restricted to standard-tier-like events (gap < 180, rms < 0.5, Nphs ≥ 6)

| ref depth bin | n | pert12 | pertA | pertS |
|---|---|---|---|---|
| 0–1 | 137 | 0.08 / 0.20 / +0.07 | 0.22 / 0.69 / −0.08 | 0.25 / 0.64 / −0.14 |
| 1–2 | 91 | 0.10 / 0.28 / +0.09 | 0.45 / 1.22 / −0.36 | 0.46 / 0.91 / −0.45 |
| 2–4 | 165 | 0.12 / 0.39 / −0.02 | 0.84 / 1.97 / +0.55 | 0.76 / 1.74 / +0.23 |
| 4–6 | 76 | 0.25 / 0.59 / −0.23 | 1.28 / 2.30 / +1.10 | 1.25 / 2.24 / +1.09 |
| 6–9 | 30 | 0.27 / 0.62 / −0.27 | 1.20 / 2.13 / +1.16 | 1.05 / 1.92 / +1.05 |
| 9–15 | 7 | 0.48 / 0.70 / −0.48 | 1.81 / 2.61 / +1.81 | 1.60 / 2.41 / +1.60 |
| ALL | 518 | 0.13 / 0.45 / +0.00 | 0.57 / 1.96 / +0.00 | 0.56 / 1.76 / +0.00 |

Envelope (per event, the largest |Δz| over the three perturbations), all located:

| ref depth bin | n | median env | p90 env |
|---|---|---|---|
| 0–1 | 758 | 0.10 | 0.77 |
| 1–2 | 292 | 0.48 | 1.36 |
| 2–4 | 375 | 0.70 | 1.87 |
| 4–6 | 190 | 1.16 | 2.36 |
| 6–9 | 116 | 1.44 | 4.94 |
| 9–15 | 54 | 1.82 | 3.85 |
| ALL | 1992 | 0.45 | 2.20 |

### NLLoc formal σz in the year catalogue (p10 / p50 / p90, km)

| tier | n | σz | σx | σy | depth BSL | depth BSF |
|---|---|---|---|---|---|---|
| loose | 15,184 | 0.27 / 0.68 / 1.33 | 0.22 / 0.46 / 0.92 | 0.22 / 0.45 / 1.05 | 1.13 / 2.68 / 6.01 | −0.01 / 1.64 / 4.90 |
| **standard** | 12,194 | **0.29 / 0.69 / 1.24** | 0.22 / 0.46 / 0.88 | 0.22 / 0.44 / 0.94 | 1.13 / 2.80 / 5.90 | 0.02 / 1.81 / 4.78 |
| **strict** | 3,864 | **0.42 / 0.71 / 1.08** | 0.26 / 0.46 / 0.73 | 0.26 / 0.40 / 0.68 | 1.66 / 3.20 / 5.22 | 0.75 / 2.29 / 4.30 |

### Reading

- Inside the family the S−P gate still accepts (pert12), the station terms contribute a
  median depth systematic of **≤ 0.12 km above 4 km, 0.23–0.41 km below it** (p90 0.20–0.84 km).
  That is comparable to, or smaller than, the formal σz at all depths.
- Taking the gate-rejected invA-scale terms as an outer bound, the systematic is
  **0.03–0.59 km above 4 km and 0.95–1.56 km below it**, with p90 up to 3.9 km in the 6–9 km
  bin. Below ~4 km this **exceeds the formal σz p90 of 1.1–1.3 km**, i.e. the NLLoc
  uncertainty under-states the true depth uncertainty for the deep part of the catalogue.
- The response is systematically depth-graded and sign-consistent: weaker S−P terms push the
  1–2 km population down (−0.37 / −0.42 km) and the > 4 km population up (+0.4 … +1.4 km);
  a stronger term does the reverse. That is the same shallow-down/deep-up lever documented in
  notes/27 I30–I33, now measured as an uncertainty rather than a tuning direction.
- The 0–1 km bin is dominated by top-face-pinned events (20.1% of the reference run sits at
  z < 0.05 km), whose depth cannot move downward; the unpinned 0.05–1 km row is the honest
  shallow number.

### Side finding (not fixed — script 67 is not mine to change)

`67_gate_table.py` builds its hyp-filename → `event_idx` map by zipping the **sorted hyp
filenames of `--ref`** against the first 2,000 rows of the obs order. That is a positional
join and it is silently wrong whenever the reference run lost an event: `abtest_v6_it1Bg`
wrote **1,999** `loc.2*.grid0.loc.hyp` files for 2,000 obs events (one event still aborts
inside NLLoc despite the guard), so running `--ref abtest_v6_it1Bg` shifts every event after
the gap and reports the reference's own S−P ratio as **1.21** instead of the correct
**0.95**. With `--ref abtest_ORCA_v5` (2,000 files) the table reproduces I33 exactly. All
numbers in this note use `--ref abtest_ORCA_v5`; `scripts/68_delay_depth_uncertainty.py`
matches on the filename itself and is unaffected. This is the same failure mode as G11/I34
and the 2026-05 v2 join bug.

---

## 3. Accounting: 79,783 associated events → each tier

`scripts/68_catalogue_accounting.py`, counted from the obs blocks, the hyp files and the
catalogue itself.

| stage | events | picks |
|---|---|---|
| associated (year, new pool) | 98,631 | |
| − airgun window 2019-01-21 → 2019-02-05 | −18,848 | |
| **into NLLoc (`year_v6.obs`)** | **79,783** | 798,627 (P 409,090 / S 389,537) |
| (before the guard, `year_v5.obs`) | 79,783 | 818,767 (P 409,090 / S 409,677) |
| − S picks dropped by the script-66 guard | 0 events deleted | **−20,140 S (4.9% of all S)** in 13,979 events (17.5%) |
| events left with no S pick at all | 141 | |
| per-event `.hyp` files written | 79,733 | |
| − obs events NLLoc could not locate | **−50** (`nlloc/output/year_v6/unmatched_obs_events.txt`) | |
| − `.hyp` files with no parsable GEOGRAPHIC line | −10 | |
| **rows in `catalogs/nlloc_year_v6.csv`** | **79,723** | |
| − `nlloc_status == REJECTED` | −220 | |
| **LOCATED solutions** | **79,503** (99.6% of 79,783) | |

Tiers (cuts of `40_filter_nlloc_reliable.py`, recomputed from the full catalogue; the
recomputed standard count **matches the written file exactly, 12,194 = 12,194**):

| tier | n | % of LOCATED | % of the 79,783 associated |
|---|---|---|---|
| loose | 15,184 | 19.1% | 19.0% |
| standard | 12,194 | 15.3% | 15.3% |
| strict | 3,864 | 4.9% | 4.8% |

Where the other 84.7% goes — each cut applied alone to the 79,503 LOCATED solutions:

| cut | survivors | | cut | survivors |
|---|---|---|---|---|
| not grid-boundary-pinned | 40,565 (51.0%) | | gap < 180° | 18,640 (23.4%) |
| `depth_bsf_km` > −0.2 km | 53,063 (66.7%) | | gap < 120° | 7,011 (8.8%) |
| gap < 200° | 23,554 (29.6%) | | rms < 0.5 s | 69,251 (87.1%) |
| rms < 0.7 s | 73,390 (92.3%) | | rms < 0.3 s | 46,749 (58.8%) |
| Nphs ≥ 4 | 79,503 (100.0%) | | Nphs ≥ 8 | 68,564 (86.2%) |
| Nphs ≥ 6 | 78,512 (98.8%) | | σz < 2 km | 50,483 (63.5%) |
| inside the ZX hull | 44,912 (56.5%) | | `depth_bsf_km` > +0.2 km | 38,916 (48.9%) |

The binding constraints are **azimuthal gap** (only 29.6% under 200°, 8.8% under 120°) and
**grid-face pinning** (49.0% pinned) — not pick count and not rms. The array is small and
most of the 79,783 associated events are outside it.

### Strict tier and near-station S

| | |
|---|---|
| strict events | 3,864 |
| with any station inside 4 km | 3,826 (99.0%) |
| **with a station inside 4 km carrying BOTH P and S** | **3,750 (97.0%)** |
| nearest-station epicentral distance p10/50/90 | 0.63 / 1.18 / 1.58 km |

So the strict tier is almost entirely made of events whose depth is constrained by a
near-station S−P, which is the observable the station terms act on. Cross-check: the
catalogue's own `dist_km` column on the strict tier gives p10/50/90 0.63/1.18/1.58 km, i.e.
identical to the independent great-circle computation — `dist_km` is the nearest-station
**epicentral** distance.

---

## 4. Catalogue README

Written as `catalogs/README_nlloc_year_v6.md`: which file to use for what, column-by-column
definitions read from `scripts/31_nlloc_hyp_to_catalog.py` and
`scripts/40_filter_nlloc_reliable.py`, the three tier definitions, datum and frame
conventions (depth below **sea level**, `depth_bsf_km` below the **local** seafloor, NLLoc
`TRANS SIMPLE -62.4413 -58.44 36` vs pyocto's different projection origin, BRA05 +0.167 s),
and nine caveats — the ~4 km-below-seafloor resolution limit of the P tomography, the station
delays as a fitted quantity with the §2 uncertainty attached, the 20,140 guard-dropped S
picks, the 50 unlocated events + 220 rejected + 10 unparsable, the excluded airgun window,
association (not picking) as the completeness bottleneck, the residual depth-axis stretch,
no magnitudes, and no dt.cc.

Files referenced there that exist as of this note: `catalogs/nlloc_year_v6.csv`,
`catalogs/nlloc_year_v6_standard.csv`, `catalogs/hypodd_year_v6_3d.csv` (10,664 relocated)
and `catalogs/hypodd_year_v6_3d_qc.csv` (**9,125** QC-pass, finished 02:01 today).
`nlloc_year_v6_loose.csv` and `nlloc_year_v6_strict.csv` have **not** been written yet; the
README says so and gives the deterministic command that produces them, with the counts
verified here (15,184 / 3,864).

hypoDD v6 3D, joined to the v6 standard tier on `event_idx` (= hypoDD `id` − 1; all 9,125
join): slope of (DD − start) on start depth **−0.275**, intercept +0.840 km, median |Δz|
0.51 km, Spearman 0.787. Median Δz by NLLoc-v6 start bin: 0–2 km **+0.21**, 2–4 **+0.12**,
4–6 **−0.47**, 6–9 **−1.96**, 9–15 **−0.42** km. The v6 station terms have already removed
much of the v4/v5 compression, but the 6–9 km population still moves up by ~2 km under
double-difference — consistent with §2, where the same bin is the least stable against the
station-term perturbation.

Depth fractions beyond the resolved part of the P model:

| tier | > 4 km below local seafloor | > 6 km below sea level | rms p50 | gap p50 | Nphs p50 |
|---|---|---|---|---|---|
| loose | 15.1% | 10.2% | 0.202 s | 139° | 11 |
| standard | 14.8% | 9.4% | 0.189 s | 127° | 11 |
| strict | 12.7% | 5.6% | 0.142 s | 99° | 12 |

---

## 5. Artefacts produced

- `scripts/68_manual_sp_vs_delays.py`, `scripts/68_perturb_delays.py`,
  `scripts/68_delay_depth_uncertainty.py`, `scripts/68_catalogue_accounting.py`
- `nlloc/delays/v6_pertA_invA.delays`, `v6_pertS_invAsp.delays`, `v6_pert12_sp1.2.delays`
- `nlloc/obs/abgrid_test_guard_{pertA,pertS,pert12}.obs`
- `nlloc/run/abtest_v6_{pertA,pertS,pert12}.in`, outputs `nlloc/output/abtest_v6_{pertA,pertS,pert12}/`
- `nlloc/output/manual_sp_pairs_v6.csv` (the 17,470 manual P+S pairs with their automatic
  twin, station delay, hypocentre and epicentral distance)
- `catalogs/README_nlloc_year_v6.md`

Nothing was committed.
