# 30. Waveform cross-correlation -> hypoDD `dt.cc` on the v5 3D tier (2026-09-13)

Append-only log. Every number below is pasted from a command's own output.

New files: `scripts/63_xcorr_dtcc.py`, `scripts/64_run_hypodd_xc.sh`,
`hypodd/year_v5_3d_xc/`. Nothing under `growclust/`, no earlier `dt.cc` and no
`pick_windows*.npy` was read; nothing in `hypodd/year_v5_3d/` was modified (it is
the catalogue-only baseline and its numbers are quoted from its own `run24.log`).

## A. Inputs and the id mapping (read, not guessed)

`scripts/22_pyocto_to_hypodd_input.py` sets

    ev["hypodd_id"] = (ev["event_idx"].astype(int) + 1)

so **hypoDD id = pyocto `event_idx` + 1**. That is the only mapping used.

`hypodd/year_v5_3d/`: 326,280 pair headers in `dt.ct`, 2,987,937 observation
rows (P 1,374,777 / S 1,613,160), 6,535 ids in `event.sel`, 6,902 event headers
in `phase.dat`, 35 distinct station codes.

Pick times come from `catalogs/pyocto_picks_year_newpool_no_shots.csv`
(818,767 rows). Its `time` column is **float64 epoch seconds**, so no datetime
round trip happens at all and the pandas-3 `.astype("int64")` unit trap cannot
fire; `timeutil.epoch_seconds` is only used if a future CSV ships ISO strings.
There are **0 duplicate (event_idx, station, phase) keys** and no bare-station
code is shared by two networks, so the key is unambiguous.

Consistency check of the pick CSV against `dt.ct` (4,000 pairs, 37,506 rows,
75,012 travel times):

    pick_CSV_time - (phase.dat origin + dt.ct travel time):
    median 0.000000 s   MAD 0.000300 s   max 0.000969 s   frac |d| > 2 ms = 0.0
    missing picks: 0

i.e. the CSV pick times reproduce exactly the travel times ph2dt wrote, to the
1 ms rounding of the `dt.ct` format. Windows are therefore cut at the CSV pick
times and the catalogue term is taken straight from `dt.ct`.

## B. Sign convention (confirmed against the user guide AND the source)

`/home/jovyan/HypoDD/doc/hypoDD2_UserGuide.pdf`, B.3.2 (p.22):

> This file stores differential travel times from cross correlation for pairs of
> earthquakes. ... OTC: Origin time correction relative to the event origin time
> reported in the catalog data. ... Set to 0.0 if cross-correlation and catalog
> origin times are identical, or if only cross-correlation data is used.
> DT: Differential time (s) between event 1 and event 2 at station STA. DT = T1-T2.

`src/hypoDD/getdata.f:402` does `dt_dt(i) = dt_dt(i) - otc`, and that same
`dt_dt` array is filled from `dt.ct` at line 488 by `dt_dt(i) = dt1 - dt2`,
i.e. a difference of **travel times**. So with OTC = 0.0 the DT column must be a
differential TRAVEL time in the catalogue's own origin-time frame, not a
differential arrival time.

Implemented:

    DT = (TT1 - TT2)_from_dt.ct  +  (off1 - off2)  -  L

with `L` the lag at which the sliding correlation peaks (positive L = event 2's
waveform arrives later, relative to its own pick, than event 1's) and `off_i`
the sub-sample rounding of each window's realised start. Derivation is in the
script's module docstring. Consequence: `dt.cc - dt.ct = (off1-off2) - L` is
exactly the cross-correlation correction, which is the quantity validated in
section E.

## C. Method

* Windows cut at the pick: **P -0.3 / +0.7 s, S -0.4 / +1.0 s** (section D shows
  these beat the longer alternatives tested). Slave window padded by +-MAXLAG.
* **3-20 Hz zero-phase Butterworth** via `xcfilter.bandpass()` (`DEFAULT_BAND`),
  applied once per station-day, so no window contains a filter transient.
* **Sliding normalised (Pearson) correlation**: master = event 1's unpadded
  window, slave = event 2's padded window, `np.correlate(..., "valid")` with the
  *moving* mean/variance of the slave in the denominator. This is deliberately
  NOT `correlate(..., "full")` on two equally-normalised windows: `xcfilter.py`
  documents that the latter applies a triangular ~(n-|L|)/n taper that drags the
  peak toward lag 0. Here every lag is normalised over exactly n samples.
* Sub-sample refinement: 3-point parabolic fit on the CC peak. The reported cc
  is the sampled peak (never the fitted value), so cc <= 1 by construction.
* P on the vertical, S on **both** horizontals with the two normalised CC
  functions stacked before peak picking.
* Keep cc >= 0.7 and |lag| <= 0.5 s.

Memory: one worker owns one station and holds at most ONE station-day
(read channel-selectively) plus that station's prepared windows. 8 workers.

## D. Self-test and the two experiments that changed the implementation

### D1. Self-test (`--self-test`) - PASS

Real BRA21 trace, 3-20 Hz, delayed by an exact Fourier phase ramp:

    imposed +0.0000 s  recovered -0.0000 s  err  -0.003 ms  cc 1.0000
    imposed +0.0040 s  recovered +0.0040 s  err  +0.015 ms  cc 0.9975
    imposed -0.0040 s  recovered -0.0040 s  err  -0.019 ms  cc 0.9975
    imposed +0.0130 s  recovered +0.0130 s  err  +0.013 ms  cc 0.9901
    imposed -0.0270 s  recovered -0.0270 s  err  +0.013 ms  cc 0.9901
    imposed +0.1110 s  recovered +0.1110 s  err  -0.019 ms  cc 0.9975
    imposed -0.2500 s  recovered -0.2500 s  err  -0.003 ms  cc 1.0000
    imposed +0.4370 s  recovered +0.4370 s  err  -0.016 ms  cc 0.9901
    worst |error| = 0.019 ms  (PASS; threshold 5 ms)
    integer check: slave delayed 17 samples (+0.085 s) -> lag +0.0850 s, cc 1.0000 (PASS)

### D2. BUG I introduced and caught: the channel picker silenced every S phase

First smoke run (BRA13 + BRA14) returned **0 S measurements** and only 50% of
windows built. Cause: my `_pick_channels` required the vertical and the
horizontals to share a band code. The Bransfield array has two OBS families and
the dominant one does NOT:

    BRA02/03/04/05/08/09/10/11 : HHZ + HH1/HH2 (+HDH), all 100 Hz
    BRA13..BRA27               : ELZ 200 Hz vertical, SL1/SL2 100 Hz horizontals
                                 (+EDH 200 Hz hydrophone, LDH 1 Hz)

The SL/EL family carries ~97% of the `dt.ct` rows, so the bug would have thrown
away essentially all S data - silently, because a station with no horizontals
just produces no S rows. Fixed: vertical and horizontal band are chosen
independently (each = highest sampling rate available), the choice is made once
per station from the headers of 5 spread-out days, each channel keeps its own
fs/t0/npts (ELZ 200 Hz and SL1/SL2 100 Hz coexist in one file), and windows are
keyed by the **full channel code** so event 1 and event 2 can never be
correlated on different components. After the fix, window build is 8,069/8,070
on BRA22 and 1,918/1,918 on BRA05.

Side effect of knowing the channel list: each day is read with libmseed's
`sourcename` selection instead of whole-file. On a 54 MB SL/EL day that is
0.17 s of decode instead of 1.24 s, because the 200 Hz hydrophone is never
unpacked.

### D3. How much of a cc >= 0.7 measurement is chance? (null test)

BRA22, juldays 200-225, 2,989 real `dt.ct` rows and the same rows with the
second window replaced by a random other window of the same phase at the same
station ("NULL"). Fraction of measurements at or above each threshold:

| window (P pre/len, S pre/len) | TRUE >=0.6 / >=0.7 / >=0.8 | NULL >=0.6 / >=0.7 / >=0.8 | purity @0.7 | purity @0.8 |
|---|---|---|---|---|
| **0.3/1.0  0.4/1.4 (chosen)** | 0.098 / 0.043 / 0.019 | 0.047 / 0.011 / 0.004 | **0.74** | 0.80 |
| 0.3/1.5  0.4/2.0 | 0.109 / 0.045 / 0.019 | 0.071 / 0.021 / 0.005 | 0.52 | 0.74 |
| 0.5/2.5  0.5/3.5 | 0.064 / 0.028 / 0.017 | 0.026 / 0.007 / 0.004 | 0.75 | 0.78 |
| 0.3/1.0  0.4/2.4 | 0.085 / 0.035 / 0.017 | 0.043 / 0.008 / 0.003 | 0.77 | 0.82 |

Two conclusions:
1. The specified windows are the right ones - longer windows do not buy purity
   and cost yield, so no deviation from the brief is justified.
2. **At cc >= 0.7 roughly one measurement in four is a chance peak** (per phase:
   P purity 0.74, S 0.73; at cc >= 0.8, P 0.79 / S 0.83). The chance floor is
   high because a 1.0 s window at 3-20 Hz has only ~17 independent samples and
   the search scans ~100 lags. This is a property of the data, not of the code -
   the median cc of unrelated windows at this station is already 0.40. It is the
   main caveat on everything below, and it is why the lag histogram has a broad
   pedestal on top of a peak at zero.

### D4. Second trap found by reading the source: `--with-xc` writes OBSCC 8, which would rebuild the clusters from 1.5% of the pairs

`scripts/24_run_hypodd.py` writes `OBSCC OBSCT = 8 8` whenever `--with-xc` is
given (`{0 if not with_xc else 8}     8`). `src/hypoDD/cluster1.f` links an event
pair when its observation count reaches **`minobs_cc + minobs_ct`** - the two are
SUMMED, and `apair_n` counts cc and ct observations together:

    if (apair_n(k).ge.(minobs_cc+minobs_ct)) then

So `--with-xc` silently raises the pair-linking threshold from 8 to 16. The
observation count per pair in `hypodd/year_v5_3d/dt.ct` (ph2dt ran with
MINOBS 8):

    pairs 326280   mean 9.16   >=8: 326280 (100%)   >=16: 4773 (1.46%)

i.e. 98.5% of the pairs would drop out of the clustering and the ct+cc run would
not be comparable with the ct baseline at all. The user guide's own IDAT=3
example (p.19) uses `OBSCC OBSCT = 0 8`. `scripts/64_run_hypodd_xc.sh` therefore
generates the control file with script 24 `--write-only` (script 24 itself is not
modified) and patches that single field to 0 in this run's own `hypoDD.inp`.

## E. Full run: throughput and dt.cc statistics

`PYTHONPATH=src python3 -u scripts/63_xcorr_dtcc.py --label year_v5_3d --out hypodd/year_v5_3d_xc/dt.cc --workers 8`
(full log: `hypodd/year_v5_3d_xc/xcorr.log`).

    tasks: 35 stations, 74,919 unique windows, 2,987,937 rows (0 rows dropped: no pick in the CSV)
    band 3.0-20.0 Hz
    P window -0.30/+0.70 s, S window -0.40/+1.00 s, max lag 0.5 s, cc >= 0.7
    ...
    total wall 23.4 min, 421 GB read

**Throughput: 7,095 station-days / 421 GB in 23.4 min wall on 8 workers =
18.0 GB/min (300 MB/s), 0.20 station-day/s, 2,130 dt.ct rows/s.** No pair
restriction by inter-event distance was needed (the brief allowed one if the run
exceeded ~4 h). Resource use during the run: 8 worker processes, ~600% CPU of
the 32-core quota, `cpu.stat` nr_throttled **0%** of periods, cgroup **anon**
memory 4.8-5.0 GiB peak (`memory.current` reads ~186 GiB but that is reclaimable
NFS page cache from streaming 421 GB, not anonymous memory - the pod never came
near an OOM). Window build succeeded for 74,916 of 74,919 windows.

Estimate made before launching, from a 3-station smoke run (966 station-days in
9.6 min on 3 workers = 1.5 s/station-day): 7,095 x 1.5 / 8 = 22 min. Actual 23.4.

### E1. dt.cc

    wrote hypodd/year_v5_3d_xc/dt.cc: 147,642 rows in 89,458 pairs

    dt.ct rows 2,987,937 -> dt.cc rows 147,642 (4.9%)   P 81,740 (5.9%)  S 65,902 (4.1%)
    pairs 326,280 -> 89,458 (27.4%)
    cc      p5/10/25/50/75/90/95 = 0.706 / 0.712 / 0.731 / 0.771 / 0.831 / 0.890 / 0.922
    |lag| s p5/10/25/50/75/90/95 = 0.0049 / 0.0100 / 0.0266 / 0.0637 / 0.1532 / 0.3142 / 0.4058
    lag   s p5/10/25/50/75/90/95 = -0.3524 / -0.2290 / -0.0783 / -0.0071 / +0.0515 / +0.1538 / +0.2646
    dt.cc - dt.ct : median +0.00710 s   MAD 0.06390 s   p99 |d| 0.4841 s
      P: n 81,740  cc p50 0.775  median +0.00794 s  MAD 0.06448 s
      S: n 65,902  cc p50 0.767  median +0.00586 s  MAD 0.06346 s

Per-pair: mean 1.65 cc observations, 7,107 pairs with >=4 and 714 with >=8.
**5,733 of the 6,535 `event.sel` events (88%) get at least one cc link.**

### E2. Consistency check dt.cc vs dt.ct, binned by cc

`d = dt.cc - dt.ct` is by construction the cross-correlation correction itself.

| cc bin | n | median d (s) | MAD d (s) | p50 \|lag\| | p90 \|lag\| |
|---|---|---|---|---|---|
| 0.70-0.75 | 55,853 | 0.0133 | 0.0779 | 0.0777 | 0.3607 |
| 0.75-0.80 | 37,978 | 0.0072 | 0.0636 | 0.0635 | 0.3171 |
| 0.80-0.85 | 25,592 | 0.0028 | 0.0546 | 0.0546 | 0.2664 |
| 0.85-0.90 | 16,167 | 0.0019 | 0.0495 | 0.0498 | 0.2346 |
| 0.90-0.95 | 9,322 | 0.0007 | 0.0537 | 0.0537 | 0.2406 |
| 0.95-1.00 | 2,730 | 0.0030 | 0.0619 | 0.0612 | 0.2547 |

| threshold | n | % of dt.ct rows | median d | MAD d | p99 \|d\| |
|---|---|---|---|---|---|
| cc >= 0.70 | 147,642 | 4.94% | +0.00710 | 0.06390 | 0.4841 |
| cc >= 0.75 | 91,789 | 3.07% | +0.00403 | 0.05698 | 0.4772 |
| cc >= 0.80 | 53,811 | 1.80% | +0.00221 | 0.05319 | 0.4681 |
| cc >= 0.85 | 28,219 | 0.94% | +0.00160 | 0.05182 | 0.4573 |
| cc >= 0.90 | 12,052 | 0.40% | +0.00107 | 0.05549 | 0.4413 |

Read this honestly:
* The median correction falls monotonically from +13.3 ms at cc 0.70-0.75 to
  +0.7 ms at cc 0.90-0.95, and the scatter falls from 78 to 50 ms. Both are the
  signature of real measurements: the better the waveform match, the less the
  cross-correlation disagrees with the catalogue pick, and the residual bias
  goes to zero.
* **MAD(dt.cc - dt.ct) = 0.064 s over all kept rows.** That is below 0.1 s but
  not "<< 0.1 s". It does not shrink below ~0.05 s at any cc: a CC correction
  with a 50 ms MAD is what a 35-40 ms per-pick scatter produces in a difference
  of two picks, so most of it is genuine pick-error correction - that is the
  point of the exercise - but it is inflated by the chance-peak population of
  section D3.
* `p99 |d|` stays near 0.47 s at every threshold: the 1-2% of rows sitting at the
  edge of the +-0.5 s search are the chance peaks and they do not clean up with
  a higher cc cut. hypoDD's own cc re-weighting (WRCC 5, WDCC 2 in the
  cross-correlation iteration blocks) is what has to absorb them.

## F. Two more hypoDD control-file traps, both found by running and reading the source

### F1. `--with-xc` warm-up sets DELETE the cross-correlation data

First ct+cc attempt (preserved as `hypodd/year_v5_3d_xc/*.wtcc_neg`) used script
24's `--with-xc` schedule verbatim:

      3     -9   -9    -999   -999   1.0    0.5    6    -999  400
      3     -9   -9    -999   -999   1.0    0.5    6    -999  400
      3     1.0   0.5    5      2    1.0    0.5    6      2   400
      3     1.0   0.5    5      2    1.0    0.5    6      2   400

hypoDD read the data correctly -

    # cross corr P dtimes =   80851 (no OTC for      0 event pairs)
    # cross corr S dtimes =   65590 (no OTC for      0 event pairs)
    # dtimes total =  3128290

- and then reported `CC = 0%` and `RMSCC = NaN` on **every one of the 25 printed
  iterations**, including the sets where WTCCP = 1.0. The relocation was a pure
catalogue run with a mangled catalogue schedule: 5,183 relocated (79%, vs the
baseline's 6,022 / 92%), CT falling to 41% because that schedule also sets
WDCT = 2 km.

Cause, in the source:

* `weighting.f:71` `dt_wt(i) = wt_ccp * dt_qual(i)` - so WTCCP = -9 makes every
  cross-correlation weight **negative**, not "unused".
* `hypoDD.f:724` `if(ineg.gt.0) call skip(...)`, and `skip.f:124` rebuilds the
  data arrays keeping only `dt_wt(i) .ge. minwght`. The deletion is permanent -
  the arrays are compacted and `ndt` reduced. Every cc observation is gone after
  the first re-weighting pass and the later cc sets have nothing left to weight.

The user guide's own IDAT=3 example (p.19) does it the other way round: cc keeps
a positive weight in every set and it is the cc **cutoffs** that are staged
(`WRCC/WDCC = -999` early, finite late). Schedule actually used (patched into
this run's own `hypoDD.inp` by `scripts/64_run_hypodd_xc.sh`; script 24 is not
modified), with the catalogue columns identical to the baseline's:

      5     1.0   0.5   -999   -999   1.0    0.5    6    -999  400
      5     1.0   0.5   -999   -999   1.0    0.5    6    -999  400
      5     1.0   0.5    5      2     1.0    0.5    4    -999  400

So the ONLY difference between the baseline and the ct+cc run is the presence of
the cross-correlation data (and OBSCC, section D4).

### F2. MISTAKE (mine): a DOTALL regex in the patch truncated the control file

My first version of the schedule patch used `(?ms)` and `[-\d].*\n`; with DOTALL
the `.*` ate newlines and the substitution removed the forward-model block and
the CID line. hypoDD died with

    error: invalid cluster number        32544
    must be between 1 and nclust (          16 )

and the surrounding shell script, having no `set -e` guard on that path,
happily re-ran the evaluation on the STALE `hypoDD.reloc` and printed the
previous run's numbers as if they were new. Caught by diffing the control file
against `hypoDD.inp.wtcc_neg` (1,238 bytes vs 1,715). Fixed by using `[^\n]*`
and by asserting that the patched file still ends in `* ID` and still contains
`IMOD` and `CID`. Recorded because it is exactly the class of silent-wrong-number
failure this log exists to catch.

## G. hypoDD 3D: catalogue-only vs catalogue + cross-correlation

Both runs: IMOD 9 (`nlloc/model/ORCA_v4.hypodd3d_fine.vel`, ray params
`2 9 2 0.5 1.0 1.35 0.0005 50`), ISTART 2, IAQ 1, DAMP 400, MAXDIST 100,
identical `dt.ct` / `event.sel` / `station.dat`, 3 sets x 5 iterations, WRCT
6/6/4, WDCT -999. Baseline numbers are quoted from `hypodd/year_v5_3d/run24.log`
and its `hypoDD.log` (run 2026-09-13 20:31-20:47); nothing in that directory was
touched. ct+cc took 18 min (23:05-23:23).

|  | ct only (`year_v5_3d`) | ct + cc (`year_v5_3d_xc`) |
|---|---|---|
| data read | 1,374,582 P + 1,613,035 S catalogue | same + **81,689 P + 65,882 S cross-corr** (0 pairs without OTC) |
| main cluster | 6,082 events | 6,082 events (16 clusters, identical) |
| trial sources | 6,082 | 6,082 |
| RMSCT first -> last iteration | 196 -> **74 ms** | 197 -> **73 ms** |
| RMSCC first -> last | - | 207 -> **59 ms** |
| CT % used, last iteration | 78% | 77% |
| CC % used, last iteration | - | 63% (WRCC 5 culls 37%) |
| CND first -> last | 117 -> **98** | 122 -> **95** |
| relocated | 6,022 / 6,535 (92%) | **6,006 / 6,535 (92%)** |
| median rct | 0.134 s | 0.133 s (median rcc 0.088 s) |
| slope (DD-start) on start depth | -0.176 | **-0.180** |
| Spearman(start, DD) | 0.867 | 0.866 |
| DD depth p10/50/90 | 0.81 / 3.30 / 7.11 km | 0.83 / 3.33 / 7.09 km |
| median \|DD - start\| | 0.58 km | 0.57 km |
| QC pass (script 50) | 5,227 (of 6,022) | **5,226 (of 6,006, 87.0%)** |
| QC depth below local seafloor p10/50/90 | 0.12 / 2.56 / 5.93 km | 0.13 / 2.62 / 5.92 km |

QC-pass CSV: **`catalogs/hypodd_year_v5_3d_xc_qc.csv`** (5,226 events; the full
flagged catalogue is `catalogs/hypodd_year_v5_3d_xc.csv`, 6,006 events).
Per-event cross-correlation links in the final solution: sum nccp 98,848,
nccs 86,390 (vs nctp 2,400,924 / ncts 2,219,476); p10/50/90 = 0 / 8 / 89 cc links
per event, and **1,702 of 5,986 events (28%) end with no cc link at all**.

### G1. Did the relative locations sharpen? Marginally, and only where cc data are dense.

Median nearest-neighbour 3D distance inside the main cluster, computed on the
**same 5,886 events** in both solutions (so the metric cannot move because the
population changed):

| event subset | ct only | ct + cc | change |
|---|---|---|---|
| all main-cluster common events (5,886) | 0.220 km | 0.217 km | **-1.7%** |
| events with >=10 cc links (2,896) | 0.203 km | 0.196 km | **-3.2%** |
| events with >=30 cc links (1,973) | 0.201 km | 0.192 km | **-4.8%** |
| events with >=50 cc links (1,329) | 0.205 km | 0.191 km | **-7.1%** |

Same statistic on each run's own QC-pass main cluster: 0.214 km (5,167 events)
-> 0.211 km (5,160 events).

Displacement between the two solutions, by how much cc data the event got:

    0 cc links   n= 1702   |ct+cc - ct| p50 0.092 km  p90 1.025 km
    1-9          n= 1388   |ct+cc - ct| p50 0.059 km  p90 0.368 km
    10-49        n= 1567   |ct+cc - ct| p50 0.055 km  p90 0.198 km
    >=50         n= 1329   |ct+cc - ct| p50 0.066 km  p90 0.198 km

**Answer: yes, but only just.** There is a clean dose-response - the more cc
links an event has, the more its nearest-neighbour spacing contracts (-1.7% over
the whole cluster, -7.1% for the best-linked quarter) - and the inversion is
slightly better conditioned (CND 98 -> 95) at the same RMSCT and the same
retention. But the effect is at the few-percent level, an order of magnitude
smaller than what cross-correlation buys in a repeating-earthquake sequence, and
every depth-axis statistic (slope -0.176 -> -0.180, Spearman 0.867 -> 0.866,
depth percentiles, QC-pass count 5,227 -> 5,226) is unchanged within noise. The
delivered position of I27 - hypoDD 3D on v5 for relative geometry, NLLoc v5 for
absolute locations - is **not** changed by the cross-correlation data.

### G2. Why the gain is small, stated plainly

1. **Coverage.** Only 4.9% of the catalogue rows get a cc measurement and the
   mean is 1.65 cc observations per pair against 9.16 catalogue observations, so
   cc is outnumbered ~20:1 in the least-squares system even at equal a-priori
   weight (WTCCP 1.0 = WTCTP 1.0). 28% of the relocated events end with no cc
   link at all.
2. **The catalogue is not a repeater catalogue.** Section D3's null test says the
   median correlation of *unrelated* windows at the same station is already 0.40
   and that ~26% of what survives cc >= 0.7 is a chance peak. hypoDD's own
   WRCC = 5 culls 37% of the cc data by the last iteration, which is the
   algorithm agreeing with that diagnosis; RMSCC ends at 59 ms, below RMSCT's
   73 ms, so what survives is good - there just is not much of it.
3. **The picks were already good.** MAD(dt.cc - dt.ct) bottoms out near 50 ms
   (section E2). The cross-correlation is correcting a ~35-40 ms per-pick
   scatter, not a 200 ms one, so the differential times it replaces were not the
   dominant error term.

Next test if this is to be pushed further (NOT run here): rebuild `dt.cc` at
cc >= 0.8 (53,811 rows, ~80% purity by D3) and re-run, to see whether the small
gain is being diluted by the chance-peak population or is simply the size of the
signal. The measurement file `hypodd/year_v5_3d_xc/dtcc_measurements.npz` holds
every cc/lag so the stricter file can be cut without re-correlating.
