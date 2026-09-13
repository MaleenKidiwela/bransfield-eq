# Session 2026-09-12 — every change, and what might be wrong with it

Written so the work can be audited rather than trusted. Each entry says what was
changed, why, what evidence supports it, and **what could still be wrong**.

Decision taken with the user mid-session: **we locate the NEW picks only. No
output from any older run is reused or trusted.** Bugs that exist only in old
products (hybrid catalogue, Stage A/B hypoDD, old GrowClust runs) are therefore
recorded but not fixed.

---

## A. Things I got wrong during this session

Listed first, deliberately.

1. **I recommended raising picker workers to 40** on the basis that the box had
   176 cores. It has a **32-CPU cgroup quota**; we were already at 98% throttling.
   Raising it would have made throughput worse. Caught by the user asking how many
   CPUs we actually have.
2. **My hypoDD 3D grid-resolution argument was wrong.** I claimed 25x25 nodes was
   already finer than the tomography resolves, reasoning from 3-4 km instrument
   spacing. That conflates receiver spacing with tomographic resolution — with
   2,426 dense airgun shots, resolution in the top 2-3 km is plausibly 1-2 km.
   I also stated a resolution the README never documents. The conclusion (don't
   recompile) may still hold, but not for the reason I gave.
3. **I oversold the cross-correlation normalisation bug.** My own table shows the
   old kernel reaching 0.34 ms RMS once bandpassed — the swell was the problem,
   the triangular taper was secondary hygiene.
4. **My reason for keeping the land stations was wrong.** I said distant stations
   constrain the depth-distance tradeoff; they don't — near stations and S-P do.
5. **I proposed reverting the 18/18b bandpass edits** because R3b would replace
   them. The user pushed back correctly: R3b is not written, so those scripts are
   still the only XC path, and the guard has standalone value.
6. **My first XC fix was bypassed entirely.** Script 18 reads pre-cut windows from
   `pick_windows.npy` when it exists and never calls the loader I patched, so it
   would have printed the band it wanted while correlating raw data. Fixed with a
   sidecar + refuse-on-mismatch, but the first version was useless.
7. **I wrote a dedup that crashed on every day** — divided a datetime column by a
   float, i.e. nearly the exact pandas-3 trap this repo already documents. Caught
   by the run failing immediately; now goes through `timeutil.epoch_seconds`.

---

## B. Code changes, in order

### B1. `src/bransfield_eq/xcfilter.py` (new) + `scripts/18`, `18b`
Bandpass before cross-correlation, default 3-20 Hz, applied per station-day.

**Why:** raw OBS is 99.9% power below 3 Hz (measured, ZX.BRA02 2019 doy 190).
Sub-5 Hz holds 27.5% of signal power but 64.4% of noise. Leakage test on 2,989
windows displaced by a known lag: unfiltered 4.58 ms RMS, 10.2% of measurements
>5 ms, at mean cc 0.993 — invisible to `CC_THRESH`.

**What could be wrong:** the band was chosen only from the noise side, using
*identical* waveforms displaced. Real event pairs decorrelate with frequency, so
the optimum on real pairs is probably narrower (Merlin suggests ~3-12 Hz P,
2-8 Hz S, and a per-phase band). **Not validated on real pairs.** Also
`xcfilter.bandpass` returns data UNFILTERED, silently, if the array is short or
contains a non-finite sample — the same silent-failure class it was written to fix.

### B2. `src/bransfield_eq/geo.py` (new) + `scripts/17`
Frozen map-projection origin (-62.5, -58.8); search box from all 38 stations;
lat/lon/depth written into the catalogue; origin sidecar; round-trip assertion.

**Why:** the origin was the mean position of *stations that had picks*, so it
moved between runs (5 km between two months, 19.9 km between two pick pools) and
was never recorded. A chunked year run would have stitched together mismatched
frames silently.

**What could be wrong:** the frozen origin differs from NLLoc's
(`TRANS SIMPLE -62.4413 -58.44 36`, rotated). That is fine *provided* every
cross-stage exchange uses lat/lon — but it means x/y from pyocto and from NLLoc
are NOT interchangeable and never were. Existing catalogues are in the legacy
frame, 3.05 km from the frozen one; do not compare their stored x/y to
`geo.to_xy(lat,lon)`.

### B3. `scripts/17` — velocity model discretisation  **[largest correctness change]**
`create_model(delta=1.0)` -> resample onto a 0.1 km grid first, slowness-averaged.

**Why:** pyocto assigns layers with `p_speeds[:, int(d1/delta):int(d2/delta)]`, so
any layer thinner than `delta` yields an empty slice and is DISCARDED. At
delta=1.0 only **15 of 68 layers survived**; the entire 1.3 km water column and
every shallow gradient vanished, and each surviving cell took its *deepest*
(fastest) velocity. Measured effect vs a correctly discretised model: S-P was
**1.4-1.5 s too small at all distances**, which maps directly into epicentral
distance and depth.

**What could be wrong:** this changes every travel time, so the new catalogue is
NOT comparable to any previous one. I have not independently verified the new
table against an analytic solution — only that it builds and that the layer count
is now 700. `delta=0.1` is a judgement call; 0.05 would be safer and slower.

### B4. `scripts/17` — water column filled with rock velocity
**Why:** the CSV gives water `vs = 0.5 km/s`. No ray in this network crosses
water (sources sub-seafloor, OBS on the seafloor, land above), but pyocto places
LAND stations at z~0, i.e. at the top of the water column, so land rays picked up
~2.6 s of fictitious S delay. This was masked while delta=1.0 deleted the layer;
at delta=0.1 it becomes real.

**What could be wrong:** this is my inference, endorsed by Merlin for NLLoc but
applied here to pyocto by analogy. If pyocto internally handles the water column
for OBS in a way I have not traced, filling it could change OBS travel times too.
I verified only that OBS sit below it (`station z ... max=1.94 km`), not the
internal ray logic.

### B5. `scripts/17` — travel-time table sized to the domain
Fixed `xdist=200` -> computed from the bbox diagonal (now 434 km for a 407 km
diagonal). pyocto's own docs: the model must exceed the search-domain diagonal.

### B6. `scripts/17` — `time_before` 40 s -> 180 s (`--time-before`)
**Why:** it was `min_node_size * 4`, i.e. derived from a *spatial* parameter. The
script's own help text says the margin should be >= max P travel time (~120 s).
Events spanning a time-slice boundary lost their late arrivals.

**What could be wrong:** 180 s is chosen to exceed max S across the domain; I did
not measure the true maximum. Too large costs runtime, not correctness.

### B7. `scripts/17` — cross-picker pick dedup (`--dedup-tol`, default 0.25 s)
**Why:** two pickers over the same stations emit the same arrival twice (~30% of
diting picks have a pnlight twin). pyocto's thresholds count PICKS, not stations,
and `--min-stations` is a dead flag, so a genuine 3-station detection whose picks
are doubled reaches `n_picks=6` and is emitted as a 5-station event. Measured on
the smoke-test day: 4,771 of 62,704 picks (7.6%) dropped.

**What could be wrong:** 0.25 s is arbitrary. Too large merges genuinely distinct
arrivals; too small leaves doubles. Keeping the highest-probability twin biases
toward whichever model is better calibrated, not necessarily better timed —
`diting` has the better timing (P bias -10 ms, MAD 25 ms) but not always the
higher probability. **A per-phase, timing-aware rule would be better.**

### B8. `scripts/17` — no silent homogeneous fallback
A missing `--velocity-model` used to print `[warn]` and continue with a constant
5.5/3.1 km/s half-space. Now exits.

### B9. `scripts/17` — cache key covers build parameters
The `.pyocto` cache was invalidated by mtime alone, so changing `delta`/`xdist`
silently reused the old table (the mechanism by which B3's fix would have failed
to take effect). Now hashed on model + parameters, written atomically.

### B10. `scripts/apply_S1_corrections.py` — BRA05 correction covers all pools
**Why:** `BRA05_DIRS` hard-coded two directories. The new pools created this
session were never corrected, and the first year run was associating **BRA05
picks 0.167 s late**. Verified by the fractional-second signature, then applied:
398 csvs in each of the two pools.

**What could be wrong:** I applied the shift to `picks_bench_bpae_stead` and
`picks_bench_eqcct_p` before restricting scope (10 csvs each). Those are
benchmark directories; the 12-picker benchmark numbers are therefore no longer
exactly reproducible from those two directories. The production pools are
correct. **This is a real mistake, recorded here.**

### B11. `scripts/17f_pyocto_year_newpool.sh` (new) — year driver
Replaces `17e`, which never passed `--pick-sources` and so would have silently
associated the **OLD** pick pool for ~12 h, and which assumed 100 cores.
Also fixed in the merge: empty days crashed `pd.read_csv`; the global key is now
`event_idx` (what every downstream stage reads) rather than `event_uid`.

**What could be wrong:** the merge pairs event and pick files by sorted position,
not by tag — if one `mv` fails, every later day joins the wrong day's picks while
the uniqueness assert still passes. **Not yet fixed.** There is also no per-day
lock, so two concurrent invocations race on the same output names.

### B12. `scripts/38_build_extended_velgrid.py` — one-character fix
`depth_km >= 1.3` -> `> 1.3`. The row at exactly 1.3 km is still water (1.4558),
so the "rock-only" profile began with a water node and all 15 land stations sat
on seawater in the NLLoc grid.

### B13. `scripts/31_nlloc_hyp_to_catalog.py`
- negative seconds in the GEOGRAPHIC line no longer drop the event (one real
  event was lost this way: NLLoc rolled the origin back past midnight)
- `nlloc_status` column records LOCATED/REJECTED. Previously 569 of 31,516
  REJECTED solutions entered the catalogue unflagged.

**What could be wrong:** the positional obs_order -> event_idx mapping is still
unguarded. A single missing per-event `.hyp` shifts every later event in that
shard — the same failure class as the 2026-05-20 scrambling. Current runs verify
clean, but **nothing enforces it**.

### B14. `scripts/40_filter_nlloc_reliable.py`
- boundary test now includes DEPTH (13.2% of the old "reliable" file sat at
  depth < 0.05 km, top-of-grid pinned with artificially small sigma_z)
- drops non-LOCATED solutions
- writes `nlloc_<label>_<tier>.csv` instead of one `_reliable.csv` for every tier

---

## C. Known-but-NOT-fixed (deliberate, because we rebuild from new picks)

- Hybrid catalogue anchors on the wrong join key (`id` vs `id-1`), Stage A ids
  come from a different numbering generation entirely.
- Stage B sub-region merge compares coordinates in two frames 8.4 km apart;
  changes the winning sub-region for 9.2% of events.
- Stage B covers only 31% of the catalogue; nothing reports the other 69%.
- GrowClust output written with integer column headers (24 names vs 25 columns).
- `18`'s `--max-dt-sec` (the 1-hour pair limit) is documented but **never applied**.
- hypoDD and GrowClust both flatten all OBS to z=0.
- hypoDD velocity decimation biases vertical travel time by +138 ms.
- dt.cc carries zero cross-correlation data in existing hypoDD runs
  (`nccp`/`nccs` = 0 throughout) — those relocations are catalogue-differential only.

## D. Open questions I could not settle

1. **Station A/B test never completed** — whether to drop 5M.BYE / 5M.TOW
   (0.06% association rate, more raw picks than any OBS) is still unanswered.
2. **Deep events**: 17-18% of events sit at >=20 km, 6-7% pile against the 40 km
   search lid, and land picks are 3x over-represented in them. Cause not isolated.
3. **NLLoc datum**: the model is "sheared" (hung from the local seafloor) but used
   as Cartesian, and every station sits at z=0. Merlin's recommended fix (un-shear
   with `srModel.elevation`, put OBS at true depth) is **not implemented**.
4. **Vp/Vs = 1.78 everywhere**, including ~2 km of sediment where 2-3 is normal.
5. Whether the new velocity discretisation actually improves locations — untested.

---

## E. v1 locations — results and what to distrust

Run 2026-09-13. Full chain on the new picks, no older product reused.

| stage | events |
|---|---|
| association | 98,631 |
| airgun window excluded | −18,848 |
| into NLLoc | 79,783 |
| located | 79,773 |
| loose / standard / **strict** | 7,675 / 6,324 / **2,152** |

### E1. The airgun window is excluded, not filtered
Shots fire every **17.7 s** (p10 16.2, p90 19.3), so a ±9 s match window already
spans the whole time axis and temporal matching saturates at 83% — it cannot
separate shots from earthquakes. Tight matching (±2 s + 15 km) caught 6,219 of
~17,000; Jan 21–25 still ran 1,161–1,994 events/day against a ~130/day baseline.
Whole window (Jan 21 – Feb 5) dropped; events are **flagged, not deleted**.

**What could be wrong:** ~1,800 genuine earthquakes are discarded with them. The
window boundaries are the first and last shot ±1 h, which is a judgement call.

### E2. Raw NLLoc output was 59.4% pinned to a grid face
45.8% at the top (< 0.05 km), 13.6% at the bottom (25.2 km). **My velocity-grid
rebuild is NOT the cause** — 2,000 events located twice, ORCA_v2 vs ORCA_v3
travel times, everything else identical:

    OLD grid (water top 1.456): 40.5% pinned, median 0.57 km
    NEW grid (rock top 2.348):  40.8% pinned, median 0.57 km

Like-for-like against the old pool at a strict cut (gap<140, ≥10 phases):

    OLD picks: 11.3% pinned, median 3.46 km
    NEW picks: 15.4% pinned, median 2.20 km

So the new pool finds a large extra population of small events whose depths are
not genuinely resolved. The depth-boundary test added to script 40 removes all of
them: the QC'd catalogue is **0.0% pinned**, depths 1–8 km, median 3.1–3.3 km,
consistent with Orca's shallow magma system (per the user — shallow seismicity is
expected there and must not be assumed erroneous).

**What could be wrong:** I have not shown the surviving shallow events are real
rather than the shallow tail of a pinned population that merely cleared QC.

### E3. The velocity model IS sheared — settled
Three independent confirmations:
1. Stingray's docs: *"The velocity model is hung from the elevation. This is
   accomplished by shearing vertically the columns of nodes."*
2. Data: depth of the 4 km/s contour vs water depth — slope **+0.007 km/km**
   (would be ~+1.0 if sea-level referenced).
3. Physics: that contour reaches 1.60 km where water reaches 1.96 km, which is
   impossible in a sea-level frame.

Stingray traced rays in true geometry, so **un-shearing is the correct inverse**,
not a double correction. Script 38's "below seafloor" comment was right.

### E4. v1 limitations — state these with the catalogue
- **Depth datum is below a flat reference seafloor at ~1.3 km**, not below sea
  level. Do not mix with any BSL product without converting.
- **All 38 stations sit at z = 0** despite 785–1943 m of relief (sd 274 m), so the
  array is flattened and absolute depths carry a bias correlated with water depth:
  edifice events biased deep, basin events shallow, up to ~0.5 km.
- **Embargoed claim:** anything comparing depths beneath the caldera with the
  flanks, or any depth-vs-bathymetry relationship. That is the signal the bias
  fakes. Epicentres, clustering, temporal behaviour and rates are usable.

### E5. v2 — the un-shear
Unblocked: GEBCO_2023 is on disk at
`/home/jovyan/ooi/rsn_cabled/SummerSchool2025/global_ocean_data/GEBCO_2023.nc`
(global 15-arcsec, covers all 38 stations); the 30 m Orca grid covers the srModel
footprint. Build is a per-column interp, under a minute. **The validation is the
work**: a shallow-water-OBS vs deep-water-OBS jackknife, which the inversion
cannot game the way it can game a residual-slope test.

### E6. Still open
- 5M.BYE / 5M.TOW: the station A/B never completed. They remain the largest noise
  sources (more raw picks than any OBS, 0.06% association rate).
- Whether strict (2,152) is the right operating point now the depth test is active.
- `--min-s` remains inert (hard-wired to `min_p`); tuning is still not possible.

---

## F. Why the depths are pinned — and why v1 depths must NOT be released

Merlin's read, cross-checked against the data. **This supersedes E2's tentative
explanation and E4's "ship with caveats".**

### F1. Two different phenomena, not one

|  | top face (41-46%) | bottom face (7-14%) |
|---|---|---|
| PDF | **sharp**: ExpectZ 0.19, sigma_z 0.19 km | flat: ExpectZ 19, sigma_z 5.7 km |
| rms | **0.32** vs 0.20 for interior events | — |
| geometry | in-network, nearest station 2.2 km median | gap p50 334 deg, outside network |

A flat PDF is depth degeneracy (what I assumed). A **sharp** minimum pressed
against a boundary with elevated rms means the true solution lies **outside the
grid — above z = 0**.

### F2. The mechanism: real shallow events the grid cannot represent
The user's point that Orca has a shallow magma chamber is the key. A source
0.2 km below the caldera floor sits at ~1.25 km BSL; BRA13/14/15/16 sit at
1.32-1.50 km BSL. **The event is physically above those stations.** The flattened
seafloor datum enforces z >= 0, so no source can ever be above any station.
NLLoc's best compromise is z = 0 with the deep-water stations predicted late.

Residual evidence (weighted picks, top-pinned vs interior):

    P, station <5 km : -0.130 (pinned) vs -0.014 (interior)
    S, station <5 km : -0.177 vs -0.011
    S, station >15 km: +0.106 vs +0.015

Near stations see arrivals ~0.13-0.18 s EARLIER than any in-grid source can
predict. Per-station medians match the geography: BRA15 -0.144, BRA16 -0.098,
BRA13 -0.092 (the deep-water western group).

So these events are not junk. They are real, and their depths are unmeasured.

### F3. What I ruled out myself
- **My velocity-grid rebuild**: A/B on 2,000 events, ORCA_v2 vs ORCA_v3 travel
  times, everything else identical -> 40.5% vs 40.8% pinned. Exonerated.
- **A second BRA05-class clock error**: per-station P residual by month over 14
  months. BRA05 +4.7 ms/month, BRA15 +4.6, BRA26 +4.0, BRA21 -1.3, BRA22 -2.1 —
  all small and NON-monotonic. BRA05/15/26 instead share a common excursion to
  -0.15..-0.19 s in Apr-Jun 2019 and recover, which is geographic/population, not
  instrumental. No drift correction is warranted on this evidence.
- **Over-correcting BRA05**: the new run is 0.05-0.08 s more negative than the old
  (which was corrected in May), not 0.167 s. Single application confirmed.
- **The LOCGRID inset**: max depth is now exactly 25.200 km = the deepest TT node,
  where the old run reached 25.599, i.e. 0.4 km beyond it. Working as intended.

### F4. REVISED shipping position
**Do not release depths from this run.** 46% of depths are boundary values, 43%
even in the well-constrained subset, and those events' epicentres are pulled
toward the deep-water stations by the same geometry. The defensible interim
product is **epicentres and origin times**, with depth flagged
`unresolved (grid face)` for z < 0.05 or z > 24.5 km, and the statement that the
depth distribution is bimodal by construction. Present it as a **detection
catalogue, not a located one**.

### F5. What v2 needs (acceptance criteria, not just a task list)
1. Un-shear to a **sea-level datum** using srModel.elevation + the 30 m Orca grid
   inside its box + GEBCO outside, blended at the edge.
2. **OBS at their true depths** (785-1943 m), rock fill above the seafloor. This
   gives 0.8-1.9 km of grid ABOVE the seafloor, so a shallow caldera source can
   finally sit above a deep-water station.
3. **Vp/Vs sweep** (1.78 / 1.95 / 2.1 / 2.3) on a 2,000-event sample. 1.78 is
   likely too low for the shallow edifice: far S late, near S early.
4. **Acceptance test**: pinned fraction at the new top face (the sea surface)
   near zero for in-network events; count of events located above the local
   seafloor becomes a pick-quality metric.
5. Cheap prior check: S-P at the nearest station gives an implied depth per event
   with almost no model dependence. Do this before rebuilding anything.

---

## G. v2 (ORCA_v4) — un-shear, the sharding bug I introduced, and the datum trap

### G1. Un-shear built and validated (commit 854a173)
`41_build_unsheared_velgrid.py`: sea-level datum, water column filled with local
top rock velocity, bathymetry from the Orca 30 m grid (35,947 nodes) + GEBCO_2023
(223,650 nodes). Validated against `station_geometry.csv`: mean |diff| **13 m**,
max 38 m. Stations at TRUE depths (BRA09 1.943 km, BRA25 0.896 km; land clamped
to 0). Acceptance on 2,000 events, sheared vs un-sheared:

    all events         48.4% -> 31.4% pinned
    gap<180, rms<0.5   38.4% -> 15.7%
    gap<140, rms<0.3   20.7% ->  5.4%   median depth 2.42 -> 3.67 km

### G2. Vp/Vs settled at 1.78 — two independent ways, NOT by sweeping
Wadati regression on the picks (479 events, r2>0.95): **1.795**, with a modest
shallow excess (1.86–1.90 where nearest-station S–P < 1 s). The sweep was a trap:
1.78→2.30 took pinning 24.1%→2.9% **while RMS rose 0.236→0.302**. Less pinning by
fitting worse. Merlin's criterion (pinning down AND rms down) fails, so the S
model is not implicated. P-only relocation confirmed it: dropping all S made
pinning WORSE (32.7% vs 24.2%) because S–P is what constrains depth.

### G3. MISTAKE (mine): every shard ran the full catalogue
I hand-edited the LOCFILES line of `nlloc/run/year_v4.in` with sed to the old obs
filename. `30_run_nlloc.py` substitutes shard paths by matching the literal string
`nlloc/obs/<label>.obs`; no match → no substitution → **all 16 shards located all
79,783 events** (16x the work; 93,606 hyp files for 79,783 inputs). Caught by
noticing per-shard hyp counts (~5,850) exceeding per-shard inputs (4,987). The
output is quarantined under `nlloc/output/QUARANTINE_year_v4_fullobs_bug/`, not
deleted. Script 30 now **refuses to start** if either substitution fails.
I also reported this run as "launched" two hours before discovering it had
crashed on a filename mismatch at the very first attempt. Two reporting failures
on one run; both recorded.

### G4. TRAP: the datum changed, and three consumers assumed the old one
v1–v3 depths are **below seafloor**; ORCA_v4 depths are **below sea level**.
- `34_animate_nlloc.py` line 89 adds local bathymetry to get BSL — correct for
  v1–v3, would plot v4 events ~1.3 km too deep. Now datum-aware.
- `40_filter_nlloc_reliable.py` strict tier tested `depth_km > 0.2` as "bsf>0.2" —
  trivially true on a BSL catalogue, i.e. the test would silently stop doing
  anything. Now computes bsf = depth − local water depth from the surface saved by
  script 41.
- `31_nlloc_hyp_to_catalog.py` now **stamps a `depth_datum` column** so no consumer
  has to guess, and both 34 and 40 read it (or fail loudly if absent).

### G5. FOUND: `nlloc_status` was parsed and then DROPPED
Script 31's output `cols` whitelist did not include `nlloc_status`, so the status
I added to `parse_hyp` never reached the v1 catalogue, and script 40's "drop
non-LOCATED" filter — guarded by `if "nlloc_status" in df.columns` — **silently did
nothing on v1**. The v1 QC tiers therefore still contain REJECTED solutions.
Fixed: the column is now written and its absence is a hard error.

### G6. What "no mistakes" means operationally from here
Every stage is gated on a check that would catch the failure class it is prone to:
1. relaunch → shard controls point at their own slice; 16 workers; output growing
2. completion → per-shard hyp count == per-shard obs count (positional-mapping precondition)
3. parse → row count, no orphans, `nlloc_status` and `depth_datum` present
4. QC → 0% grid-face pinning in standard/strict; bsf computed on the right datum
5. acceptance → v4 vs v1 pinning and depth distributions
6. movie → a frame visually inspected, datum confirmed from the column

### G7. MISTAKE (mine): two instances of the v4 run started simultaneously
My first relaunch attempt (`rm -rf` old output + relaunch, run in the background)
appeared blocked — its output file was empty when I checked — so I quarantined the
old output by `mv` and relaunched by hand (instance A, 06:57). The background task
had in fact been stuck in the `rm -rf` on ~280k NFS files; when it finished it
launched a second instance (B, 07:01). 32 NLLoc workers were writing the same
shard directories. Caught by the worker count (32 vs 16). B was killed by PID;
A retained; verified 16 workers, all children of A. Because filenames are
deterministic per event and both instances had identical inputs and grids, any
files B wrote were byte-identical overwrites — Gate 2 (per-shard count == input
count) is the confirming check. Lesson recorded: never `rm -rf` a large NFS tree
in a fire-and-forget background command; `mv` to quarantine, and never launch a
second run without confirming the first is dead.

### G8. CORRECTION to G3 and to commit 3ce7d58: the broken output was NOT preserved
Both say the full-obs-bug output was "quarantined, not deleted". That is wrong.
The background `rm -rf` (see G7) had already opened the old directory when I
renamed it to QUARANTINE_...; it continued deleting through the open handle, so
the quarantine ended up empty and the broken output is gone. Nothing of value was
lost — that output was 16 copies of the full catalogue and useless — but the
record must say what actually happened. The `rm` did NOT recurse into instance
A's new `nlloc/output/year_v4/` (its "Directory not empty" error is the rmdir at
the end hitting A's live directory by path); A's files all postdate 06:57 and
Gate 2 is the definitive check of A's completeness. Empty quarantine dirs removed.

---

## H. v2 FINAL — ORCA_v4 located catalogue, all gates passed (2026-09-13)

| gate | check | result |
|---|---|---|
| 1 | each shard control reads its own slice; 16 workers | PASS |
| 2 | per-shard hyp count == obs count, all 16 shards | PASS — 79,783 == 79,783, 0 mismatches |
| 3 | parse: rows, unique, no orphans, status + datum columns | PASS — 79,773 rows; 340 REJECTED flagged; datum=sealevel |
| 4 | QC: 0 REJECTED, 0% grid-face pinned, bsf on right datum | PASS — standard 11,376; strict 2,277 |
| 5 | v4 beats v1 pinning at every matched cut | PASS |
| 6 | movie frame inspected, datum from column | see below |

### Acceptance numbers (raw catalogues, matched cuts)
| cut | v1 pinned | v4 pinned | v1 med z (bsf) | v4 med z (bsl) |
|---|---|---|---|---|
| all | 59.4% | 42.7% | 0.24 | 1.07 |
| gap<180 rms<0.5 N≥6 | 42.3% | **16.1%** | 0.29 | 1.20 |
| gap<140 rms<0.3 N≥10 | 15.4% | **4.4%** | 2.20 | 2.98 |
| gap<120 rms<0.3 N≥8 | 16.4% | **4.3%** | 2.00 | 2.86 |

### Gate 6 (visual) caught two QC gaps, both fixed before release
**Gap 1 — water-column events.** The first movie frame showed a dense band of
events above the seafloor line. The standard tier tested "not pinned at a grid
face" but never "below the local seafloor". Quantified: **25.0% of standard
(2,842 events) had bsf < 0**, 16.2% clearly so. Those events carried rms 0.298 vs
0.199 and an artificially tight σ_z 0.30 vs 0.77 — a forced minimum, the same
signature as pinning on the sheared grid: picks inconsistent with any
sub-seafloor source. Merlin had named exactly this count as the v2 pick-quality
metric. Added `depth_bsf_km > -0.2` (half a 0.4 km grid cell) to loose and
standard; strict already required bsf > +0.2.

**Gap 2 — bathymetry lookup too coarse.** An *independent* recheck against the
30 m Orca bathymetry (different data source from script 41's surface) still found
**67 standard events > 0.2 km above their exact local seafloor** after Gap 1 was
closed. Cause: QC read the seafloor at the nearest 0.4 km grid node, and on the
caldera walls the seafloor changes by hundreds of metres within one cell
(0.35–1.69 km BSL across the caldera strip). Bilinear interpolation of the same
0.4 km surface only reduced it to 49 (worst +0.28 km) — the surface itself is the
limit. Fix: QC now reads the 30 m Orca bathymetry directly wherever it covers the
event (49,095 of 79,433 events), and the 0.4 km surface only outside its
footprint, where the basin floor is smooth. Re-verified: **0 events above the
seafloor by > 0.2 km in any tier.**

Why the movie still shows points above the black line: the section draws the
**median** seafloor across the latitude strip and a gray min–max envelope; a
caldera-floor event (seafloor 1.69 km) legitimately plots above the median
(~1.3 km). Per-event bsf is the authoritative test, not the 2-D projection.

### FINAL QC tiers (catalogs/nlloc_year_v4_{loose,standard,strict}.csv)
- **loose 11,682** — 0 above seafloor by >0.2 km; 1,261 (11.0%) within 0–0.2 km of it
- **standard 9,517** (v1: 6,324) — bsf median ~1.4 km; σ_z 0.73; rms 0.203.
  **2,005 events (21%) sit within ±0.2 km of the local seafloor**, 87% of them under
  the Orca edifice. The user's point stands: near-seafloor seismicity is real on an
  unstable volcano, and these are a *continuum* with the shallow population (rms and
  σ_z vary smoothly deep→shallow→seafloor; no discontinuity), not an artefact class.
  The honest caveat is narrower than I first wrote: near-seafloor depths are resolved
  only to ~0.2 km (half a 0.4 km cell), and the top kilometre is where the velocity
  model is weakest (strongest 3D variation, starting-model fill at the top node,
  measured Vp/Vs 1.86–1.90 vs 1.78). Use **standard** for the shallow population —
  strict excludes it by construction (bsf > 0.2). Refinement if it matters: a finer
  grid in the top ~3 km and a better shallow velocity model.
- **strict 2,278** (v1: 2,152) — bsf 0.53/**2.93**/6.33 km; σ_z 0.78; rms 0.170;
  every event ≥ 0.2 km below the seafloor
- all tiers: 0% grid-face pinned, 0 REJECTED, `depth_datum=sealevel` stamped

The standard tier is 50% larger than v1's because events the sheared grid jammed
onto z=0 now have real interior depths. Its shallow population — p10 at the
seafloor, median ~1.4 km below it — is the Orca shallow system, measured rather
than truncated.

### Still open (not blocking)
- 5M.BYE / 5M.TOW exclusion never A/B-tested on the corrected catalogue
- `--min-s` still inert in the associator (hard-wired to `min_p`)
- Vp/Vs shallow excess (1.86–1.90 for near paths vs 1.78 model) is real but small; not applied
- relative relocation (hypoDD/GrowClust) not rebuilt — this is an absolute catalogue

### G9. Process note: two commits carry a Co-Authored-By trailer against this repo's rule
Commits 3ce7d58 and 1396ffd end with a "Co-Authored-By: Claude ..." trailer. This
repo's standing rule (memory note, and the clean history before this session) is
never to include one. Recorded here rather than rewriting history; no further
commits carry it.

---

## I. hypoDD 1D baseline on the v4 standard tier (2026-09-13)

Scope: relative relocation with **catalogue** differential times only (dt.ct). The
cross-correlation rebuild (dt.cc) and the IMOD=9 3D experiment are later steps.

### I1. Datum decision — seafloor, rigid shift of 1.0 km
hypoDD clamps every negative station elevation to 0 (`getdata.f`) and `ray_3d.f`
never reads elevation, so it cannot represent OBS at depth. The only consistent
frame is a seafloor datum. **Shift = 1.0 km**, not the model's 1.3: the median
water depth *under the events* is 1.02 km (they sit on the edifice), and a 1.3 km
shift would have put 25.6% of events above the model top vs 8.8% at 1.0 km. The
rock-only 1D model is shifted by the same 1.0 km with the first rock Vp extended
to the datum. Relative geometry is preserved exactly (rigid shift); the residual
error is the ±0.5 km bathymetric spread across the array, which double-difference
cancels for close pairs. 871 events (9.2%) shallower than the datum start at 0.01
km; IAQ=0 keeps them if the inversion pushes them back up. Output carries
`depth_bsl_km = dep + 1.0` and `depth_datum` so nothing downstream guesses.

Land stations are written at **+1.0 km + elevation** above the datum (IMOD=1
honours positive elevation). Previously they were written at ~0 m, i.e. sitting
on the seafloor 1 km too deep.

### I2. Audited bugs fixed in 22 / 23 / 24
- 22: depth read from `depth`/`z` — the NLLoc file has `depth_km`, so every
  starting depth would have been **0.0**. Now `--depth-col` (auto-detected).
- 22: numeric `origin_time` dispatch could never fire (numeric epochs parse to
  valid 1970 timestamps, not NaT). Now dtype-dispatched, with a hard error if any
  date parses before 2000. Same for pick times.
- 22: picks dropped for tt≤0 / tt>60 s were silent → counted and printed
  (200 / 0 here).
- 23: ph2dt drops whole events below MINOBS and only says so in its log → parsed
  and reported; empty dt.ct is now a hard error.
- 24: velocity decimation took every other distinct-Vp row (systematically slow,
  +138 ms over 0–31 km, sediment layer dropped) → rock-only, shifted, merged by
  <1% Vp contrast, capped by smallest-contrast merging, monotonicity asserted.
- 24: water rows kept with vs=0.5 so the vs≤0.05 "no-S" guard never fired and S
  propagated through water at Vp/Vs 2.91 → water rows removed entirely.
- 24: missing velocity model silently substituted a 5-layer continental default →
  hard error.
- 24: event.sel → reloc loss never reported → reported.

### I3. Gates
| gate | check | result |
|---|---|---|
| H1 | 9,517 headers; years 2019–20; depths ≥0.01; OBS elev 0; land +1000–1030 m; ids unique | PASS |
| H2 | ph2dt 9,517 → 8,992 (5.5% < MINOBS); 462,024 pairs; 4.17M observations | PASS |
| H3 | IMOD=1; 22 layers 0–30.3 km monotonic; Vp 2.35–7.00; Vp/Vs 1.78; no water, no 200 sentinel | PASS |
| H4 | reloc count, airquakes, shift vs NLLoc start, rct | pending |

### I4. Observations during the run (before H4)
- **Cluster structure: 23 clusters, one of 8,449 events (94%)**, next largest 55.
  This is the single-big-cluster LSQR regime the repo's own notes flag as slow
  (~5 days at 42k events). Smaller here, but expect hours, not minutes. If a 1D
  baseline on the full tier proves too slow to iterate on, the proven fallback is
  the existing pruned-backbone + sub-cluster approach (~15 min).
- **2,069 distinct events warned "negative depth"** during the inversion — 23% of
  the cluster, vs 871 that were clamped to the datum at input. So the inversion is
  pushing ~1,200 events that *started* below the 1.0 km datum up above it. On a
  rigid flat datum this is NOT the same as "in the water column": local seafloor
  under the edifice is as shallow as 0.35 km BSL. The correct plausibility test at
  H4 is `depth_bsl_km` against the **30 m local bathymetry**, exactly as script 40
  does for NLLoc — not the flat-datum `physical` flag. Two candidate explanations
  to weigh at H4: (a) the genuine shallow edifice population that NLLoc also
  found; (b) a too-fast shallow 1D model biasing depths upward (the same "top
  layer too fast" hypothesis raised earlier). DD cancels smooth model error for
  close pairs, so (b) should be small; the local-bathymetry count decides.

### I5. Gate H4 and post-H4 findings — hypoDD 1D baseline DONE, with a clear role
Run: 8,449-event main cluster, 15 LSQR iterations, 17 minutes.
| gate/check | result |
|---|---|
| H4 | PASS — 8,992 → 7,296 relocated (18.9% lost); ids unique; all join to NLLoc start; rct 0.231 s; nctp/ncts p50 336/400 |
| physical (30 m local bathymetry) | 59 events (0.81%) > 0.2 km above their seafloor → flagged |
| links | 200 events (2.7%) with < 20 catalogue links → flagged (the 10 that moved > 10 km had a median of **3** links) |
| **QC pass** | **7,044 (96.5%)** → `catalogs/hypodd_year_v4_1d_qc.csv`; depth below local seafloor p10/50/90 0.42/2.24/3.30 km |

New script `50_hypodd_qc.py` does this reproducibly (flags added to the full CSV,
subset written separately). hypoDD's own `physical` flag tests the flat datum and
is the wrong criterion; `physical_local` is the one to use.

**Loss is biased against the shallow population.** Dropped events: median start
depth 0.53 km below seafloor vs 1.90 for kept; 49% shallow (bsf < 0.5) vs 31%.
Also fewer phases (10 vs 12), wider gap (154° vs 124°). hypoDD drops poorly-linked
events and the shallow edifice events are the least well recorded. **The
relocated catalogue under-represents exactly the shallow population the user
cares about.** A limitation to state, not a bug.

**hypoDD and NLLoc disagree on depth at the km level, systematically.**
| set | NLLoc p10/50/90 | hypoDD p10/50/90 | median shift |
|---|---|---|---|
| all 7,296 | 1.02 / 2.92 / 7.33 | 1.38 / 3.21 / 4.36 | −0.07 km |
| strict tier only (2,212) | 1.57 / 3.92 / 7.23 | 1.84 / 3.31 / 4.10 | **−0.55 km** |
Compression is concentrated where NLLoc was uncertain (|dz| 2.05 km at σ_z > 1.5
vs 0.51 at σ_z ≤ 0.5) — that is DD collapsing a weak tail, expected. But even the
strict tier compresses (p90 7.2 → 4.1 km), so this is not only noise. Mechanism
consistent with the flat datum: pairs up to MAXSEP = 5 km apart span > 1 km of
bathymetric relief across the caldera, so the flattened-station error does NOT
cancel between them, and DD on a 1D model has weak depth leverage in the top km.

**Role of each product (state this with the catalogue):**
- **NLLoc v4** (3D, un-sheared, true station depths, per-event σ) → absolute
  positions and depths. The depth product.
- **hypoDD 1D** → relative geometry within clusters (fault planes, lineations,
  fine structure). Do **not** quote its absolute depths over NLLoc's.
This is the reason a hybrid (relative geometry anchored to absolute frame) exists;
the old `32_hybrid_catalog.py` did it with a wrong join key and no datum
conversion, and is not reused.

Not traced: the 2,069 mid-run "negative depth" warnings — hypoDD rewrites its log
at the end and my regex found none afterwards. Final state is what matters and is
characterised above (0 above the flat datum; 59 above the local seafloor).

Not done in this pass: dt.cc (new correlator), IMOD=9 experiment, any hybrid.

### I6. CORRECTION: the catalogue CSVs are NOT in git
`.gitignore` line 5 ignores `catalogs/*.csv`. Commits 0662675 and c4288d8 name the
NLLoc catalogue CSVs as if committed; the `git add` was refused (stderr hidden,
`;` not `&&`) and the commits went ahead without them. Same for the hypoDD CSVs
(caught this time because the chain used `&&`). So: **every catalogue CSV lives on
disk and was delivered to the user as a file; none is version-controlled.** That
is the repo's standing convention (outputs are not versioned), not a new decision
— but my commit messages overstated it. Scripts, grids headers, notes and the
movie ARE committed, and every CSV is reproducible from them. If the catalogues
should be versioned, that is a deliberate `.gitignore` change for the user to make.

The same applies to **every movie** (`notes/figures/**/*.mp4`, .gitignore line 53) and
**every NLLoc grid/header** (`nlloc/`, line 83). Commits ced9c31, 854a173 and
c4288d8 name mp4 and `.hdr` files they do not contain. What IS in git, in full:
scripts, `src/`, notes (incl. this log), `configs/velocity_model.csv`,
`catalogs/station_geometry.csv`, `catalogs/manual_picks.csv`. What is disk-only
and was **delivered to the user as files**: all `catalogs/*.csv` products, both
movies, all grids. Everything disk-only is reproducible from what is committed:
scripts 17f → discriminate_shots → 41 → 39 → 29/30 → 31 → 40 → 34 (NLLoc chain)
and 22 → 23 → 24 → 50 (hypoDD chain), with the parameters recorded in this log.

### I7. RETRACTION of I5's conclusion, and the damping sweep (Merlin review)
**The DAMP=20 run is an ill-conditioned inversion and its output must not be
used for relative geometry either.** Verified in `hypodd/year_v4_1d/hypoDD.log`:
main-cluster condition numbers in the **thousands** against a user-guide target of
~40–80 (two orders of magnitude under-damped); mean |DZ| of 1.8–5.1 km per
iteration through set 2 (the depth axis reshuffled every pass, not converging);
600–1,450 airquakes per iteration clamped to the datum with IAQ=0 (the Fortran
fields run together at that magnitude). In this regime LSQR fits noise along the
least-resolved direction — relative depth — and the −0.80 dz slope is
**regression to the mean, not a measurement.** I5 §3 stood that run up as "for
relative geometry"; withdrawn. The delivered file is renamed
`hypodd_year_v4_1d_qc_RETRACTED_underdamped_DAMP20.csv`.

My flat-datum explanation for the compression (I5) was **wrong in sign and ~50×
too small**: with the datum at the median event water depth the pairwise error is
~0.05 s per station per 5 km pair and pushes toward *expansion*.

**NLLoc's depth ordering is real** (Merlin's model-light test on phase.dat:
observed near-station S−P rises monotonically with NLLoc depth, 0.58 → 1.19 s,
four times the pick noise; the DAMP=20 run flattened those events to a constant
0.62–0.77 s). NLLoc's 6–12 km bins are probably **~1.5–2 km over-deep** (it
over-predicts near S−P by ~0.2 s there) — a modest correction to check on the
NLLoc side (Vp/Vs, top layer), not the 3 km the bad run implied.

Also wrong-rather-than-limited in the DAMP=20 setup: WRCT 10/8 (effectively no
outlier rejection on picks associated at 0.5 s tolerance; conventional 6→4);
WDCT=4 < MAXSEP=5 (a mid-run cull that removed 21% of pairs and preferentially
the sparse ends of the depth distribution); IAQ=0 with 7–17% airquakes per pass.
The shallow-biased loss is partly inherent (few S, short chains) and partly these
settings plus a datum at the *median* event seafloor that sits above the
shallowest edifice population (local seafloor to 0.35 km BSL).

**Kept from the run:** the seafloor rigid-shift datum (the only frame hypoDD's
code allows) and script 50's local-bathymetry plausibility test.

**Sweep launched:** DAMP ∈ {100, 200, 400, 800}, IAQ=1, WRCT 6/4, WDCT −999, same
inputs, same datum, same ph2dt (one variable at a time; MINOBS/datum changes
deferred). Script 24 now takes `--damp --iaq --wrct --wrct-last --wdct-last`; the
first control file was gated (IAQ=1; DAMP in all 3 sets; no cull; WRCT 6/6/4)
before the other three launched. Acceptance: CND ~40–80; dz-vs-depth slope well
above −0.80; p90 depth close to NLLoc's; retention not collapsed. Then Merlin's
discriminating test — forward-model dt.ct from the NLLoc locations through the same
1D model, run the identical control, and see whether the depth spread survives.

### I8. Damping sweep results (DAMP 100/200/400; 800 running) — compression is NOT a damping artefact
CND read from the raw main-cluster iteration table (my earlier fused-field parse
was unreliable and is superseded):

| run | CND it1→last | RMSCT ms | AQ removed | DZ/iter m | reloc | retain | dz slope | p90 NLLoc | p90 DD |
|---|---|---|---|---|---|---|---|---|---|
| DAMP 20 | 1,151→3,744 | 203→128 | kept | 2,479→421 | 7,296 | 81% | −0.53 | 7.33 | 4.36 |
| DAMP 100 | 382→485 | 193→117 | 1,270→18 | 2,319→468 | 5,734 | 64% | −0.56 | 8.14 | 4.73 |
| DAMP 200 | 225→202 | | | | 6,125 | 68% | −0.55 | 8.02 | 4.84 |
| DAMP 400 | **106→117** | 205→135 | 674→12 | 1,739→304 | 6,758 | 75% | **−0.53** | 7.87 | 5.23 |

Damping fixed what it should: CND down 30×, airquakes to ~0 by iteration 4,
per-iteration depth adjustment 2.3 km → 0.3 km. **The depth-compression slope did
not move: −0.53 at CND 3,744 and −0.53 at CND ~110.** A signal that survives a 30×
change in conditioning is in the differential times, not in LSQR noise-fitting —
so Merlin's diagnosis that under-damping *caused* the compression does not hold,
even though under-damping was real and needed fixing.

Tentative reconciliation (sent to Merlin to attack): Merlin's own model-light S−P
test found NLLoc's 6–12 km bins ~1.5–2 km over-deep. A −0.53 slope on events ~4 km
above the median moves them ~2 km — the same magnitude — and DAMP 400's p90 of
5.2 km is roughly where the S−P table placed those events. So hypoDD may be
partly right about the deep tail and the truth sits between the catalogues.
Alternative not excluded: a damping-independent 1D/flat-datum bias in the dt.ct
forward model, which only the synthetic relocation test can discriminate.

Retention rises with damping under IAQ=1 (64→75%) because a stiffer solution
generates fewer airquakes. DAMP 800 pending; CND expected inside 40–80.

### I9. Merlin on the sweep — my point (3) retracted; NLLoc's depth axis is stretched
Model-light S−P test against every run (NLLoc σ_z ≤ 0.5, nearest P+S station < 4 km):

| run | Spearman(NLLoc z, DD z) | median |S−P misfit| at NLLoc z | at DD z | DD z for NLLoc 5–8 km | for 8–12 km |
|---|---|---|---|---|---|---|
| DAMP 20 | 0.49 | 0.210 s | 0.223 | 2.58 km | 2.94 |
| DAMP 100 | 0.39 | 0.201 | 0.275 | 2.01 | 2.37 |
| DAMP 200 | 0.45 | 0.205 | 0.265 | 2.21 | 2.66 |
| DAMP 400 | 0.57 | 0.212 | 0.241 | 2.83 | 3.76 |

1. **hypoDD depths fit near-station S−P worse than NLLoc's at every damping.** DD is
   not vindicated by the data that most directly constrain depth.
2. **The deep-bin DD depths rise monotonically with DAMP** (2.01→2.21→2.83 km). LSQR
   damping penalises movement from the *initial* (NLLoc) positions, so a solution
   that drifts toward NLLoc as DAMP rises is **leakage, not information**. My I8
   reading that "DAMP 400 lands where the S−P table put those events" mistook
   leakage for agreement. **Retracted.** The data-dominated DD answer is the
   low-damping one (~2 km for events NLLoc puts at 5–12 km) and the S−P contradicts it.
3. **The −0.53 slope is a bulk average of two opposite errors.** NLLoc's 0–1 km events
   (σ_z ≤ 0.5) show observed near S−P 0.69–0.75 s vs 0.50 predicted at NLLoc depth —
   too shallow; its deep bins are too deep. **NLLoc's depth axis is stretched.** DD
   compresses the whole axis; the regression averages the regimes.
4. Model-robust quantity: **S−P spread ratio** 0–1 → 8–12 km bins. Observed ×1.8.
   NLLoc predicts ×2.8. DD predicts ×1.36. Truth is between, ≈ geometric mean. Quote
   this, not the slope.
5. Intrinsic DD compression is only ~10–20% (flat-datum ~10%; too-fast source-depth
   model adds some). The rest of −0.53 is NLLoc's true depth errors being much larger
   than its formal σ_z (model error is not in the PDF) → regression to the mean.

**Consequences**
- For depth: **NLLoc v4 remains the product, with a stated stretch caveat**
  (shallow too shallow, deep too deep). It is an NLLoc-side velocity problem — the
  same top-layer-too-fast / Vp/Vs suspicion — with a cheap test: the 2,000-event
  sample with Vp/Vs 1.9, and with the top 2 km slowed 10%, scored by near S−P.
- For hypoDD: **with catalogue-only dt at rct 0.23 s it will not beat NLLoc on depth
  at any damping.** Its value arrives with dt.cc. Choose DAMP by CND (stop at 40–80,
  likely 800), never by preferred depths. The IAQ=1 stiff run is the honest one for
  relative geometry, but the 1.0 km datum makes the shallow edifice population
  airquakes by construction → re-run with the datum at the shallowest event
  seafloor (~0.4 km).
- **Decisive test (next):** synthetic dt.ct forward-modelled from NLLoc starts
  through hypoDD's own 22-layer flat model; three runs (noise-free; 0.15 s/pick
  Gaussian; +5% ±0.5 s outliers); identical DAMP-400 control. Acceptance for the
  noise-free run: slope of (DD−start) on start within ±0.05, p90 within 0.2 km —
  else the forward model or inversion is broken. The noisy runs give the
  inversion's own compression baseline; real-data −0.53 minus that baseline =
  genuine data–model inconsistency.

### I10. DAMP 800 hung; sweep closed at 400
The DAMP 800 run produced iterations 1–8 at ~1 min each, then sat 55 minutes in
iteration 9 at 99.8% CPU with no new output (last reloc file and log line both
14:47). Killed by PID after confirming the process cwd. Not diagnosed further:
the sweep's conclusion — depth compression is damping-independent — stands on
20/100/200/400, and Merlin's rule is to choose DAMP by CND, where 400 (CND
~100–120) is the closest to the 40–80 target that completed. **DAMP 400, IAQ=1
is the reference control for the synthetic test.** If a run in the 40–80 range
is wanted later, try 600 and watch for the same stall.

Launched: (a) the synthetic-dt.ct travel-time table (TauPy on hypoDD's own
22-layer flat model, 300 depths × 521 distances × P,S, 16 workers), gated by
three self-checks before any dt.ct is generated; (b) NLLoc Vp/Vs 1.90 on the
2,000-event `abgrid_test` sample, ORCA_v4 grids, everything else identical to
`abtest_ORCA_v4` — the cheap half of Merlin's NLLoc depth-stretch test, to be
scored on near-station S−P.

### I11. Independent S−P depth check (script 52) — the stretch is real; Merlin's table not reproducible
`52_sp_depth_check.py`: observed nearest-station (<4 km) S−P from the picks vs the
S−P predicted at each catalogue's depth (straight ray, path-average rock model,
station at its true depth). Crude, but identical for every catalogue scored; the
**spread ratio** across depth bins is the model-robust quantity. Scorer verified:
pick times are epoch s, zero duplicate (event,station) pairs, obs S−P correlates
0.66 with hypocentral distance.

- I could not reproduce Merlin's I9 table. Its stated subset (σ_z ≤ 0.5, n=6,472)
  matches neither the standard tier (3,455 such events, almost all shallow) nor
  the full catalogue (31,676, of which **48% are grid-face-pinned** with
  artificially small σ_z). A table on the latter tests artefacts. Merlin asked.
- **On the STRICT tier (2,222 events with a near P+S station, depths 1.5–12 km)
  the stretch is confirmed independently:** observed S−P grows ×2.14 from the
  1–2 km bin to 8–12 km; NLLoc's depths predict ×3.79. Observed S−P is nearly
  flat (0.79→0.88 s) across NLLoc depths 2→8 km — five kilometres of catalogue
  depth the nearest station barely sees. NLLoc's depth axis is stretched ~1.8×.
  The 1–2 km bin fits (+0.02 s); 2–5 km under-predicted by 0.15–0.30 s (too
  shallow *or* model too fast); 8–12 km over-predicted by +0.16 (too deep).
- **hypoDD DAMP 400 (QC-pass, strict-tier events) fits worse:** median misfit
  −0.32 s vs NLLoc's −0.13 s; under-predicts everywhere 1–8 km. Confirms Merlin's
  "DD not vindicated by the data that most directly constrain depth".
- Both catalogues under-predict S−P at 2–5 km → the shared forward model is
  likely too fast for S in the shallow crust (Wadati gave Vp/Vs 1.86–1.90 for near
  paths vs 1.78 in the model). The Vp/Vs 1.90 NLLoc sample run is the direct test.
