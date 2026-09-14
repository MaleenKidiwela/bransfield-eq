# Methods and Results — manuscript draft

Seismicity of the Orca submarine volcano, Bransfield Strait, from a 14-month ocean-bottom
seismometer deployment: absolute locations and double-difference geometry.

*Draft prose for the manuscript. Every quantitative statement is traceable to a source
listed in "Data and code availability" or to a computation reported in the text. Placeholders of the
form `[value: compute from X]` mark quantities that are not yet established; they must be filled or
deleted before submission.*

---

# METHODS

## 1. Network and data

### 1.1 Instrumentation

The data were recorded by a 22-station ocean-bottom seismometer (OBS) array (network code ZX)
deployed across and around the Orca volcanic edifice in the central Bransfield Strait, supplemented
by 16 permanent or campaign land stations on the South Shetland Islands and the Antarctic Peninsula
(14 stations of network 5M, one of AI, one of AM). All 22 ZX instruments sat on the seafloor, in
water depths of 785–1943 m (median 1336 m); the array therefore has 1.16 km of internal
receiver-depth relief, which is a first-order control on the location problem and is treated
explicitly throughout (Sections 4 and 5). Fourteen instruments (BRA13–BRA27) form a dense aperture
of roughly 12 km across the edifice and carry five channels — a 200 Hz vertical (ELZ), two 100 Hz
horizontals (SL1/SL2), a 200 Hz differential pressure gauge (EDH) and a 1 Hz hydrophone channel
(LDH). Eight broadband instruments (BRA02–BRA11) are distributed over the wider basin at 20–90 km
offsets and carry four channels (HHZ, HH1, HH2, HDH), all at 100 Hz. Land stations are
three-component at elevations of 0–30 m. Station coordinates and bathymetric depths are those of
`catalogs/station_geometry.csv`.

The mixture of channel families matters operationally: the vertical and horizontal band codes differ
within a single instrument for the dense OBS family (ELZ at 200 Hz against SL1/SL2 at 100 Hz), so
any processing step that requires the three components to share a band code silently discards the
S-wave information of ~97% of the array. Vertical and horizontal channels were therefore selected
independently at every stage, each retaining its own sampling rate and time base.

### 1.2 Recording period and completeness

Continuous data span 2019-01-10 to 2020-02-19 (403 days). Per-station recording was close to
complete: the ZX instruments contributed 385–404 station-days each, 8,786 ZX station-days in total,
and the land network a further 5,168 station-days, giving 13,954 three-component station-days and
8,786 four-component (hydrophone-bearing) OBS station-days available for picking. Zero-length daily
files (2,195 placeholders in the archive) were skipped automatically.

One timing correction is applied: the internal clock of ZX.BRA05 ran fast, and a constant +0.167 s
correction is added to every arrival time recorded on that station before any downstream use. The
correction is applied once, at the pick stage, to every pick pool and also to the analyst pick set
used for validation, so that both are in the same time frame. No other station required a clock
correction.

**Figure 1:** Location map. Bathymetry of the central Bransfield Strait with the Orca
edifice and its caldera; ZX OBS positions colour-coded by water depth; land stations of networks
5M/AI/AM; inset showing the regional setting. Produced from `catalogs/station_geometry.csv` with the
base-map code of `scripts/33_plot_nlloc.py`.

## 2. Phase picking

### 2.1 Choice of picker

Twelve automatic pickers were benchmarked against an analyst-picked reference set before any
production picking was done. The reference comprises 46,339 hand picks made on this deployment; the
benchmark was run on the ten days from 2019-10-01 onward carrying the most analyst picks (3,278
picks — 1,236 P and 2,042 S — 447 events, 20 ZX stations, 220 station-days per model). Every model
was run at a common detection-probability floor of 0.1 so that no picker is flattered by a tuned
threshold, and recall was scored by greedy nearest-time matching per station and phase.

| picker | P recall | S recall | P bias | P MAD | S bias | S MAD | picks/station-day |
|---|---|---|---|---|---|---|---|
| PhaseNet, DiTing weights | **80.6%** | 84.6% | −10 ms | **25 ms** | +23 ms | **47 ms** | 947 |
| OBSTransformer (obst2024) | 73.4% | 92.4% | +11 ms | 40 ms | +70 ms | 43 ms | 1,714 |
| PickBlue PhaseNetLight (OBS) | 73.2% | **93.2%** | +3 ms | 49 ms | +21 ms | 59 ms | 2,069 |
| PickBlue PhaseNet (OBS) | 65.9% | 75.0% | −1 ms | 32 ms | +41 ms | 51 ms | 1,186 |
| STA/LTA reference | 60.2% | 79.9% | +60 ms | 36 ms | +90 ms | 122 ms | 2,567 |
| PhaseNet, STEAD weights | 59.4% | 63.6% | +7 ms | 25 ms | +62 ms | 52 ms | 198 |
| PickBlue EQTransformer | 57.7% | 81.3% | −29 ms | 49 ms | +58 ms | 74 ms | 1,004 |
| PhaseNet, INSTANCE weights | 51.5% | 52.2% | −10 ms | 38 ms | +18 ms | 60 ms | 210 |
| EQTransformer, INSTANCE | 50.7% | 47.8% | +21 ms | 50 ms | +88 ms | 71 ms | 272 |
| EQCCT | 47.5% | 57.8% | — | — | — | — | 226 |
| BasicPhaseAE, STEAD | 43.0% | 26.9% | −34 ms | 130 ms | −108 ms | 272 ms | 3,339 |

Models fail on different arrivals, so an exhaustive search over all one-, two- and three-model pools
was carried out. The two-model pool **PhaseNet (DiTing weights) + PickBlue PhaseNetLight**, both
contributing P and S, reaches 89.2% P and 96.0% S recall — 11.4 and 3.6 points above the best single
model and above any alternative pair — for 3,016 picks per station-day. A third model
(OBSTransformer) adds 1.9 points of P for a further 1,700 picks per station-day and was rejected on
that trade; its S picks were rejected outright because they run +70 ms late, a bias that would enter
S−P times and therefore depth. This pool was adopted for production.

Sampling was checked: for the two models with full-year picks, the ten benchmark days reproduce the
recall measured over all 123 days of the late period (11,308 picks) to within 1.1 percentage points.
Differences below ~2 points are not resolved by 447 events.

### 2.2 Production picking

Both models were run over the full deployment at a P and S probability threshold of 0.1,
deliberately low so that downstream stages filter on probability rather than re-pick.
PhaseNet/DiTing is three-component (ZNE, 50 Hz internal resampling) and was run on every network;
PickBlue PhaseNetLight is four-component (Z12H — three seismometer channels plus the hydrophone,
with a 0.5 Hz highpass on the pressure channel and per-channel standard deviation normalisation) and
was run on ZX only, since only the OBS carry a pressure channel. An ablation on one station-day
confirms that the fourth channel does real work: P recall 50.7% with the hydrophone against 46.3%
without, S recall 67.0% against 56.0%.

The two pools yield 9,638,456 and 23,043,906 candidate picks respectively above the 0.1 threshold
(32.68 M in total, counted directly from the pick archives). Roughly 30% of the DiTing picks have a
PickBlue twin; duplicates are removed at association time (Section 3).

**Figure 2:** Picker benchmark. Panel (a) P and S recall against the analyst set for the
twelve models; panel (b) pick-time residual distributions (bias and MAD) for the four leading
models; panel (c) recall of the exhaustive model pools against picks per station-day, with the
adopted pool marked. Source: `notes/picker_benchmark.html` and the benchmark metrics behind it.

**Figure 3:** Example events: raw picker output, associated picks and analyst picks. Four strict-tier events (a shallow caldera event at 1.41 km below the seafloor, a deeper event at 4.39 km, a small eight-phase event, and an event from the analyst's trusted window) shown as 3–20 Hz vertical-component record sections of the nearest stations. Faded dotted ticks: every pick the two pickers produced in the window (P above the trace, S below); solid ticks: the picks pyocto associated to the event; black dashed ticks: the analyst's manual picks; thin grey: the final predicted arrival times. Across the four windows 161 raw picks were reduced to 78 associated picks, the surplus being the two pickers detecting the same arrival and unassociated detections; automatic minus manual pick-time medians are 45, 30, 35 and 46 ms. Produced by `scripts/72_example_waveforms.py`.

## 3. Association

Phase association used pyocto with a 1D reference velocity model (`configs/velocity_model.csv`;
1.456 km/s water to 1.30 km, rock top 2.35 km/s, 6.51 km/s by 7.3 km depth below sea level, Vp/Vs
1.78).

Parameters: at least 3 P picks, 2 S picks and 6 phases in total per event; pick match tolerance 0.5
s; EDT pick standard deviation 0.5 s; time slicing 1200 s; pre-event search window 180 s; source
depth range 0–40 km; three refinement iterations. Cross-picker duplicates are removed before
association with a 0.25 s tolerance, keeping the higher-probability member of each pair; on a
representative day this removes 4,771 of 62,704 picks (7.6%).

Three implementation choices are load-bearing and are recorded because each of them is silent when
wrong. (i) The map projection is frozen — a transverse Mercator with origin (−62.5°, −58.8°) on
WGS84 — rather than recomputed from whichever stations happen to carry picks in a given chunk, which
otherwise moves the frame by up to 19.9 km between pools. This projection is **not** the same as the
one used for location (Section 5), so only geographic coordinates and the event index may cross
stage boundaries. (ii) The velocity model is resampled onto a 0.1 km slowness-averaged grid before
the travel-time table is built; at coarser sampling only 15 of 68 layers survive and predicted S−P
is 1.4–1.5 s too small at all distances. (iii) The water column is filled with rock velocity in the
association model, because otherwise land stations at zero depth acquire ~2.6 s of fictitious S
delay. The year was associated in daily chunks with a 120 s margin, six processes of five threads
each.

**Airgun survey.** An active-source survey operated inside the array in early 2019. Shots
fire every 17.7 s on average (10th–90th percentile 16.2–19.3 s), so a temporal matching window wide
enough to catch them spans the entire time axis, and tight matching (±2 s and 15 km) recovered only
6,219 of ~17,000 shots while leaving 1,161–1,994 events per day against a background of ~130 per
day. The whole interval **2019-01-21 to 2019-02-05** (first and last shot ±1 h) is therefore
excluded from the catalogue rather than filtered. Events in that window are flagged and retained in
the association output, not deleted. The cost is estimated at ~1,800 genuine earthquakes discarded
with the shots, and the catalogue is explicitly incomplete over that window.

| stage | events |
|---|---|
| associated over the full deployment | 98,631 |
| excluded with the airgun window | −18,848 |
| **input to location** | **79,783** |

The 79,783 events carry 818,767 associated picks (409,090 P and 409,677 S).

## 4. Velocity model and travel times

### 4.1 The 3D P model and its datum

Absolute locations use a 3D P-wave tomographic model of the Orca edifice derived from this
experiment by Stingray-style ray tomography. The model is honestly resolved to approximately **4 km
below the seafloor**; below that, values relax toward the 1D starting model and are not
independently constrained. This limit is carried through the whole analysis and is quoted with every
depth statement below.

The tomographic cube is *sheared*: its node columns hang from the local seafloor, so its vertical
axis is depth below the seafloor and the cube drapes over bathymetry. Three independent
confirmations were required before acting on this — the tomography package's own documentation; the
depth of the 4 km/s contour, whose regression on water depth has slope +0.007 km/km against ≈ +1.0
for a sea-level frame; and that same contour reaching 1.60 km depth where the water column reaches
1.96 km, which a sea-level frame cannot produce. Rays were traced in true geometry, so the shear is
undone to recover a sea-level-referenced model:

    V(x, y, z) = V_sheared(x, y, z − w(x,y))   for z ≥ w(x,y)
    V(x, y, z) = V_sheared(x, y, 0)            for z <  w(x,y)

where w(x,y) is the water depth, taken from a 30 m multibeam grid of the Orca edifice inside its
footprint (35,947 nodes) and from GEBCO_2023 outside it (223,650 nodes), blended across the
boundary. The water column is filled with the local top-rock velocity rather than with 1.5 km/s: no
first arrival in this experiment crosses seawater (sources are sub-seafloor, OBS sit on the
seafloor, land stations are above it), and a true water layer would act as a slow trap in the
eikonal solution. The un-sheared surface reproduces the independently measured station depths to a
mean absolute difference of 13 m (maximum 38 m).

The resulting grid is 576 × 451 × 64 nodes at 0.4 km spacing, origin (−150, −110, 0) km in a
transverse-Mercator frame with origin (−62.4413°, −58.44°) rotated 36°. Stations are placed at their
**true** depths (0.785–1.943 km for the OBS, land stations clamped to 0), which leaves 0.8–1.9 km of
model above the seafloor at each OBS and allows a shallow caldera source to sit legitimately above a
deep-water receiver — a configuration the sheared model cannot represent, and one that is physically
required under a volcanic edifice surrounded by deeper water.

### 4.2 Vp/Vs

S travel times are computed from P times through a constant ratio. The ratio was fixed at
**1.78** from a Wadati regression over 479 events with r² > 0.95, which returns 1.795, and
not by sweeping the ratio for a preferred location outcome: raising it from 1.78 to 2.30 reduces
grid-face pinning from 24.1% to 2.9% while the root-mean-square residual *rises* from 0.236 to 0.302
s, and a P-only relocation makes pinning worse (32.7% against 24.2%), confirming that S−P is what
constrains depth here. The Wadati excess of 1.86–1.90 seen on paths whose nearest-station S−P is
below 1 s is not applied as a bulk ratio; it is absorbed by the per-station S terms of Section 6,
which is where the data place it (Section 6.2).

### 4.3 Travel-time grids

Travel times were computed with a **point-source fast-marching eikonal solver** (pykonal) rather
than with the finite-difference scheme distributed with the location code, because the latter was
found to be biased against the model it was computed from. Three independent checks establish this:
the finite-difference grids run +66 ms slower (median; range −5 to +134 ms by station) than the
vertical slowness integral beneath each station, constant with depth below 1.6 km; re-integrating
with the cell-slowness convention used internally by that scheme (each node's slowness applied to
the cell below it, i.e. a half-cell downward shift of the model) removes the median offset but not
the station-to- station spread; and, on 4,954 real source–receiver paths, the finite-difference
times exceed the fast-marching times by +137 ms (MAD 19 ms), growing from +108 ms at 0–2 km to +161
ms at 10–25 km. A third, structurally different forward solver — the pseudo-bending ray tracer used
for the double-difference relocation (Section 7) — agrees with the fast-marching solution to +6.5 ms
(MAD 8.5 ms). Two independent solvers agreeing with each other and with the model to under 10 ms,
against one that does not, is the basis for the choice. Because S is computed as P × Vp/Vs at
location time, the bias would have entered S times amplified by 1.78.

The fast-marching solution is itself discretisation-limited at the native 0.4 km node spacing, where
the first-order scheme runs up to 60 ms *fast* at stations sitting in the steep near-seafloor
velocity gradient (e.g. −56, −61, −63 ms at 3, 6 and 10 km depth beneath one station). The grids are
therefore solved on a 0.2 km trilinear upsample of the velocity field and decimated back to 0.4 km
for storage, which reduces the same offsets to +1 to +7 ms. Each station's grid takes 224–323 s to
build with a peak of 13.7 GB.

Two stations remain outliers against the reference tracer after refinement and are flagged rather
than corrected: one inside the filled water column, where both solvers disagree (−35.4 ms, MAD
23.6 ms), and one far station at 32 km where the comparison tracer's own nodes are 10–20 km apart
(+68.9 ms, n = 452), so the tracer rather than the grid is the suspect. All other stations agree with
the tracer to within −13.6 to +13.3 ms.

**Figure 4:** Velocity model. Panel (a) map of the un-sheared model at 2 km below sea
level with the 1000 m bathymetric contour; panels (b–c) two orthogonal cross-sections through the
caldera showing the seafloor surface, the resolved depth interval (to 4 km below seafloor) shaded,
and the OBS at their true depths. Produced by `scripts/37_plot_velocity_slice.py` on the grid built
by `scripts/41_build_unsheared_velgrid.py`.

**Figure 5:** Forward-model verification. Travel-time difference against source–receiver
distance for (a) the finite-difference grids minus the fast-marching grids on real paths, (b) the
pseudo-bending tracer minus the fast-marching grids, and (c) per-station offsets of each solver
against the vertical slowness integral. Produced by `scripts/58_eikonal_check.py` and
`scripts/57_compare_3d_tt.py`.

## 5. Absolute location

Absolute hypocentres were computed with NonLinLoc using the octree importance-sampling search
(`LOCSEARCH OCT 9 7 5 0.001 100000 5000 0 0`) and the Gaussian-analytic likelihood with a fixed
Vp/Vs of 1.78 (`LOCMETH GAU_ANALYTIC 100 4 -1 -1 1.78 -1 -1 1`), an a-priori pick uncertainty of 0.1
s (`LOCGAU 0.1 0.0`), and quality-to-error mapping `LOCQUAL2ERR 0.1 0.5 1.0 2.0 99999.9`. The search
grid spans the full travel-time grid: 576 × 451 × 63 nodes at 0.4 km, i.e. x ∈ [−150, +80] km, y ∈
[−110, +70] km and z ∈ [0, 24.8] km below sea level in the frame `TRANS SIMPLE −62.4413 −58.44 36`.
Station depths in the control file are the true bathymetric depths of Section 4.1, positive down;
land stations are clamped to zero elevation. All 38 stations with travel-time grids are used. The
year was located in 16 shards, each shard's control file verified by textual substitution before
launch.

Each located event is written with its maximum-likelihood hypocentre, origin time, the marginal 1σ
standard deviations √Cov_XX, √Cov_YY, √Cov_ZZ from the posterior, the horizontal error ellipse, the
weighted-residual RMS, the phase count, the azimuthal gap, the epicentral distance to the nearest
station, and the solver's own LOCATED/REJECTED flag. Depths are **below sea level**; a second depth
column gives depth below the **local** seafloor, computed from the 30 m bathymetric grid where it
covers the epicentre (49,224 of 79,503 events) and from the 0.4 km model water-depth surface
elsewhere. Structural statements are made on the below-seafloor depth, since depth below sea level
mixes in 0.8–1.9 km of bathymetric relief.

**Hypocentre-to-event mapping.** Each solver output file is matched to the association
event that shares its picks (station, phase, date, hour-minute and second), not to the k-th event in
the input order. Positional joins fail silently and catastrophically whenever the solver drops an
event: a positional mapping shifts every subsequent event in a shard onto the wrong solution, which
in a trial run left 64% of origin times displaced by a median of 2,271 s while depths and residuals
appeared normal. Events with no solution are reported explicitly and excluded rather than absorbed.
A regression against a run with no dropped events reproduces the positional mapping slot-for-slot,
so the content-based mapping is not itself introducing an offset.

**Quality tiers.** Three nested tiers are defined. All require a LOCATED status and a
hypocentre more than 0.5 km from every face of the search grid, since boundary-pinned solutions are
unconstrained in that direction and carry artificially small formal errors.

| tier | cuts |
|---|---|
| loose | gap < 200°, RMS < 0.7 s, N_phases ≥ 4, depth below local seafloor > −0.2 km |
| standard | gap < 180°, RMS < 0.5 s, N_phases ≥ 6, depth below local seafloor > −0.2 km |
| strict | gap < 120°, RMS < 0.3 s, N_phases ≥ 8, σx < 1 km, σy < 1 km, σz < 2 km, inside the convex hull of the OBS array, depth below local seafloor > +0.2 km |

The −0.2 km seafloor tolerance is half a grid cell; an event above the local seafloor is in the
water column and cannot be an earthquake. This test is applied against the 30 m bathymetry directly
where available, because bilinear interpolation of the 0.4 km water surface alone leaves tens of
events above their true local seafloor. After the correction, no event in any tier lies more than
0.2 km above its local seafloor.

## 6. Station terms

### 6.1 Why station terms, and how they were derived

The tomographic model is resolved only in the shallow section, the S structure is not independently
resolved at all, and the array sits on 1.16 km of seafloor relief above sediments of unknown and
laterally variable thickness. Rather than edit the velocity model outside its resolved interval, the
unmodelled near-receiver structure is carried by per-station P and S time terms.

The terms were derived from a joint hypocentre–velocity–station-correction inversion of VELEST type,
run on a sea-level datum with stations at their true depths (the code truncates its top layer at the
receiver depth, so no datum shift is required) and on a rock-only column, the interval above the
model's rock top being filled with the shallowest rock velocity as in the location model. Sixteen
layers were used, with tops at −0.05, 0.78, 1.30, 1.70, 2.10, 2.50, 3.00, 3.50, 4.00, 5.00, 6.50,
8.00, 11.00, 16.00, 21.00 and 31.00 km below sea level and travel-time-preserving
(slowness-averaged) layer velocities. An independent S model and independent S station corrections
were allowed, so that the depth dependence of Vp/Vs is an output rather than an assumption.

The input is the strict tier of the reference location run made without station terms: 1,576 events
and 23,116 observations (10,880 P, 12,236 S) within 150 km. Because the inversion solves a single
normal system of 4·N_events + N_layers + N_corrections unknowns in single precision, the full set
(6,374 unknowns) does not factorise; following standard practice the minimum-1D model was derived on
a well-recorded, depth-stratified subset of 377 events (median 18 observations each, ≥ 4 stations
with both P and S, at most 6 events per 1.5 km cell) and the full set was then relocated with the
inverted model and corrections held fixed. Damping was chosen by sweep over twelve configurations
(final accepted RMS 0.214 s at velocity damping 1, 0.170 at 3, 0.153 at 10, 0.151 at 30, divergence
at 100); the adopted configuration damps origin time, epicentre, depth and station terms at 0.01 and
velocity at 30, with low-velocity layers disallowed. All inversions terminate on the code's
step-halving convergence test.

### 6.2 What the inversion resolves, and what it does not

**The absolute velocity is not resolved.** Starting models perturbed by ±10% survive the
inversion almost untouched (Vp at 4 km below sea level: 5.39 / 5.99 / 6.61 km/s from the slow,
reference and fast starts — a 15–21% spread from a 20% spread of starts) and trade directly against
the station corrections (OBS P corrections span −1.23 to +0.15 s from the slow start and −0.05 to
+0.98 s from the fast start, against −0.21 to +0.19 s from the reference start). The classic
velocity / origin-time / station-term degeneracy is fully present, and the ±10% cross-start spread
is quoted as the honest uncertainty. The inverted 1D model is used for relocation and for relative
structure; it is not quoted as an absolute crustal velocity profile, and the low Vp/Vs it returns
for the top 5 km (1.60–1.70) is a fitting parameter, not a measurement — it disagrees with the
Wadati estimates from the same data (1.795 globally, 1.86–1.90 on near paths) and arises because the
inversion has two ways to represent one observation.

**The S station delay is the robust output.** Every OBS in the array requires a large
positive S correction (+0.21 to +1.25 s) while its P correction stays small (−0.21 to +0.19 s). The
OBS P corrections correlate with water depth at r = +0.71 (r = +0.76 when hypocentres are held
fixed), which is the expected signature of a 1D model spanning 1.16 km of seafloor relief. The S
excess is not: it is a station-side delay of roughly half a second beneath the whole array that P
does not see.

This is confirmed without any relocation. Regressing the observed nearest-station (< 4 km) S−P on
hypocentral distance gives a slope of 0.063 s/km with an intercept of +0.539 s using the reference
locations, and 0.105 s/km with +0.439 s using the relocated set. For any plausible crustal Vp/Vs the
slope should be (Vp/Vs − 1)/Vp = 0.29 s/km at Vp = 3 km/s, 0.176 at 5 km/s and 0.147 at 6 km/s. The
observed near-station S−P is dominated by a constant station term of +0.44 to +0.54 s, not by path
length — the classic signature of a low-Vs sedimentary layer beneath an OBS. Whether the delay is
sediment Vs or a systematic S *pick* bias (a converted or later phase picked at the OBS) is not
resolved by these data; both produce this signature, and both are absorbed identically by a station
term.

### 6.3 Transfer to the location, and the S-before-P guard

The corrections are transferred to the absolute location as per-station P and S delays applied as
`observed − delay` (positive = station observes late), which is the same sign convention the joint
inversion uses, so they map one-to-one. One refinement iteration was then applied: the per-station
median weighted residual of a location run using the inversion terms is added to those terms, with a
guard requiring at least 50 readings per station and clipping any term at 1 s in absolute value
(without it, a far land station with few readings ran away to −2.06 s over successive iterations).
One refinement is used, not more: further iterations let the location progressively re-absorb the
terms (the residual shift per station is only 20–30% of the imposed delay), converging to small S−P
terms that lose the physical signal.

The adopted set is 76 delay lines over 38 stations. On the OBS the P terms span −0.124 to +0.157 s
(median +0.046 s), the S terms +0.174 to +1.000 s (median +0.438 s), and the S−P terms +0.168 to
+0.933 s (median +0.372 s). Land-station terms are smaller and of mixed sign (P median −0.340 s, S−P
median +0.025 s).

A constant S term can exceed the observed S−P for an event very close to its nearest station. The
corrected S then lands before its own P and the location code aborts the whole event ("cannot find
companion arrival"), which removed 4.6% of a trial set. Since such a pick pair is by construction
inconsistent with a constant station S delay, the S pick is the one to drop: S picks whose corrected
S−P would fall at or below 0.05 s are deleted
**before** location. Over the year this removes **20,140 S picks (4.9% of all S picks) in
13,979 of 79,783 events (17.5%)**, and leaves 141 events with no S pick at all. No event is deleted
by the guard. The consequence is stated as a caveat, not hidden: the guard removes the S constraint
preferentially from the closest station of the shallowest events, which is exactly the geometry that
best constrains depth.

### 6.4 Gate protocol and the choice of terms

No station-term variant was adopted on the strength of its residuals. A fixed sample of 2,000 events
was relocated with each candidate set, every run scored with its own delays, and all variants
compared on the same five diagnostics: the number of events located, the fraction pinned against the
top face of the search grid, the near-station P residual of the pinned population, the median RMS,
and — the decisive one — the ratio of the *predicted* to the *observed* spread of nearest-station
S−P between the shallow and deep depth bins. That ratio is the model-robust form of the question "is
the depth axis the right length?": it compares the S−P moveout the locations predict with the S−P
moveout actually measured, and it is insensitive to the absolute S−P level that the station terms
themselves set.

| variant | located | top-pinned | near-station P residual (pinned) | RMS p50 (s) | S−P pred / obs | **ratio** | depth p10/50/90 (km) |
|---|---|---|---|---|---|---|---|
| reference run, no station terms | 1,994 | 28.5% | −0.160 | 0.239 | 3.51 / 2.80 | 1.25 | 0.00 / 1.20 / 19.2 |
| inversion A terms (corrections only) | 1,906 | 24.4% | −0.170 | 0.258 | 3.14 / 2.58 | 1.22 | 0.00 / 1.68 / 19.1 |
| inversion A + 2 refinements | 1,874 | 23.2% | −0.138 | 0.234 | 3.26 / 2.66 | 1.23 | 0.00 / 1.70 / 18.7 |
| inversion B terms (joint) | 1,743 | 18.4% | −0.129 | 0.257 | 3.00 / 3.08 | 0.97 | 0.01 / 2.40 / 19.6 |
| **inversion B + 1 refinement (adopted)** | 1,775 | 18.9% | −0.127 | **0.237** | 3.14 / 3.04 | **1.03** | 0.01 / 2.09 / 17.8 |
| inversion B, Vp/Vs 1.70 | 1,751 | 25.4% | −0.151 | 0.258 | 2.89 / 2.83 | 1.02 | 0.00 / 1.87 / 19.5 |
| S−P terms scaled ×0.4, guarded | 1,993 | 23.9% | −0.164 | 0.251 | 3.09 / 2.75 | 1.12 | 0.00 / 1.60 / 18.0 |
| S−P terms scaled ×0.6, guarded | 1,994 | 22.2% | −0.148 | 0.252 | 2.93 / 2.83 | 1.04 | 0.00 / 1.66 / 17.4 |
| S−P terms scaled ×0.8, guarded | 1,997 | 21.0% | −0.137 | 0.254 | 2.85 / 3.01 | 0.95 | 0.00 / 1.87 / 16.9 |
| Vp/Vs 1.70 + terms ×0.4 | 1,994 | 34.7% | −0.177 | 0.256 | 3.24 / 2.88 | 1.13 | 0.00 / 0.86 / 17.1 |
| Vp/Vs 1.70, no terms | 1,997 | 38.2% | −0.162 | 0.249 | 3.22 / 2.34 | 1.38 | 0.00 / 0.65 / 20.0 |

Two results follow. First, changing the bulk Vp/Vs ratio does not do the job: lowering it to 1.70
without station terms makes every diagnostic worse (ratio 1.38, pinning 38.2%). The lever is the
*per-station* S term, not the ratio, and the tomographic P model and Vp/Vs = 1.78 are left
untouched. Second, the S−P spread ratio — which had resisted five separate model changes, including
a 10% slower and a 15% faster shallow P section, a Vp/Vs of 1.90, and the change of travel-time
solver, all of which left it at 1.25–1.31 — moves to 0.95–1.04 with station S terms of the inverted
scale.

The adopted variant is the joint-inversion terms with one refinement, applied together with the
S-before-P guard: it gives the best RMS of any variant (0.238 s, marginally below the 0.239 s of the
run without terms), reduces top-face pinning from 28.5% to 20.1%, reduces the near-station P
residual of the pinned population from −0.160 to −0.129 s, and brings the S−P spread ratio to
**0.95** on the guarded confirmation run, with no loss of located events (1,992 of 2,000 against
1,994 for the reference run).

**Figure 6:** Station terms. Panel (a) map of the array with P and S delay terms as scaled
symbols; panel (b) OBS P term against water depth with the r = +0.71 regression; panel (c) observed
nearest-station S−P against hypocentral distance with the fitted line, the zero-distance intercept,
and the slopes expected for Vp/Vs = 1.78 at Vp = 3, 5 and 6 km/s. Produced by
`scripts/65_make_locdelay.py` output and `scripts/68_manual_sp_vs_delays.py`.

**Figure 7:** Gate protocol. Predicted against observed nearest-station S−P by depth bin,
for the reference run without station terms and for the adopted run, with the spread ratio
annotated. Produced by `scripts/52_sp_depth_check.py` driven by `scripts/67_gate_table.py`.

## 7. Relative relocation

Relative geometry was obtained by double-difference relocation (hypoDD 2.1b) of the standard tier,
in 3D mode (IMOD = 9), using the pseudo-bending ray tracer of the simulps family with ray parameters
`2 9 2 0.5 1.0 1.35 0.0005 50` and a path scale of 0.5 km.

**Node model.** The 3D tracer interpolates trilinearly between nodes, so the 0.4 km
location grid was resampled onto a graded node set in the relocation code's own projection (x east,
y north, no rotation, origin −62.4413° / −58.44°, verified by evaluating the projection through the
code's own routines), with z positive down from **sea level**. The production model uses 1.0 km
lateral nodes over a ±20 km core and 0.2 km vertical nodes to 4 km depth, 59 × 59 × 37 nodes in
total, with far padding nodes so that no ray point out to the most distant station falls off the
model — the tracer halts if one does. Node spacing matters: a first, coarser model (2.5 km lateral,
0.5 km vertical) ran ~125 ms fast against the location grids and was rejected. Vp/Vs is a constant
1.78 supplied as its own grid, because the alternative input mode hard-codes 1.73 internally.

**Two source-level requirements.** The relocation code zeroes negative station elevations
for every mode except one, which in 3D mode would have placed every OBS at sea level, 1.5–1.9 km
above the seafloor, while the 3D partial-derivative routine correctly places the receiver at z =
−elevation; the elevation handling was patched for the 3D mode so that stations enter at their true
depths. The ray tracer also caps path subdivisions at seven (≤ 129 ray points) irrespective of the
path scale, which is what keeps long land-station rays from overflowing its fixed-size path array.
The 3D binary is built separately with enlarged model arrays (80 × 80 × 40) and reproduces the
production binary exactly on a frozen 1D test case.

**Trial sources.** Trial sources are the catalogue hypocentres (ISTART = 2). This is
verified, not assumed: with a noise-free synthetic differential-time dataset built from the input
locations, a frozen run (damping 10⁶, one iteration) returns RMSCT = 4 ms and moves nothing, whereas
the alternative single-trial-source mode returns RMSCT = 283 ms with the residual equal to the
observed differential time for 100% of 4.13 M rows — i.e. predicted differential times identically
zero, because the code collapses to one source at the cluster centroid. Relocating the noise-free
synthetic at production damping recovers the input depths with a regression slope of (relocated −
start) on start depth of **+0.001**; adding 0.15 s Gaussian pick noise gives −0.014, and adding a
further 5% of ±0.5 s outliers gives −0.023. These figures set the inversion's own depth compression
baseline at −0.01 to −0.02, against which the real-data value in Section 12 must be read. The
synthetic chain was itself validated: a travel-time table built on the relocation model passes three
self-checks (interpolation error ≤ 3 ms beyond 2 km), the synthetic differential times correlate at
0.76 (P) and 0.81 (S) with the observed ones, and an independent driver built on the code's own
travel-time routines reproduces them to −0.4 ms median and 12 ms RMS over 300,000 rows.

**Damping** was chosen by the condition number of the least-squares system, which hypoDD
reports at every iteration and for which its manual recommends 40–80. Damping 200 leaves the
condition number at 247 (under-damped) and visibly inflates the depth response; damping 400 gives
141 → 109 on the same data; damping 800 did not complete in 82 minutes and was abandoned. Damping
400 is used throughout, and the 200/400 spread is carried as part of the uncertainty on the relative
depth scale.

**Run and QC.** Pair formation (ph2dt) reduced 12,194 standard-tier events to 11,644 events
and 11,061 trial sources. The inversion relocated **10,664 events (92%)**, taking RMSCT from 174 ms
to 59 ms and the condition number from 159 to 137. Quality control (script 50) then removes events
that finish above their local seafloor — 1,317 (12%), a real failure mode in 3D mode since it has no
datum clamp — and events with too few differential-time links (256), leaving **9,125 events** in the
delivered relative catalogue. Depth below local seafloor of the surviving set has 10th/50th/90th
percentiles of 0.15 / 2.15 / 3.89 km. This is the catalogue-differential solution, and it is the
more complete of the two relative products; a second relocation using the same configuration plus
waveform cross-correlation differential times is described in Section 8 and delivers 8,005 events
over a better-linked core.

## 8. Cross-correlation differential times

Waveform cross-correlation differential times were measured on the same event pairs as the
catalogue-differential run of Section 7 and used for a second, refined relative relocation. Both
relative catalogues are delivered: the catalogue-only solution for completeness, and the
cross-correlation-refined solution for cluster-scale geometry.

**Measurement.** Windows are cut at the pick — P from −0.3 to +0.7 s, S from −0.4 to +1.0 s, with
the slave window padded by the maximum lag — and a 3–20 Hz zero-phase Butterworth filter is applied
once per station-day, so that no window contains a filter transient. The correlation is a *sliding*
normalised Pearson coefficient in which the denominator uses the moving mean and variance of the
slave under the window. This is deliberately not a conventional full-mode correlation of two
separately normalised windows, which applies a triangular (n−|L|)/n taper and drags the apparent
peak toward zero lag; here every lag is normalised over exactly the same number of samples. The peak
is refined by a three-point parabolic fit, but the reported coefficient is always the sampled peak,
so cc ≤ 1 by construction. P is measured on the vertical and S on both horizontals, with the two
normalised correlation functions stacked before peak picking, keeping |lag| ≤ 0.5 s. A self-test
that imposes an exact fractional delay on a real trace recovers it to a worst-case error of
0.019 ms, and the pick times used to cut the windows reproduce the travel times already written into
the differential-time file to 0.3 ms (MAD, over 75,012 travel times).

Vertical and horizontal band codes are selected independently per station and the window is keyed by
the full channel code, so that two events are never correlated on different components. This is not
cosmetic: requiring the three components to share a band code — which they do not on the dense OBS
family — returns zero S measurements and does so silently.

**Threshold, set by a null test.** The coefficient threshold was chosen by measuring the chance
floor rather than by convention. At one station over 26 days, 2,989 real pairs were compared against
the same rows with the second window replaced by a random other window of the same phase at the same
station. At cc ≥ 0.7 the adopted windows give a purity of 0.74 (P 0.74, S 0.73) — roughly one
measurement in four is a chance peak — rising to 0.80 at cc ≥ 0.8. Longer windows do not improve
purity and cost yield. The chance floor is high because a 1 s window at 3–20 Hz carries only ~17
independent samples while the search scans ~100 lags, and the median correlation of *unrelated*
windows at this station is already 0.40. The production threshold is therefore **cc ≥ 0.8**.

**Yield.** On the delivered event pairs the correlator produced **103,056 differential times in
64,994 pairs** (473 GB read in 36.2 minutes on six workers): 55,463 P rows at a median coefficient
of 0.860 and 47,593 S rows at 0.847. The correction the correlation applies to the catalogue
differential time has a median of +2.0 ms with a MAD of 55 ms for P and +4.4 ms with a MAD of 51 ms
for S — a near-zero bias, as it should be if the picks are unbiased, and a scatter consistent with a
35–40 ms per-pick error entering a difference of two picks.

**Refined relocation.** The relocation was repeated with the same 3D model, trial sources, damping
and quality control, adding the correlation data with positive weights in every iteration block and
with the correlation minimum-observation parameter set to zero (see below). Of 11,061 trial sources,
**9,378 events (81%) relocated**, against 92% for the catalogue-only run: the correlation-weighted
iteration sets drop weakly linked events, which is the intended behaviour and the reason both
catalogues are delivered. RMSCT went from 174 to 106 ms, the correlation RMS from 189 ms to 1 ms,
and the condition number from 160 to 129. Quality control removed 894 events (9.5%) finishing above
their local seafloor and 689 with fewer than 20 links, leaving **8,005 events**, with depth below
local seafloor at 0.20 / 2.32 / 4.10 km (p10/p50/p90).

**Two control-file behaviours** were established by reading the source, because each silently
produces a plausible wrong answer. Setting the correlation weight negative in warm-up iteration
blocks — a common way to express "unused" — makes every correlation weight negative, after which the
code permanently compacts those observations out of its arrays, leaving the later blocks nothing to
weight; the correlation data must instead keep a positive weight throughout while the *cutoffs* are
staged. Separately, the pair-linking threshold is the **sum** of the catalogue and correlation
minimum-observation parameters, so raising the correlation minimum from 0 to 8 raises the linking
threshold from 8 to 16 and drops 98.5% of pairs from the clustering.

**Figure 8:** Cross-correlation. Panel (a) coefficient distribution for true and null pairs with the
purity curve and the adopted threshold; panel (b) dt.cc − dt.ct against coefficient, binned, for P
and S; panel (c) nearest-neighbour separation against correlation link count for the refined
solution. Produced by `scripts/63_xcorr_dtcc.py` and `scripts/64_run_hypodd_xc.sh`.

## 9. Uncertainty assessment

Three distinct contributions are quantified separately and are not interchangeable.

**Formal uncertainty.** The posterior marginal standard deviations σx, σy, σz are Gaussian
approximations *conditional on* the velocity model, the travel-time grids and the station terms. For
the standard tier they are 0.29 / 0.69 / 1.24 km (p10/p50/p90) in depth and 0.22 / 0.46 / 0.88 and
0.22 / 0.44 / 0.94 km in the two horizontal directions. They contain no model error.

**Station-term systematic.** Because the station terms are a fitted quantity carrying a
large part of the depth axis, their contribution to depth was measured directly. Three perturbed
term sets were built, each guarded with its own delays and relocated on the same 2,000-event sample
with control files differing from the reference *only* in the delay block and the input/output paths
(verified by diff): the corrections-only inversion terms; a set in which the P terms are kept and
the S term is set to P plus the corrections-only S−P difference (isolating the S−P term); and the
adopted terms with the S−P part scaled by 1.2. Scored on the same gate, the ×1.2 set is admissible
(spread ratio 0.92) while the corrections-only sets are rejected by it (1.12–1.19, with worse RMS
for one of them), so the perturbation family brackets the adopted choice from both sides.

| reference depth bin | n | median / p90 absolute Δz, ×1.2 terms | median / p90, corrections-only terms | median / p90, S−P-isolated terms |
|---|---|---|---|---|
| 0–1 km | 758 | 0.00 / 0.20 | 0.03 / 0.63 | 0.04 / 0.56 |
| 1–2 km | 292 | 0.12 / 0.30 | 0.43 / 1.30 | 0.44 / 0.88 |
| 2–4 km | 375 | 0.11 / 0.40 | 0.59 / 1.72 | 0.52 / 1.55 |
| 4–6 km | 189 | 0.23 / 0.57 | 1.00 / 2.23 | 0.95 / 2.03 |
| 6–9 km | 116 | 0.24 / 0.72 | 1.00 / 3.86 | 0.96 / 2.68 |
| 9–15 km | 54 | 0.41 / 0.84 | 1.56 / 3.21 | 1.55 / 3.24 |
| all | 1,990 | 0.08 / 0.43 | 0.37 / 1.91 | 0.38 / 1.63 |

Inside the family the gate still accepts, the station terms therefore contribute a median depth
systematic of ≤ 0.12 km above 4 km and 0.23–0.41 km below it (p90 0.20–0.84 km), comparable to or
smaller than the formal σz at all depths. Taking the gate-rejected sets as an outer bound, the
systematic is 0.03–0.59 km above 4 km and 0.95–1.56 km below it, with p90 reaching 3.9 km in the 6–9
km bin — **below ~4 km this exceeds the formal σz p90 of 1.1–1.3 km, so the formal uncertainty
understates the true depth uncertainty for the deep part of the catalogue.** The response is
sign-consistent and depth-graded: weaker S−P terms push the 1–2 km population down by 0.37–0.42 km
and the > 4 km population up by 0.4–1.4 km, and stronger terms do the reverse. The per-event
envelope over the three perturbations has a median of 0.10 km in the 0–1 km bin, rising to 0.70 km
at 2–4 km, 1.16 km at 4–6 km and 1.82 km at 9–15 km (p90 2.4–4.9 km in the deepest bins). The 0–1 km
bin is dominated by top-face-pinned events, whose depth cannot move downward; the unpinned 0.05–1 km
subset (median 0.15–0.34 km, p90 0.28–0.87 km) is the honest shallow figure.

**Estimator disagreement on the depth scale.** The absolute and relative estimators do not
agree on the length of the depth axis (Section 12). This disagreement — not the formal σz — is the
quantity to quote for depths in the 4–9 km range.

**Model resolution.** Independently of all of the above, the P model is resolved to ~4 km
below the seafloor and the S structure is not independently resolved at all, being carried by a
constant Vp/Vs and the per-station S terms. Depths beyond that interval relax toward the 1D starting
model and are an upper bound rather than a measurement. The joint inversion reaches the same
conclusion from the other direction: absolute Vp below ~4 km is unresolved, with ±10% starting
perturbations surviving and being absorbed by the station terms.

**Figure 9:** Depth uncertainty budget. Formal σz, the station-term envelope, and the
absolute-versus-relative depth difference, all as functions of depth, on one axis. Produced by
`scripts/68_delay_depth_uncertainty.py` and the catalogue join of Section 12.

---

# RESULTS

## 10. Catalogue statistics and completeness of processing

The processing chain is fully accounted for, event by event, from association to the delivered
tiers.

| stage | events | picks |
|---|---|---|
| associated over the deployment | 98,631 | |
| − airgun window 2019-01-21 to 2019-02-05 | −18,848 | |
| **input to location** | **79,783** | 818,767 (P 409,090 / S 409,677) |
| − S picks removed by the S-before-P guard | 0 events removed | −20,140 S (4.9% of all S) in 13,979 events (17.5%) |
| picks actually located | | 798,627 (P 409,090 / S 389,537) |
| events left with no S pick | 141 | |
| per-event solution files written | 79,733 | |
| − events the solver could not locate | −50 | |
| − solution files with no parsable geographic line | −10 | |
| **rows in the full catalogue** | **79,723** | |
| − solutions flagged REJECTED by the solver | −220 | |
| **LOCATED solutions** | **79,503 (99.6% of 79,783)** | |

| tier | n | % of LOCATED | % of associated |
|---|---|---|---|
| loose | 15,184 | 19.1% | 19.0% |
| standard | 12,194 | 15.3% | 15.3% |
| strict | 3,864 | 4.9% | 4.8% |

The tier counts were reproduced independently from the full catalogue; the recomputed standard-tier
count matches the delivered file exactly (12,194 = 12,194).

**Where the other 84.7% goes.** Applying each tier cut alone to the 79,503 LOCATED
solutions isolates the binding constraints:

| cut | survivors | cut | survivors |
|---|---|---|---|
| not pinned to a grid face | 40,565 (51.0%) | gap < 200° | 23,554 (29.6%) |
| depth below seafloor > −0.2 km | 53,063 (66.7%) | gap < 180° | 18,640 (23.4%) |
| depth below seafloor > +0.2 km | 38,916 (48.9%) | gap < 120° | 7,011 (8.8%) |
| RMS < 0.7 s | 73,390 (92.3%) | N_phases ≥ 4 | 79,503 (100.0%) |
| RMS < 0.5 s | 69,251 (87.1%) | N_phases ≥ 6 | 78,512 (98.8%) |
| RMS < 0.3 s | 46,749 (58.8%) | N_phases ≥ 8 | 68,564 (86.2%) |
| σz < 2 km | 50,483 (63.5%) | inside the OBS hull | 44,912 (56.5%) |

The binding constraints are **azimuthal gap** (only 29.6% of solutions fall below 200°, 8.8% below
120°) and **grid-face pinning** (49.0% pinned), not phase count and not residual. The array is small
relative to the region that produces associable detections, and most associated events lie outside
it.

**Completeness is set by association, not by picking.** Against the analyst reference, the
pickers recall 86.7% of P and 93.1% of S picks in the trusted window (from 2019-10-01), with timing
bias ≤ 20 ms and MAD 23–37 ms; but only 62.7% of P and 57.0% of S analyst picks reach an associated
event. Per station within that window (≥ 100 picks), raw recall ranges from 85% to 95% and catalogue
recall from 51% to 77%. A third of analyst-confirmed picks are therefore lost at association, not at
detection.

| subset | n (P / S) | P raw / in catalogue | S raw / in catalogue | P bias, MAD | S bias, MAD |
|---|---|---|---|---|---|
| all analyst picks | 18,572 / 27,767 | 77.6% / 52.1% | 90.9% / 48.8% | −16 ms, 30 ms | 0 ms, 37 ms |
| trusted window (≥ 2019-10-01) | 4,474 / 6,956 | **86.7% / 62.7%** | **93.1% / 57.0%** | −8 ms, 23 ms | +11 ms, 34 ms |
| earlier period | 14,098 / 20,811 | 74.8% / 48.7% | 90.2% / 46.1% | −19 ms, 31 ms | −6 ms, 38 ms |
| land stations | 135 / 124 | 68.1% / 46.7% | 60.5% / 24.2% | −30 ms, 70 ms | −10 ms, 100 ms |

No magnitudes were computed, so no magnitude of completeness can be quoted: `[value: compute from a
magnitude stage that does not yet exist]`.

**Figure 10:** Processing accounting. Sankey or waterfall diagram from associated events
through the airgun exclusion, the guard, the location failures and each tier cut. Produced by
`scripts/68_catalogue_accounting.py`.

## 11. Spatial, depth and temporal distribution

The standard tier contains 12,194 events between 2019-01-11 and 2020-02-18. Descriptive statistics
below were computed directly from the delivered catalogue files.

| quantity (p10 / p50 / p90) | loose (15,184) | standard (12,194) | strict (3,864) |
|---|---|---|---|
| depth below sea level, km | 1.13 / 2.68 / 6.02 | 1.13 / 2.80 / 5.91 | 1.66 / 3.20 / 5.22 |
| depth below local seafloor, km | −0.01 / 1.64 / 4.90 | 0.02 / 1.81 / 4.78 | 0.75 / 2.29 / 4.30 |
| σz, km | 0.28 / 0.68 / 1.33 | 0.29 / 0.69 / 1.24 | 0.42 / 0.71 / 1.08 |
| σx, km | 0.22 / 0.46 / 0.92 | 0.22 / 0.46 / 0.88 | 0.26 / 0.46 / 0.73 |
| σy, km | 0.22 / 0.45 / 1.05 | 0.23 / 0.44 / 0.94 | 0.26 / 0.40 / 0.68 |
| RMS residual, s | 0.085 / 0.202 / 0.383 | 0.081 / 0.189 / 0.361 | 0.072 / 0.142 / 0.236 |
| azimuthal gap, ° | 88.7 / 138.9 / 188.5 | 84.7 / 127.1 / 170.3 | 72.4 / 98.8 / 115.1 |
| phase count | 7 / 11 / 15 | 8 / 11 / 16 | 9 / 12 / 19 |
| nearest-station distance, km | 0.47 / 1.20 / 3.33 | 0.50 / 1.18 / 2.54 | 0.63 / 1.18 / 1.58 |
| local water depth, km | 0.77 / 1.04 / 1.45 | 0.76 / 1.00 / 1.40 | 0.73 / 0.91 / 1.12 |
| epicentral distance from the caldera centre (−62.4413, −58.44), km | 1.42 / 4.13 / 7.46 | 1.25 / 4.02 / 6.38 | 0.63 / 3.73 / 4.76 |
| fraction inside the OBS hull | 97.9% | 98.8% | 100% |

**Epicentres.** The seismicity is strongly concentrated on and around the edifice: 77.9% of
standard-tier epicentres lie within 5 km of the caldera centre and 95.2% within 10 km; 16.2% lie
within 2 km. The strict tier, which by construction is confined to the OBS hull, is tighter still
(95.4% within 5 km, 24.4% within 2 km). The 5th–95th percentile range of standard-tier epicentres is
−62.478° to −62.408° in latitude and −58.545° to −58.334° in longitude, i.e. a source region roughly
8 km north–south by 11 km east–west.

**Depths.** Median depth below the local seafloor is 1.81 km in the standard tier and
2.29 km in the strict tier, with 90th percentiles of 4.78 and 4.30 km. **14.8% of the standard tier
and 12.7% of the strict tier lie deeper than 4 km below the local seafloor, i.e. outside the
resolved part of the P model** (9.4% and 5.6% respectively lie deeper than 6 km below sea level).
Those depths are quoted as upper bounds. The shallow end is real and not an artefact of the seafloor
test: in the strict tier, which requires the hypocentre to be at least 0.2 km below the local
seafloor, the 10th percentile is still 0.75 km below the seafloor, and residual and formal
uncertainty vary smoothly from the deep population into the near-seafloor one.

**Temporal behaviour.** The standard tier is active on 390 of 402 days spanned, with a
median of 28 events per active day and a maximum of 100 (2019-02-11). Monthly counts rise through
the austral winter and peak in late 2019 and January 2020:

| month | 2019-01\* | 02\* | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 2020-01 | 02\* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard tier | 360 | 879 | 891 | 646 | 713 | 820 | 1,141 | 1,439 | 950 | 885 | 812 | 1,167 | 1,258 | 233 |
| strict tier | 129 | 185 | 166 | 124 | 97 | 214 | 379 | 525 | 364 | 322 | 342 | 434 | 501 | 82 |

\* January and February 2019 are incomplete (deployment start and the excluded airgun window);
February 2020 is truncated at recovery.

**Figure 11:** Epicentral map of the standard and strict tiers over the 30 m bathymetry,
with the 1000 m contour outlining the edifice, OBS positions, and events coloured by depth below the
local seafloor. Produced by `scripts/33_plot_nlloc.py`.

**Figure 12:** Depth distribution. Panel (a) histogram of depth below local seafloor per
tier with the 4 km resolution limit marked; panels (b–c) two orthogonal cross-sections with the
seafloor profile band (minimum/median/maximum across the swath) and formal σz error bars. Produced
by `scripts/33_plot_nlloc.py` and `scripts/plot_hypodd_with_seafloor.py`.

**Figure 13:** Temporal behaviour. Daily event counts for each tier, cumulative count, and
a depth-versus-time scatter coloured by distance from the caldera centre, with the excluded airgun
window shaded. Produced by `scripts/33_plot_nlloc.py` and `scripts/34_animate_nlloc.py`.

## 12. Comparison of the estimators

**The two relative catalogues.** The catalogue-differential solution contains 9,125 events and the
cross-correlation-refined solution 8,005, of which 7,612 are common; both join the standard tier on
the event index (relocation id = event index + 1) and both are dominated by a single cluster (9,057
of 9,125 and 7,969 of 8,005). The catalogue-only solution has 330 / 676 / 1,449 differential-time
links per event (p10/p50/p90), a median per-event catalogue-residual RMS of 0.106 s, and formal
relative errors of 24.5 / 34.6 / 68.8 m in x, 24.7 / 35.7 / 70.0 m in y and 28.8 / 42.7 / 76.4 m in
z. The refined solution carries 57,444 P and 41,110 S correlation links in total, a median of 1 P
and 2 S correlation links per event (p90 18 and 16), and 1,805 of its 8,005 events end with no
correlation link at all — the correlation data tighten a well-linked core rather than the whole
catalogue.

The refinement measurably sharpens relative geometry. Median nearest-neighbour 3D separation is
0.130 km in the catalogue-only solution (p10 0.054, p90 0.414 km) and 0.114 km in the refined one
(p10 0.043, p90 0.296 km); on the 7,612 events common to both — so that the statistic cannot move
because the population changed — it contracts from 0.1154 to 0.1116 km. That, not the absolute
depth, is what the refined catalogue is for: **0.11–0.13 km is the scale at which relative structure
is resolved.**

**Absolute against relative.** Relative to their absolute starting positions the catalogue-only
relocations move by a median of 0.709 km horizontally (p90 2.406 km) and 0.507 km in depth (p90
1.838 km), with a median signed depth change of only +0.013 km and a rank correlation of 0.787
between absolute and relative depth. The refined solution moves less in depth (median |Δz| 0.424 km)
and correlates better with the absolute depths (Spearman 0.886).

The estimators nevertheless disagree systematically on the *length* of the depth axis. Regressing
the depth change on the starting depth gives a slope of −0.192 over all catalogue-only relocated
events and **−0.275** (intercept +0.840 km) over the quality-controlled subset; with correlation data
the same regressions give −0.128 over all relocated events and −0.182 (intercept +0.715 km) over the
quality-controlled subset. The relative solution is thus 13–28% shorter in depth than the absolute
one, with the disagreement roughly halved once correlation data are included. Binned, for the
catalogue-only solution:

| absolute depth bin | n | median depth change |
|---|---|---|
| 0–2 km | 2,512 | +0.21 km |
| 2–4 km | 4,031 | +0.12 km |
| 4–6 km | 1,858 | −0.47 km |
| 6–9 km | 586 | −1.96 km |
| 9–15 km | 125 | −0.42 km |

Three points make this interpretable. First, the inversion's own compression under realistic noise
is only −0.01 to −0.02 (Section 7), so these slopes are a data–model signal, not an inversion
artefact; they do depend on damping, being steeper at lower damping, which is why the damping choice
is documented and the 200/400 spread is carried as uncertainty. Second, constant per-station terms
cancel exactly in a differential time, so the relative solution is blind to the station terms that
carry much of the absolute depth axis — the disagreement is precisely the part of the depth scale
those terms supply. Third, the 6–9 km bin, which moves up by ~2 km, is also the bin least stable
against the station-term perturbation of Section 9; both diagnostics point at the same population.

The delivered position follows: **absolute position and depth are quoted from the absolute
catalogue; the relative catalogues are used for geometry within clusters only**, the
catalogue-differential one where completeness matters and the cross-correlation-refined one where
cluster-scale resolution matters. Relative depths inherit the absolute frame, and their value is
relative precision, not absolute accuracy. Both relative solutions are also depth-selectively
retained — quality control removes events finishing above their local seafloor (12% and 9.5%
respectively), which are preferentially the least well recorded and shallowest edifice events — so
their depth distributions are not like-for-like samples of the absolute one.

**Figure 14:** Estimator comparison. Panel (a) relative depth against absolute depth with the 1:1
line and the fitted regression, for both relative solutions; panel (b) median depth change by
absolute depth bin with the station-term perturbation envelope overlaid; panel (c) map of horizontal
displacement vectors. Produced from the catalogue join
(`catalogs/nlloc_year_v6_standard.csv` × `catalogs/hypodd_year_v6_3d_qc.csv` and
`catalogs/hypodd_year_v6_3d_xc_qc.csv`).

**Figure 15:** Relative geometry. Map and two cross-sections of the cross-correlation-refined
catalogue at full resolution, coloured by time, with the seafloor profile, illustrating lineations
and sub-cluster structure at the 0.11 km nearest-neighbour scale. Produced by
`scripts/plot_hypodd_with_seafloor.py` and `scripts/animate_hypodd.py`.

## 13. Validation outcomes

**The station S terms are supported by the analyst's own picks.** The terms are large and
carry much of the depth axis, so they were tested against data that played no part in deriving them.
Of 46,339 analyst picks, 17,470 form event/station pairs with both P and S (17,347 of them on the
OBS array). At **every** OBS the analyst's 5th-percentile hand-measured S−P sits *above* that
station's S−P term, by margins from +0.07 s (at the station with the largest term, 0.578 s) to +0.53
s. Across the whole set only **83 of 17,347 pairs (0.5%)** measure an S−P smaller than their own
station's S−P term, and 117 (0.7%) fall at or below the guard threshold. A term that was an artefact
would be contradicted routinely by hand picks; it is not, and the margin is smallest exactly where
the term is largest, which is where it should be tight.

The same comparison exonerates the automatic picks: over 7,515 pairs with an automatic twin,
automatic minus analyst S−P has a median of **+0.030 s with a MAD of 0.070 s**, and per station lies
between −0.112 and +0.052 s. The automatic picks are not inflating S−P relative to the analyst, and
the difference is far smaller than the station terms themselves.

The *mechanism* previously assumed for the sub-threshold pairs is, however, **not** supported. Those
pairs are not events sitting almost directly beneath their station: hypocentral distance p50 is 4.00
km against 3.87 km for the whole population, essentially identical. They are *shallower* (source
depth p50 1.06 km against 3.20 km) with anomalously small S−P (0.250 s against 0.950 s) at ordinary
distance. The honest reading is station/path variability that a single constant term cannot
represent, and which the guard then deletes. The residual risk is that the guard removes real,
depth-informative short S−P measurements at a rate of ~0.5–0.7% of near-station S pairs in the
analyst set — higher over the full year (Section 10), because the year pool reaches closer distances
than the analyst's set does.

**The strict tier is made of events whose depth the terms can actually act on.** 99.0% of
strict-tier events have a station within 4 km, and **97.0% have a station within 4 km carrying both
P and S**; the nearest-station epicentral distance is 0.63 / 1.18 / 1.58 km (p10/p50/p90). An
independent great-circle computation reproduces the catalogue's own nearest-station distance column
exactly on that tier, confirming that the column is the epicentral, not hypocentral, distance.

**Forward-model consistency.** After the change of travel-time solver, two structurally
independent forward solvers agree with each other and with the velocity model to under 10 ms on real
ray paths, with two flagged station exceptions (Section 4.3). This is a prerequisite for any of the
residual-based reasoning above, and it was not true of the finite-difference grids.

**Frozen-run and synthetic validation of the relative inversion.** The trial-source mode,
the differential-time construction and the travel-time table were each validated against noise-free
synthetic data before the production run (Section 7), and the production configuration recovers
input depths with a slope of +0.001.

**Figure 16:** Validation. Panel (a) per-station distribution of analyst-measured S−P with
the station's S−P term marked; panel (b) automatic minus analyst S−P per station; panel (c) raw and
catalogue recall per station in the trusted window. Produced by `scripts/68_manual_sp_vs_delays.py`
and `scripts/54_manual_pick_recall.py`.

## 14. Limitations

1. **Depth below ~4 km beneath the seafloor is model-dependent.** Deeper values relax toward the 1D
   starting model and the S structure is not independently resolved at all. 14.8% of the standard
   tier and 12.7% of the strict tier lie beyond that limit; those depths are upper bounds.
2. **The station terms are fitted, not observed.** They carry much of the depth axis. Their
   contribution to depth is ≤ 0.12 km (median) above 4 km and 0.23–0.41 km below it inside the
   admissible family, and 0.03–0.59 / 0.95–1.56 km against the rejected outer bound — which below
   ~4 km exceeds the formal σz (Section 9).
3. **20,140 S picks (4.9%) were removed before location** by the S-before-P guard, in 17.5% of
   events, preferentially at the nearest station of the shallowest events — the geometry that best
   constrains depth. Those depths are less well constrained than their formal σz suggests.
4. **The airgun window (2019-01-21 to 2019-02-05) is excluded entirely**, with ~1,800 genuine
   earthquakes discarded alongside ~17,000 shots. Rate statistics must exclude it.
5. **Completeness is limited by association, not detection**: 87% P and 93% S analyst-pick recall in
   the trusted window, but only 63% and 57% reaching an associated event. The binding location
   constraints are azimuthal gap and grid-face pinning, not pick availability.
6. **50 events could not be located**, 10 solution files are unparsable and 220 solutions are
   REJECTED; 99.6% of associated events have a LOCATED solution.
7. **The absolute and relative estimators disagree on the depth scale by 13–28%**, concentrated in
   the 4–9 km interval (slope −0.192 / −0.275 without correlation data, −0.128 / −0.182 with it,
   over all relocated and quality-controlled events respectively). Neither is preferred on depth:
   the absolute solution owns the frame, the relative solutions own the geometry.
8. **No magnitudes** are computed, so no magnitude–frequency distribution, completeness magnitude or
   moment release can be reported.
9. **Cross-correlation coverage is partial and these are not repeaters.** The chance-peak rate is
   ~26% at cc ≥ 0.7 and ~20% at the adopted cc ≥ 0.8; 1,805 of 8,005 refined events end with no
   correlation link, the refined run retains 81% of trial sources against 92%, and the refinement
   contracts nearest-neighbour spacing by only ~3% on common events. Relative precision remains
   largely pick-precision-limited outside the well-linked core.
10. **The pre-October-2019 analyst reference is less reliable than the later one** — a property of
    the reference, not the pickers (recall scored against it is 12–19 points lower for every model).
    Validation statements are made on the trusted window.
11. **Two stations are flagged forward-model outliers** (one inside the filled water column, one far
    outside the array where the comparison tracer is itself coarse). They are flagged, not corrected.

---

# Data and code availability

All processing scripts are in the project repository under `scripts/`, in pipeline order. Catalogue
CSVs, travel-time grids and animations are delivered as files and are reproducible from the
committed scripts, `configs/velocity_model.csv`, `catalogs/station_geometry.csv` and
`catalogs/manual_picks.csv`.

*Picking and association.* `03_run_phasenet.py` (deep-learning picking, arbitrary SeisBench
model, `--picking-channels` to admit the hydrophone); `33_stalta_baseline.py` (classical reference
picker); `34_run_full_year_picks.sh` (production picking driver); `35_preflight_gpu_run.py` (pre-run
checks and wall-time projection); `apply_S1_corrections.py` (BRA05 clock correction);
`17_pyocto_associate.py` with `17f_pyocto_year_newpool.sh` (association); `discriminate_shots.py`
(airgun identification); `54_manual_pick_recall.py` (recall against the analyst set).

*Velocity model and travel times.* `41_build_unsheared_velgrid.py` (un-shearing to a
sea-level datum, bathymetry blending, water-column fill); `59_build_eikonal_ttgrids.py`
(fast-marching travel-time grids at 0.2 km, decimated to 0.4 km); `58_eikonal_check.py` and
`57_compare_3d_tt.py` (solver cross-validation); `53_perturb_velgrid.py` (controlled model
perturbations); `37_plot_velocity_slice.py`.

*Absolute location.* `28_pyocto_to_nlloc_obs.py` (observation files);
`29_write_nlloc_control.py` (control file, station geometry at true depths); `30_run_nlloc.py`
(sharded execution); `31_nlloc_hyp_to_catalog.py` (content-based solution-to-event mapping,
catalogue assembly); `40_filter_nlloc_reliable.py` (quality tiers, seafloor test, hull test);
`33_plot_nlloc.py`, `34_animate_nlloc.py`.

*Station terms.* `62_velest_prep.py`, `62_run_velest.py`, `62_velest_post.py`,
`62_velest_gate.py` (joint 1D inversion and its gate); `65_make_locdelay.py` (delay construction and
refinement); `66_guard_obs_for_delays.py` (S-before-P guard); `61_sample_gates.py`,
`60_compare_sample_runs.py`, `52_sp_depth_check.py`, `67_gate_table.py` (gate diagnostics);
`68_perturb_delays.py`, `68_delay_depth_uncertainty.py` (station-term depth uncertainty);
`68_manual_sp_vs_delays.py` (analyst validation of the terms).

*Relative relocation.* `22_pyocto_to_hypodd_input.py` (event/phase files, id mapping);
`23_run_ph2dt.py` (pair formation); `56_build_hypodd_3dmodel.py` (graded 3D node model);
`24_run_hypodd.py` (relocation, trial-source and damping control); `51_synthetic_dtct.py`,
`55_synth_eval.py` (synthetic validation); `50_hypodd_qc.py` (seafloor and link-count quality
control); `animate_hypodd.py`, `plot_hypodd_with_seafloor.py`.

*Cross-correlation.* `63_xcorr_dtcc.py` (correlator, self-test, null test);
`64_run_hypodd_xc.sh` (relocation with correlation data).

*Accounting and release.* `68_catalogue_accounting.py` (event- and pick-level accounting);
`catalogs/README_nlloc_year_v6.md` (column definitions, tier definitions, datum and frame
conventions, caveats).

Delivered catalogues: the full absolute catalogue and its three quality tiers
(`catalogs/nlloc_year_v6.csv`, `_loose`, `_standard`, `_strict`); the quality-controlled
catalogue-differential relative catalogue (`catalogs/hypodd_year_v6_3d_qc.csv`, 9,125 events, use
for completeness); and the cross-correlation-refined relative catalogue
(`catalogs/hypodd_year_v6_3d_xc_qc.csv`, 8,005 events, use for cluster-scale geometry). The association output the locations were built
from is `catalogs/pyocto_events_year_newpool_no_shots.csv` and its companion pick file. The event
index is the join key across all stages.

---

## Open items for this draft

- No magnitudes exist, so Section 10 cannot report a completeness magnitude and Section 14
  item 8 stands: `[value: compute from a magnitude stage that does not yet exist]`.
- The search-grid vertical extent is quoted from the control file as z ∈ [0, 24.8] km
  (63 nodes × 0.4 km); the catalogue release notes state [0, 25.2] km. The control file is
  authoritative; the release notes should be corrected.
- The gate-table utility joins solution files to events positionally and is silently wrong
  whenever the reference run lost an event; all gate numbers quoted here were produced with
  a reference run that lost none, and the station-term uncertainty analysis matches on the
  filename instead. This should be fixed before the tool is reused.
- Figures are specified but not yet produced; the scripts named under each placeholder
  generate the underlying data and, in most cases, a first-pass version of the panel.
