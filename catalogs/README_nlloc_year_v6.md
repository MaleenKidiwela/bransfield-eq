# Orca / Bransfield Strait OBS earthquake catalogue — NLLoc **v6** (release notes)

Deployment 2019-01-01 → 2020-03-01, ZX OBS array over the Orca volcanic edifice plus
regional land stations (5M, AI, AM). Absolute locations by NLLoc on a 3D P model with
per-station P/S time delays. Built 2026-09-14; supersedes v1–v5 and every earlier
hybrid / GrowClust / pre-2026-09 hypoDD product.

---

## 1. Which file to use for what

| question | file | n |
|---|---|---|
| **absolute locations, default catalogue** | `catalogs/nlloc_year_v6_standard.csv` | 12,194 |
| absolute locations, maximum completeness | `catalogs/nlloc_year_v6_loose.csv` \* | 15,184 |
| absolute locations, paper-defensible subset | `catalogs/nlloc_year_v6_strict.csv` \* | 3,864 |
| everything NLLoc returned, incl. rejected/unconstrained | `catalogs/nlloc_year_v6.csv` | 79,723 |
| **relative geometry / structure** (lineations, planes, clusters) | `catalogs/hypodd_year_v6_3d_qc.csv` | 9,125 |
| association output the locations were built from | `catalogs/pyocto_events_year_newpool_no_shots.csv`, `..._picks_...csv` | 79,783 |

\* Only the **standard** tier file has been written so far. The loose and strict files are
produced by rerunning the same filter, which is deterministic:
`python scripts/40_filter_nlloc_reliable.py --label year_v6 --tt-prefix ORCA_v5 --tier loose`
(and `--tier strict`). The counts above were reproduced independently by
`scripts/68_catalogue_accounting.py` from `nlloc_year_v6.csv`, and its standard-tier count
matches the written file exactly (12,194).

- Use the **tier files, not the full file**. `nlloc_year_v6.csv` contains every parsed
  solution including 220 NLLoc-REJECTED ones and ~49% that are pinned against a face of
  the search grid; it exists for accounting and reprocessing, not for interpretation.
- Use **NLLoc for absolute position and depth**; use **hypoDD for relative geometry only**.
  hypoDD depths are relocations of the v6 standard tier and inherit its absolute frame;
  their value is the *relative* precision inside a cluster.
- `event_idx` is the join key across every stage (association → NLLoc → hypoDD, where
  hypoDD's `id` = `event_idx + 1`).

## 2. Columns

### `nlloc_year_v6*.csv` — written by `scripts/31_nlloc_hyp_to_catalog.py`

| column | meaning |
|---|---|
| `event_idx` | pyocto association event index. Stable join key to `pyocto_events_year_newpool_no_shots.csv` and `pyocto_picks_year_newpool_no_shots.csv`. Mapping from NLLoc `.hyp` files is **content-based** (each hyp is matched to the obs block sharing its picks), never positional. |
| `origin_time` | UTC, ISO-8601 with microseconds, from the NLLoc `GEOGRAPHIC OT` field. |
| `lat`, `lon` | degrees, WGS84, from `GEOGRAPHIC Lat`/`Long`. |
| `depth_km` | maximum-likelihood hypocentre depth, km, **positive down from the datum named in `depth_datum`**. |
| `depth_datum` | `sealevel` for v6 (the ORCA_v4/v5 grids are un-sheared to a sea-level datum). Never assume; read this column. |
| `sigma_x_km`, `sigma_y_km`, `sigma_z_km` | √CovXX, √CovYY, √CovZZ from the NLLoc `STATISTICS` line — 1σ marginal standard deviations of the posterior in the NLLoc x/y/z frame, km. **Formal, Gaussian-approximation errors** conditional on the velocity model and the station delays; they do not contain model error (§5). |
| `semi_minor_km`, `semi_major_km`, `az_max_horunc_deg` | `QML_OriginUncertainty` minHorUnc / maxHorUnc / azMaxHorUnc — the horizontal error ellipse and the azimuth of its major axis (deg). |
| `rms_s` | `QUALITY RMS` — rms of the weighted travel-time residuals, s (residuals are computed **after** the LOCDELAY station terms are applied). |
| `n_phases` | `QUALITY Nphs` — number of phases used. |
| `gap_deg` | `QUALITY Gap` — largest azimuthal station gap, degrees. |
| `dist_km` | `QUALITY Dist` — epicentral distance to the **closest** station, km (verified against an independent great-circle computation on the strict tier: p10/50/90 0.63/1.18/1.58 km both ways). |
| `nlloc_x_km`, `nlloc_y_km`, `nlloc_z_km` | hypocentre in NLLoc's own Cartesian frame, `TRANS SIMPLE -62.4413 -58.44 36` (origin lat/lon, rotation 36°). `nlloc_z_km` == `depth_km`. Use these only for grid-geometry tests; cross-stage joins go through lat/lon or `event_idx` (pyocto's projection origin is a *different* one, −62.5/−58.8). |
| `nlloc_status` | NLLoc's own flag from the `NLLOC` header line: `LOCATED` or `REJECTED`. NLLoc writes a full `GEOGRAPHIC` line even when it rejects a solution, so this column must be checked — 220 rejected solutions are present in the full file. |

### extra columns in the tier files — written by `scripts/40_filter_nlloc_reliable.py`

| column | meaning |
|---|---|
| `on_boundary` | True if the hypocentre is within 0.5 km of any face of the `LOCGRID` (x ∈ [−150, 80], y ∈ [−110, 70], z ∈ [0, 25.2] km). Boundary-pinned solutions are unconstrained in that direction and carry artificially small σ; all tiers exclude them. |
| `local_water_km` | water depth at the epicentre, km, from the 30 m Orca bathymetry grid where it covers the event (49,224 of 79,503) and from the 0.4 km ORCA_v5 water-depth surface elsewhere. |
| `depth_bsf_km` | `depth_km − local_water_km` — depth **below the local seafloor**. This is the quantity to plot against structure; `depth_km` (below sea level) mixes in 0.8–1.9 km of bathymetric relief. |
| `in_hull` | epicentre inside the convex hull of the ZX OBS array. |
| `qc_tier` | the tier this file represents (`loose` / `standard` / `strict`). |

### `hypodd_year_v6_3d_qc.csv` — hypoDD 2.1b, IMOD=9 (3D), `scripts/24_run_hypodd.py` + `scripts/50_hypodd_qc.py`

`id` (= `event_idx + 1`), `lat`, `lon`, `dep`, `x`, `y`, `z`, `ex`, `ey`, `ez` (hypoDD's own
formal errors, m), `yr…sc` origin time, `mag`, `nccp`/`nccs` (cross-correlation links — **0**,
no dt.cc was built), `nctp`/`ncts` (catalogue differential-time links), `rcc`/`rct` (rms of
cc / ct residuals), `cid` (cluster id), plus QC columns added by script 50:
`depth_datum`, `datum_shift_km`, `depth_bsl_km` (depth on the sea-level datum — **use this,
not `dep`**), `local_water_km`, `depth_bsf_local_km`, `physical_local`, `n_links`,
`well_linked`, `hypodd_qc_pass`. The `_qc` file already contains only `hypodd_qc_pass` rows.

## 3. Tier definitions

All tiers additionally require `nlloc_status == LOCATED` and `on_boundary == False`.

| tier | cuts | n | % of LOCATED |
|---|---|---|---|
| **loose** | gap < 200°, rms < 0.7 s, Nphs ≥ 4, `depth_bsf_km` > −0.2 km | 15,184 | 19.1% |
| **standard** | gap < 180°, rms < 0.5 s, Nphs ≥ 6, `depth_bsf_km` > −0.2 km | 12,194 | 15.3% |
| **strict** | gap < 120°, rms < 0.3 s, Nphs ≥ 8, σx < 1 km, σy < 1 km, σz < 2 km, inside the ZX hull, `depth_bsf_km` > +0.2 km | 3,864 | 4.9% |

Tiers are nested: strict ⊂ standard ⊂ loose. The −0.2 km seafloor tolerance is half a
0.4 km grid cell; it exists because an event above the local seafloor is in the water column
and cannot be an earthquake.

## 4. Datum and frame conventions

- `depth_km` is **below sea level**, positive down (`depth_datum = sealevel`). The travel-time
  grids (`nlloc/time/ORCA_v5.*`) were un-sheared from the Stingray seafloor-hung frame to a
  sea-level datum, the water column filled with the local top-rock velocity, and every OBS
  placed at its **true** depth (0.785–1.943 km), so a shallow caldera source can legitimately
  sit above a deep-water station.
- `depth_bsf_km` is below the **local** seafloor. Quote this for anything structural.
- Station elevations/depths: `catalogs/station_geometry.csv` (ZX rows from Kidiwela+ SI
  Table S1, bathymetric depth).
- NLLoc Cartesian frame: `TRANS SIMPLE -62.4413 -58.44 36`. pyocto's association frame is a
  *different* transverse Mercator (origin −62.5/−58.8), 3.05 km away — only lat/lon and
  `event_idx` may cross stage boundaries.
- Times are UTC. **ZX.BRA05 clock correction +0.167 s** is applied at the pick stage to every
  pool; do not apply it again.

## 5. Known caveats

1. **Depth below ~6 km BSL is model-dependent.** The 3D P tomography is honestly resolved to
   about **4 km below the seafloor**; deeper values relax toward the 1D starting model, and
   the S structure is not independently resolved at all (Vp/Vs is a constant 1.78, and the
   per-station S terms absorb the near-receiver S structure). 14.8% of the standard tier and
   12.7% of the strict tier sit deeper than 4 km BSF (9.4% / 5.6% deeper than 6 km BSL).
   Treat those depths as an upper bound, not a measurement.
2. **The station delays are a fitted quantity, not an independent observation.**
   `nlloc/delays/v6_it1B.delays` (76 LOCDELAY terms, applied as obs − delay; positive = the
   station observes late) comes from a VELEST joint inversion (invB) refined by one iteration
   of NLLoc's own per-station median residuals. The OBS S−P terms are large (+0.17 … +0.93 s,
   median +0.37 s) and they carry a large part of the depth axis. Perturbing them on the
   2,000-event sample (`notes/31_final_catalogue_v6.md` §2): inside the family the S−P gate
   still accepts (S−P terms ×1.2) median |Δz| is 0.00–0.12 km above 4 km and 0.23–0.41 km
   below it (p90 0.20–0.84 km); against invA-scale terms — which the gate rejects, so this is
   an outer bound — median |Δz| is 0.03–0.59 km above 4 km and 0.95–1.56 km below it (p90 up
   to 3.9 km). Below ~4 km that systematic is larger than the NLLoc formal σz (standard tier
   p10/50/90 = 0.29/0.69/1.24 km), which does not contain it.
   The terms are, however, *supported* by the analyst's own picks: over 17,347 manual P+S
   pairs on the ZX array only 0.5% measure an S−P smaller than their station's S−P delay
   (`notes/31_final_catalogue_v6.md` §1).
3. **Guard-dropped S picks.** A constant station S delay can exceed the observed S−P for an
   event very close to the station, which puts the corrected S before its P and makes NLLoc
   abort the whole event. `scripts/66_guard_obs_for_delays.py` therefore deletes S picks whose
   corrected S−P ≤ 0.05 s **before** location: **20,140 S picks (4.9% of all S) in 13,979 of
   79,783 events (17.5%)**. This removes the S constraint preferentially from the closest
   station of the shallowest events — exactly the geometry that best constrains depth — so
   those events' depths are less well constrained than their formal σz suggests. 141 events
   ended up with no S pick at all.
4. **50 events could not be located** by NLLoc even after the guard
   (`nlloc/output/year_v6/unmatched_obs_events.txt`); a further 10 `.hyp` files carry no
   parsable `GEOGRAPHIC` line and 220 solutions are `REJECTED`. 79,503 of 79,783 associated
   events (99.6%) have a LOCATED solution.
5. **The airgun window 2019-01-21 → 2019-02-05 is excluded entirely** (18,848 associated
   events), not filtered. An estimated ~1,800 genuine earthquakes were discarded with the
   shots. The catalogue is not complete over that window.
6. **Association, not picking, is the completeness bottleneck.** Against the analyst's picks
   the pickers recall 87% (P) / 93% (S) in the trusted window with ≤20 ms bias, but only
   63% / 57% of those picks reach an associated event.
7. **Depth-axis stretch.** Relative to the near-station S−P observable the v5 depth axis was
   ~1.25× too long; the v6 station terms bring the predicted/observed S−P spread ratio to
   0.95–1.03 on the 2,000-event sample. hypoDD 3D still compresses the remaining v6 axis
   (slope −0.275 of Δz on start depth, median |Δz| 0.51 km, n = 9,125), mostly in the 6–9 km
   bin (−1.96 km). Depths in that range should be quoted with that spread.
8. No magnitudes are computed in v6 (`mag` in the hypoDD file is a placeholder).
9. No cross-correlation differential times (`dt.cc`) were built; hypoDD used catalogue
   differential times only, so relative precision is pick-precision-limited.

## 6. Provenance

Picks: PhaseNet/DiTing (3C, all networks) + PickBlue PhaseNetLight (4C, ZX), threshold 0.1,
deduplicated at 0.25 s. Association: pyocto, `min_p 3 / min_s 2 / min_total 6`. Location:
NLLoc `OCT` search, `LOCMETH GAU_ANALYTIC 100 4 -1 -1 1.78 -1 -1 1`, travel-time grids
`nlloc/time/ORCA_v5.*` (pykonal eikonal FMM on the 3D P model, 0.2 km solve decimated to
0.4 km), station terms `nlloc/delays/v6_it1B.delays`, control `nlloc/run/year_v6.in`.
Validation package: `notes/31_final_catalogue_v6.md`, `scripts/68_*.py`.
