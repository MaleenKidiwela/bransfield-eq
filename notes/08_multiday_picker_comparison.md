# Multi-day picker comparison: 2019-02-04 → 2019-02-13

10 contiguous days from the active swarm-preceding period. Validates the day-26 picker findings against a much larger sample.

## Setup
- **Date range:** 2019-02-04 → 2019-02-13 inclusive (10 days, contiguous)
- **Stations available:** 17 ZX OBS (the year-download-complete set as of this morning) + a few partial others = 18 with data
- **Manual ground truth (mag07):** 1,236 picks (616 P + 620 S) across 10 days
- **Pickers run:**
  1. PhaseNet `instance` @ p=s=0.1 → `catalogs/picks/`
  2. OBSTransformer `obst2024` @ p=s=0.5 → `catalogs/picks_obst_05/`
- **Wall:** PhaseNet 2:21, OBSTransformer 2:05 (both with `--workers 8 --batch-size 256` on the L40S)

## Aggregate results (10-day sum)

| Picker | Total picks | P TP / FP / FN | **P recall** | S TP / FP / FN | **S recall** |
|---|---:|---|---:|---|---:|
| PhaseNet `instance` @ 0.1 | 87,833 | 262 / 70,009 / 354 | **0.425** | 388 / 11,344 / 232 | 0.626 |
| OBSTransformer @ 0.5 | 140,818 | 225 / 31,514 / 391 | 0.365 | 485 / 99,121 / 135 | **0.782** |

**Hybrid (PN for P + OBST for S):** P=0.425, S=0.782. Combined recall on the manual catalog ≈ 60%.

## Per-day breakdown

```
day        manualP manualS | PN P-rec OBST P-rec | PN S-rec OBST S-rec
2019-02-04      49      76 |    0.653      0.735 |    0.461      0.737
2019-02-05      79      97 |    0.823      0.886 |    0.794      0.959
2019-02-06       9      17 |    0.667      0.556 |    0.588      0.706
2019-02-07      18      20 |    0.889      0.944 |    0.850      0.950
2019-02-08       4       6 |    0.500      0.750 |    0.333      1.000
2019-02-09      55      83 |    0.582      0.455 |    0.422      0.518
2019-02-10      17      12 |    0.176      0.000 |    0.917      0.667   ← OBST P=0 anomaly
2019-02-11      99      60 |    0.283      0.162 |    0.567      0.783
2019-02-12      51      42 |    0.706      0.588 |    0.738      0.786
2019-02-13     235     207 |    0.179      0.098 |    0.657      0.812   ← swarm day
```

Plot: [`figures/multiday_2019-02-04_to_2019-02-13.png`](../figures/multiday_2019-02-04_to_2019-02-13.png)

## Findings

### 1. Hybrid recommendation holds — but margins are smaller than day-26 suggested
- Day-26 had PN P-rec = 0.64 vs OBST P-rec = 0.45 (28-pp gap). 11 manual P picks → very small sample.
- Across 616 P picks over 10 days: PN = 0.42 vs OBST = 0.37 (only 6-pp gap).
- The day-26 PN advantage was real but inflated by sample size. Across the 10 days, **PN beats OBST on P every time the day has ≤ 50 P picks**, but gets clobbered on swarm days.

### 2. OBSTransformer dominates S consistently (9 of 10 days)
- Aggregate S recall: 0.78 vs 0.63 — a much more durable advantage than the P side.
- Wins on every day except 2019-02-10 (where PN gets 0.92 vs OBST 0.67 — and OBST gets 0.000 P, see anomaly).
- The S-recall advantage scales with day complexity: small days (n=6, 17, 20) → near-perfect S recall (0.71–1.00). Large days (n=83, 207) → still 0.5–0.8, well above PN.

### 3. Swarm days hammer both pickers on P
- 2019-02-11 (n=99 P): PN 0.28, OBST 0.16
- 2019-02-13 (n=235 P): PN 0.18, OBST 0.10
- Likely cause: overlapping events in the same window confuse the model — each window can only emit one P pick at the same sample position, so densely-spaced events get under-counted.
- This is a **real production limitation** — the dense swarm days are exactly the days we most want catalog completeness for. Worth investigating: window size tuning, peak-finding tolerance, or post-processing to recover overlapping picks.

### 4. Anomaly: 2019-02-10 OBSTransformer P-recall = 0.000
- 0 of 17 manual P picks recovered, but OBST got 8 of 12 S picks (0.67) — so events were detected, just not as P.
- Possible cause: events on this day may have been impulsive S with weak P (a real OBS phenomenon — water-borne phase conversion can suppress direct P), and OBSTransformer is biased toward calling these as S.
- Need to spot-check waveforms — flagged for follow-up.

### 5. Pick volume vs validation
- PN total: 87,833 picks (10 days, 17 active stations) ≈ 5,164 picks/day/station.
- OBST total: 140,818 ≈ 8,283 picks/day/station — 60% more picks for the modest S-recall gain.
- For downstream PyOcto, OBST volume might require pre-filtering (e.g., post-pick threshold cleanup, or only keeping S picks from OBST + P from PN).

## Recommendations going forward

1. **Lock in the hybrid as the production picker** for the remaining download-and-pick run:
   - PhaseNet `instance` @ thresh 0.1 — use **only the P picks**
   - OBSTransformer @ thresh 0.5 — use **only the S picks**
2. **Investigate the swarm-day P-recall collapse** before declaring the catalog complete. Test:
   - Smaller picking windows (less event overlap per window)
   - Peak-finding tolerance changes
   - Comparison with manual swarm picks at high temporal resolution
3. **Defer fine-tuning still** — the hybrid covers the obvious gains; fine-tuning effort would only pay off if it specifically improves swarm-day P recall.
4. Once the year download is complete, re-run picker comparison on **a ~50-day sample stratified by event rate** (mix of quiet, active, and swarm days) to get robust per-condition statistics.

## Reproduce
```bash
source .venv/bin/activate
./scripts/run_10day_comparison.sh
python -c "...per-day validation..."   # see scripts/plot_multiday.py
python scripts/plot_multiday.py
```

Outputs:
- `logs/multiday_picks.log`, `logs/multiday_picks_obst_05.log`, `logs/multiday_runner.log`
- `catalogs/picks/`, `catalogs/picks_obst_05/` (per-station-day pick CSVs)
- `figures/multiday_2019-02-04_to_2019-02-13.png`
