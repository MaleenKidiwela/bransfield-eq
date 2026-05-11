# Validation refinement: mag07 as trusted ground truth

## The three manual-pick sources
`PYTHONPATH=src python -m bransfield_eq.manual_picks` merges and dedupes:

| Source | Picks | Events | Trusted? |
|---|---|---|---|
| `nllmaleen_mag07_202210.out` | 12,072 (5,443 P + 6,629 S) | 1,124 | **✓ ML ≥ 0.7 high-quality subset** |
| `nllmaleen_magall_202210.out` | 33,545 (12,735 P + 20,810 S) | 4,668 | mixed quality |
| `collect_regional.out` | 817 (442 P + 375 S) | 48 | regional, different scope |

Total after dedup: 46,434 picks / 5,840 events / 39 stations.

## Why this matters
Validating against all 46k picks deflates recall — the model is penalized for missing low-confidence manual picks that aren't really ground truth.

For the 2019-12-26 verification it didn't change anything (all 23 day-26 manual picks happened to be from mag07), but it'll matter for the full year.

## New flag
Added `--manual-source` to `scripts/05_validate_picks.py`:
- `--manual-source all` (default) — current behavior, no filter
- `--manual-source mag07` — filters to `nllmaleen_mag07_202210.out` (substring match)
- Bogus values give a clean error listing available sources

## Standard validation runs going forward
```bash
# Trusted recall (use this for primary metrics):
python scripts/05_validate_picks.py --manual-source mag07

# Broad sanity check across all sources:
python scripts/05_validate_picks.py
```

⚠️ Both runs write to the same `catalogs/validation_report.csv` and `catalogs/validation_per_pick.csv` — they overwrite each other. If you need both side-by-side, `cp` the output between runs (or we can extend with `--out-suffix`).
