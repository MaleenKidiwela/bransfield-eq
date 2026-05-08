# Manual picks — drop zone

Drop your manual P/S pick files here in whatever format you have them.
Anything is fine: CSV, Excel, NLLoc `.obs`, SeisAn S-file, QuakeML, plain text, etc.

## What's expected to live here
- Pick times (P and S)
- Per-pick station/network/channel
- Optional: event ID / origin time, magnitude

## What happens next
The loader at `src/bransfield_eq/manual_picks.py` parses whatever's here
and writes a normalized table to `catalogs/manual_picks.csv` with this schema:

| column | type | notes |
|---|---|---|
| event_id | str / int | groups picks belonging to the same event (NaN if loose picks) |
| origin_time | ISO8601 UTC | event origin if known |
| magnitude | float | event magnitude if known |
| network | str | SEED net code (e.g. `ZX`, `5M`, `AI`, `AM`) |
| station | str | SEED sta code |
| location | str | location code, or "" |
| channel | str | usually Z for P, E/N/1/2 for S |
| phase | str | "P" or "S" |
| pick_time | ISO8601 UTC | arrival time |
| analyst | str | analyst initials if known |
| source_file | str | original filename, for traceability |

Downstream consumers:
- Stage 1 — validation against PhaseNet picks (precision/recall, threshold tuning)
- Stage 3 — associator sanity check
- Stage 4 — location ground truth
- Stage 5 — polarity validation (if polarities are included later)

## Provenance
Keep the original files unchanged in this folder. The CSV is the derived view;
re-run the loader if originals change.
