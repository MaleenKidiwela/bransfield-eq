"""
Loader for user-provided manual P/S picks.

Reads from `data/manual_picks/` and writes a normalized
`catalogs/manual_picks.csv` with a schema shared across stages.

Per-format parsers are added as needed once we see actual files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import REPO

INPUT_DIR = REPO / "data" / "manual_picks"
OUTPUT_CSV = REPO / "catalogs" / "manual_picks.csv"

CANONICAL_COLUMNS = [
    "event_id", "origin_time", "magnitude",
    "network", "station", "location", "channel",
    "phase", "pick_time", "analyst", "source_file",
]


def list_input_files() -> list[Path]:
    """Return everything in the drop zone (excluding the README)."""
    return [p for p in INPUT_DIR.iterdir()
            if p.is_file() and p.name.lower() != "readme.md"]


def parse_file(path: Path) -> pd.DataFrame:
    """
    Dispatch to a format-specific parser based on file extension or content.
    Implementations are added as user-provided files arrive.
    """
    raise NotImplementedError(
        f"No parser registered for {path.name}. "
        "Inspect the file and add a parser branch in manual_picks.parse_file."
    )


def load() -> pd.DataFrame:
    """Concatenate all parsed manual-pick files into the canonical schema."""
    frames = []
    for f in list_input_files():
        df = parse_file(f)
        df["source_file"] = f.name
        frames.append(df[CANONICAL_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def write_normalized() -> Path:
    df = load()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return OUTPUT_CSV


if __name__ == "__main__":
    out = write_normalized()
    n = sum(1 for _ in open(out)) - 1
    print(f"Wrote {out.relative_to(REPO)} ({n} picks)")
