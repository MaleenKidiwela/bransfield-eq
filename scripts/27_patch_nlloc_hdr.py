"""Patch sign-stripped grid origins in nlloc/time/ORCA.P.BRA*.time.hdr.

The shipped .hdr files declare grid origin (29.8, 20.0, 0) but the .buf data
was actually built for origin (-29.8, -20.0, 0): the argmin of the BRA05 P
travel-time grid lies at indices (158, 115, 0), which under origin -29.8,-20.0
maps to (1.8, 3.0) km -- the projected position of BRA05. Under the shipped
+29.8,+20.0 origin, argmin maps to (61.4, 43.0) km, far outside the array and
inconsistent with the station listed on hdr line 2 (BRA05  1.7355  3.0241 0).

This script rewrites the first header line in place with the negative-sign
origin so NLLoc reads the grid in the same coordinate frame in which it was
built. Idempotent: re-running on already-patched files is a no-op.
"""
from __future__ import annotations

from pathlib import Path

NLLOC_TIME_DIR = Path(__file__).resolve().parent.parent / "nlloc" / "time"

WRONG_PREFIX = "301    201    126    29.8    20.0    0"
RIGHT_PREFIX = "301    201    126    -29.8    -20.0    0"


def patch_one(hdr_path: Path) -> str:
    text = hdr_path.read_text()
    lines = text.splitlines()
    if not lines:
        return "empty"
    line0 = lines[0]
    if line0.startswith(RIGHT_PREFIX):
        return "already_patched"
    if not line0.startswith(WRONG_PREFIX):
        return f"unexpected: {line0!r}"
    lines[0] = RIGHT_PREFIX + line0[len(WRONG_PREFIX):]
    hdr_path.write_text("\n".join(lines) + "\n")
    return "patched"


def main() -> None:
    hdrs = sorted(NLLOC_TIME_DIR.glob("ORCA.P.BRA*.time.hdr"))
    if not hdrs:
        raise SystemExit(f"no .hdr files in {NLLOC_TIME_DIR}")
    for h in hdrs:
        print(f"{h.name}: {patch_one(h)}")


if __name__ == "__main__":
    main()
