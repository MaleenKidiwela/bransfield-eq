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
    "phase", "pick_time", "uncertainty_s", "analyst", "source_file",
]


def list_input_files() -> list[Path]:
    """Return parseable files in the drop zone."""
    skip = {"readme.md", ".ds_store"}
    return [p for p in INPUT_DIR.iterdir()
            if p.is_file() and p.name.lower() not in skip
            and not p.name.startswith(".")]


def parse_file(path: Path) -> pd.DataFrame:
    """Dispatch to a format-specific parser by inspecting file contents."""
    head = path.read_text(errors="replace").splitlines()[:5]
    head = [ln for ln in head if ln.strip()]
    if not head:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    first = head[0]
    # NLLoc obs format: starts with station code then SP/BB/HH instrument code.
    if " GAU " in first or " BOX " in first:
        return _parse_nlloc_obs(path)
    # Nordic S-file: first line of an event ends with "1" in column 80,
    # has lat/lon/depth columns. We detect by length and trailing digit.
    return _parse_nordic(path)


# ----- NLLoc -----------------------------------------------------------------

def _parse_nlloc_obs(path: Path) -> pd.DataFrame:
    """
    Parse NLLoc observation file. Each non-blank line is one pick; blank lines
    separate events. Columns (whitespace-delimited):
        STA INST COMP ONSET PHASE FM YYYYMMDD HHMM SECONDS ERRTYPE SIGMA ...
    """
    rows = []
    event_idx = 0
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            event_idx += 1
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        sta, inst, comp, onset, phase, fm, ymd, hm, sec = parts[:9]
        if phase not in ("P", "S"):
            continue
        # Optional NLLoc error fields after the 9 mandatory ones:
        #   parts[9]  = ERRTYPE (e.g. 'GAU')
        #   parts[10] = SIGMA   (uncertainty in seconds, e.g. '1.00e-01')
        uncertainty_s = None
        if len(parts) >= 11 and parts[9] in ("GAU", "BOX"):
            try:
                uncertainty_s = float(parts[10])
            except ValueError:
                pass
        try:
            year = int(ymd[:4]); mo = int(ymd[4:6]); da = int(ymd[6:8])
            hh = int(hm[:2]); mm = int(hm[2:4])
            ss = float(sec)
        except (ValueError, IndexError):
            continue
        # Build ISO time, handling sec >= 60 by carrying over.
        from obspy import UTCDateTime
        t = UTCDateTime(year, mo, da, hh, mm, 0) + ss
        rows.append({
            "event_id": f"{path.stem}_e{event_idx}",
            "origin_time": None,
            "magnitude": None,
            "network": None,           # not in NLLoc obs; resolved later
            "station": sta,
            "location": "",
            "channel": comp,
            "phase": phase,
            "pick_time": str(t),
            "uncertainty_s": uncertainty_s,
            "analyst": None,
            "source_file": path.name,
        })
    return pd.DataFrame(rows)


# ----- Nordic / SeisAn S-file -----------------------------------------------

def _parse_nordic(path: Path) -> pd.DataFrame:
    """
    Minimal Nordic 'collect' parser. Robust to small column-alignment quirks
    in SeisAn output that ObsPy's strict reader rejects.

    Type-1 (origin) line ends with '1' in column 80 and gives YEAR MO DA HH MI SS LAT LON DEPTH ... MAG.
    Phase lines have STA in cols 2-6, phase indicator in cols 11-12, and
    HHMM SS.SS at known offsets. We tokenize on whitespace and rely on
    field positions of the parsed tokens, which is more forgiving than
    fixed-column slicing.
    """
    from obspy import UTCDateTime
    rows = []
    current_origin = None
    current_mag = None
    current_eid = None
    event_idx = 0

    for raw in path.read_text(errors="replace").splitlines():
        if not raw.strip():
            continue
        # Type-1 origin line: ends with '1' in col 80 (right-justified).
        if len(raw) >= 80 and raw[79] == "1":
            tokens = raw.split()
            try:
                year = int(tokens[0]); mo = int(tokens[1]); da = int(tokens[2])
                hm = tokens[3]                       # HHMM, possibly with leading space stripped → e.g. "510" = 0510
                sec = float(tokens[4][:-2]) if tokens[4].endswith("BL") else float(tokens[4])
                # robust HHMM: zero-pad on left
                hm = hm.zfill(4)
                hh, mm = int(hm[:2]), int(hm[2:])
                current_origin = UTCDateTime(year, mo, da, hh, mm, 0) + sec
                current_eid = f"{path.stem}_{current_origin.strftime('%Y%m%d%H%M%S')}"
                # Magnitude in Nordic type-1 line: cols 56-59 (value) + col 59 (type).
                # Take the last <num><L|b|B|w|s|S|c> match in the line, since the
                # earliest one might match the seconds field's location/agency code.
                # Bound to physically plausible magnitudes [<= 9.9].
                import re
                matches = re.findall(r"\s(-?\d*\.\d+)([LbBwsSc])", raw)
                current_mag = None
                for val, _typ in reversed(matches):
                    try:
                        v = float(val)
                        if -2 < v < 10:
                            current_mag = v
                            break
                    except ValueError:
                        pass
                event_idx += 1
            except (ValueError, IndexError):
                current_origin = None
                current_eid = f"{path.stem}_e{event_idx}"
            continue

        # Skip header / action / comment lines.
        if (len(raw) < 30 or raw.lstrip().startswith("GAP=")
                or "OLDACT" in raw or "ACTION:" in raw
                or raw.lstrip().startswith("STAT")):
            continue

        # Nordic phase line — fixed columns (1-based per spec, 0-based here):
        #   2-5   station (5 chars)
        #   7-8   component
        #   11-12 phase indicator (EP/IS/ES/IP/P/S)
        #   19-22 hour+minute (HHMM)
        #   23-28 seconds
        # Lines may have trailing column-80 type code; padding to 80 first.
        padded = raw.ljust(80)
        sta = padded[1:6].strip()
        comp = padded[6:8].strip()
        ph_field = padded[10:14].strip()
        hm = padded[18:22]
        ss_str = padded[22:28]

        phase = None
        for ch in ph_field:
            if ch in ("P", "S"):
                phase = ch
                break
        if phase is None or not sta:
            continue
        try:
            ss = float(ss_str)
            hm = hm.replace(" ", "0")
            hh, mm = int(hm[:2]), int(hm[2:])
        except (ValueError, IndexError):
            continue
        if current_origin is None:
            continue

        try:
            t = UTCDateTime(current_origin.year, current_origin.month,
                            current_origin.day, hh, mm, 0) + ss
            if t < current_origin - 3600:
                t += 86400
        except (ValueError, IndexError):
            continue

        rows.append({
            "event_id": current_eid,
            "origin_time": str(current_origin),
            "magnitude": current_mag,
            "network": None,
            "station": sta,
            "location": "",
            "channel": comp,
            "phase": phase,
            "pick_time": str(t),
            "uncertainty_s": None,    # Nordic format does not carry per-pick sigma
            "analyst": None,
            "source_file": path.name,
        })
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    """Concatenate all parsed manual-pick files into the canonical schema."""
    frames = []
    for f in list_input_files():
        try:
            df = parse_file(f)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue
        if df.empty:
            print(f"  [empty] {f.name}")
            continue
        df["source_file"] = f.name
        frames.append(df[CANONICAL_COLUMNS])
        print(f"  [ok]   {f.name}: {len(df)} picks")
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def resolve_networks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing `network` codes by joining against `catalogs/station_geometry.csv`.
    Manual picks from NLLoc / Nordic don't carry network codes; we infer them
    from the station inventory we already pulled from FDSN.
    """
    geo_csv = REPO / "catalogs" / "station_geometry.csv"
    if not geo_csv.exists():
        print("  [warn] station_geometry.csv not found — run scripts/04_station_geometry.py first")
        return df
    geo = pd.read_csv(geo_csv)[["station", "network"]].drop_duplicates(subset=["station"])
    sta_to_net = dict(zip(geo["station"], geo["network"]))
    if "network" not in df.columns:
        df["network"] = None
    missing = df["network"].isna()
    df.loc[missing, "network"] = df.loc[missing, "station"].map(sta_to_net)
    n_resolved = missing.sum() - df["network"].isna().sum()
    n_unresolved = df["network"].isna().sum()
    if n_resolved or n_unresolved:
        print(f"  network resolution: {n_resolved} filled, "
              f"{n_unresolved} still missing "
              f"({df.loc[df.network.isna(), 'station'].value_counts().to_dict()})")
    return df


def backfill_origin_times(df: pd.DataFrame) -> pd.DataFrame:
    """For events with no origin_time (NLLoc), use earliest pick time as a proxy.
    Within a few seconds of the true origin for local events — adequate for
    event-window validation modes that use tolerances of order tens of seconds.
    """
    if "origin_time" not in df.columns or "event_id" not in df.columns:
        return df
    missing = df["origin_time"].isna() & df["event_id"].notna()
    if not missing.any():
        return df
    earliest = (df.loc[missing].groupby("event_id")["pick_time"].min())
    df.loc[missing, "origin_time"] = df.loc[missing, "event_id"].map(earliest)
    n_filled = (~df.loc[missing, "origin_time"].isna()).sum()
    if n_filled:
        print(f"  backfilled origin_time for {n_filled} picks "
              f"({earliest.size} events) using earliest pick per event")
    return df


def dedup_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate picks across source files. mag07 ⊆ magall, so each NLLoc
    pick appears in both files; dedup on (station, phase, pick_time) at
    millisecond precision.
    """
    if df.empty:
        return df
    n0 = len(df)
    df = df.copy()
    df["_t_ms"] = pd.to_datetime(df["pick_time"], errors="coerce").astype("int64") // 10**6
    df = df.drop_duplicates(subset=["station", "phase", "_t_ms"], keep="first")
    df = df.drop(columns=["_t_ms"])
    print(f"  deduped {n0 - len(df)} duplicate picks across source files")
    return df


def write_normalized() -> Path:
    df = load()
    df = resolve_networks(df)
    df = backfill_origin_times(df)
    df = dedup_picks(df)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    return OUTPUT_CSV


if __name__ == "__main__":
    out = write_normalized()
    n = sum(1 for _ in open(out)) - 1
    print(f"Wrote {out.relative_to(REPO)} ({n} picks)")
