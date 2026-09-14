"""Parse NLLoc per-event .hyp files into a single catalog CSV.

Joins back to pyocto event_idx via the obs_order sidecar written by script 28.
NLLoc processes events serially in the order they appear in the input .obs
file, so the i-th per-event .hyp file (sorted by filename, which encodes the
first observation time) corresponds to obs_order = i.

Sharded runs (script 30 with --shards N): script 30 distributes obs blocks
round-robin (block k goes to shard k % N), so for the j-th hyp within
shard s the global obs_order is j * N + s. We detect shard_XX/ subdirs
and apply the shard-aware mapping; otherwise we treat the run as a single
input stream.

DO NOT sort the global hyp list by computed origin-time -- NLLoc can shift
OT by hours for marginal events, which scrambles the join. (That was the
2026-05-20 bug: ~half of v2 mappings were wrong.)

Output columns:
    event_idx, origin_time (UTC ISO), lat, lon, depth_km,
    sigma_x_km, sigma_y_km, sigma_z_km,
    semi_minor_km, semi_major_km, az_max_horunc_deg,
    rms_s, n_phases, gap_deg, dist_km,
    nlloc_x_km, nlloc_y_km, nlloc_z_km
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

STATUS_RE = re.compile(r'^NLLOC\s+"[^"]*"\s+"(\w+)"')
GEO_RE = re.compile(
    r"GEOGRAPHIC\s+OT\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)"
    r"\s+Lat\s+(-?[\d.]+)\s+Long\s+(-?[\d.]+)\s+Depth\s+(-?[\d.]+)")
HYPO_RE = re.compile(r"HYPOCENTER\s+x\s+(-?[\d.eE+-]+)\s+y\s+(-?[\d.eE+-]+)\s+z\s+(-?[\d.eE+-]+)")
QUAL_RE = re.compile(r"QUALITY.*?RMS\s+([\d.eE+-]+).*?Nphs\s+(\d+).*?Gap\s+([\d.]+)\s+Dist\s+([\d.]+)")
STAT_RE = re.compile(r"STATISTICS.*?CovXX\s+(-?[\d.eE+-]+).*?YY\s+(-?[\d.eE+-]+).*?ZZ\s+(-?[\d.eE+-]+)")
QML_RE  = re.compile(r"QML_OriginUncertainty.*?minHorUnc\s+([\d.eE+-]+)\s+maxHorUnc\s+([\d.eE+-]+)\s+azMaxHorUnc\s+([\d.]+)")


def parse_hyp(path: Path) -> dict | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    g = GEO_RE.search(text)
    if g is None:
        return None
    yr, mo, dy, hr, mn = (int(g.group(i)) for i in range(1, 6))
    sec = float(g.group(6))
    lat, lon, depth = (float(g.group(i)) for i in (7, 8, 9))
    # NLLoc can emit a NEGATIVE seconds field (origin rolled back past the minute,
    # e.g. "00 00 -17.703086"). Building Timestamp(...) with a negative component
    # raised, so the event was silently dropped -- one real event was lost this way.
    # Adding a Timedelta handles negatives and >=60 s alike.
    ot = pd.Timestamp(yr, mo, dy, hr, mn, tz="UTC") + pd.Timedelta(seconds=sec)
    st = STATUS_RE.search(text)
    # NLLoc writes a full GEOGRAPHIC line even for REJECTED solutions, so they parsed
    # as clean locations and entered the catalogue unflagged (569 of 31,516 in the v2
    # run, incl. boundary-pinned events with sigma_z of 7.5 km).
    rec: dict = {"origin_time": ot, "lat": lat, "lon": lon, "depth_km": depth,
                 "nlloc_status": (st.group(1) if st else "UNKNOWN")}
    h = HYPO_RE.search(text)
    if h:
        rec.update(nlloc_x_km=float(h.group(1)),
                   nlloc_y_km=float(h.group(2)),
                   nlloc_z_km=float(h.group(3)))
    q = QUAL_RE.search(text)
    if q:
        rec.update(rms_s=float(q.group(1)),
                   n_phases=int(q.group(2)),
                   gap_deg=float(q.group(3)),
                   dist_km=float(q.group(4)))
    s = STAT_RE.search(text)
    if s:
        rec.update(sigma_x_km=np.sqrt(float(s.group(1))),
                   sigma_y_km=np.sqrt(float(s.group(2))),
                   sigma_z_km=np.sqrt(float(s.group(3))))
    qml = QML_RE.search(text)
    if qml:
        rec.update(semi_minor_km=float(qml.group(1)),
                   semi_major_km=float(qml.group(2)),
                   az_max_horunc_deg=float(qml.group(3)))
    return rec


def _pick_keys_from_obs_block(block: list[str]) -> set:
    keys = set()
    for l in block:
        f = l.split()
        if len(f) >= 9 and not f[0].startswith("#"):
            keys.add((f[0], f[4], f[6], f[7], f"{float(f[8]):.2f}"))
    return keys


def _pick_keys_from_hyp(path: Path) -> set:
    keys, inblock = set(), False
    for l in path.read_text().split("\n"):
        if l.startswith("PHASE ID"): inblock = True; continue
        if l.startswith("END_PHASE"): break
        if inblock:
            f = l.split()
            if len(f) >= 9:
                try: keys.add((f[0], f[4], f[6], f[7], f"{float(f[8]):.2f}"))
                except ValueError: pass
    return keys


def _match_hyps_to_obs(out_dir: Path, obs_root: Path, label: str) -> list:
    """Content-based mapping hyp -> global obs_order: each hyp is matched to the obs event
    block sharing its picks (station, phase, date, hrmn, sec). Order is preserved within a
    shard, so a forward pointer suffices; obs blocks with no hyp (events NLLoc failed on,
    e.g. 'cannot find companion arrival' under LOCDELAY) are skipped and REPORTED instead of
    shifting every later event onto the wrong hyp -- which is what the positional mapping
    did on year_v6 (50 failed events scrambled 64% of origin times by hours)."""
    shard_dirs = sorted(d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("shard_"))
    ordered: list = []
    if shard_dirs:
        n_shards = len(shard_dirs); slots = {}
        unmatched = []
        for s_idx, sh in enumerate(shard_dirs):
            hyps = sorted(h for h in sh.glob("loc.20*.grid0.loc.hyp") if "last" not in h.name)
            obs = (obs_root / f"{label}_shards" / f"shard_{s_idx:02d}.obs").read_text().split("\n")
            blocks, cur = [], []
            for l in obs:
                if l.strip() == "":
                    if cur: blocks.append(cur); cur = []
                else: cur.append(l)
            if cur: blocks.append(cur)
            bkeys = [_pick_keys_from_obs_block(b) for b in blocks]
            j = 0
            for h in hyps:
                hk = _pick_keys_from_hyp(h)
                j0 = j
                while j < len(blocks) and not (hk & bkeys[j]):
                    j += 1
                if j >= len(blocks):
                    raise SystemExit(f"{h.name}: no obs block in shard {s_idx} shares its picks (searched from block {j0})")
                for k in range(j0, j):
                    unmatched.append((s_idx, k, sorted(bkeys[k])[0] if bkeys[k] else None))
                slots[j * n_shards + s_idx] = h
                j += 1
            for k in range(j, len(blocks)):
                unmatched.append((s_idx, k, sorted(bkeys[k])[0] if bkeys[k] else None))
        n = max(slots) + 1
        ordered = [slots.get(i) for i in range(n)]
        print(f"content-based mapping: {len(slots):,} hyps matched; {len(unmatched)} obs events without a hyp"
              + (f" (first: {unmatched[:3]})" if unmatched else ""))
        (out_dir / "unmatched_obs_events.txt").write_text("\n".join(f"{s} {k} {key}" for s, k, key in unmatched) + "\n")
        return ordered
    return sorted([h for h in out_dir.glob("loc.20*.grid0.loc.hyp") if "last" not in h.name])


def _collect_hyps_in_obs_order(out_dir: Path) -> list[Path]:
    """Return per-event .hyp files indexed by global obs_order. Index `i`
    of the returned list corresponds to obs_order == i (or None when that
    slot has no parsed hyp -- e.g. an event NLLoc failed to locate)."""
    shard_dirs = sorted(d for d in out_dir.iterdir()
                        if d.is_dir() and d.name.startswith("shard_"))
    if shard_dirs:
        n_shards = len(shard_dirs)
        per_shard: list[list[Path]] = []
        for sh in shard_dirs:
            hyps = sorted([h for h in sh.glob("loc.20*.grid0.loc.hyp")
                           if "last" not in h.name])
            per_shard.append(hyps)
        max_local = max(len(hs) for hs in per_shard)
        ordered: list[Path | None] = [None] * (max_local * n_shards)
        for s_idx, hyps in enumerate(per_shard):
            for local_i, h in enumerate(hyps):
                ordered[local_i * n_shards + s_idx] = h
        # trim trailing Nones
        while ordered and ordered[-1] is None:
            ordered.pop()
        return ordered
    # Single-shard run: per-event hyps live directly under out_dir.
    return sorted([h for h in out_dir.glob("loc.20*.grid0.loc.hyp")
                   if "last" not in h.name])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="picker_only_no_shots")
    p.add_argument("--allow-positional", action="store_true", help="skip the positional-vs-content comparison print")
    p.add_argument("--tt-prefix", default=None,
                   help="travel-time grid prefix the run used; sets depth_datum")
    p.add_argument("--depth-datum", default=None, choices=["seafloor", "sealevel"],
                   help="datum of depth_km. Sheared v1-v3 grids (ORCA, ORCA_v2, ORCA_v3) "
                        "are 'seafloor'; the un-sheared ORCA_v4+ is 'sealevel'. Written as "
                        "a column so downstream consumers cannot double-convert.")
    args = p.parse_args()
    if args.depth_datum is None:
        if args.tt_prefix is None:
            raise SystemExit("pass --tt-prefix (or --depth-datum) so the catalogue records "
                             "its depth datum")
        args.depth_datum = "sealevel" if args.tt_prefix >= "ORCA_v4" else "seafloor"

    out_dir = REPO / "nlloc" / "output" / args.label
    obs_root = REPO / "nlloc" / "obs"
    if (obs_root / f"{args.label}_shards").exists():
        ordered_hyps = _match_hyps_to_obs(out_dir, obs_root, args.label)
        if not args.allow_positional:
            pos = _collect_hyps_in_obs_order(out_dir)
            same = sum(1 for a_, b_ in zip(pos, ordered_hyps) if a_ == b_)
            print(f"positional vs content mapping agree on {same:,} of {len(ordered_hyps):,} slots")
    else:
        ordered_hyps = _collect_hyps_in_obs_order(out_dir)
    if not ordered_hyps:
        raise SystemExit(f"no .hyp files in {out_dir}")
    n_slots = len(ordered_hyps)
    n_have = sum(h is not None for h in ordered_hyps)
    print(f"obs slots: {n_slots}; per-event hyp files present: {n_have}")

    recs: list[dict] = []
    obs_orders: list[int] = []
    for oo, h in enumerate(ordered_hyps):
        if h is None:
            continue
        r = parse_hyp(h)
        if r is None:
            continue
        recs.append(r)
        obs_orders.append(oo)
    nlloc = pd.DataFrame.from_records(recs)
    nlloc["obs_order"] = obs_orders
    print(f"parsed {len(nlloc)} events")

    order_path = REPO / "nlloc" / "obs" / f"{args.label}.event_order.csv"
    order = pd.read_csv(order_path)
    nlloc = nlloc.merge(order, on="obs_order", how="left")
    missing = nlloc.event_idx.isna().sum()
    if missing:
        print(f"WARN: {missing} parsed hyps have no event_order entry")
    nlloc = nlloc.dropna(subset=["event_idx"]).copy()
    nlloc["event_idx"] = nlloc["event_idx"].astype(int)
    nlloc = nlloc.sort_values("origin_time").reset_index(drop=True)
    cols = ["event_idx", "origin_time", "lat", "lon", "depth_km",
            "sigma_x_km", "sigma_y_km", "sigma_z_km",
            "semi_minor_km", "semi_major_km", "az_max_horunc_deg",
            "rms_s", "n_phases", "gap_deg", "dist_km",
            "nlloc_x_km", "nlloc_y_km", "nlloc_z_km",
            # nlloc_status was being parsed but DROPPED here by this whitelist, so the
            # v1 catalogue had no status column and script 40's REJECTED filter silently
            # did nothing. Keep it, and stamp the datum.
            "nlloc_status", "depth_datum"]
    nlloc["depth_datum"] = args.depth_datum
    nlloc = nlloc[[c for c in cols if c in nlloc.columns]]
    if "nlloc_status" not in nlloc.columns:
        raise SystemExit("nlloc_status missing from parsed output -- parse_hyp regressed")
    n_rej = int((nlloc.nlloc_status != "LOCATED").sum())
    print(f"  depth_datum={args.depth_datum}   non-LOCATED solutions kept+flagged: {n_rej:,}")
    out_csv = REPO / "catalogs" / f"nlloc_{args.label}.csv"
    nlloc.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(nlloc.describe(include="all").iloc[:5])


if __name__ == "__main__":
    main()
