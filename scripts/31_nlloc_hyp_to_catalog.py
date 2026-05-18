"""Parse NLLoc per-event .hyp files into a single catalog CSV.

Joins back to pyocto event_idx via the obs_order sidecar written by script 28:
the .sum.grid0.loc.hyp concatenated SUM file (or the per-shard SUMs combined
by script 30) lists events in the same order as the input .obs file, which
in turn matches the obs_order column of <label>.event_order.csv.

For sharded runs we still get per-event hyp files (loc.YYYYMMDD.HHMMSS.*.hyp)
distributed across shard_XX/ subdirs; these are the authoritative source.
We sort them by origin-time and match by index after also sorting the
event_order map by pyocto origin_time. That works because there are no
duplicate origin-times at sub-second resolution in either side.

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
    isec = int(sec)
    micro = int(round((sec - isec) * 1e6))
    if micro >= 1_000_000:  # clamp roundoff at 60s edge
        micro -= 1_000_000
        isec += 1
    ot = pd.Timestamp(yr, mo, dy, hr, mn, isec, micro, tz="UTC")
    rec: dict = {"origin_time": ot, "lat": lat, "lon": lon, "depth_km": depth}
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="picker_only_no_shots")
    args = p.parse_args()

    out_dir = REPO / "nlloc" / "output" / args.label
    hyps = sorted([h for h in out_dir.rglob("loc.20*.grid0.loc.hyp")
                   if "last" not in h.name])
    if not hyps:
        raise SystemExit(f"no .hyp files in {out_dir}")
    print(f"parsing {len(hyps)} hyp files...")
    recs = []
    for h in hyps:
        r = parse_hyp(h)
        if r is not None:
            recs.append(r)
    nlloc = pd.DataFrame.from_records(recs).sort_values("origin_time").reset_index(drop=True)
    print(f"parsed {len(nlloc)} events")

    # Match by sorted-time order to event_order sidecar (also sort that by pyocto OT)
    order_path = REPO / "nlloc" / "obs" / f"{args.label}.event_order.csv"
    order = pd.read_csv(order_path)
    ev_pyocto = pd.read_csv(REPO / "catalogs" / f"pyocto_events_{args.label}.csv",
                            usecols=["event_idx", "origin_time"])
    order = order.merge(ev_pyocto, on="event_idx")
    order["origin_time"] = pd.to_datetime(order["origin_time"], utc=True)
    order = order.sort_values("origin_time").reset_index(drop=True)

    if len(order) != len(nlloc):
        print(f"WARN: order map has {len(order)} events, NLLoc parsed {len(nlloc)}")
        n = min(len(order), len(nlloc))
        order = order.iloc[:n]
        nlloc = nlloc.iloc[:n]

    nlloc["event_idx"] = order["event_idx"].values
    cols = ["event_idx", "origin_time", "lat", "lon", "depth_km",
            "sigma_x_km", "sigma_y_km", "sigma_z_km",
            "semi_minor_km", "semi_major_km", "az_max_horunc_deg",
            "rms_s", "n_phases", "gap_deg", "dist_km",
            "nlloc_x_km", "nlloc_y_km", "nlloc_z_km"]
    nlloc = nlloc[[c for c in cols if c in nlloc.columns]]
    out_csv = REPO / "catalogs" / f"nlloc_{args.label}.csv"
    nlloc.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(nlloc.describe(include="all").iloc[:5])


if __name__ == "__main__":
    main()
