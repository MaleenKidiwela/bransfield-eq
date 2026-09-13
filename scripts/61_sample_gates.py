"""Four-number gate for an NLLoc sample run (Merlin's protocol, 2026-09-13), read
straight from the .hyp files:
  1. top-face pinned fraction (depth < 0.05 km on the sea-level grid)
  2. near-station (< 4 km) P residual of the pinned set (v4: about -0.13 s, observed early)
  3. per-station median P residual vs station water depth (slope; v4: -0.036 s/km)
  4. S-P spread ratio -> script 52 (run separately, same sample)
Also: depth p10/50/90, RMS, and residual medians for all events, so runs can be compared.
"""
from __future__ import annotations
import argparse, glob, re
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
GEO = re.compile(r'GEOGRAPHIC\s+OT\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?[\d.]+)\s+Lat\s+(-?[\d.]+)\s+Long\s+(-?[\d.]+)\s+Depth\s+(-?[\d.]+)')
QUAL = re.compile(r'QUALITY\s+Pmax\s+\S+\s+MFmin\s+\S+\s+MFmax\s+\S+\s+RMS\s+(\S+)\s+Nphs\s+(\d+)\s+Gap\s+(\S+)\s+Dist\s+(\S+)')


def parse(path):
    txt = open(path).read()
    g = GEO.search(txt); q = QUAL.search(txt)
    if not g or "LOCATED" not in txt.split("\n")[0]: return None, []
    ev = dict(lat=float(g.group(2)), lon=float(g.group(3)), depth=float(g.group(4)), rms=float(q.group(1)), nphs=int(q.group(2)), gap=float(q.group(3)))
    ph = []
    inblock = False
    for line in txt.split("\n"):
        if line.startswith("PHASE ID"): inblock = True; continue
        if line.startswith("END_PHASE"): break
        if inblock and line.strip():
            f = line.split()
            if len(f) < 22: continue
            try:
                ph.append(dict(sta=f[0], pha=f[4], ttpred=float(f[15]), res=float(f[16]), wt=float(f[17]), sz=float(f[20]), sdist=float(f[21])))
            except ValueError:
                continue
    return ev, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--near-km", type=float, default=4.0)
    a = ap.parse_args()
    for r in a.runs:
        evs, phs = [], []
        for i, f in enumerate(sorted(glob.glob(str(REPO / "nlloc" / "output" / r / "loc.2*.grid0.loc.hyp")))):
            ev, ph = parse(f)
            if ev is None: continue
            ev["i"] = i; evs.append(ev)
            for p in ph: p["i"] = i; phs.append(p)
        E = pd.DataFrame(evs); P = pd.DataFrame(phs); P = P[P.wt > 0]
        E["pinned"] = E.depth < 0.05
        P = P.merge(E[["i", "pinned", "depth"]], on="i")
        near = P[(P.sdist < a.near_km)]
        nearP_pin = near[(near.pha == "P") & near.pinned].res; nearS_pin = near[(near.pha == "S") & near.pinned].res
        nearP_all = near[near.pha == "P"].res
        sta = P[P.pha == "P"].groupby("sta").agg(res=("res", "median"), n=("res", "size"), wz=("sz", "first"))
        sta = sta[(sta.n >= 30) & (sta.wz > 0.3)]                     # OBS only
        slope = np.polyfit(sta.wz, sta.res, 1)[0] if len(sta) > 3 else np.nan
        print(f"\n=== {r}: {len(E):,} located events, {len(P):,} weighted phases ===")
        print(f"  1. top-pinned (depth<0.05 km): {E.pinned.mean()*100:5.1f}%      depth p10/50/90 {E.depth.quantile(.1):.2f}/{E.depth.median():.2f}/{E.depth.quantile(.9):.2f} km   rms p50 {E.rms.median():.3f} s")
        print(f"  2. near(<{a.near_km:.0f} km) P residual, pinned set: median {nearP_pin.median():+.3f} s (n={len(nearP_pin)})   S: {nearS_pin.median():+.3f} (n={len(nearS_pin)})   | all near P: {nearP_all.median():+.3f}")
        print(f"  3. per-OBS median P residual vs water depth: slope {slope:+.4f} s/km over {len(sta)} stations; station medians p10/50/90 {sta.res.quantile(.1):+.3f}/{sta.res.median():+.3f}/{sta.res.quantile(.9):+.3f} s")
        for ph in ("P", "S"):
            k = P.pha == ph; print(f"     all {ph} residuals: median {P.res[k].median():+.3f}  MAD {np.median(np.abs(P.res[k]-P.res[k].median())):.3f} s  (n={k.sum():,})")
        print("  4. S-P spread ratio: run 52_sp_depth_check.py --hyp-dir nlloc/output/" + r)


if __name__ == "__main__":
    main()
