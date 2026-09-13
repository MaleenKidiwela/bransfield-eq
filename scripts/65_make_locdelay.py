"""Build NLLoc LOCDELAY lines (station P/S time delays).

Two sources, combinable (iteration of station terms):
  --velest-csv  velest/<run>/station_corrections.csv  (ptcor_s, stcor_s; key = NET.STA)
  --from-hyp-dir <nlloc output dir>  per-station median residual of weighted phases from
                 the .hyp files of a previous NLLoc run, ADDED to --base (previous delays)
NLLoc semantics (NLLocLib.c DELAY_CORR): obs_time -= delay, i.e. delay = mean (O-C) of a
station, positive = station observed LATE. VELEST's station corrections are also
"observed minus computed" per station (positive = late), so they map 1:1.
Format: LOCDELAY <label> <phase> <nReadings> <delay_s>
"""
from __future__ import annotations
import argparse, glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def parse_delays(path):
    d = {}
    for l in Path(path).read_text().split("\n"):
        f = l.split()
        if len(f) == 5 and f[0] == "LOCDELAY":
            d[(f[1], f[2])] = float(f[4])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velest-csv")
    ap.add_argument("--from-hyp-dir")
    ap.add_argument("--base", help="existing LOCDELAY file to add residual medians to")
    ap.add_argument("--min-readings", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    delays, nread = {}, {}
    if a.base:
        delays.update(parse_delays(a.base))
    if a.velest_csv:
        d = pd.read_csv(a.velest_csv)
        for _, r in d.iterrows():
            sta = r["key"].split(".")[-1]
            delays[(sta, "P")] = float(r.ptcor_s); delays[(sta, "S")] = float(r.stcor_s)
        print(f"velest corrections: {len(d)} stations from {a.velest_csv}")
    if a.from_hyp_dir:
        from importlib import import_module
        s61 = import_module("61_sample_gates")
        rows = []
        for f in sorted(glob.glob(str(Path(a.from_hyp_dir) / "loc.2*.grid0.loc.hyp"))):
            ev, ph = s61.parse(f)
            if ev is None: continue
            rows += [p for p in ph if p["wt"] > 0]
        P = pd.DataFrame(rows)
        med = P.groupby(["sta", "pha"]).res.agg(["median", "size"])
        upd = 0
        for (sta, pha), r in med.iterrows():
            if r["size"] >= a.min_readings:
                delays[(sta, pha)] = delays.get((sta, pha), 0.0) + float(r["median"]); nread[(sta, pha)] = int(r["size"]); upd += 1
        print(f"residual update from {a.from_hyp_dir}: {upd} station/phase terms (>= {a.min_readings} readings); "
              f"|median residual| p50 {med['median'].abs().median():.3f} s, max {med['median'].abs().max():.3f} s")
    lines = [f"LOCDELAY {sta:<6s} {pha} {nread.get((sta, pha), 1):4d} {v:8.4f}" for (sta, pha), v in sorted(delays.items())]
    Path(a.out).write_text("\n".join(lines) + "\n")
    v = np.array(list(delays.values()))
    print(f"wrote {a.out}: {len(lines)} LOCDELAY lines; P range {min(x for (s,p),x in delays.items() if p=='P'):+.2f}..{max(x for (s,p),x in delays.items() if p=='P'):+.2f}, "
          f"S range {min(x for (s,p),x in delays.items() if p=='S'):+.2f}..{max(x for (s,p),x in delays.items() if p=='S'):+.2f} s")


if __name__ == "__main__":
    main()
