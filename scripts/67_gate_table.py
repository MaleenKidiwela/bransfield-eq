"""One table for several NLLoc sample runs (same 2,000-event obs, possibly guarded):
located, pinned fraction, near-station P/S residual of the pinned set, rms, S-P spread
ratio scored consistently with each run's own LOCDELAY file, depth quantiles, and depth
change vs a reference run by reference-depth bin. Events are matched by hyp filename
(origin-time), so runs that lost events still compare.
usage: 67_gate_table.py --ref abtest_ORCA_v5 run1[:delays] run2[:delays] ...
"""
from __future__ import annotations
import argparse, glob, os, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
GEO = re.compile(r'GEOGRAPHIC\s+OT\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?[\d.]+)\s+Lat\s+(-?[\d.]+)\s+Long\s+(-?[\d.]+)\s+Depth\s+(-?[\d.]+)')


def load(run):
    from importlib import import_module
    s61 = import_module("61_sample_gates")
    evs, phs = {}, []
    for f in glob.glob(str(REPO / "nlloc" / "output" / run / "loc.2*.grid0.loc.hyp")):
        ev, ph = s61.parse(f)
        if ev is None: continue
        b = os.path.basename(f); ev["fn"] = b; evs[b] = ev
        for p in ph: p["fn"] = b; phs.append(p)
    E = pd.DataFrame(evs.values()).set_index("fn"); P = pd.DataFrame(phs); P = P[P.wt > 0].merge(E[["depth"]], left_on="fn", right_index=True)
    return E, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="abtest_ORCA_v5")
    ap.add_argument("--obs-order", default="nlloc/obs/year_v4.event_order.csv")
    ap.add_argument("runs", nargs="+", help="run[:delays_file]")
    a = ap.parse_args()
    order = pd.read_csv(REPO / a.obs_order).head(2000)
    ref_files = sorted(os.path.basename(f) for f in glob.glob(str(REPO / "nlloc" / "output" / a.ref / "loc.2*.grid0.loc.hyp")))
    fn2idx = dict(zip(ref_files, order.event_idx.values))
    Eref, _ = load(a.ref)
    rows = []
    for spec in [a.ref] + a.runs:
        run, _, dl = spec.partition(":")
        E, P = load(run)
        pin = E.depth < 0.05
        near = P[P.sdist < 4.0]; nP = near[(near.pha == "P") & (near.depth < 0.05)].res; nS = near[(near.pha == "S") & (near.depth < 0.05)].res
        # S-P gate via script 52 on a mapped catalogue
        cat = pd.DataFrame([dict(event_idx=fn2idx[b], lat=r.lat, lon=r.lon, depth_km=r.depth) for b, r in E.iterrows() if b in fn2idx])
        cp = REPO / "nlloc" / "output" / run / "catalog_mapped.csv"; cat.to_csv(cp, index=False)
        cmd = [sys.executable, str(REPO / "scripts" / "52_sp_depth_check.py"), "--catalog", str(cp), "--frame", "sealevel", "--scorer-vpvs", "1.88", "--label", run]
        if dl: cmd += ["--delays", dl]
        out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}).stdout
        m = re.search(r"observed x([\d.]+)\s+predicted x([\d.]+)", out); obs_r, pred_r = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
        j = pd.concat([Eref.depth.rename("ref"), E.depth.rename("x")], axis=1).dropna()
        bins = {f"{lo}-{hi}": (j.x[(j.ref >= lo) & (j.ref < hi)] - j.ref[(j.ref >= lo) & (j.ref < hi)]).median() for lo, hi in [(0.05, 1), (1, 2), (2, 4), (4, 6), (6, 9), (9, 15)]}
        rows.append(dict(run=run, located=len(E), pinned=f"{pin.mean()*100:.1f}%", nearP_pin=f"{nP.median():+.3f}", nearS_pin=f"{nS.median():+.3f}", rms=f"{E.rms.median():.3f}",
                         sp_pred_obs=f"{pred_r:.2f}/{obs_r:.2f}", ratio=f"{pred_r/obs_r:.2f}", z10_50_90=f"{E.depth.quantile(.1):.2f}/{E.depth.median():.2f}/{E.depth.quantile(.9):.2f}",
                         **{f"dz {k}": f"{v:+.2f}" for k, v in bins.items()}))
    pd.set_option("display.width", 250); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
