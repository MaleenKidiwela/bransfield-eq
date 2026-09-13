"""Evaluate hypoDD depth recovery for one or more run labels.

For each label: join hypoDD.reloc (DD depth, flat frame) to event.sel (start depth,
same frame) on the hypoDD id and report
  * retention (relocated / event.sel)
  * slope of (DD depth - start depth) on start depth  (0 = spread preserved,
    -1 = all depth information destroyed)
  * p10/p50/p90 of start and DD depth, Spearman(start, DD)
  * median |DD - start|
Gate for the noise-free synthetic run (Merlin's protocol): |slope| <= 0.05 and
|p90_DD - p90_start| <= 0.2 km. Used on the real runs too, so the numbers are
directly comparable with the change-log tables.
"""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("s24", REPO / "scripts" / "24_run_hypodd.py")
s24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s24)


def evaluate(label: str, gate: bool) -> dict:
    run = REPO / "hypodd" / label
    ev = pd.read_csv(run / "event.sel", sep=r"\s+", header=None,
                     names=["date", "time", "lat", "lon", "dep", "mag", "eh", "ez", "rms", "id"])
    if not ev.id.is_unique:
        raise SystemExit(f"{label}: event.sel ids not unique")
    dd = s24.parse_reloc(run / "hypoDD.reloc")
    j = dd.merge(ev[["id", "dep"]].rename(columns={"dep": "z0"}), on="id", how="inner", validate="1:1")
    if len(j) != len(dd):
        raise SystemExit(f"{label}: {len(dd)-len(j)} relocated ids missing from event.sel")
    dz = j.dep - j.z0
    slope, icpt = np.polyfit(j.z0, dz, 1)
    q = lambda s: np.percentile(s, [10, 50, 90])
    r = dict(label=label, n_in=len(ev), n_reloc=len(j), retain=len(j) / len(ev), slope=slope,
             p_start=q(j.z0), p_dd=q(j.dep), spearman=spearmanr(j.z0, j.dep).statistic,
             med_abs_dz=np.median(np.abs(dz)), rct=np.median(j.rct[j.rct > -9]))
    print(f"\n=== {label} ===")
    print(f"  relocated {r['n_reloc']:,} / {r['n_in']:,} ({r['retain']*100:.0f}%)   median rct {r['rct']:.3f} s")
    print(f"  slope of (DD - start) on start depth: {slope:+.3f}   (intercept {icpt:+.2f} km)")
    print(f"  start depth p10/50/90: {r['p_start'][0]:.2f} / {r['p_start'][1]:.2f} / {r['p_start'][2]:.2f} km")
    print(f"  DD    depth p10/50/90: {r['p_dd'][0]:.2f} / {r['p_dd'][1]:.2f} / {r['p_dd'][2]:.2f} km")
    print(f"  Spearman(start, DD) {r['spearman']:.3f}   median |DD - start| {r['med_abs_dz']:.2f} km")
    if gate:
        ok1 = abs(slope) <= 0.05; ok2 = abs(r['p_dd'][2] - r['p_start'][2]) <= 0.2
        print(f"  GATE noise-free: slope within +-0.05 [{'PASS' if ok1 else 'FAIL'}]   "
              f"p90 within 0.2 km [{'PASS' if ok2 else 'FAIL'}]  -> {'PASS' if ok1 and ok2 else 'FAIL'}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--gate-first", action="store_true", help="apply the noise-free gate to the first label")
    a = ap.parse_args()
    for i, lab in enumerate(a.labels):
        evaluate(lab, gate=a.gate_first and i == 0)


if __name__ == "__main__":
    main()
