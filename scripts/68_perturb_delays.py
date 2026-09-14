"""Build perturbed LOCDELAY sets for the v6 station-term depth-uncertainty test.

Two perturbations of the production terms (nlloc/delays/v6_it1B.delays), chosen to
bracket the amplitude of the per-station S-P term that carries the v6 depth axis:

  pertA  VELEST invA station corrections mapped exactly as 65_make_locdelay.py maps them
         (delay = ptcor_s / stcor_s, key = last field of NET.STA, |delay| clipped to 1 s).
         invA held hypocentres and the 1.78 start model fixed, so its S-P terms are the
         SMALLER, "hypocentres-fixed" end of the plausible range (OBS S-P mean +0.21 s
         against +0.54 s for invB, notes/27 I30/I31).
  pertS  the invA S-P DIFFERENCE grafted onto the production P terms:
         S' = P_it1B + (stcor_s - ptcor_s)_invA.  Isolates the S-P term (the quantity the
         depth axis responds to) from the P terms, which also shift the origin time.
  pert12 production terms with the S-P difference scaled x1.2:
         S' = P_it1B + 1.2 * (S_it1B - P_it1B).  P terms unchanged.

Writes LOCDELAY files only; does not run NLLoc.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MAX_ABS = 1.0          # same clip as 65_make_locdelay.py


def parse_delays(path):
    d = {}
    for l in Path(path).read_text().split("\n"):
        f = l.split()
        if len(f) == 5 and f[0] == "LOCDELAY":
            d[(f[1], f[2])] = float(f[4])
    return d


def write_delays(d, out, nread=None):
    nread = nread or {}
    lines = [f"LOCDELAY {sta:<6s} {pha} {nread.get((sta, pha), 1):4d} {max(-MAX_ABS, min(MAX_ABS, v)):8.4f}"
             for (sta, pha), v in sorted(d.items())]
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(lines)} LOCDELAY lines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="nlloc/delays/v6_it1B.delays")
    ap.add_argument("--velest-csv", default="velest/invA/station_corrections.csv")
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--outdir", default="nlloc/delays")
    a = ap.parse_args()

    base = parse_delays(REPO / a.base)
    nread = {}
    for l in (REPO / a.base).read_text().split("\n"):
        f = l.split()
        if len(f) == 5 and f[0] == "LOCDELAY":
            nread[(f[1], f[2])] = int(f[3])

    v = pd.read_csv(REPO / a.velest_csv)
    inva = {}
    for _, r in v.iterrows():
        sta = r["key"].split(".")[-1]
        inva[(sta, "P")] = float(r.ptcor_s)
        inva[(sta, "S")] = float(r.stcor_s)
    write_delays(inva, REPO / a.outdir / "v6_pertA_invA.delays")

    pertS = dict(base)
    n = 0
    for (sta, pha) in list(base):
        if pha != "P":
            continue
        if (sta, "P") in inva and (sta, "S") in inva and (sta, "S") in base:
            pertS[(sta, "S")] = base[(sta, "P")] + (inva[(sta, "S")] - inva[(sta, "P")])
            n += 1
    print(f"pertS: {n} stations took the invA S-P difference")
    write_delays(pertS, REPO / a.outdir / "v6_pertS_invAsp.delays", nread)

    p12 = dict(base)
    n = 0
    for (sta, pha) in list(base):
        if pha != "S" or (sta, "P") not in base:
            continue
        p12[(sta, "S")] = base[(sta, "P")] + a.scale * (base[(sta, "S")] - base[(sta, "P")])
        n += 1
    print(f"pert12: {n} stations scaled S-P by x{a.scale}")
    write_delays(p12, REPO / a.outdir / f"v6_pert12_sp{a.scale:.1f}.delays", nread)

    obs = [s for (s, p) in base if p == "P" and s.startswith("BRA")]
    rows = []
    for s in sorted(obs):
        rows.append(dict(sta=s,
                         it1B=base[(s, "S")] - base[(s, "P")],
                         pertA=inva.get((s, "S"), float("nan")) - inva.get((s, "P"), float("nan")),
                         pertS=pertS[(s, "S")] - pertS[(s, "P")],
                         pert12=p12[(s, "S")] - p12[(s, "P")]))
    t = pd.DataFrame(rows)
    print("\nOBS S-P delay (s) per station:")
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print("\nmedian S-P delay: " + "  ".join(f"{c} {t[c].median():+.3f}" for c in ["it1B", "pertA", "pertS", "pert12"]))


if __name__ == "__main__":
    main()
