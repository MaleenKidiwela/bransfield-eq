"""Task 2 of the v6 validation package: how much of the v6 depth is carried by the
fitted station terms?

Relocates the same 2,000-event sample with perturbed LOCDELAY sets (built by
68_perturb_delays.py, guarded by 66_guard_obs_for_delays.py) and compares each
event with the production reference run abtest_v6_it1Bg. Events are matched by hyp
FILENAME (the first observation time), exactly as 67_gate_table.py does, so runs
that lost events still compare like with like.

Reports median |dz| and p90 |dz| per REFERENCE-depth bin -- the systematic depth
uncertainty contributed by the station terms -- alongside the signed median shift,
and the NLLoc formal sigma_z of the year catalogue tiers for comparison.
"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
from importlib import import_module
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
s61 = import_module("61_sample_gates")

BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 9.0), (9.0, 15.0)]


def load(run):
    """hyp filename -> event record, same parser 67_gate_table.py uses."""
    rows = {}
    for f in glob.glob(str(REPO / "nlloc" / "output" / run / "loc.2*.grid0.loc.hyp")):
        if "last" in os.path.basename(f):
            continue
        ev, _ = s61.parse(f)
        if ev is None:
            continue
        rows[os.path.basename(f)] = ev
    return pd.DataFrame(rows.values(), index=list(rows))


def dz_table(ref, run, label, mask=None):
    j = ref[["depth", "rms", "gap", "nphs"]].join(run[["depth"]], rsuffix="_x", how="inner").dropna()
    if mask is not None:
        j = j[mask(j)]
    j["dz"] = j.depth_x - j.depth
    out = []
    for lo, hi in BINS:
        g = j[(j.depth >= lo) & (j.depth < hi)]
        out.append(dict(run=label, bin=f"{lo:g}-{hi:g}", n=len(g),
                        med_abs_dz=g.dz.abs().median() if len(g) else np.nan,
                        p90_abs_dz=g.dz.abs().quantile(0.9) if len(g) else np.nan,
                        med_dz=g.dz.median() if len(g) else np.nan))
    g = j[(j.depth >= 0.05) & (j.depth < 1.0)]
    out.append(dict(run=label, bin="0.05-1 (unpinned)", n=len(g),
                    med_abs_dz=g.dz.abs().median() if len(g) else np.nan,
                    p90_abs_dz=g.dz.abs().quantile(0.9) if len(g) else np.nan,
                    med_dz=g.dz.median() if len(g) else np.nan))
    out.append(dict(run=label, bin="ALL", n=len(j), med_abs_dz=j.dz.abs().median(),
                    p90_abs_dz=j.dz.abs().quantile(0.9), med_dz=j.dz.median()))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="abtest_v6_it1Bg")
    ap.add_argument("--runs", nargs="+", default=["abtest_v6_pertA", "abtest_v6_pertS", "abtest_v6_pert12"])
    ap.add_argument("--catalog", default="catalogs/nlloc_year_v6.csv")
    a = ap.parse_args()

    ref = load(a.ref)
    print(f"reference {a.ref}: {len(ref):,} located; depth p10/50/90 "
          f"{ref.depth.quantile(.1):.2f}/{ref.depth.median():.2f}/{ref.depth.quantile(.9):.2f} km; "
          f"pinned (z<0.05) {100*(ref.depth < 0.05).mean():.1f}%; rms p50 {ref.rms.median():.3f} s")

    all_t, all_ts = [], []
    for r in a.runs:
        run = load(r)
        common = ref.index.intersection(run.index)
        print(f"\n{r}: {len(run):,} located, {len(common):,} share a hyp filename with the reference; "
              f"depth p10/50/90 {run.depth.quantile(.1):.2f}/{run.depth.median():.2f}/{run.depth.quantile(.9):.2f} km; "
              f"pinned {100*(run.depth < 0.05).mean():.1f}%; rms p50 {run.rms.median():.3f} s")
        all_t.append(dz_table(ref, run, r))
        all_ts.append(dz_table(ref, run, r,
                               mask=lambda j: (j.gap < 180) & (j.rms < 0.5) & (j.nphs >= 6)))

    pd.set_option("display.width", 200)
    T = pd.concat(all_t).pivot(index="bin", columns="run", values=["med_abs_dz", "p90_abs_dz", "med_dz"])
    order = [f"{lo:g}-{hi:g}" for lo, hi in BINS] + ["0.05-1 (unpinned)", "ALL"]
    print("\n=== |dz| vs the production run, by REFERENCE depth bin (km), all located events ===")
    print(T.reindex(order).to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n   n per bin: " + ", ".join(f"{r['bin']} {r['n']}" for _, r in all_t[0].iterrows()))

    Ts = pd.concat(all_ts).pivot(index="bin", columns="run", values=["med_abs_dz", "p90_abs_dz", "med_dz"])
    print("\n=== same, restricted to standard-tier-like events (gap<180, rms<0.5, Nphs>=6) ===")
    print(Ts.reindex(order).to_string(float_format=lambda x: f"{x:+.2f}"))
    print("\n   n per bin: " + ", ".join(f"{r['bin']} {r['n']}" for _, r in all_ts[0].iterrows()))

    # envelope across the perturbations: worst-case |dz| per event, per bin
    print("\n=== envelope over the perturbation set (per event, max |dz| over the runs) ===")
    runs = {r: load(r) for r in a.runs}
    D = pd.DataFrame({r: (v.depth.reindex(ref.index) - ref.depth) for r, v in runs.items()})
    D["ref"] = ref.depth
    D["env"] = D[list(runs)].abs().max(axis=1)
    rows = []
    for lo, hi in BINS:
        g = D[(D.ref >= lo) & (D.ref < hi)].dropna(subset=["env"])
        rows.append(dict(bin=f"{lo:g}-{hi:g}", n=len(g), med_env=g.env.median(), p90_env=g.env.quantile(0.9)))
    g = D.dropna(subset=["env"])
    rows.append(dict(bin="ALL", n=len(g), med_env=g.env.median(), p90_env=g.env.quantile(0.9)))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== NLLoc formal sigma_z (km) in the year catalogue, for comparison ===")
    acc = import_module("68_catalogue_accounting")
    cat = pd.read_csv(REPO / a.catalog)
    cat = cat[cat.nlloc_status == "LOCATED"].copy()
    cat, _ = acc.add_bsf_and_hull(cat)
    loose, standard, strict = acc.tiers(cat)
    for name, m in [("loose", loose), ("standard", standard), ("strict", strict)]:
        q = cat.sigma_z_km[m].quantile([0.1, 0.5, 0.9])
        print(f"  {name:9s} n={int(m.sum()):6,}  sigma_z p10/50/90 "
              f"{q.iloc[0]:.2f} / {q.iloc[1]:.2f} / {q.iloc[2]:.2f} km")


if __name__ == "__main__":
    main()
