"""Compare hypoDD IMOD=9 travel times (hypoDD.src from a frozen run) with NLLoc's
ORCA_v4 time grids at the SAME source-station pairs.

This is the direct fidelity test of the simulps node resampling + pseudo-bending
tracer against the finite-difference times NLLoc located with. The frozen-run RMSCT
cannot do that job: NLLoc's own per-event rms is ~0.2 s, so residuals of ~0.2 s are
expected under any model that resembles the one NLLoc used.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def read_time_grid(prefix: Path):
    f = Path(str(prefix) + ".hdr").read_text().split("\n")[0].split()
    nx, ny, nz = map(int, f[:3]); org = tuple(map(float, f[3:6])); sp = tuple(map(float, f[6:9]))
    assert f[9] == "TIME", f
    t = np.fromfile(str(prefix) + ".buf", dtype=np.float32).reshape(nx, ny, nz)
    return t, org, sp


def trilinear(g, org, sp, gx, gy, gz):
    nx, ny, nz = g.shape
    fx = np.clip((gx - org[0]) / sp[0], 0, nx - 1.0001); fy = np.clip((gy - org[1]) / sp[1], 0, ny - 1.0001)
    fz = np.clip((gz - org[2]) / sp[2], 0, nz - 1.0001)
    i, j, k = fx.astype(int), fy.astype(int), fz.astype(int); tx, ty, tz = fx - i, fy - j, fz - k
    v = 0
    for di, wx in ((0, 1 - tx), (1, tx)):
        for dj, wy in ((0, 1 - ty), (1, ty)):
            for dk, wz in ((0, 1 - tz), (1, tz)):
                v = v + wx * wy * wz * g[i + di, j + dj, k + dk]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="year_v4_3d_frozen")
    ap.add_argument("--tt-prefix", default="ORCA_v4")
    a = ap.parse_args()
    from importlib import import_module
    s41 = import_module("41_build_unsheared_velgrid")
    run = REPO / "hypodd" / a.label
    src = pd.read_csv(run / "hypoDD.src", sep=r"\s+", header=None,
                      names=["id", "lat", "lon", "sta", "azp", "ainp", "azs", "ains", "ttp", "tts"])
    ev = pd.read_csv(run / "event.sel", sep=r"\s+", header=None,
                     names=["d", "t", "lat", "lon", "dep", "m", "eh", "ez", "rms", "id"]).set_index("id")
    st = pd.read_csv(run / "station.dat", sep=r"\s+", header=None, names=["sta", "lat", "lon", "elev_m"]).set_index("sta")
    src = src[src.ttp > 0].reset_index(drop=True)
    src["dep"] = ev.dep.reindex(src.id).values
    src["land"] = st.elev_m.reindex(src.sta).values > 0
    gx, gy = s41.stingray_xy(src.lat.values, src.lon.values)
    src["dist"] = np.hypot(*(np.array(s41.stingray_xy(st.lat.reindex(src.sta).values, st.lon.reindex(src.sta).values)) - np.array([gx, gy])))
    src["nllP"] = np.nan; src["nllS"] = np.nan
    for sta, g in src.groupby("sta"):
        for ph, col in (("P", "nllP"), ("S", "nllS")):
            p = REPO / "nlloc" / "time" / f"{a.tt_prefix}.{ph}.{sta}.time"
            if not Path(str(p) + ".hdr").exists():
                print(f"  no NLLoc grid for {sta} {ph}"); continue
            t, org, sp = read_time_grid(p)
            ix = g.index.values
            src.loc[ix, col] = trilinear(t, org, sp, gx[ix], gy[ix], g.dep.values)
    src["dP"] = (src.ttp - src.nllP) * 1000; src["dS"] = (src.tts - src.nllS) * 1000
    print(f"{len(src):,} source-station rays, {src.sta.nunique()} stations, {src.id.nunique():,} sources; depth {src.dep.min():.2f}-{src.dep.max():.2f} km")
    def line(name, k):
        if k.sum() == 0: return
        dP, dS = src.dP[k].dropna(), src.dS[k].dropna()
        def stats(d):
            return f"median {d.median():+6.1f} MAD {np.median(np.abs(d-d.median())):5.1f} p90|.| {np.percentile(np.abs(d),90):6.1f}" if len(d) else "no grid"
        print(f"  {name:28s} n={k.sum():6,}  P: {stats(dP)}   S: {stats(dS)}")
    print("hypoDD-3D minus NLLoc travel time (ms):")
    line("ALL", np.ones(len(src), bool)); line("OBS", ~src.land.values); line("land", src.land.values)
    for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 20), (20, 40), (40, 100)]:
        line(f"OBS dist {lo}-{hi} km", (~src.land.values) & (src.dist >= lo).values & (src.dist < hi).values)
    for lo, hi in [(0, 1.5), (1.5, 2.5), (2.5, 4), (4, 6), (6, 9), (9, 15), (15, 30)]:
        line(f"OBS depth {lo}-{hi} km BSL", (~src.land.values) & (src.dep >= lo).values & (src.dep < hi).values)
    print("per station median dP (ms):", src.groupby("sta").dP.median().round(0).sort_values().to_dict())
    out = run / "tt_compare_3d_vs_nlloc.csv"; src.to_csv(out, index=False); print("wrote", out)


if __name__ == "__main__":
    main()
