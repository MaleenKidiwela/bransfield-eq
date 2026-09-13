"""Independent eikonal travel times on the ORCA_v4 grid (pykonal point-source FMM),
compared with (i) the slowness integral along the vertical column under the station,
(ii) NLLoc's Grid2Time (Podvin-Lecomte) TIME grid, (iii) hypoDD IMOD=9 path times.

Why: NLLoc's P time grids are 60-130 ms slower than the slowness integral through their
own velocity column (station-specific, constant with depth). Two candidate causes: the
FD scheme applying each node's slowness to the cell below it (half-cell shift) and a
near-source FD error in coarse, high-contrast cells. A second solver on the SAME grid,
with node-value = velocity-at-node, arbitrates.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent


def read_grid(prefix):
    f = Path(str(prefix) + ".hdr").read_text().split("\n"); a = f[0].split()
    nx, ny, nz = map(int, a[:3]); org = np.array(list(map(float, a[3:6]))); sp = np.array(list(map(float, a[6:9])))
    buf = np.fromfile(str(prefix) + ".buf", dtype=np.float32).reshape(nx, ny, nz)
    return buf, org, sp, f


def tri(g, org, sp, p):
    f = (np.asarray(p) - org) / sp; i = f.astype(int); t = f - i; v = 0.0
    for di, wx in ((0, 1 - t[0]), (1, t[0])):
        for dj, wy in ((0, 1 - t[1]), (1, t[1])):
            for dk, wz in ((0, 1 - t[2]), (1, t[2])):
                v += wx * wy * wz * g[i[0] + di, i[1] + dj, i[2] + dk]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sta", default="BRA19")
    ap.add_argument("--half", type=float, default=24.0, help="half-width of the sub-grid (km)")
    ap.add_argument("--zmax", type=float, default=16.0)
    ap.add_argument("--refine", type=float, default=1.0, help="1 = native 0.4 km; 2 = 0.2 km (trilinear upsample)")
    a = ap.parse_args()
    import pykonal
    slow, org, sp, hdr = read_grid(REPO / "nlloc" / "model" / "ORCA_v4.P.mod"); vp = sp[0] / slow
    T, _, _, thdr = read_grid(REPO / "nlloc" / "time" / f"ORCA_v4.P.{a.sta}.time")
    hx, hy, hz = map(float, thdr[1].split()[1:4])
    # sub-grid around the station
    i0 = int((hx - a.half - org[0]) / sp[0]); i1 = int((hx + a.half - org[0]) / sp[0]) + 1
    j0 = int((hy - a.half - org[1]) / sp[1]); j1 = int((hy + a.half - org[1]) / sp[1]) + 1
    k1 = int(a.zmax / sp[2]) + 1
    sub = vp[i0:i1, j0:j1, :k1]; sorg = org + np.array([i0, j0, 0]) * sp
    if a.refine > 1:
        from scipy.ndimage import zoom
        sub = zoom(sub, a.refine, order=1); ssp = sp / a.refine
    else:
        ssp = sp.copy()
    print(f"{a.sta} at ({hx:.3f},{hy:.3f},{hz:.3f}); sub-grid {sub.shape} spacing {ssp[0]:.2f} km, origin {sorg}")
    solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    solver.vv.min_coords = sorg; solver.vv.node_intervals = ssp; solver.vv.npts = sub.shape
    solver.vv.values = sub.astype(np.float64)
    solver.src_loc = np.array([hx, hy, hz])
    solver.solve()
    tt = solver.tt.values
    print(f"pykonal done; tt range {np.nanmin(tt):.3f}-{np.nanmax(tt):.3f} s")
    # (i) vertical column integral, (ii) NLLoc FD grid, (iii) pykonal at points below the station
    print(f"{'z':>5} {'column ∫s':>10} {'pykonal':>9} {'NLLoc FD':>9}  {'pyk-col':>8} {'FD-col':>7}")
    for z in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0):
        zz = np.linspace(hz, z, 2000); s = np.array([tri(slow, org, sp, (hx, hy, q)) for q in zz]) / sp[0]
        ti = np.trapezoid(s, zz); tp = tri(tt, sorg, ssp, (hx, hy, z)); tf = tri(T, org, sp, (hx, hy, z))
        print(f"{z:5.1f} {ti:10.4f} {tp:9.4f} {tf:9.4f}  {(tp-ti)*1000:+8.1f} {(tf-ti)*1000:+7.1f}")
    # (iii) hypoDD IMOD=9 (fine node model) path times for this station's rays
    src = pd.read_csv(REPO / "hypodd" / "year_v4_3d_frozen_fine" / "hypoDD.src", sep=r"\s+", header=None,
                      names=["id", "lat", "lon", "sta", "azp", "ainp", "azs", "ains", "ttp", "tts"])
    src = src[(src.sta == a.sta) & (src.ttp > 0)]
    ev = pd.read_csv(REPO / "hypodd" / "year_v4_3d" / "event.sel", sep=r"\s+", header=None,
                     names=["d", "t", "lat", "lon", "dep", "m", "eh", "ez", "rms", "id"]).set_index("id")
    sys.path.insert(0, str(REPO / "scripts")); from importlib import import_module; s41 = import_module("41_build_unsheared_velgrid")
    gx, gy = s41.stingray_xy(src.lat.values, src.lon.values); dep = ev.dep.reindex(src.id).values
    inside = (gx > sorg[0] + 1) & (gx < sorg[0] + ssp[0] * (sub.shape[0] - 1) - 1) & (gy > sorg[1] + 1) & (gy < sorg[1] + ssp[1] * (sub.shape[1] - 1) - 1) & (dep < a.zmax - 1)
    d_pyk = np.array([tri(tt, sorg, ssp, (x, y, z)) for x, y, z in zip(gx[inside], gy[inside], dep[inside])])
    d_fd = np.array([tri(T, org, sp, (x, y, z)) for x, y, z in zip(gx[inside], gy[inside], dep[inside])])
    hd = src.ttp.values[inside]; dist = np.hypot(gx[inside] - hx, gy[inside] - hy)
    def st(x): return f"median {np.median(x)*1000:+7.1f}  MAD {np.median(np.abs(x-np.median(x)))*1000:5.1f} ms"
    print(f"\n{len(hd)} real rays to {a.sta} (dist {dist.min():.1f}-{dist.max():.1f} km, depth {dep[inside].min():.1f}-{dep[inside].max():.1f}):")
    print(f"  hypoDD-3D − pykonal : {st(hd - d_pyk)}")
    print(f"  NLLoc FD  − pykonal : {st(d_fd - d_pyk)}")
    print(f"  hypoDD-3D − NLLoc FD: {st(hd - d_fd)}")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 25)]:
        k = (dist >= lo) & (dist < hi)
        if k.sum(): print(f"    dist {lo:>2}-{hi:<2} km n={k.sum():5d}: FD−pyk {st(d_fd[k]-d_pyk[k])} | hDD−pyk {st(hd[k]-d_pyk[k])}")


if __name__ == "__main__":
    main()
