"""Rebuild the NLLoc P travel-time grids with a point-source fast-marching eikonal
solver (pykonal) on the ORCA_v4 velocity grid -> prefix ORCA_v5.

Why (2026-09-13): NLLoc's Grid2Time (Podvin-Lecomte) grids are 60-160 ms SLOWER than
the model they were computed from -- station- and distance-dependent. Two independent
solvers on the same grid (pykonal FMM and hypoDD's simulps pseudo-bending tracer) agree
with each other to ~6 ms and with the vertical slowness integral to ~5 ms; Grid2Time
does not. Time_3d_NLL.c states that hs[][][] are "(constant) slownesses in cells", so the
node-value model is effectively shifted half a cell (0.2 km) downward, and the coarse
near-source cells add a station-specific error on top. Every NLLoc location in v1-v4
inherited these forward-model errors (and S = P x Vp/Vs inherited them x1.78).

Output: nlloc/time/ORCA_v5.P.<STA>.time.{hdr,buf} with the SAME header lines as the v4
grids (dimensions, origin, spacing, station line, TRANS), plus the model copied as
ORCA_v5.P.mod.* so NLLoc's GTFILES/LOCFILES roots stay consistent. Node value =
velocity AT the node (trilinear between nodes), the convention every other tool here uses.
Gate per station: |grid - vertical column slowness integral| at z = 3, 6, 10 km < 15 ms
(the column is not always the first arrival, so a CHECK is a flag, not a failure; the
real arbiter is agreement with hypoDD's independent tracer on real rays, script 58).
"""
from __future__ import annotations
import argparse, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC, DST = "ORCA_v4", "ORCA_v5"          # overridable with --src/--dst (module globals used by the workers)


def read_grid(prefix):
    lines = Path(str(prefix) + ".hdr").read_text().split("\n"); a = lines[0].split()
    nx, ny, nz = map(int, a[:3]); org = np.array(list(map(float, a[3:6]))); sp = np.array(list(map(float, a[6:9])))
    return np.fromfile(str(prefix) + ".buf", dtype=np.float32).reshape(nx, ny, nz), org, sp, lines


def tri(g, org, sp, p):
    f = (np.asarray(p) - org) / sp; i = f.astype(int); t = f - i; v = 0.0
    for di, wx in ((0, 1 - t[0]), (1, t[0])):
        for dj, wy in ((0, 1 - t[1]), (1, t[1])):
            for dk, wz in ((0, 1 - t[2]), (1, t[2])):
                v += wx * wy * wz * g[i[0] + di, i[1] + dj, i[2] + dk]
    return v


def one_station(sta: str, refine: int = 2, src: str = SRC, dst: str = DST) -> str:
    """refine=2: solve on a 0.2 km trilinear upsample of the model, then keep every 2nd node.
    At the native 0.4 km the first-order FMM is up to 60 ms FAST at some stations (BRA20 -60,
    BRA23 -52 ms vs the column integral); at 0.2 km BRA20 is +5 ms. Verified 2026-09-13."""
    import pykonal
    from scipy.ndimage import zoom
    t0 = time.time()
    slow, org, sp, _ = read_grid(REPO / "nlloc" / "model" / f"{src}.P.mod"); vp = (sp[0] / slow).astype(np.float64)
    if refine > 1:
        vp = zoom(vp, refine, order=1); sp_s = sp / refine
    else:
        sp_s = sp
    _, _, _, thdr = read_grid(REPO / "nlloc" / "time" / f"ORCA_v4.P.{sta}.time")   # station line + dims from the v4 headers
    hx, hy, hz = map(float, thdr[1].split()[1:4])
    solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    solver.vv.min_coords = org; solver.vv.node_intervals = sp_s; solver.vv.npts = vp.shape
    solver.vv.values = vp
    solver.src_loc = np.array([hx, hy, hz])
    solver.solve()
    tt = np.asarray(solver.tt.values, dtype=np.float32)
    if refine > 1:
        tt = np.ascontiguousarray(tt[::refine, ::refine, ::refine])   # back on the 0.4 km nodes
        assert tt.shape == slow.shape, (tt.shape, slow.shape)
    del solver, vp
    if not np.all(np.isfinite(tt)):
        return f"{sta}: FAIL non-finite travel times ({(~np.isfinite(tt)).sum()} nodes)"
    out = REPO / "nlloc" / "time" / f"{dst}.P.{sta}.time"
    tt.tofile(str(out) + ".buf")
    Path(str(out) + ".hdr").write_text("\n".join(thdr))          # identical header (dims, station, TRANS)
    # gate: vertical column
    checks = []
    for z in (3.0, 6.0, 10.0):
        if z <= hz + 0.5: continue
        zz = np.linspace(hz, z, 2000); s = np.array([tri(slow, org, sp, (hx, hy, q)) for q in zz]) / sp[0]
        checks.append((tri(tt, org, sp, (hx, hy, z)) - np.trapezoid(s, zz)) * 1000)
    worst = max(abs(c) for c in checks)
    return f"{sta}: {time.time()-t0:6.0f} s  tt max {tt.max():.2f} s  column offsets {[f'{c:+.1f}' for c in checks]} ms -> {'OK' if worst < 15 else 'CHECK'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="*", default=None, help="default: every station with a v4 grid")
    ap.add_argument("--workers", type=int, default=5, help="refine=2 needs ~7 GB per worker")
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--src", default=SRC, help="velocity model prefix (nlloc/model/<src>.P.mod)")
    ap.add_argument("--dst", default=DST, help="time-grid prefix to write")
    a = ap.parse_args()
    stations = a.stations or sorted(p.name.split(".")[2] for p in (REPO / "nlloc" / "time").glob("ORCA_v4.P.*.time.hdr"))
    for ext in ("hdr", "buf"):
        srcf = REPO / "nlloc" / "model" / f"{a.src}.P.mod.{ext}"; dstf = REPO / "nlloc" / "model" / f"{a.dst}.P.mod.{ext}"
        if a.src != a.dst and not dstf.exists(): shutil.copy(srcf, dstf)
    print(f"{len(stations)} stations, {a.workers} workers; model {a.src} -> time grids {a.dst}")
    n = len(stations)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for msg in ex.map(one_station, stations, [a.refine] * n, [a.src] * n, [a.dst] * n):
            print("  " + msg, flush=True)


if __name__ == "__main__":
    main()
