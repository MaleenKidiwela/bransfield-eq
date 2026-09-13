"""Build a simulps-format 3D velocity model for hypoDD IMOD=9 from the ORCA_v4 grid.

hypoDD's 3D mode (partials_3d.f -> ray_3d.f, the simul2000 pseudo-bending tracer)
interpolates trilinearly between NODES, so the 0.4 km NLLoc grid is resampled onto
a graded node set: 2.5 km spacing over the Orca network, coarsening outward, with
far padding nodes so no ray point (stations to 145 km) falls off the model -- the
tracer STOPs on that. Frame: hypoDD's own setorg/dist projection (x east, y north,
km, rot 0, origin = NLLoc TRANS origin), evaluated through `proj_driver` (built
from hypoDD's setorg.o/dist.o/redist.o) so the nodes sit exactly where hypoDD
looks for them. z positive down from SEA LEVEL: stations enter at z = -elev, i.e.
OBS at their true depth, and event depths are NLLoc's below-sea-level depths.

Sampling: node lat/lon -> NLLoc SIMPLE frame (script 41's stingray_xy) -> trilinear
in ORCA_v4.P.mod (SLOW_LEN buffer -> Vp). Nodes above sea level take the z=0
value; nodes outside the grid take the nearest edge (the far field is ~1D anyway).
ORCA_v4 has the water column filled with rock; with stations at true depth no ray
travels there, so that fill is inert. Vp/Vs constant (IPHA=2 with a vpvs grid --
IPHA=1 would hard-code 1.73 in partials_3d.f).

Format (get_vel3d.f): "bld nx ny nz" / xn / yn / zn / two "0 0 0" lines /
vel rows (per z, per y: nx values) / vpvs rows. bld=0.1 -> nodes on 0.1 km
multiples; ixloc arrays (10000) then allow a span of 1000 km.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
ORIGIN_LAT, ORIGIN_LON = -62.4413, -58.44        # NLLoc TRANS SIMPLE origin (script 41)

FINE = np.arange(-25.0, 25.01, 2.5)
XY_NODES = np.unique(np.concatenate([[-450, -200, -140, -100, -75, -55, -42, -32],
                                     FINE, [32, 42, 55, 75, 100, 140, 200, 450]]))
Z_NODES = np.array([-3.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0,
                    7.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 40.0, 400.0])


def read_nlloc_grid(prefix: Path):
    hdr = Path(str(prefix) + ".hdr").read_text().split("\n")
    f = hdr[0].split()
    nx, ny, nz = map(int, f[:3]); x0, y0, z0 = map(float, f[3:6]); dx, dy, dz = map(float, f[6:9])
    assert f[9] == "SLOW_LEN", f
    buf = np.fromfile(str(prefix) + ".buf", dtype=np.float32).reshape(nx, ny, nz)
    vp = dx / buf                                           # SLOW_LEN = dx / v
    return vp, (x0, y0, z0), (dx, dy, dz)


def trilinear(vp, org, sp, gx, gy, gz):
    nx, ny, nz = vp.shape
    fx = np.clip((gx - org[0]) / sp[0], 0, nx - 1.0001); fy = np.clip((gy - org[1]) / sp[1], 0, ny - 1.0001)
    fz = np.clip((gz - org[2]) / sp[2], 0, nz - 1.0001)
    i, j, k = fx.astype(int), fy.astype(int), fz.astype(int)
    tx, ty, tz = fx - i, fy - j, fz - k
    v = 0
    for di, wx in ((0, 1 - tx), (1, tx)):
        for dj, wy in ((0, 1 - ty), (1, ty)):
            for dk, wz in ((0, 1 - tz), (1, tz)):
                v = v + wx * wy * wz * vp[i + di, j + dj, k + dk]
    return v


def hypodd_xy_to_latlon(x, y, driver: Path):
    inp = f"{ORIGIN_LAT} {ORIGIN_LON} 0.0\n1\n{len(x)}\n" + "\n".join(f"{a:.4f} {b:.4f}" for a, b in zip(x, y)) + "\n"
    out = subprocess.run([str(driver)], input=inp, capture_output=True, text=True, check=True).stdout
    arr = np.array([list(map(float, l.split())) for l in out.strip().split("\n")])
    return arr[:, 0], arr[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-prefix", default="ORCA_v4")
    ap.add_argument("--vpvs", type=float, default=1.78)
    ap.add_argument("--out", default=None)
    ap.add_argument("--driver", default=str(REPO / "hypodd" / "_synth_tables" / "proj_driver"))
    a = ap.parse_args()
    from importlib import import_module
    s41 = import_module("41_build_unsheared_velgrid")
    vp, org, sp = read_nlloc_grid(REPO / "nlloc" / "model" / f"{a.src_prefix}.P.mod")
    print(f"{a.src_prefix}: {vp.shape} nodes, origin {org}, spacing {sp}, Vp {vp.min():.2f}-{vp.max():.2f}")

    xn, yn, zn = XY_NODES, XY_NODES, Z_NODES
    X, Y = np.meshgrid(xn, yn, indexing="ij")
    lat, lon = hypodd_xy_to_latlon(X.ravel(), Y.ravel(), Path(a.driver))
    gx, gy = s41.stingray_xy(lat, lon)                    # NLLoc frame of the grid
    gx, gy = gx.reshape(X.shape), gy.reshape(X.shape)
    inside = (gx >= org[0]) & (gx <= org[0] + sp[0] * (vp.shape[0] - 1)) & \
             (gy >= org[1]) & (gy <= org[1] + sp[1] * (vp.shape[1] - 1))
    print(f"xy nodes: {len(xn)}x{len(yn)}; inside the NLLoc grid: {inside.sum()} of {inside.size} "
          f"(outside -> edge value); fine core {FINE[0]:+.0f}..{FINE[-1]:+.0f} km @ {FINE[1]-FINE[0]} km")
    V = np.empty((len(xn), len(yn), len(zn)), dtype=np.float32)
    for k, z in enumerate(zn):
        V[:, :, k] = trilinear(vp, org, sp, gx, gy, np.full_like(gx, max(z, 0.0)))
    # self-checks -------------------------------------------------------------
    # 1: node at the origin, z=2 km equals a direct trilinear read of the grid there
    ox, oy = s41.stingray_xy(np.array([ORIGIN_LAT]), np.array([ORIGIN_LON]))
    ref = trilinear(vp, org, sp, ox, oy, np.array([2.0]))[0]
    i0, j0, k0 = np.argmin(np.abs(xn)), np.argmin(np.abs(yn)), list(zn).index(2.0)
    print(f"  check 1  origin node z=2: model {V[i0, j0, k0]:.3f} vs grid {ref:.3f}  "
          f"({'OK' if abs(V[i0, j0, k0]-ref) < 1e-3 else 'FAIL'})")
    # 2: monotone-ish with depth on average, no NaN, plausible range
    prof = V[i0, j0, :]
    print(f"  check 2  origin profile z={list(zn[:12])} ->\n           Vp={np.round(prof[:12], 2).tolist()}")
    print(f"  check 3  NaN: {np.isnan(V).sum()}  range {V.min():.2f}-{V.max():.2f} km/s  "
          f"({'OK' if np.isnan(V).sum() == 0 and V.min() > 1.3 and V.max() < 8.5 else 'FAIL'})")
    # 3: lateral heterogeneity in the fine core at 1 km depth (should exist), far field ~1D
    core = V[np.abs(xn) <= 25][:, np.abs(yn) <= 25, list(zn).index(1.0)]
    far = V[np.abs(xn) >= 100][:, np.abs(yn) >= 100, list(zn).index(1.0)]
    print(f"  check 4  Vp at z=1 km: core std {core.std():.3f} km/s (range {core.min():.2f}-{core.max():.2f}); "
          f"far field std {far.std():.3f}")

    out = Path(a.out) if a.out else REPO / "nlloc" / "model" / f"{a.src_prefix}.hypodd3d.vel"
    L = [f" 0.1 {len(xn)} {len(yn)} {len(zn)}",
         " ".join(f"{v:.1f}" for v in xn), " ".join(f"{v:.1f}" for v in yn), " ".join(f"{v:.1f}" for v in zn),
         "  0  0  0", "  0  0  0"]
    for k in range(len(zn)):
        for j in range(len(yn)):
            L.append(" ".join(f"{V[i, j, k]:.3f}" for i in range(len(xn))))
    for k in range(len(zn)):
        for j in range(len(yn)):
            L.append(" ".join(f"{a.vpvs:.3f}" for _ in range(len(xn))))
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}  ({len(L)} lines; nx ny nz = {len(xn)} {len(yn)} {len(zn)}; vpvs {a.vpvs})")
    print(f"control lines: LAT3D LON3D ROT3D = {ORIGIN_LAT} {ORIGIN_LON} 0.0")


if __name__ == "__main__":
    main()
