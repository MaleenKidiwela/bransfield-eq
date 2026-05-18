"""Build an extended NLLoc VELOCITY grid covering all BRAVOSEIS stations.

Inputs:
    nlloc/srModel_3DEQ.mat              3D Orca Vp from Stingray (0.2 km, 60x40x25 km)
    configs/velocity_model.csv          1D Vp(depth) used by pyocto/GrowClust

Output:
    nlloc/model/ORCA_v2.P.mod.hdr       NLLoc VELOCITY grid header
    nlloc/model/ORCA_v2.P.mod.buf       float32 Vp(km/s), ix*ny*nz + iy*nz + iz

Grid layout (Stingray frame, x East-ish, y North-ish, z down):
    x:  -150 to +80 km     (576 nodes @ 0.4 km)   covers all 5M land stations
    y:  -110 to +70 km     (451 nodes @ 0.4 km)
    z:  0    to +25.2 km   ( 64 nodes @ 0.4 km)

Velocity assignment per node:
  - Inside the srModel extent: linear interpolation from srModel.P.u (1/u).
  - Outside, with smooth blend over a 10 km buffer ring: weight w(d) =
    max(0, 1 - d/buffer) where d is distance to srModel edge in km;
    Vp(node) = w * Vp_srModel_edge + (1-w) * Vp_1D(depth).
  - Outside the buffer ring: Vp_1D(depth) only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parent.parent
SR_MAT = REPO / "nlloc" / "srModel_3DEQ.mat"
VMOD1D = REPO / "configs" / "velocity_model.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ORCA_v2")
    ap.add_argument("--dx", type=float, default=0.4)
    ap.add_argument("--x-min", type=float, default=-150.0)
    ap.add_argument("--x-max", type=float, default=80.0)
    ap.add_argument("--y-min", type=float, default=-110.0)
    ap.add_argument("--y-max", type=float, default=70.0)
    ap.add_argument("--z-max", type=float, default=25.2)
    ap.add_argument("--buffer-km", type=float, default=10.0,
                    help="Blend distance from srModel edge to 1D background")
    args = ap.parse_args()

    # --- load srModel ---
    sr = sio.loadmat(SR_MAT, squeeze_me=True, struct_as_record=False)["srModel"]
    u = np.asarray(sr.P.u)                  # (nxs, nys, nzs) slowness s/km
    Vp_sr = 1.0 / u
    xg = np.asarray(sr.xg, dtype=float)     # 0.2-km nodes
    yg = np.asarray(sr.yg, dtype=float)
    zg = np.asarray(sr.zg, dtype=float)     # 0 -> -25 (negative depth)
    # NLLoc uses positive-down z. Build a srModel sampler in (x, y, depth) space.
    depth_sr = -zg                          # 0 -> +25 (down positive)
    order_z = np.argsort(depth_sr)
    depth_sr_sorted = depth_sr[order_z]
    Vp_sr_zsort = Vp_sr[:, :, order_z]
    sr_interp = RegularGridInterpolator(
        (xg, yg, depth_sr_sorted), Vp_sr_zsort,
        bounds_error=False, fill_value=np.nan,
    )
    print(f"srModel: x {xg.min():.1f}..{xg.max():.1f}  "
          f"y {yg.min():.1f}..{yg.max():.1f}  "
          f"depth {depth_sr_sorted.min():.1f}..{depth_sr_sorted.max():.1f}  km")

    # --- 1D background Vp(depth-below-seafloor) ---
    # The shipped velocity_model.csv is sea-level-referenced with a 1.3 km
    # water layer at the top (rock starts at z=1.301). srModel uses
    # depth-BELOW-SEAFLOOR, so we strip the water layer and shift depths
    # so 0 = top of rock = seafloor.
    v1d_full = pd.read_csv(VMOD1D)
    SEAFLOOR_SHIFT_KM = 1.3
    v1d = v1d_full[v1d_full.depth_km >= SEAFLOOR_SHIFT_KM].copy()
    v1d["depth_km"] = v1d["depth_km"] - SEAFLOOR_SHIFT_KM
    print(f"1D rock-only model (shifted to depth-below-seafloor): "
          f"depth {v1d.depth_km.min():.2f}..{v1d.depth_km.max():.1f} km, "
          f"Vp {v1d.vp_kms.min():.2f}..{v1d.vp_kms.max():.2f} km/s")

    def vp_1d(depth: np.ndarray) -> np.ndarray:
        return np.interp(depth, v1d.depth_km.values, v1d.vp_kms.values)

    # --- new extended grid ---
    nx = int(round((args.x_max - args.x_min) / args.dx)) + 1
    ny = int(round((args.y_max - args.y_min) / args.dx)) + 1
    nz = int(round(args.z_max / args.dx)) + 1
    xs = args.x_min + np.arange(nx) * args.dx
    ys = args.y_min + np.arange(ny) * args.dx
    zs = np.arange(nz) * args.dx
    print(f"new grid: nx={nx} ny={ny} nz={nz}  ({nx*ny*nz/1e6:.1f} M nodes, "
          f"{nx*ny*nz*4/1e6:.0f} MB per grid)")

    sr_xmin, sr_xmax = xg.min(), xg.max()
    sr_ymin, sr_ymax = yg.min(), yg.max()
    sr_zmax = depth_sr_sorted.max()

    Vp_new = np.empty((nx, ny, nz), dtype=np.float32)
    for iz, z in enumerate(zs):
        v1d_z = float(vp_1d(z))
        # 3D srModel sample on the (xs, ys) plane at this depth
        # If depth outside srModel range, sample falls back to 1D.
        if z > sr_zmax:
            Vp_new[:, :, iz] = v1d_z
            continue
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        sample_pts = np.stack(
            (XX.ravel(), YY.ravel(),
             np.full(XX.size, z, dtype=float)), axis=-1
        )
        vp_sr_layer = sr_interp(sample_pts).reshape(nx, ny)
        # Distance to srModel edge (positive outside, zero inside)
        dx_edge = np.maximum(0.0, np.maximum(sr_xmin - XX, XX - sr_xmax))
        dy_edge = np.maximum(0.0, np.maximum(sr_ymin - YY, YY - sr_ymax))
        d_edge = np.sqrt(dx_edge ** 2 + dy_edge ** 2)
        # Use nearest srModel-edge value where outside srModel laterally
        outside = np.isnan(vp_sr_layer)
        if outside.any():
            x_clip = np.clip(XX, sr_xmin, sr_xmax)
            y_clip = np.clip(YY, sr_ymin, sr_ymax)
            edge_pts = np.stack((x_clip.ravel(), y_clip.ravel(),
                                 np.full(XX.size, z, dtype=float)), axis=-1)
            vp_edge = sr_interp(edge_pts).reshape(nx, ny)
            vp_sr_layer = np.where(outside, vp_edge, vp_sr_layer)
        # Blend weight: 1 inside srModel, decays to 0 over buffer-km outside
        w = np.where(d_edge <= 0, 1.0,
                     np.clip(1.0 - d_edge / args.buffer_km, 0.0, 1.0))
        Vp_new[:, :, iz] = (w * vp_sr_layer + (1 - w) * v1d_z).astype(np.float32)

    print(f"Vp range: {Vp_new.min():.2f} .. {Vp_new.max():.2f} km/s")

    # --- write .hdr + .buf ---
    out_dir = REPO / "nlloc" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = out_dir / f"{args.prefix}.P.mod.hdr"
    buf = out_dir / f"{args.prefix}.P.mod.buf"
    # Grid2Time's Podvin-Lecomte solver needs SLOW_LEN grids (slowness * dx).
    slow_len = (args.dx / Vp_new).astype(np.float32)
    with hdr.open("w") as fh:
        fh.write(f"{nx} {ny} {nz}  "
                 f"{args.x_min:.3f} {args.y_min:.3f} 0.000  "
                 f"{args.dx:.3f} {args.dx:.3f} {args.dx:.3f}  "
                 f"SLOW_LEN FLOAT\n")
    slow_len.tofile(buf)
    print(f"wrote {hdr}")
    print(f"wrote {buf} ({buf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
