"""Perturb an NLLoc velocity grid in the top N km of ROCK below the LOCAL seafloor.

Why
---
NLLoc v4's depth axis is stretched ~1.5-1.8x (observed nearest-station S-P is
nearly flat across catalogue depths 2-8 km). Vp/Vs is not the cause (I13). The
remaining velocity hypothesis is that the shallow P model is too fast: refraction
rays sample the top few hundred metres poorly and the tomography's top node is the
starting-model fill. This scales Vp by `--factor` for nodes within `--top-km` of
the local seafloor (from the water-depth surface script 41 saved), leaves
everything else untouched, and writes a new prefix. Rebuild travel times with
script 39 and relocate the sample; score with script 52 at a fixed Vp/Vs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-prefix", default="ORCA_v4")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--factor", type=float, default=0.90, help="multiply Vp by this in the perturbed zone")
    ap.add_argument("--top-km", type=float, default=2.0, help="thickness of rock below the local seafloor to perturb")
    args = ap.parse_args()
    mdir = REPO / "nlloc" / "model"
    hdr = (mdir / f"{args.src_prefix}.P.mod.hdr").read_text().split()
    nx, ny, nz = int(hdr[0]), int(hdr[1]), int(hdr[2]); z0 = float(hdr[5]); dx = float(hdr[6])
    buf = np.fromfile(mdir / f"{args.src_prefix}.P.mod.buf", dtype=np.float32).reshape(nx, ny, nz)
    V = dx / buf
    w = np.load(mdir / f"{args.src_prefix}.water_depth_km.npy")           # (nx, ny) local water depth BSL
    z = z0 + np.arange(nz) * dx                                             # BSL
    depth_below_sf = z[None, None, :] - w[:, :, None]
    zone = (depth_below_sf > 0.0) & (depth_below_sf <= args.top_km)
    V2 = V.copy(); V2[zone] *= args.factor
    print(f"{args.src_prefix} -> {args.out_prefix}: Vp x{args.factor} where 0 < depth below local seafloor <= {args.top_km} km")
    print(f"  nodes perturbed: {zone.sum():,} of {zone.size:,} ({zone.mean()*100:.1f}%)")
    print(f"  Vp in zone: {V[zone].min():.2f}-{V[zone].max():.2f} -> {V2[zone].min():.2f}-{V2[zone].max():.2f} km/s")
    print(f"  unchanged elsewhere: {np.array_equal(V[~zone], V2[~zone])}")
    (mdir / f"{args.out_prefix}.P.mod.hdr").write_text(" ".join(hdr) + "\n")
    (dx / V2).astype(np.float32).tofile(mdir / f"{args.out_prefix}.P.mod.buf")
    np.save(mdir / f"{args.out_prefix}.water_depth_km.npy", w)
    print(f"wrote {args.out_prefix}.P.mod.{{hdr,buf}} + water-depth surface")

if __name__ == "__main__":
    main()
