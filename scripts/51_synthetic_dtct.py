"""Synthetic dt.ct for the hypoDD self-consistency test.

Why
---
On the v4 standard tier hypoDD compresses the depth axis (slope -0.53 of
[DD depth - start depth] on start depth) and the compression did NOT respond to a
30x change in LSQR conditioning. Two readings remain: the differential times
genuinely want NLLoc's deep events shallower (NLLoc's depth axis is stretched), or
the inversion compresses depth on its own under realistic noise. Only a synthetic
test separates them: forward-model dt.ct from the NLLoc starting locations through
hypoDD's OWN 1D flat-station model, run the identical control, and see whether the
starting depth spread survives.

Forward model: obspy TauPy on hypoDD's 22-layer model (from script 24's
write_velocity with the same datum shift), receivers at the flat datum, first
arrival = min over direct/refracted. Land stations (elev > 0) get a vertical
elevation correction. Travel times are precomputed on a (distance, depth) grid and
interpolated -- 4.16 M lookups in seconds.

Runs (Merlin's protocol): (0) noise-free  -> must reproduce the starting depth
spread (slope within +-0.05, p90 within 0.2 km) or the chain is broken;
(1) Gaussian 0.15 s per pick; (2) as (1) plus 5% outliers at +-0.5 s. The noisy
runs give the inversion's own compression baseline; real-data slope minus that
baseline is genuine data-model inconsistency.
"""
from __future__ import annotations

import argparse, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
R_EARTH_KM = 6371.0


def hypodd_layers(datum_shift_km: float):
    spec = importlib.util.spec_from_file_location("s24", REPO / "scripts" / "24_run_hypodd.py")
    s24 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s24)
    n, tops, vp, vs = s24.write_velocity(REPO / "configs" / "velocity_model.csv", Path("/tmp"), datum_shift_km)
    return np.array(tops), np.array(vp), np.array(vs)


def build_taup_model(tops, vp, vs, work: Path) -> str:
    """Write an .nd model and compile it for TauPy. hypoDD's bottom layer is a
    halfspace; extend it to 400 km before joining a generic deep Earth so no
    mantle refraction can become a first arrival inside MAXDIST."""
    from obspy.taup import taup_create
    rho = lambda v: 0.32 * v + 0.77                       # Gardner-ish, irrelevant to times
    lines = []
    for i, (t, p, s) in enumerate(zip(tops, vp, vs)):
        if i > 0:                                           # close previous layer at this top
            lines.append(f"{t:.4f} {vp[i-1]:.4f} {vs[i-1]:.4f} {rho(vp[i-1]):.3f}")
        lines.append(f"{t:.4f} {p:.4f} {s:.4f} {rho(p):.3f}")
    lines.append(f"400.0000 {vp[-1]:.4f} {vs[-1]:.4f} {rho(vp[-1]):.3f}")
    lines.append("mantle")
    lines += ["400.0000 8.9000 4.8000 3.550", "2889.0000 13.7000 7.2500 5.550", "outer-core",
              "2889.0000 8.0000 0.0000 9.900", "5153.9000 10.3000 0.0000 12.170", "inner-core",
              "5153.9000 11.0000 3.5000 12.760", "6371.0000 11.2600 3.6700 13.090"]
    nd = work / "hypodd_flat.nd"; nd.write_text("\n".join(lines) + "\n")
    taup_create.build_taup_model(str(nd), output_folder=str(work))
    return str(work / "hypodd_flat.npz")


def build_table(model_path: str, dists_km, depths_km, workers: int):
    """First-arrival P and S travel times on a (dist, depth) grid via TauPy, parallel."""
    from concurrent.futures import ProcessPoolExecutor
    deg = dists_km / (2 * np.pi * R_EARTH_KM / 360.0)
    chunks = np.array_split(np.arange(len(depths_km)), workers)
    args = [(model_path, deg, depths_km[c]) for c in chunks if len(c)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_table_rows, args))
    P = np.vstack([p for p, _ in parts]); S = np.vstack([s for _, s in parts])
    return P, S                                          # shape (n_depth, n_dist)


def _table_rows(a):
    model_path, deg, depths = a
    from obspy.taup import TauPyModel
    m = TauPyModel(model=model_path)
    P = np.full((len(depths), len(deg)), np.nan); S = np.full_like(P, np.nan)
    for i, z in enumerate(depths):
        for j, d in enumerate(deg):
            for arr_list, out in ((m.get_travel_times(max(z, 0.001), d, phase_list=["p", "P"]), P),
                                  (m.get_travel_times(max(z, 0.001), d, phase_list=["s", "S"]), S)):
                if arr_list:
                    out[i, j] = min(x.time for x in arr_list)
    return P, S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-label", default="year_v4_1d", help="run dir with event.sel, dt.ct, station.dat")
    ap.add_argument("--out-label", required=True)
    ap.add_argument("--noise-s", type=float, default=0.0, help="Gaussian sigma per PICK, seconds")
    ap.add_argument("--outlier-frac", type=float, default=0.0)
    ap.add_argument("--outlier-s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--table-only", action="store_true", help="build+verify the travel-time table and stop")
    args = ap.parse_args()

    src = REPO / "hypodd" / args.src_label
    shift = json.loads((src / "hypodd_datum.json").read_text())["datum_shift_km"]
    tops, vp, vs = hypodd_layers(shift)
    print(f"hypoDD model: {len(tops)} layers, Vp {vp[0]:.2f}..{vp[-1]:.2f}, datum shift {shift} km")

    work = REPO / "hypodd" / "_synth_tables"; work.mkdir(exist_ok=True)
    cache = work / "tt_table.npz"
    dists = np.arange(0.0, 130.0001, 1.0); depths = np.arange(0.0, 30.0001, 0.25)   # 131x121; the 0.25x0.1 grid (313k TauPy calls) blew a 40-min timeout
    if cache.exists():
        z = np.load(cache); P, S = z["P"], z["S"]; print(f"loaded cached table {P.shape}")
    else:
        mp = build_taup_model(tops, vp, vs, work)
        print(f"TauPy model built; computing {len(depths)}x{len(dists)} table x2 phases on {args.workers} workers ...")
        P, S = build_table(mp, dists, depths, args.workers)
        np.savez(cache, P=P, S=S, dists=dists, depths=depths)
        print("table cached")

    # ---- verification 1: vertical travel time == integral of slowness ----
    def vertical_tt(z, v):
        edges = np.append(tops, np.inf); t = 0.0
        for k in range(len(tops)):
            lo, hi = edges[k], min(edges[k+1], z)
            if hi > lo: t += (hi - lo) / v[k]
            if edges[k+1] >= z: break
        return t
    errs = []
    for z in (2.0, 5.0, 10.0, 20.0):
        i = int(round(z / 0.25)); errs.append(abs(P[i, 0] - vertical_tt(z, vp)))
    print(f"  check 1  vertical P (dist 0) vs slowness integral: max |diff| {max(errs)*1000:.1f} ms  "
          f"({'OK' if max(errs) < 0.02 else 'FAIL'})")
    # ---- verification 2: table finite where it should be ----
    bad = np.isnan(P).sum() + np.isnan(S).sum()
    print(f"  check 2  NaN cells: {bad} ({'OK' if bad == 0 else 'FAIL - inspect'})")
    # ---- verification 3: monotone in distance at fixed depth (first arrivals) ----
    mono = np.all(np.diff(P[20], axis=0) >= -1e-6) and np.all(np.diff(S[20], axis=0) >= -1e-6)
    print(f"  check 3  monotone with distance at 5 km depth: {'OK' if mono else 'FAIL'}")
    if args.table_only:
        return

    from scipy.interpolate import RegularGridInterpolator
    fP = RegularGridInterpolator((depths, dists), P, bounds_error=False, fill_value=None)
    fS = RegularGridInterpolator((depths, dists), S, bounds_error=False, fill_value=None)

    # ---- events (flat-frame depths) and stations ----
    ev = pd.read_csv(src / "event.sel", sep=r"\s+", header=None,
                     names=["date", "time", "lat", "lon", "dep", "mag", "eh", "ez", "rms", "id"])
    ev = ev.set_index("id")
    st = pd.read_csv(src / "station.dat", sep=r"\s+", header=None, names=["sta", "lat", "lon", "elev_m"]).set_index("sta")

    def dist_km(lat1, lon1, lat2, lon2):
        p1, p2 = np.radians(lat1), np.radians(lat2); dl = np.radians(lon2 - lon1)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * R_EARTH_KM * np.arcsin(np.sqrt(a))

    # ---- rewrite dt.ct ----
    rng = np.random.default_rng(args.seed)
    out_dir = REPO / "hypodd" / args.out_label; out_dir.mkdir(exist_ok=True)
    for f in ("phase.dat", "station.dat", "event.sel", "hypodd_datum.json"):
        (out_dir / f).write_bytes((src / f).read_bytes())
    n_rows = n_pairs = n_skip = 0
    with (src / "dt.ct").open() as fin, (out_dir / "dt.ct").open("w") as fout:
        id1 = id2 = None
        for line in fin:
            if line.startswith("#"):
                _, a, b = line.split(); id1, id2 = int(a), int(b); n_pairs += 1
                fout.write(line); continue
            sta, _, _, w, ph = line.split()
            if sta not in st.index or id1 not in ev.index or id2 not in ev.index:
                n_skip += 1; continue
            e1, e2, s = ev.loc[id1], ev.loc[id2], st.loc[sta]
            f = fP if ph == "P" else fS
            vtop = vp[0] if ph == "P" else vs[0]
            tt = []
            for e in (e1, e2):
                d = dist_km(e.lat, e.lon, s.lat, s.lon)
                t = float(f([[max(e.dep, 0.0), min(d, 130.0)]])[0])
                t += max(s.elev_m, 0.0) / 1000.0 / vtop        # land station above the datum
                if args.noise_s: t += rng.normal(0.0, args.noise_s)
                if args.outlier_frac and rng.random() < args.outlier_frac:
                    t += rng.choice([-1.0, 1.0]) * args.outlier_s
                tt.append(t)
            fout.write(f"{sta:<6s}{tt[0]:9.3f}{tt[1]:8.3f} {float(w):.4f} {ph}\n"); n_rows += 1
    print(f"wrote {out_dir/'dt.ct'}: {n_pairs:,} pairs, {n_rows:,} rows, {n_skip} skipped   "
          f"noise {args.noise_s}s  outliers {args.outlier_frac*100:.0f}% @ +-{args.outlier_s}s")
    (out_dir / "synthetic.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
