"""Model-light depth check: observed near-station S-P vs the S-P predicted at a
catalogue's own depth.

The single most depth-sensitive observable a shallow OBS array has is S-P at the
nearest station. It is read straight from the picks -- no location, no inversion.
A catalogue whose depths are right predicts it; one whose depths are stretched or
compressed does not. Straight-ray hypocentral distance with a path-average
velocity from the rock-only 1D model: crude, but IDENTICAL for every catalogue
scored, so the comparison is fair even if the absolute misfit is not.

Two frames:
  sealevel  depth_km is below sea level; station is at its water depth
  seafloor  depth_km is below a flat datum; station is at 0  (hypoDD runs)
"""
from __future__ import annotations

import argparse, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
R_EARTH = 6371.0


def rock_model(vpvs: float | None = None):
    """Rock-only 1D model. If `vpvs` is given, Vs = Vp / vpvs so every catalogue is
    scored with the SAME ratio (positions-only comparison); otherwise the file's Vs."""
    vm = pd.read_csv(REPO / "configs" / "velocity_model.csv")
    r = vm[vm.vp_kms > 1.6].copy(); r["z_bsl"] = r.depth_km       # rock rows, below sea level
    vs = (r.vp_kms / vpvs).values if vpvs else r.vs_kms.values
    return r.z_bsl.values, r.vp_kms.values, vs


def path_avg_slowness(z_top, z_bot, z_grid, v):
    """1/v averaged over the vertical extent [z_top, z_bot] of the 1D model (BSL)."""
    lo, hi = np.minimum(z_top, z_bot), np.maximum(z_top, z_bot)
    zz = np.linspace(0, 1, 32)
    out = np.empty_like(lo, dtype=float)
    for k in range(len(lo)):
        zs = lo[k] + (hi[k] - lo[k]) * zz
        out[k] = np.mean(1.0 / np.interp(zs, z_grid, v))
    return out


def load_hypdir(hyp_dir: Path, obs_order_csv: Path, n_first: int):
    """Sample runs: hyp files in first-arrival-time order = obs order (1 shard, in-order)."""
    GEO = re.compile(r'GEOGRAPHIC\s+OT\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(-?[\d.]+)\s+Lat\s+(-?[\d.]+)\s+Long\s+(-?[\d.]+)\s+Depth\s+(-?[\d.]+)')
    files = sorted(glob.glob(str(hyp_dir / "loc.2*.grid0.loc.hyp")))
    rows = []
    for f in files:
        g = GEO.search(open(f).read())
        if g: rows.append((float(g.group(2)), float(g.group(3)), float(g.group(4))))
    order = pd.read_csv(obs_order_csv).head(n_first)
    if len(rows) != len(order):
        raise SystemExit(f"{len(rows)} hyps vs {len(order)} obs -- cannot map positionally")
    d = pd.DataFrame(rows, columns=["lat", "lon", "depth_km"]); d["event_idx"] = order.event_idx.values
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", help="CSV with event_idx (or id=event_idx+1), lat, lon, depth_km")
    ap.add_argument("--hyp-dir", help="sample run output dir (positional mapping via --obs-order)")
    ap.add_argument("--obs-order", default="nlloc/obs/year_v4.event_order.csv")
    ap.add_argument("--n-first", type=int, default=2000)
    ap.add_argument("--frame", required=True, choices=["sealevel", "seafloor"])
    ap.add_argument("--datum-shift-km", type=float, default=1.0)
    ap.add_argument("--depth-col", default="depth_km")
    ap.add_argument("--max-sta-km", type=float, default=4.0)
    ap.add_argument("--restrict-to", help="CSV of event_idx to restrict to (e.g. NLLoc sigma_z<=0.5)")
    ap.add_argument("--label", default="")
    ap.add_argument("--delays", default=None,
                    help="LOCDELAY file: subtract (S delay - P delay) per station from the OBSERVED S-P, "
                         "so a catalogue located with station delays is scored consistently")
    ap.add_argument("--scorer-vpvs", type=float, default=None,
                    help="fix the scorer Vp/Vs (e.g. Wadati 1.88) so runs differ only by positions")
    args = ap.parse_args()

    if args.catalog:
        c = pd.read_csv(args.catalog)
        if "event_idx" not in c.columns and "id" in c.columns: c["event_idx"] = c["id"] - 1
        c = c.rename(columns={args.depth_col: "depth_km"}) if args.depth_col != "depth_km" else c
    else:
        c = load_hypdir(REPO / args.hyp_dir, REPO / args.obs_order, args.n_first)
    if args.restrict_to:
        keep = set(pd.read_csv(args.restrict_to).event_idx); c = c[c.event_idx.isin(keep)]
    # depth below sea level for the geometry, whichever frame the catalogue is in
    c["z_bsl"] = c.depth_km + (args.datum_shift_km if args.frame == "seafloor" else 0.0)

    pk = pd.read_csv(REPO / "catalogs" / "pyocto_picks_year_newpool_no_shots.csv")
    pk = pk[pk.event_idx.isin(set(c.event_idx))]
    P = pk[pk.phase == "P"][["event_idx", "station", "time"]].rename(columns={"time": "tp"})
    S = pk[pk.phase == "S"][["event_idx", "station", "time"]].rename(columns={"time": "ts"})
    sp = P.merge(S, on=["event_idx", "station"]); sp["sp_obs"] = sp.ts - sp.tp
    if args.delays:
        dl = {}
        for l in Path(args.delays).read_text().split("\n"):
            f = l.split()
            if len(f) == 5 and f[0] == "LOCDELAY": dl[(f[1], f[2])] = float(f[4])
        corr = sp.station.map(lambda st: dl.get((str(st).split(".")[-1], "S"), 0.0) - dl.get((str(st).split(".")[-1], "P"), 0.0))
        sp["sp_obs"] = sp.sp_obs - corr
        print(f"  observed S-P corrected by station (S-P) delays from {args.delays}: median correction {corr.median():+.3f} s")
    sp = sp[(sp.sp_obs > 0) & (sp.sp_obs < 10)]
    st = pd.read_csv(REPO / "catalogs" / "station_geometry.csv"); st["station"] = st.network + "." + st.station
    st["z_sta_bsl"] = np.where(st.on_seafloor, st.water_depth_m / 1000.0, -st.elevation_m / 1000.0)
    m = sp.merge(c[["event_idx", "lat", "lon", "z_bsl"]], on="event_idx").merge(
        st[["station", "latitude", "longitude", "z_sta_bsl"]], on="station")
    p1, p2 = np.radians(m.lat), np.radians(m.latitude); dl = np.radians(m.longitude - m.lon)
    m["epi_km"] = 2 * R_EARTH * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))
    m = m[m.epi_km <= args.max_sta_km]
    m = m.sort_values("epi_km").groupby("event_idx").head(1)          # nearest station per event
    zg, vp, vs = rock_model(args.scorer_vpvs)
    z_sta = m.z_sta_bsl.values.clip(min=zg[0]); z_src = m.z_bsl.values.clip(min=zg[0])
    dz = m.z_bsl.values - m.z_sta_bsl.values
    R = np.sqrt(m.epi_km.values ** 2 + dz ** 2)
    m["sp_pred"] = R * (path_avg_slowness(z_sta, z_src, zg, vs) - path_avg_slowness(z_sta, z_src, zg, vp))
    m["misfit"] = m.sp_pred - m.sp_obs
    bins = [0, 1, 2, 3, 4, 5, 6, 8, 12, 40]
    m["zbin"] = pd.cut(m.z_bsl, bins)
    g = m.groupby("zbin", observed=True).agg(n=("misfit", "size"), sp_obs=("sp_obs", "median"),
                                              sp_pred=("sp_pred", "median"), misfit=("misfit", "median"))
    print(f"\n=== {args.label or (args.catalog or args.hyp_dir)}  [{args.frame}; scorer Vp/Vs {args.scorer_vpvs or 'file'}]  n={len(m):,} events with a P+S station <{args.max_sta_km} km ===")
    print(f"{'depth BSL bin':>15}{'n':>7}{'obs S-P':>10}{'pred S-P':>10}{'pred-obs':>10}")
    for b, r in g.iterrows():
        print(f"{str(b):>15}{r.n:>7.0f}{r.sp_obs:>10.3f}{r.sp_pred:>10.3f}{r.misfit:>+10.3f}")
    print(f"{'ALL':>15}{len(m):>7,}{m.sp_obs.median():>10.3f}{m.sp_pred.median():>10.3f}{m.misfit.median():>+10.3f}   median |misfit| {m.misfit.abs().median():.3f} s")
    lo, hi = g.iloc[0], g.iloc[-1]
    print(f"  spread ratio (deepest bin / shallowest bin): observed x{hi.sp_obs/lo.sp_obs:.2f}   predicted x{hi.sp_pred/lo.sp_pred:.2f}"
          f"   -> {'depth axis STRETCHED' if hi.sp_pred/lo.sp_pred > 1.15*hi.sp_obs/lo.sp_obs else 'depth axis COMPRESSED' if hi.sp_pred/lo.sp_pred < 0.85*hi.sp_obs/lo.sp_obs else 'consistent'}")


if __name__ == "__main__":
    main()
