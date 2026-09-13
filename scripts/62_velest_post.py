"""Parse a VELEST run directory: relocated hypocentres, station corrections,
final 1D model, and the rms trajectory.

Writes velest/<run>/relocated.csv in the same column convention as the NLLoc
catalogues (event_idx, origin_time, lat, lon, depth_km below SEA LEVEL) so it
can be fed straight to 52_sp_depth_check.py --frame sealevel.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# (1x,i3,1x,3i2.2,1x,2i2.2,f6.2,1x,f7.4,a1,1x,f8.4,a1,1x,f6.2,f5.2,i4,f6.3,1x,3f7.2)
COLS = dict(idx=(1, 4), yy=(5, 7), mo=(7, 9), dd=(9, 11), hh=(12, 14), mi=(14, 16),
            sec=(16, 22), lat=(23, 30), ns=(30, 31), lon=(32, 40), ew=(40, 41),
            dep=(42, 48), mag=(48, 53), nobs=(53, 57), rms=(57, 63),
            x=(64, 71), y=(71, 78), z=(78, 85))


def read_out(run: Path) -> str:
    return (run / "velest.OUT").read_text(errors="replace")


def rms_trajectory(txt: str) -> pd.DataFrame:
    rows = []
    for m in re.finditer(r"(\(?Iteration nr\s+(\d+)\)?(?:\s+BACKUP nr\s+(\d+))?) obtained:\s*\n"
                         r"\s*DATVAR=\s*([\d.E+-]+)\s+mean sqrd residual=\s*([\d.E+-]+)"
                         r"\s+RMS RESIDUAL=\s*([\d.E+-]+)", txt):
        rows.append(dict(iteration=int(m.group(2)), backup=int(m.group(3) or 0),
                         datvar=float(m.group(4)), msr=float(m.group(5)),
                         rms=float(m.group(6))))
    return pd.DataFrame(rows)


def final_hypocenters(txt: str) -> pd.DataFrame:
    i = txt.rfind("output final hypocenters")
    if i < 0:
        raise SystemExit("no final hypocentre block in velest.OUT")
    block = txt[i:].splitlines()
    rows = []
    for ln in block:
        if len(ln) < 85:
            continue
        g = {k: ln[a:b] for k, (a, b) in COLS.items()}
        # The listing index is i3, so events 1000+ print as "***" -- never key
        # off it; the row order IS the input order.
        try:
            idx = int(g["idx"])
        except ValueError:
            idx = -1
        try:
            lat = float(g["lat"]) * (-1 if g["ns"].strip() == "S" else 1)
            lon = float(g["lon"]) * (-1 if g["ew"].strip() == "W" else 1)
            rows.append(dict(vel_idx=idx,
                             yy=int(g["yy"]), mo=int(g["mo"]), dd=int(g["dd"]),
                             hh=int(g["hh"]), mi=int(g["mi"]), sec=float(g["sec"]),
                             lat=lat, lon=lon, depth_km=float(g["dep"]),
                             n_obs=int(g["nobs"]), rms_s=float(g["rms"]),
                             x_km=float(g["x"]), y_km=float(g["y"])))
        except ValueError:
            continue
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("final hypocentre block parsed to zero rows")
    base = pd.to_datetime(
        dict(year=2000 + d.yy, month=d.mo, day=d.dd, hour=d.hh, minute=d.mi), utc=True)
    d["origin_time"] = base + pd.to_timedelta(d.sec, unit="s")
    return d.drop(columns=["yy", "mo", "dd", "hh", "mi", "sec"])


def final_model(run: Path, vpvs_fixed: float | None = None) -> pd.DataFrame:
    txt = (run / "velout.mod").read_text().splitlines()
    blocks, cur = [], None
    for ln in txt:
        s = ln.strip()
        if not s or s.lower().startswith("output"):
            continue
        if re.fullmatch(r"\d+", s):
            cur = []
            blocks.append(cur)
            continue
        if cur is not None and len(ln) >= 26:
            cur.append((float(ln[0:5]), float(ln[10:17]), float(ln[19:26])))
    p = pd.DataFrame(blocks[0], columns=["vp_kms", "top_km_bsl", "vdamp"])
    if len(blocks) == 1:          # nsp=3: S is tied to P by a fixed Vp/Vs
        m = p[["top_km_bsl", "vp_kms", "vdamp"]].copy()
        m["vs_kms"] = m.vp_kms / (vpvs_fixed or 1.78)
        m["vpvs"] = m.vp_kms / m.vs_kms
        return m
    s = pd.DataFrame(blocks[1], columns=["vs_kms", "top_km_bsl", "vdamp"])
    m = p[["top_km_bsl", "vp_kms", "vdamp"]].join(s[["vs_kms"]])
    m["vpvs"] = m.vp_kms / m.vs_kms
    return m


def station_corrections(run: Path) -> pd.DataFrame:
    """From final.STA, written with the format string on its own first line."""
    lines = (run / "final.STA").read_text().splitlines()[1:]
    rows = []
    for ln in lines:
        if len(ln) < 47 or not ln[:4].strip():
            continue
        rows.append(dict(alias=ln[0:4].strip(),
                         elev_m=int(ln[23:28]), icc=int(ln[31:34]),
                         ptcor_s=float(ln[35:40]), stcor_s=float(ln[42:47])))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", default="velest/input")
    ap.add_argument("--order", default="", help="event_order file; default: inferred from the run's .cnv")
    ap.add_argument("--start-catalog", default="catalogs/nlloc_year_v5_standard.csv")
    args = ap.parse_args()

    run = REPO / "velest" / args.run
    inp = REPO / args.input
    txt = read_out(run)

    traj = rms_trajectory(txt)
    traj.to_csv(run / "rms_trajectory.csv", index=False)
    acc = traj[traj.backup == 0]
    print(f"\n=== {args.run} ===")
    print("  rms trajectory (accepted iterations):")
    for _, r in traj.iterrows():
        tag = f" BACKUP {r.backup:.0f}" if r.backup else ""
        print(f"    it {r.iteration:.0f}{tag:<9s}  rms {r.rms:.4f} s   datvar {r.datvar:.6f}")
    print(f"  rms before {traj.rms.iloc[0]:.4f} s  ->  after {traj.rms.iloc[-1]:.4f} s")

    hyp = final_hypocenters(txt)
    ordf = args.order or ("event_order_sub.csv"
                          if (run / "orca.cnv").read_text().count("9999")
                          and len(pd.read_csv(inp / "event_order_sub.csv")) == len(hyp)
                          else "event_order.csv")
    order = pd.read_csv(inp / ordf)
    if len(hyp) != len(order):
        raise SystemExit(f"{len(hyp)} final hypocentres vs {len(order)} input events")
    hyp["event_idx"] = order.event_idx.values
    hyp["depth_datum"] = "sealevel"

    start = pd.read_csv(REPO / args.start_catalog)
    start = start[start.event_idx.isin(set(hyp.event_idx))][
        ["event_idx", "lat", "lon", "depth_km", "rms_s"]].rename(
        columns={"lat": "lat0", "lon": "lon0", "depth_km": "depth0_km", "rms_s": "rms0_s"})
    out = hyp.merge(start, on="event_idx", how="left")
    out["dz_km"] = out.depth_km - out.depth0_km
    cols = ["event_idx", "origin_time", "lat", "lon", "depth_km", "rms_s", "n_obs",
            "depth_datum", "x_km", "y_km", "lat0", "lon0", "depth0_km", "dz_km"]
    out[cols].to_csv(run / "relocated.csv", index=False)

    q = [10, 50, 90]
    print(f"  events relocated: {len(out):,}")
    print(f"  depth km BSL  start p10/50/90 "
          f"{'/'.join(f'{v:.2f}' for v in np.percentile(out.depth0_km, q))}"
          f"   ->  VELEST {'/'.join(f'{v:.2f}' for v in np.percentile(out.depth_km, q))}")
    print(f"  per-event rms  NLLoc p50 {out.rms0_s.median():.3f} s"
          f"   ->  VELEST p50 {out.rms_s.median():.3f} s")
    print(f"  depth change dz p10/50/90 "
          f"{'/'.join(f'{v:+.2f}' for v in np.percentile(out.dz_km, q))} km"
          f"   median |dz| {out.dz_km.abs().median():.2f} km")

    import json
    cfgp = run / "run_config.json"
    vpvs_fixed = json.loads((REPO / args.input / "meta.json").read_text())["vpvs"]
    mod = final_model(run, vpvs_fixed)
    mod.to_csv(run / "final_model.csv", index=False)
    print("\n  final 1D model:")
    print(f"    {'top km BSL':>11}{'Vp':>8}{'Vs':>8}{'Vp/Vs':>8}{'vdamp':>8}")
    for _, r in mod.iterrows():
        print(f"    {r.top_km_bsl:>11.2f}{r.vp_kms:>8.2f}{r.vs_kms:>8.2f}"
              f"{r.vpvs:>8.3f}{r.vdamp:>8.0f}")

    st = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    st["key"] = st.network + "." + st.station
    st["alias"] = ["B" + s[3:] if s.startswith("BRA") else s[:4] for s in st.station]
    st["water_km"] = np.where(st.on_seafloor, st.water_depth_m / 1000.0,
                              -st.elevation_m / 1000.0)
    sc = station_corrections(run).merge(
        st[["alias", "key", "water_km", "on_seafloor"]], on="alias", how="left")
    sc.to_csv(run / "station_corrections.csv", index=False)
    obs = sc[sc.on_seafloor]
    print(f"\n  station corrections: P {sc.ptcor_s.min():+.3f} .. {sc.ptcor_s.max():+.3f} s"
          f"   S {sc.stcor_s.min():+.3f} .. {sc.stcor_s.max():+.3f} s")
    nz = obs[(obs.ptcor_s != 0) | (obs.stcor_s != 0)].copy()
    if len(nz) > 2:
        # The absolute level of the corrections trades against origin times and
        # the overall model speed, so quote the DEMEANED OBS spread as well.
        nz["dp"] = nz.ptcor_s - nz.ptcor_s.mean()
        nz["ds"] = nz.stcor_s - nz.stcor_s.mean()
        print(f"  OBS (n={len(nz)}) demeaned: P {nz.dp.min():+.3f} .. {nz.dp.max():+.3f} s"
              f" (sd {nz.dp.std():.3f})   S {nz.ds.min():+.3f} .. {nz.ds.max():+.3f} s"
              f" (sd {nz.ds.std():.3f})")
        print(f"  OBS corr vs water depth: "
              f"P r={np.corrcoef(nz.water_km, nz.ptcor_s)[0, 1]:+.2f}"
              f"  S r={np.corrcoef(nz.water_km, nz.stcor_s)[0, 1]:+.2f}")
    print(f"\n  wrote {run}/relocated.csv, final_model.csv, station_corrections.csv, "
          f"rms_trajectory.csv")


if __name__ == "__main__":
    main()
