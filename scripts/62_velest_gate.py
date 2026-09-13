"""Run 52_sp_depth_check.py on a set of catalogues and tabulate the gate.

The gate for this project (notes/27 I27) is the ratio of the PREDICTED to the
OBSERVED near-station S-P spread across depth bins: it must move from ~1.25-1.3
toward 1.0, and the rms must drop.

52_sp_depth_check.py prints its own spread ratio from the FIRST and LAST
populated bin, which on the strict tier are (0,1] with n~7 and (12,40] with n~2
-- too few to carry a conclusion.  This wrapper re-reads the printed table and
also reports a robust ratio over the two best-populated end bins, (1,2] and
(8,12], which is the pair the 1.25-1.3 figure in notes/27 was built on.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ROBUST = ("(1, 2]", "(8, 12]")


def run_one(catalog: Path, label: str, vpvs: float) -> tuple[pd.DataFrame, dict]:
    cmd = [sys.executable, str(REPO / "scripts" / "52_sp_depth_check.py"),
           "--catalog", str(catalog), "--frame", "sealevel",
           "--scorer-vpvs", str(vpvs), "--label", label]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                         env={**__import__("os").environ,
                              "PYTHONPATH": str(REPO / "src")})
    if out.returncode != 0:
        raise SystemExit(out.stdout + out.stderr)
    rows = []
    for ln in out.stdout.splitlines():
        m = re.match(r"\s*(\([\d, ]+\])\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)", ln)
        if m:
            rows.append(dict(bin=m.group(1), n=int(m.group(2)), sp_obs=float(m.group(3)),
                             sp_pred=float(m.group(4)), misfit=float(m.group(5))))
    tab = pd.DataFrame(rows)
    m = re.search(r"ALL\s+([\d,]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+"
                  r"median \|misfit\|\s+([\d.]+)", out.stdout)
    r = re.search(r"observed x([\d.]+)\s+predicted x([\d.]+)", out.stdout)
    s = tab.set_index("bin")
    lo, hi = ROBUST
    rob = (float("nan") if lo not in s.index or hi not in s.index else
           (s.loc[hi, "sp_pred"] / s.loc[lo, "sp_pred"]) /
           (s.loc[hi, "sp_obs"] / s.loc[lo, "sp_obs"]))
    summ = dict(label=label, n=int(m.group(1).replace(",", "")),
                median_misfit=float(m.group(4)), median_abs_misfit=float(m.group(5)),
                obs_spread_edge=float(r.group(1)), pred_spread_edge=float(r.group(2)),
                ratio_edge=float(r.group(2)) / float(r.group(1)),
                obs_spread_robust=(s.loc[hi, "sp_obs"] / s.loc[lo, "sp_obs"]),
                pred_spread_robust=(s.loc[hi, "sp_pred"] / s.loc[lo, "sp_pred"]),
                ratio_robust=rob)
    return tab, summ


# --------------------------------------------------------------------------- #
# Bin-free version of the same test.
# 52's spread ratio divides the LAST populated depth bin by the FIRST, and on a
# relocated catalogue those bins can hold a handful of events (allB: n=8 in
# (8,12]).  The slope of near-station S-P against catalogue depth over the
# well-populated 1-8 km range uses every event and measures the same thing:
# ratio > 1 = the catalogue's depth axis is stretched, 1.0 = consistent.
# --------------------------------------------------------------------------- #
R_EARTH = 6371.0


def _slowness(z_top, z_bot, zg, v):
    lo, hi = np.minimum(z_top, z_bot), np.maximum(z_top, z_bot)
    zz = np.linspace(0, 1, 32)
    return np.array([np.mean(1.0 / np.interp(lo[k] + (hi[k] - lo[k]) * zz, zg, v))
                     for k in range(len(lo))])


def slope_metric(catalogs, vpvs, zlo=1.0, zhi=8.0, max_sta_km=4.0):
    vm = pd.read_csv(REPO / "configs" / "velocity_model.csv")
    rock = vm[vm.vp_kms > 1.6]
    zg, vp = rock.depth_km.to_numpy(float), rock.vp_kms.to_numpy(float)
    vs = vp / vpvs
    st = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    st["station"] = st.network + "." + st.station
    st["zs"] = np.where(st.on_seafloor, st.water_depth_m / 1000.0, -st.elevation_m / 1000.0)
    pk = pd.read_csv(REPO / "catalogs" / "pyocto_picks_year_newpool_no_shots.csv")
    P = pk[pk.phase == "P"][["event_idx", "station", "time"]].rename(columns={"time": "tp"})
    S = pk[pk.phase == "S"][["event_idx", "station", "time"]].rename(columns={"time": "ts"})
    sp = P.merge(S, on=["event_idx", "station"])
    sp["sp_obs"] = sp.ts - sp.tp
    sp = sp[(sp.sp_obs > 0) & (sp.sp_obs < 10)]
    rows = []
    for label, path in catalogs:
        c = pd.read_csv(REPO / path)
        m = sp.merge(c[["event_idx", "lat", "lon", "depth_km"]], on="event_idx") \
              .merge(st[["station", "latitude", "longitude", "zs"]], on="station")
        p1, p2 = np.radians(m.lat), np.radians(m.latitude)
        dl = np.radians(m.longitude - m.lon)
        m["epi"] = 2 * R_EARTH * np.arcsin(np.sqrt(
            np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))
        m = m[m.epi <= max_sta_km].sort_values("epi").groupby("event_idx").head(1).copy()
        za = m.zs.values.clip(min=zg[0])
        zb = m.depth_km.values.clip(min=zg[0])
        rr = np.hypot(m.epi.values, m.depth_km.values - m.zs.values)
        m["sp_pred"] = rr * (_slowness(za, zb, zg, vs) - _slowness(za, zb, zg, vp))
        q = m[(m.depth_km > zlo) & (m.depth_km <= zhi)]
        a_obs = np.polyfit(q.depth_km, q.sp_obs, 1)[0]
        a_pred = np.polyfit(q.depth_km, q.sp_pred, 1)[0]
        rows.append(dict(label=label, n=len(q), dobs_dz=a_obs, dpred_dz=a_pred,
                         ratio_slope=a_pred / a_obs,
                         med_abs_misfit=float(np.median(np.abs(m.sp_pred - m.sp_obs))),
                         med_misfit=float(np.median(m.sp_pred - m.sp_obs))))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogs", nargs="+", required=True,
                    help="label=path pairs")
    ap.add_argument("--scorer-vpvs", type=float, default=1.88)
    ap.add_argument("--out", default="velest/sp_gate.csv")
    args = ap.parse_args()

    summaries, tables = [], []
    for spec in args.catalogs:
        label, path = spec.split("=", 1)
        tab, summ = run_one(REPO / path, label, args.scorer_vpvs)
        tab.insert(0, "label", label)
        tables.append(tab)
        summaries.append(summ)
        print(f"\n=== {label} ===")
        print(tab.to_string(index=False))

    s = pd.DataFrame(summaries)
    pd.concat(tables).to_csv(REPO / args.out.replace(".csv", "_bins.csv"), index=False)
    s.to_csv(REPO / args.out, index=False)
    print(f"\n=== S-P spread gate (scorer Vp/Vs {args.scorer_vpvs}) ===")
    print(f"{'catalogue':>22}{'n':>7}{'obs':>8}{'pred':>8}{'ratio':>8}"
          f"{'|misfit|':>10}{'misfit':>9}")
    print(f"{'':>22}{'':>7}{'(1,2]->(8,12]':>24}")
    for _, r in s.iterrows():
        print(f"{r.label:>22}{r.n:>7,}{r.obs_spread_robust:>8.2f}"
              f"{r.pred_spread_robust:>8.2f}{r.ratio_robust:>8.2f}"
              f"{r.median_abs_misfit:>10.3f}{r.median_misfit:>+9.3f}")
    print(f"\n  (52's own edge-bin ratio, for reference: "
          + "; ".join(f"{r.label} {r.ratio_edge:.2f}" for _, r in s.iterrows()) + ")")
    sl = slope_metric([tuple(x.split("=", 1)) for x in args.catalogs], args.scorer_vpvs)
    sl.to_csv(REPO / args.out.replace(".csv", "_slope.csv"), index=False)
    print(f"\n=== bin-free stretch metric: d(S-P)/dz over 1-8 km BSL "
          f"(scorer Vp/Vs {args.scorer_vpvs}) ===")
    print(f"{'catalogue':>22}{'n':>7}{'obs s/km':>10}{'pred s/km':>11}{'ratio':>8}"
          f"{'|misfit|':>10}{'misfit':>9}")
    for _, r in sl.iterrows():
        print(f"{r.label:>22}{r.n:>7,}{r.dobs_dz:>10.4f}{r.dpred_dz:>11.4f}"
              f"{r.ratio_slope:>8.2f}{r.med_abs_misfit:>10.3f}{r.med_misfit:>+9.3f}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
