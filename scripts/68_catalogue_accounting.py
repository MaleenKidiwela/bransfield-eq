"""Task 3 of the v6 validation package: exact event/pick accounting from the
associated catalogue to each QC tier, plus the near-station P+S check on strict.

Every number is counted from the files themselves (obs blocks, hyp files, the
catalogue), not copied from a log. The QC tier masks reproduce
40_filter_nlloc_reliable.py exactly; the standard-tier count is gated against the
already-written catalogs/nlloc_year_v6_standard.csv so a divergence is loud.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

GRID_XY = ((-150.0, 80.0), (-110.0, 70.0))
GRID_Z = (0.0, 25.2)
SEAFLOOR_TOL_KM = 0.2
R_EARTH = 6371.0


def obs_blocks(path):
    blocks, cur = [], []
    for l in Path(path).read_text().split("\n"):
        if l.strip() == "":
            if cur: blocks.append(cur); cur = []
        else: cur.append(l)
    if cur: blocks.append(cur)
    return blocks


def pick_counts(blocks):
    n = p = s = 0
    for b in blocks:
        for l in b:
            f = l.split()
            if len(f) >= 9 and not f[0].startswith("#"):
                n += 1
                if f[4] == "P": p += 1
                elif f[4] == "S": s += 1
    return n, p, s


def add_bsf_and_hull(df, tt_prefix="ORCA_v5"):
    """depth below the LOCAL seafloor + ZX-hull flag, exactly as script 40 computes them."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MplPath
    w = np.load(REPO / "nlloc" / "model" / f"{tt_prefix}.water_depth_km.npy")
    hdr = (REPO / "nlloc" / "model" / f"{tt_prefix}.P.mod.hdr").read_text().split()
    hx0, hy0, hdx = float(hdr[3]), float(hdr[4]), float(hdr[6])
    gx = hx0 + np.arange(w.shape[0]) * hdx
    gy = hy0 + np.arange(w.shape[1]) * hdx
    fw = RegularGridInterpolator((gx, gy), w, method="linear", bounds_error=False, fill_value=None)
    coarse = fw(np.c_[np.clip(df.nlloc_x_km.values, gx[0], gx[-1]),
                      np.clip(df.nlloc_y_km.values, gy[0], gy[-1])])
    fine = np.full(len(df), np.nan)
    orca = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
    if orca.exists():
        import xarray as xr
        o = xr.open_dataset(orca)
        olat = np.asarray(o.latitude.values, float); olon = np.asarray(o.longitude.values, float)
        oz = np.asarray(o["data"].values, float)
        if oz.shape != (len(olat), len(olon)): oz = oz.T
        fo = RegularGridInterpolator((olat, olon), np.clip(-oz / 1000.0, 0.0, None),
                                     bounds_error=False, fill_value=np.nan)
        fine = fo(np.c_[df.lat.values, df.lon.values])
    df["local_water_km"] = np.where(np.isfinite(fine), fine, coarse)
    df["depth_bsf_km"] = df.depth_km - df.local_water_km
    st = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    zx = st[st.network == "ZX"][["longitude", "latitude"]].to_numpy()
    hull = ConvexHull(zx)
    df["in_hull"] = MplPath(zx[hull.vertices]).contains_points(np.c_[df.lon, df.lat])
    (gx0, gx1), (gy0, gy1) = GRID_XY
    gz0, gz1 = GRID_Z
    df["on_boundary"] = ((df.nlloc_x_km - gx0 < 0.5) | (gx1 - df.nlloc_x_km < 0.5) |
                         (df.nlloc_y_km - gy0 < 0.5) | (gy1 - df.nlloc_y_km < 0.5) |
                         (df.depth_km - gz0 < 0.5) | (gz1 - df.depth_km < 0.5))
    return df, int(np.isfinite(fine).sum())


def tiers(df):
    below = df.depth_bsf_km > -SEAFLOOR_TOL_KM
    loose = ~df.on_boundary & below & (df.gap_deg < 200) & (df.rms_s < 0.7) & (df.n_phases >= 4)
    standard = ~df.on_boundary & below & (df.gap_deg < 180) & (df.rms_s < 0.5) & (df.n_phases >= 6)
    strict = (~df.on_boundary & df.in_hull & (df.gap_deg < 120) & (df.rms_s < 0.3) &
              (df.n_phases >= 8) & (df.sigma_x_km < 1.0) & (df.sigma_y_km < 1.0) &
              (df.sigma_z_km < 2.0) & (df.depth_bsf_km > 0.2))
    return loose, standard, strict


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="year_v6")
    ap.add_argument("--prev-obs", default="nlloc/obs/year_v5.obs")
    ap.add_argument("--near-km", type=float, default=4.0)
    a = ap.parse_args()
    lab = a.label

    print("=== 1. association -> NLLoc input ===")
    assoc = pd.read_csv(REPO / "catalogs" / "pyocto_events_year_newpool.csv")
    noshot = pd.read_csv(REPO / "catalogs" / f"pyocto_events_year_newpool_no_shots.csv")
    print(f"  associated events (year, new pool)      {len(assoc):,}")
    print(f"  after the airgun-window exclusion       {len(noshot):,}  (-{len(assoc)-len(noshot):,})")

    v6 = obs_blocks(REPO / "nlloc" / "obs" / f"{lab}.obs")
    v5 = obs_blocks(REPO / a.prev_obs)
    n6, p6, s6 = pick_counts(v6); n5, p5, s5 = pick_counts(v5)
    print(f"  events written to {lab}.obs        {len(v6):,}")
    print(f"\n=== 2. the script-66 S-before-P guard ===")
    print(f"  picks before guard ({Path(a.prev_obs).name})   {n5:,}  (P {p5:,}  S {s5:,})")
    print(f"  picks after  guard ({lab}.obs)      {n6:,}  (P {p6:,}  S {s6:,})")
    ndrop = s5 - s6
    nev = sum(1 for b5, b6 in zip(v5, v6) if len(b6) < len(b5))
    print(f"  S picks dropped                         {ndrop:,} ({100*ndrop/s5:.1f}% of S)"
          f" in {nev:,} events ({100*nev/len(v6):.1f}%)")
    print(f"  events left with no S pick at all       "
          f"{sum(1 for b in v6 if not any(l.split()[4] == 'S' for l in b if len(l.split()) >= 9)):,}")

    print(f"\n=== 3. NLLoc ===")
    hyps = [h for h in (REPO / "nlloc" / "output" / lab).rglob("loc.2*.grid0.loc.hyp") if "last" not in h.name]
    unm = [l for l in (REPO / "nlloc" / "output" / lab / "unmatched_obs_events.txt").read_text().split("\n") if l.strip()]
    print(f"  obs events in                           {len(v6):,}")
    print(f"  per-event .hyp files written            {len(hyps):,}")
    print(f"  obs events NLLoc could not locate       {len(unm):,}  (unmatched_obs_events.txt)")

    cat = pd.read_csv(REPO / "catalogs" / f"nlloc_{lab}.csv")
    print(f"  rows in catalogs/nlloc_{lab}.csv   {len(cat):,}"
          f"  (-{len(hyps)-len(cat):,} hyp files with no parsable GEOGRAPHIC line)")
    vc = cat.nlloc_status.value_counts()
    for k, v in vc.items():
        print(f"    nlloc_status {k:12s} {v:,}")
    cat = cat[cat.nlloc_status == "LOCATED"].copy()
    print(f"  LOCATED solutions                       {len(cat):,}")

    cat, n_fine = add_bsf_and_hull(cat)
    print(f"  local seafloor: {n_fine:,} from the 30 m Orca grid, {len(cat)-n_fine:,} from the 0.4 km surface")
    loose, standard, strict = tiers(cat)
    print(f"\n=== 4. QC tiers (cumulative cuts of 40_filter_nlloc_reliable.py) ===")
    for name, m in [("loose   (gap<200, rms<0.7, N>=4, bsf>-0.2)", loose),
                    ("standard(gap<180, rms<0.5, N>=6, bsf>-0.2)", standard),
                    ("strict  (gap<120, rms<0.3, N>=8, sig<1/1/2, hull, bsf>0.2)", strict)]:
        print(f"  {name:60s} {int(m.sum()):,} ({100*m.mean():.1f}% of LOCATED)")
    # why events fall out of 'loose'
    below = cat.depth_bsf_km > -SEAFLOOR_TOL_KM
    print("\n  cut-by-cut survival from LOCATED (each cut applied alone):")
    for name, m in [("not grid-boundary-pinned", ~cat.on_boundary), ("bsf > -0.2 km", below),
                    ("gap < 200", cat.gap_deg < 200), ("rms < 0.7", cat.rms_s < 0.7),
                    ("n_phases >= 4", cat.n_phases >= 4), ("gap < 180", cat.gap_deg < 180),
                    ("rms < 0.5", cat.rms_s < 0.5), ("n_phases >= 6", cat.n_phases >= 6),
                    ("gap < 120", cat.gap_deg < 120), ("rms < 0.3", cat.rms_s < 0.3),
                    ("n_phases >= 8", cat.n_phases >= 8), ("sigma_z < 2 km", cat.sigma_z_km < 2.0),
                    ("inside the ZX hull", cat.in_hull), ("bsf > +0.2 km", cat.depth_bsf_km > 0.2)]:
        print(f"    {name:28s} {int(m.sum()):,} ({100*m.mean():.1f}%)")

    std_file = REPO / "catalogs" / f"nlloc_{lab}_standard.csv"
    if std_file.exists():
        n_std = sum(1 for _ in open(std_file)) - 1
        ok = "OK" if n_std == int(standard.sum()) else "MISMATCH"
        print(f"\n  gate: standard tier recomputed {int(standard.sum()):,} vs written file {n_std:,} -> {ok}")

    print(f"\n=== 5. formal sigma_z (km) by tier ===")
    for name, m in [("loose", loose), ("standard", standard), ("strict", strict)]:
        q = cat.sigma_z_km[m].quantile([0.1, 0.5, 0.9])
        qx = cat.sigma_x_km[m].quantile([0.1, 0.5, 0.9]); qy = cat.sigma_y_km[m].quantile([0.1, 0.5, 0.9])
        print(f"  {name:9s} n={int(m.sum()):6,}  sigma_z p10/50/90 {q.iloc[0]:.2f}/{q.iloc[1]:.2f}/{q.iloc[2]:.2f}"
              f"   sigma_x {qx.iloc[0]:.2f}/{qx.iloc[1]:.2f}/{qx.iloc[2]:.2f}"
              f"   sigma_y {qy.iloc[0]:.2f}/{qy.iloc[1]:.2f}/{qy.iloc[2]:.2f}")
        z = cat.depth_km[m].quantile([0.1, 0.5, 0.9]); b = cat.depth_bsf_km[m].quantile([0.1, 0.5, 0.9])
        print(f"            depth BSL p10/50/90 {z.iloc[0]:.2f}/{z.iloc[1]:.2f}/{z.iloc[2]:.2f} km"
              f"   BSF {b.iloc[0]:.2f}/{b.iloc[1]:.2f}/{b.iloc[2]:.2f} km")

    print(f"\n=== 6. strict-tier events with a near (< {a.near_km:g} km) station carrying BOTH P and S ===")
    order = pd.read_csv(REPO / "nlloc" / "obs" / f"{lab}.event_order.csv")
    idx2block = dict(zip(order.event_idx.values, order.obs_order.values))
    st = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    stlat = dict(zip(st.station, st.latitude)); stlon = dict(zip(st.station, st.longitude))
    S = cat[strict]
    have_near_ps = have_near = 0
    dmins = []
    for ei, lat, lon in zip(S.event_idx.values, S.lat.values, S.lon.values):
        b = v6[idx2block[ei]]
        ph = {}
        for l in b:
            f = l.split()
            if len(f) >= 9 and not f[0].startswith("#"):
                ph.setdefault(f[0], set()).add(f[4])
        best = np.inf; best_ps = False
        for sta, phs in ph.items():
            if sta not in stlat: continue
            d = haversine_km(lat, lon, stlat[sta], stlon[sta])
            if d < best: best = d
            if d < a.near_km and {"P", "S"} <= phs: best_ps = True
        dmins.append(best)
        have_near += best < a.near_km
        have_near_ps += best_ps
    dmins = np.array(dmins)
    print(f"  strict events                                   {len(S):,}")
    print(f"  with any station inside {a.near_km:g} km              {have_near:,} ({100*have_near/len(S):.1f}%)")
    print(f"  with a station inside {a.near_km:g} km having P AND S  {have_near_ps:,} ({100*have_near_ps/len(S):.1f}%)")
    print(f"  nearest-station epicentral distance p10/50/90   "
          f"{np.percentile(dmins,10):.2f}/{np.percentile(dmins,50):.2f}/{np.percentile(dmins,90):.2f} km")


if __name__ == "__main__":
    main()
