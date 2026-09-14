"""Task 1 of the v6 validation package: do the analyst's own picks support the
per-station S-P station terms in nlloc/delays/v6_it1B.delays?

The v6 delays are applied by NLLoc as obs - delay, so a station's S-P delay
(S_delay - P_delay) is subtracted from every observed S-P at that station. If the
analyst routinely measured S-P SMALLER than the station's S-P delay, the term is
not a station property and v6 is over-correcting (and the script-66 guard is
hiding it by deleting those S picks).

Per OBS station this script reports:
  n manual P+S pairs, manual S-P min / p5 / p50,
  automatic (pyocto) S-P p5 / p50 on the SAME event/station pairs,
  median(auto - manual) S-P,
  the station's S-P delay,
  the fraction of manual pairs with S-P below the delay (+ margin), and for those
  the epicentral distance to the station and the source depth, so a sub-delay
  measurement can be checked against the "event vertically under the station"
  explanation.

Manual events are matched to pyocto events by origin time (+-3 s); pyocto event_idx
then gives the v6 hypocentre for the geometry check.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.timeutil import epoch_seconds, assert_nanosecond_sanity

R_EARTH = 6371.0


def load_delays(p):
    d = {}
    for l in Path(p).read_text().split("\n"):
        f = l.split()
        if len(f) == 5 and f[0] == "LOCDELAY":
            d[(f[1], f[2])] = float(f[4])
    return d


def sp_pairs(df, tcol, keycols):
    """(event, station) rows with both P and S -> S-P in seconds."""
    g = df.pivot_table(index=keycols, columns="phase", values=tcol, aggfunc="min")
    g = g.dropna(subset=["P", "S"])
    return (g["S"] - g["P"]).rename("sp").reset_index()


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", default="catalogs/manual_pick_recall_year_newpool.csv")
    ap.add_argument("--auto-picks", default="catalogs/pyocto_picks_year_newpool_no_shots.csv")
    ap.add_argument("--auto-events", default="catalogs/pyocto_events_year_newpool_no_shots.csv")
    ap.add_argument("--catalog", default="catalogs/nlloc_year_v6.csv")
    ap.add_argument("--stations", default="catalogs/station_geometry.csv")
    ap.add_argument("--delays", default="nlloc/delays/v6_it1B.delays")
    ap.add_argument("--ot-tol", type=float, default=3.0)
    ap.add_argument("--min-pairs", type=int, default=20)
    a = ap.parse_args()
    assert_nanosecond_sanity()

    dl = load_delays(REPO / a.delays)
    st = pd.read_csv(REPO / a.stations).set_index(["network", "station"])

    # ---------- manual S-P ----------
    M = pd.read_csv(REPO / a.manual)
    print(f"manual picks: {len(M):,} rows, {M.event_id.nunique():,} events, "
          f"{M.sta_key.nunique()} stations; t dtype {M.t.dtype}")
    msp = sp_pairs(M, "t", ["event_id", "sta_key"])
    print(f"manual event/station pairs with both P and S: {len(msp):,}")

    # ---------- manual event -> pyocto event (origin time) ----------
    mev = M.groupby("event_id").origin_time.first().reset_index()
    mev["t0"] = epoch_seconds(mev.origin_time)
    E = pd.read_csv(REPO / a.auto_events)
    E = E.sort_values("time").reset_index(drop=True)
    j = pd.merge_asof(mev.sort_values("t0"), E[["time", "event_idx"]].rename(columns={"time": "t0_auto"}),
                      left_on="t0", right_on="t0_auto", direction="nearest", tolerance=a.ot_tol)
    j = j.dropna(subset=["event_idx"])
    j["event_idx"] = j.event_idx.astype(int)
    print(f"manual events matched to a pyocto event within +-{a.ot_tol:g} s: "
          f"{len(j):,} / {len(mev):,} ({100*len(j)/len(mev):.1f}%); "
          f"|dOT| p50 {np.abs(j.t0 - j.t0_auto).median():.3f} s")

    # ---------- automatic S-P on the same event/station pairs ----------
    A = pd.read_csv(REPO / a.auto_picks)
    print(f"pyocto picks: {len(A):,} rows; time dtype {A.time.dtype}")
    at = A.time if np.issubdtype(A.time.dtype, np.number) else epoch_seconds(A.time)
    A = A.assign(t=at).rename(columns={"station": "sta_key"})
    asp = sp_pairs(A, "t", ["event_idx", "sta_key"]).rename(columns={"sp": "sp_auto"})

    P = msp.merge(j[["event_id", "event_idx", "t0"]], on="event_id", how="left")
    P = P.merge(asp, on=["event_idx", "sta_key"], how="left")

    # ---------- v6 hypocentre for the geometry check ----------
    C = pd.read_csv(REPO / a.catalog, usecols=["event_idx", "lat", "lon", "depth_km", "nlloc_status"])
    C = C[C.nlloc_status == "LOCATED"]
    P = P.merge(C.rename(columns={"lat": "ev_lat", "lon": "ev_lon"}), on="event_idx", how="left")
    P["net"] = P.sta_key.str.split(".").str[0]
    P["sta"] = P.sta_key.str.split(".").str[1]
    P["st_lat"] = [st.latitude.get((n, s), np.nan) for n, s in zip(P.net, P.sta)]
    P["st_lon"] = [st.longitude.get((n, s), np.nan) for n, s in zip(P.net, P.sta)]
    P["epi_km"] = haversine_km(P.ev_lat, P.ev_lon, P.st_lat, P.st_lon)
    P["sp_delay"] = [dl.get((s, "S"), np.nan) - dl.get((s, "P"), np.nan) for s in P.sta]

    P.to_csv(REPO / "nlloc" / "output" / "manual_sp_pairs_v6.csv", index=False)

    rows = []
    for sta, g in P[P.net == "ZX"].groupby("sta"):
        if len(g) < a.min_pairs:
            continue
        gm = g.sp.values
        ga = g.sp_auto.dropna()
        both = g.dropna(subset=["sp_auto"])
        below = g[g.sp < g.sp_delay]
        guard = g[g.sp <= g.sp_delay + 0.05]
        rows.append(dict(
            sta=sta, n=len(g), n_neg=int((gm < 0).sum()),
            man_min=np.min(gm), man_p1=np.percentile(gm, 1),
            man_p5=np.percentile(gm, 5), man_p50=np.median(gm),
            n_auto=len(ga),
            auto_p5=np.percentile(ga, 5) if len(ga) else np.nan,
            auto_p50=np.median(ga) if len(ga) else np.nan,
            auto_minus_man=np.median(both.sp_auto - both.sp) if len(both) else np.nan,
            sp_delay=g.sp_delay.iloc[0],
            p5_minus_delay=np.percentile(gm, 5) - g.sp_delay.iloc[0],
            pct_below=100 * len(below) / len(g),
            pct_guard=100 * len(guard) / len(g),
            below_epi_p50=below.epi_km.median() if len(below) else np.nan,
            below_epi_p90=below.epi_km.quantile(0.9) if len(below) else np.nan,
            below_z_p50=below.depth_km.median() if len(below) else np.nan,
        ))
    T = pd.DataFrame(rows).sort_values("n", ascending=False)
    pd.set_option("display.width", 220)
    print("\n=== manual vs automatic S-P vs station S-P delay (OBS, >= "
          f"{a.min_pairs} manual P+S pairs) ===")
    print(T.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    all_zx = P[(P.net == "ZX") & P.sp_delay.notna()]
    nb = (all_zx.sp < all_zx.sp_delay).sum()
    ng = (all_zx.sp <= all_zx.sp_delay + 0.05).sum()
    nneg = (all_zx.sp < 0).sum()
    print(f"\nAll ZX manual pairs: {len(all_zx):,}; below the station S-P delay: {nb:,} "
          f"({100*nb/len(all_zx):.1f}%); at or below delay+0.05 s (the script-66 guard "
          f"threshold): {ng:,} ({100*ng/len(all_zx):.1f}%); physically impossible "
          f"(manual S-P < 0, i.e. analyst mis-order): {nneg:,} ({100*nneg/len(all_zx):.2f}%)")
    for lim in (4.0, 6.0, 10.0):
        sub = all_zx[all_zx.sp < all_zx.sp_delay].dropna(subset=["epi_km"])
        print(f"  of those with a v6 hypocentre ({len(sub):,}): epicentral distance < {lim:g} km: "
              f"{100*(sub.epi_km < lim).mean():.1f}%")
    sub = all_zx[all_zx.sp < all_zx.sp_delay].dropna(subset=["epi_km"])
    ref = all_zx.dropna(subset=["epi_km"])
    print(f"  epicentral distance p50/p90: below-delay {sub.epi_km.median():.2f}/{sub.epi_km.quantile(.9):.2f} km "
          f"vs all {ref.epi_km.median():.2f}/{ref.epi_km.quantile(.9):.2f} km")
    print(f"  source depth p50: below-delay {sub.depth_km.median():.2f} km vs all {ref.depth_km.median():.2f} km")
    both = all_zx.dropna(subset=["sp_auto"])
    print(f"\nauto - manual S-P over all ZX pairs with both ({len(both):,}): "
          f"median {np.median(both.sp_auto - both.sp):+.3f} s, "
          f"MAD {np.median(np.abs(both.sp_auto - both.sp - np.median(both.sp_auto - both.sp))):.3f} s")
    print(f"wrote {REPO / 'nlloc' / 'output' / 'manual_sp_pairs_v6.csv'}")


if __name__ == "__main__":
    main()
