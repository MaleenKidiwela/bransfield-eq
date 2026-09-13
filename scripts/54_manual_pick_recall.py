"""Recall of the analyst (manual) picks by the new automatic pick pool, full year.

For every manual pick (station, phase, time) ask: did the automatic pool contain a
pick of the same phase at the same station within `--tol` seconds? Two levels:
  1. RAW pool  - union of catalogs/picks_pn_diting and picks_pnlight_obs for that
                 station-day (what fed association)
  2. CATALOGUE - the associated picks of the located catalogue
                 (catalogs/pyocto_picks_year_newpool_no_shots.csv)
Also reports the timing residual (auto - manual) of matched picks.

Caveats handled:
  - manual_picks.csv duplicates mag07 inside magall -> dedup on (sta, phase, time)
  - ZX.BRA05 manual picks are NOT clock-corrected (audit); the auto pool is ->
    subtract 0.167 s from manual BRA05 before matching
  - 2018 manual picks predate the deployment -> excluded and counted
  - the analyst says picks are most careful from 2019-10-01 -> reported separately
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
POOLS = ["picks_pn_diting", "picks_pnlight_obs"]
BRA05_OFFSET_S = 0.167


def epoch(s):
    # NEVER .astype("int64") on a datetime here: pandas 3 returns MICROseconds for
    # us-resolution datetimes and every date lands in 1970. The first run of this
    # script did exactly that and found 0 usable manual picks. Use the repo helper.
    from bransfield_eq.timeutil import epoch_seconds
    return epoch_seconds(pd.to_datetime(s, utc=True, format="mixed")).values


def load_auto_day(sta_key: str, day: pd.Timestamp) -> pd.DataFrame:
    tag = f"{day.year}-{day.dayofyear:03d}.csv"
    parts = []
    for pool in POOLS:
        f = REPO / "catalogs" / pool / sta_key / tag
        if f.exists() and f.stat().st_size > 0:
            try:
                d = pd.read_csv(f, usecols=["time", "phase", "prob"])
            except Exception:
                continue
            if len(d):
                d["t"] = epoch(d.time); d["pool"] = pool; parts.append(d[["t", "phase", "prob", "pool"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["t", "phase", "prob", "pool"])


def nearest(sorted_vals: np.ndarray, q: np.ndarray):
    if len(sorted_vals) == 0:
        return np.full(len(q), np.nan)
    i = np.searchsorted(sorted_vals, q)
    lo = np.clip(i - 1, 0, len(sorted_vals) - 1); hi = np.clip(i, 0, len(sorted_vals) - 1)
    dlo = np.abs(sorted_vals[lo] - q); dhi = np.abs(sorted_vals[hi] - q)
    best = np.where(dlo <= dhi, sorted_vals[lo], sorted_vals[hi])
    return best - q                                  # auto - manual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--trusted-from", default="2019-10-01")
    args = ap.parse_args()

    m = pd.read_csv(REPO / "catalogs" / "manual_picks.csv")
    n0 = len(m)
    m = m[m.network.notna()].copy(); n_nonet = n0 - len(m)
    m["sta_key"] = m.network + "." + m.station
    m["t"] = epoch(m.pick_time)
    m.loc[m.sta_key == "ZX.BRA05", "t"] -= BRA05_OFFSET_S     # audit: manual BRA05 uncorrected
    m["tr"] = m.t.round(2)
    n_before = len(m); m = m.drop_duplicates(["sta_key", "phase", "tr"]); n_dup = n_before - len(m)
    m["day"] = pd.to_datetime(m.t, unit="s", utc=True).dt.floor("D")
    in_dep = (m.day >= "2019-01-01") & (m.day < "2020-03-01")
    n_predep = int((~in_dep).sum()); m = m[in_dep].copy()
    print(f"manual picks: {n0:,} rows -> {n_nonet} no network, {n_dup:,} duplicates (mag07 in magall), "
          f"{n_predep:,} outside the deployment -> {len(m):,} usable   (P {int((m.phase=='P').sum()):,} / S {int((m.phase=='S').sum()):,})")

    # ---- level 1: raw pool ----
    res = np.full(len(m), np.nan); prob = np.full(len(m), np.nan)
    m = m.reset_index(drop=True)
    groups = m.groupby(["sta_key", "day"]).indices
    for (sta, day), idx in groups.items():
        a = load_auto_day(sta, day)
        for ph in ("P", "S"):
            sel = idx[m.phase.values[idx] == ph]
            if len(sel) == 0: continue
            aa = a[a.phase == ph].sort_values("t")
            r = nearest(aa.t.values, m.t.values[sel]); res[sel] = r
    m["res_raw"] = res
    m["hit_raw"] = np.abs(m.res_raw) <= args.tol

    # ---- level 2: catalogue (associated picks) ----
    pk = pd.read_csv(REPO / "catalogs" / "pyocto_picks_year_newpool_no_shots.csv", usecols=["station", "phase", "time"])
    res2 = np.full(len(m), np.nan)
    for (sta, ph), idx in m.groupby(["sta_key", "phase"]).indices.items():
        c = np.sort(pk[(pk.station == sta) & (pk.phase == ph)].time.values)
        res2[idx] = nearest(c, m.t.values[idx])
    m["res_cat"] = res2; m["hit_cat"] = np.abs(m.res_cat) <= args.tol

    trusted = m.day >= args.trusted_from
    def block(title, sub):
        if len(sub) == 0: return
        print(f"\n--- {title}  (n={len(sub):,}) ---")
        print(f"{'phase':>6}{'n':>8}{'RAW recall':>12}{'CAT recall':>12}{'bias ms':>9}{'MAD ms':>8}")
        for ph in ("P", "S"):
            s = sub[sub.phase == ph]
            if len(s) == 0: continue
            r = s.res_raw[s.hit_raw] * 1000
            print(f"{ph:>6}{len(s):>8,}{s.hit_raw.mean()*100:>11.1f}%{s.hit_cat.mean()*100:>11.1f}%{np.median(r):>+9.0f}{np.median(np.abs(r-np.median(r))):>8.0f}")
    block("ALL usable manual picks", m)
    block(f"analyst-trusted window (>= {args.trusted_from})", m[trusted])
    block(f"earlier window (< {args.trusted_from}, analyst says less careful)", m[~trusted])
    block("land stations only (5M/AI)", m[~m.sta_key.str.startswith("ZX")])

    print(f"\n--- per station, trusted window, RAW recall (stations with >= 100 manual picks) ---")
    t = m[trusted].groupby("sta_key").agg(n=("hit_raw", "size"), P=("phase", lambda p: (p == "P").sum()),
                                           raw=("hit_raw", "mean"), cat=("hit_cat", "mean"))
    t = t[t.n >= 100].sort_values("raw")
    for sta, r in t.iterrows():
        print(f"  {sta:10s} n={r.n:>6,}  raw {r["raw"]*100:5.1f}%  catalogue {r["cat"]*100:5.1f}%")
    out = REPO / "catalogs" / "manual_pick_recall_year_newpool.csv"
    m.drop(columns=["tr"]).to_csv(out, index=False); print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
