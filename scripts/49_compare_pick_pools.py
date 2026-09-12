#!/usr/bin/env python3
"""
Head-to-head comparison of two pick pools, at the *catalogue* level.

This is the Step-3 validation gate of notes/25_gpu_run_runbook.md: higher pick
recall is not automatically a better catalogue.  Run one month through
`17_pyocto_associate.py` with each pool, then run this.

    | metric                        | good result                            |
    |-------------------------------|----------------------------------------|
    | event count                   | up, or flat with better-constrained ev. |
    | picks per event               | up                                      |
    | post-fit RMS                  | flat or lower (a jump = noise assoc.)   |
    | stations per event            | up                                      |
    | NLLoc sigma (after stage 5)   | DOWN - this is what decides the paper   |

WHY EVERY METRIC IS BROKEN DOWN BY STATION COUNT
------------------------------------------------
The one-day smoketest (2020-01-25) gave the new pool +60% events while the
count of well-constrained (>=8 station) events stayed FLAT: 59 vs 60.  All 70
extra events were marginal 3-7 station detections.  A comparison that reports
only totals hides exactly the effect we are testing for, so every table here is
split by station-count class and NLLoc sigma is reported on the
well-constrained subset, *not* pooled over everything.

INPUTS (per pool, `--*-label` resolves all three under catalogs/)
    catalogs/pyocto_events_<label>.csv   required
    catalogs/pyocto_picks_<label>.csv    strongly recommended - the ONLY source
                                         of stations/event and post-fit RMS
    catalogs/nlloc_<label>.csv           optional, after stage 5 (script 31)

USAGE
    PYTHONPATH=src python3 scripts/49_compare_pick_pools.py \
        --new-label newpool_2020_01 --prod-label prodpool_2020_01

    # explicit paths instead of labels, and dump the tables as CSV
    PYTHONPATH=src python3 scripts/49_compare_pick_pools.py \
        --new-events catalogs/pyocto_events_a.csv \
        --prod-events catalogs/pyocto_events_b.csv \
        --out-dir notes/pool_compare

SCHEMA NOTES (see the module docstring of the parser below - these bit us)
  * Two event-catalogue schemas exist in this repo.  Single-run outputs are
    `idx,time,x,y,z,picks`; chunked/merged year runs add
    `event_idx,longitude,latitude,depth,origin_time` and their `idx` is
    CHUNK-LOCAL and NOT UNIQUE.  The join key to the assoc-pick and NLLoc files
    is `event_idx` when present, `idx` otherwise.  Joining on `idx` for a merged
    catalogue silently matches ~2% of rows.
  * The event catalogue carries NO rms, NO station count and NO lat/lon in the
    single-run schema.  RMS and station counts are computed from the assoc
    picks file.
  * x/y are NOT on a shared grid between two runs.  Script 17 sets the tmerc
    origin to the mean lat/lon of the stations that have picks, so two pools
    with different station coverage get origins ~20 km apart (measured on the
    2020-01-25 smoketest).  Matching therefore prefers lat/lon haversine, and
    falls back to x/y only after estimating and removing that offset.
  * pandas-3: never `.astype("int64")` a datetime.  All epoch conversion goes
    through src/bransfield_eq/timeutil.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bransfield_eq.timeutil import assert_nanosecond_sanity, epoch_seconds  # noqa: E402

# Quality tiers, copied from scripts/40_filter_nlloc_reliable.py so the two
# stay comparable.  (on_boundary / below-seafloor tests are not applied here -
# they need the grid extent and bathymetry; these are the location-quality
# parts of the tiers.)
NLLOC_TIERS = {
    "loose   (gap<200, RMS<0.7, N>=4)":
        lambda d: (d.gap_deg < 200) & (d.rms_s < 0.7) & (d.n_phases >= 4),
    "standard(gap<180, RMS<0.5, N>=6)":
        lambda d: (d.gap_deg < 180) & (d.rms_s < 0.5) & (d.n_phases >= 6),
    "strict  (gap<120, RMS<0.3, N>=8, sig<1/1/2)":
        lambda d: (d.gap_deg < 120) & (d.rms_s < 0.3) & (d.n_phases >= 8)
                  & (d.sigma_x_km < 1.0) & (d.sigma_y_km < 1.0) & (d.sigma_z_km < 2.0),
}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
class Pool:
    """One pick pool's catalogue, normalised to a common schema.

    After load, `self.ev` always has columns:
        key        unique event id, joins to assoc picks and NLLoc event_idx
        t          origin time, float epoch seconds (unit-safe)
        x, y, z    km in the run's tmerc CRS (z positive down)
        n_picks    picks associated to the event
        n_sta      distinct stations (NaN if no assoc file)
        n_p, n_s   P and S pick counts (NaN if no assoc file)
        rms        post-fit RMS of the association residuals (NaN if no assoc)
        lat, lon   WGS84 if the catalogue carries them, else NaN
    """

    def __init__(self, name, events_path, picks_path=None, nlloc_path=None):
        self.name = name
        self.events_path = Path(events_path)
        self.picks_path = Path(picks_path) if picks_path else None
        self.nlloc_path = Path(nlloc_path) if nlloc_path else None
        self.nlloc = None
        self.has_assoc = False
        self._load()

    def _load(self):
        if not self.events_path.exists():
            raise SystemExit(f"[{self.name}] event catalogue not found: {self.events_path}")
        raw = pd.read_csv(self.events_path)

        # --- the join key.  'event_idx' wins when present: in chunked/merged
        # runs 'idx' restarts per chunk and is NOT unique.
        if "event_idx" in raw.columns:
            key_col = "event_idx"
        elif "idx" in raw.columns:
            key_col = "idx"
        else:
            raise SystemExit(f"[{self.name}] no 'event_idx' or 'idx' column in {self.events_path}")
        if not raw[key_col].is_unique:
            raise SystemExit(
                f"[{self.name}] join key '{key_col}' is not unique in {self.events_path} "
                f"({len(raw)} rows, {raw[key_col].nunique()} distinct). "
                f"This catalogue looks like merged chunks without a global event_idx; "
                f"station counts and NLLoc sigma cannot be joined safely."
            )

        ev = pd.DataFrame({"key": raw[key_col].astype("int64")})

        # --- origin time.  'time' from pyocto is already float epoch seconds.
        # Only a datetime column goes through timeutil (never a raw int cast).
        if "time" in raw.columns and pd.api.types.is_numeric_dtype(raw["time"]):
            ev["t"] = raw["time"].astype(float)
        elif "origin_time" in raw.columns:
            ev["t"] = epoch_seconds(raw["origin_time"])
        else:
            raise SystemExit(f"[{self.name}] no usable origin-time column in {self.events_path}")

        for c in ("x", "y", "z"):
            ev[c] = raw[c].astype(float) if c in raw.columns else np.nan
        if "depth" in raw.columns:
            ev["z"] = raw["depth"].astype(float)
        ev["lat"] = raw["latitude"].astype(float) if "latitude" in raw.columns else np.nan
        ev["lon"] = raw["longitude"].astype(float) if "longitude" in raw.columns else np.nan
        ev["n_picks"] = raw["picks"].astype(float) if "picks" in raw.columns else np.nan

        # --- per-event station count, phase split and post-fit RMS.
        ev["n_sta"] = np.nan
        ev["n_p"] = np.nan
        ev["n_s"] = np.nan
        ev["rms"] = np.nan
        if self.picks_path and self.picks_path.exists():
            a = pd.read_csv(self.picks_path)
            if "event_idx" not in a.columns:
                print(f"  ! [{self.name}] {self.picks_path.name} has no event_idx; ignoring")
            else:
                a["event_idx"] = a["event_idx"].astype("int64")
                g = a.groupby("event_idx")
                agg = pd.DataFrame({
                    "n_sta": g["station"].nunique(),
                    "n_assoc": g.size(),
                })
                if "residual" in a.columns:
                    agg["rms"] = g["residual"].apply(
                        lambda r: float(np.sqrt(np.mean(np.square(r.to_numpy(float)))))
                    )
                if "phase" in a.columns:
                    ph = a.assign(_one=1).pivot_table(
                        index="event_idx", columns="phase", values="_one", aggfunc="sum"
                    )
                    for src, dst in (("P", "n_p"), ("S", "n_s")):
                        if src in ph.columns:
                            agg[dst] = ph[src]
                ev = ev.drop(columns=["n_sta", "n_p", "n_s", "rms"]).merge(
                    agg, left_on="key", right_index=True, how="left"
                )
                for c in ("n_sta", "n_p", "n_s", "rms"):
                    if c not in ev.columns:
                        ev[c] = np.nan
                # prefer the assoc count; it is what the station count is drawn from
                if "n_assoc" in ev.columns:
                    ev["n_picks"] = ev["n_assoc"].fillna(ev["n_picks"])
                self.has_assoc = True
                unmatched = int(ev["n_sta"].isna().sum())
                if unmatched:
                    print(f"  ! [{self.name}] {unmatched}/{len(ev)} events have no rows in "
                          f"{self.picks_path.name} (station counts missing for those)")
        else:
            print(f"  ! [{self.name}] no assoc-pick file -> stations/event and RMS unavailable. "
                  f"Expected {self.picks_path if self.picks_path else '(none given)'}")

        self.ev = ev.sort_values("t").reset_index(drop=True)

        # --- NLLoc (stage 5), joined on event_idx
        if self.nlloc_path and self.nlloc_path.exists():
            nl = pd.read_csv(self.nlloc_path)
            if "event_idx" not in nl.columns:
                print(f"  ! [{self.name}] {self.nlloc_path.name} has no event_idx; ignoring")
            else:
                nl["event_idx"] = nl["event_idx"].astype("int64")
                matched = nl["event_idx"].isin(set(ev["key"])).mean() if len(nl) else 0.0
                if matched < 0.5:
                    print(f"  ! [{self.name}] only {matched:.1%} of NLLoc event_idx values match "
                          f"the event catalogue - wrong pairing, or the script-31 join bug. "
                          f"Not using this NLLoc file.")
                else:
                    self.nlloc = nl.merge(
                        ev[["key", "n_sta", "n_picks"]], left_on="event_idx",
                        right_on="key", how="left"
                    )
                    if matched < 0.99:
                        print(f"  ! [{self.name}] {1-matched:.1%} of NLLoc rows do not join to an "
                              f"event; they are kept but have no station count")
        elif self.nlloc_path:
            print(f"  ! [{self.name}] NLLoc catalogue not found: {self.nlloc_path} "
                  f"(stage 5 not run yet - sigma comparison skipped)")


# --------------------------------------------------------------------------
# station-count classes
# --------------------------------------------------------------------------
def parse_classes(spec):
    """'3-5,6-7,8+' -> [('3-5',3,5), ('6-7',6,7), ('>=8',8,inf)]"""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok.endswith("+"):
            lo = int(tok[:-1])
            out.append((f">={lo}", lo, np.inf))
        elif "-" in tok:
            lo, hi = tok.split("-")
            out.append((f"{int(lo)}-{int(hi)}", int(lo), int(hi)))
        else:
            n = int(tok)
            out.append((f"{n}", n, n))
    return out


def class_mask(n_sta, lo, hi):
    v = n_sta.to_numpy(float)
    return (v >= lo) & (v <= hi)


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------
def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}"


def delta_str(new, old, nd=2, pct=True):
    if new is None or old is None or not np.isfinite(new) or not np.isfinite(old):
        return "-"
    d = new - old
    s = f"{d:+.{nd}f}"
    if pct and old != 0:
        s += f" ({d/abs(old)*100:+.0f}%)"
    return s


def table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def med(s):
    s = pd.Series(s).dropna()
    return float(s.median()) if len(s) else np.nan


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def section_headline(new: Pool, prod: Pool, wc_min: int):
    print("\n" + "=" * 78)
    print("1. HEADLINE  (totals - read section 2 before trusting any of this)")
    print("=" * 78)
    rows = []

    def row(label, fn, nd=2, pct=True):
        a, b = fn(new.ev), fn(prod.ev)
        rows.append([label, fmt(b, nd), fmt(a, nd), delta_str(a, b, nd, pct)])

    row("events", lambda d: float(len(d)), 0)
    row("picks/event (median)", lambda d: med(d.n_picks), 1)
    row("picks/event (mean)", lambda d: float(d.n_picks.mean()), 1)
    row("stations/event (median)", lambda d: med(d.n_sta), 1)
    row("P picks/event (median)", lambda d: med(d.n_p), 1)
    row("S picks/event (median)", lambda d: med(d.n_s), 1)
    row("post-fit RMS s (median)", lambda d: med(d.rms), 3)
    row("post-fit RMS s (90th pct)",
        lambda d: float(d.rms.dropna().quantile(0.90)) if d.rms.notna().any() else np.nan, 3)
    row(f"events with >={wc_min} stations", lambda d: float(class_mask(d.n_sta, wc_min, np.inf).sum()), 0)
    row("total associated picks", lambda d: float(d.n_picks.sum()), 0)
    print(table(rows, ["metric", f"prod [{prod.name}]", f"new [{new.name}]", "delta"]))

    n_a = class_mask(new.ev.n_sta, wc_min, np.inf).sum()
    n_b = class_mask(prod.ev.n_sta, wc_min, np.inf).sum()
    if new.has_assoc and prod.has_assoc:
        ev_gain = (len(new.ev) - len(prod.ev)) / max(len(prod.ev), 1)
        wc_gain = (n_a - n_b) / max(n_b, 1)
        if ev_gain > 0.10 and wc_gain < 0.05:
            print(f"\n  >> The event count is up {ev_gain:+.0%} but the >={wc_min}-station count is "
                  f"{wc_gain:+.0%}.\n     The gain is marginal detections, not better locations. "
                  f"This is the 2020-01-25 pattern.")


def section_by_class(new: Pool, prod: Pool, classes):
    print("\n" + "=" * 78)
    print("2. BY STATION-COUNT CLASS  (the headline metric is NOT the total)")
    print("=" * 78)
    if not (new.has_assoc and prod.has_assoc):
        print("  skipped - needs the assoc-pick file (catalogs/pyocto_picks_<label>.csv) "
              "for both pools")
        return
    rows = []
    for label, lo, hi in classes:
        a = new.ev[class_mask(new.ev.n_sta, lo, hi)]
        b = prod.ev[class_mask(prod.ev.n_sta, lo, hi)]
        rows.append([
            f"{label} sta",
            len(b), len(a), delta_str(float(len(a)), float(len(b)), 0),
            fmt(med(b.n_picks), 1), fmt(med(a.n_picks), 1),
            fmt(med(b.rms), 3), fmt(med(a.rms), 3),
            fmt(med(b.n_sta), 1), fmt(med(a.n_sta), 1),
        ])
    print(table(rows, ["class", "n prod", "n new", "delta n",
                       "pk/ev prod", "pk/ev new",
                       "RMS prod", "RMS new",
                       "sta prod", "sta new"]))


def section_nlloc(new: Pool, prod: Pool, classes, wc_min: int):
    print("\n" + "=" * 78)
    print(f"3. NLLoc SIGMA  (stage 5) - THE METRIC THAT DECIDES THE PAPER")
    print("=" * 78)
    if new.nlloc is None or prod.nlloc is None:
        missing = [p.name for p in (new, prod) if p.nlloc is None]
        print(f"  skipped - no usable NLLoc catalogue for: {', '.join(missing)}")
        print(f"  Run scripts 28-31 on both labels, then re-run this with --new-nlloc/--prod-nlloc")
        print(f"  (or just --*-label, which looks for catalogs/nlloc_<label>.csv).")
        return

    sig = ["sigma_x_km", "sigma_y_km", "sigma_z_km"]

    def block(title, a, b):
        print(f"\n  {title}   [prod n={len(b)}, new n={len(a)}]")
        rows = []
        for c in sig + ["rms_s", "gap_deg", "n_phases", "depth_km"]:
            if c not in a.columns or c not in b.columns:
                continue
            ma, mb = med(a[c]), med(b[c])
            nd = 3 if c in sig + ["rms_s"] else 1
            rows.append([c, fmt(mb, nd), fmt(ma, nd), delta_str(ma, mb, nd)])
        print("    " + table(rows, ["median", "prod", "new", "delta"]).replace("\n", "\n    "))

    block("ALL located events", new.nlloc, prod.nlloc)

    # The one that matters: restricted to well-constrained events.
    if new.nlloc["n_sta"].notna().any() and prod.nlloc["n_sta"].notna().any():
        a = new.nlloc[class_mask(new.nlloc.n_sta, wc_min, np.inf)]
        b = prod.nlloc[class_mask(prod.nlloc.n_sta, wc_min, np.inf)]
        block(f"WELL-CONSTRAINED SUBSET (>={wc_min} stations)  <-- decide on this", a, b)
        if len(a) and len(b):
            dz = med(a.sigma_z_km) - med(b.sigma_z_km)
            dn = len(a) - len(b)
            print(f"\n    verdict: sigma_z {delta_str(med(a.sigma_z_km), med(b.sigma_z_km), 3)}, "
                  f"well-constrained count {dn:+d}")
            if dz >= -0.01 and abs(dn) <= 0.05 * max(len(b), 1):
                print("    >> sigma flat AND the well-constrained count flat: the pool change "
                      "buys\n       nothing for the paper's catalogue, only a detection-rate "
                      "argument.")
    else:
        print(f"\n  ! cannot restrict to >={wc_min} stations - no assoc-pick file joined. "
              f"Falling back to NLLoc n_phases.")
        a = new.nlloc[new.nlloc.n_phases >= wc_min]
        b = prod.nlloc[prod.nlloc.n_phases >= wc_min]
        block(f"n_phases >= {wc_min} (weaker proxy for well-constrained)", a, b)

    # per station class
    if new.nlloc["n_sta"].notna().any() and prod.nlloc["n_sta"].notna().any():
        print("\n  median sigma by station class")
        rows = []
        for label, lo, hi in classes:
            a = new.nlloc[class_mask(new.nlloc.n_sta, lo, hi)]
            b = prod.nlloc[class_mask(prod.nlloc.n_sta, lo, hi)]
            rows.append([f"{label} sta", len(b), len(a)]
                        + [fmt(med(b[c]), 3) for c in sig]
                        + [fmt(med(a[c]), 3) for c in sig])
        print("  " + table(rows, ["class", "n prod", "n new",
                                  "sx prod", "sy prod", "sz prod",
                                  "sx new", "sy new", "sz new"]).replace("\n", "\n  "))

    # quality tiers
    print("\n  quality-tier counts (thresholds from scripts/40_filter_nlloc_reliable.py)")
    rows = []
    for name, fn in NLLOC_TIERS.items():
        try:
            na, nb = int(fn(new.nlloc).sum()), int(fn(prod.nlloc).sum())
        except Exception:
            continue
        rows.append([name, nb, na, delta_str(float(na), float(nb), 0)])
    if rows:
        print("  " + table(rows, ["tier", "prod", "new", "delta"]).replace("\n", "\n  "))


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _time_candidates(ta, tb, t_tol):
    lo = np.searchsorted(tb, ta - t_tol, side="left")
    hi = np.searchsorted(tb, ta + t_tol, side="right")
    return [(i, j) for i in range(len(ta)) for j in range(lo[i], hi[i])]


def match_events(new: Pool, prod: Pool, t_tol, d_tol, calibrate=True, cal_dt=0.5):
    """Greedy 1:1 match on origin time + epicentral proximity.

    Candidates are all pairs within `t_tol` seconds AND `d_tol` km; they are
    consumed best-first on a normalised (dt, dist) score so each event is used
    once.

    THE x/y ORIGIN TRAP.  `17_pyocto_associate.py` builds its local tmerc CRS as
    `lat_0 = stations.latitude.mean()` over *the stations that have picks*.  Two
    pools with different station coverage therefore get DIFFERENT map origins,
    and their `x`/`y` columns are not on the same grid.  On the 2020-01-25
    smoketest the shift is 19.9 km - larger than any sane match radius, so a
    naive x/y comparison matches zero events.

    So: use lat/lon (haversine) when both catalogues carry them; otherwise use
    x/y after estimating the constant offset from pairs that are within
    `cal_dt` seconds of each other (those are all but certainly the same
    physical event) and subtracting it.
    """
    a, b = new.ev, prod.ev
    ta, tb = a["t"].to_numpy(float), b["t"].to_numpy(float)
    use_ll = a["lat"].notna().all() and b["lat"].notna().all()
    use_xy = (not use_ll) and a["x"].notna().all() and b["x"].notna().all()
    off = (0.0, 0.0)

    if use_ll:
        la, lo_a = a["lat"].to_numpy(float), a["lon"].to_numpy(float)
        lb, lo_b = b["lat"].to_numpy(float), b["lon"].to_numpy(float)
        mode = "lat/lon haversine"
    elif use_xy:
        xa, ya = a["x"].to_numpy(float), a["y"].to_numpy(float)
        xb, yb = b["x"].to_numpy(float), b["y"].to_numpy(float)
        mode = "projected x/y (km)"
        if calibrate:
            cal = _time_candidates(ta, tb, cal_dt)
            if len(cal) >= 5:
                off = (float(np.median([xa[i] - xb[j] for i, j in cal])),
                       float(np.median([ya[i] - yb[j] for i, j in cal])))
                mode += f", origin de-biased by ({off[0]:+.2f}, {off[1]:+.2f}) km"
            else:
                mode += ", NOT de-biased (too few close-in-time pairs to calibrate)"
    else:
        mode = "origin time only (no usable coordinates)"

    cand = []
    for i, j in _time_candidates(ta, tb, t_tol):
        dt = abs(ta[i] - tb[j])
        if use_ll:
            dist = float(_haversine_km(la[i], lo_a[i], lb[j], lo_b[j]))
        elif use_xy:
            dist = float(np.hypot(xa[i] - xb[j] - off[0], ya[i] - yb[j] - off[1]))
        else:
            dist = np.nan
        if np.isfinite(dist) and dist > d_tol:
            continue
        score = (dt / t_tol) ** 2 + ((dist / d_tol) ** 2 if np.isfinite(dist) else 0.0)
        cand.append((score, i, j, dt, dist))

    cand.sort(key=lambda c: c[0])
    used_a, used_b, pairs = set(), set(), []
    for score, i, j, dt, dist in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, dt, dist))
    return pairs, used_a, used_b, mode, off


def section_match(new: Pool, prod: Pool, t_tol, d_tol, wc_min, out_dir=None, calibrate=True):
    print("\n" + "=" * 78)
    print(f"4. EVENT MATCHING  (|dt| <= {t_tol}s and epicentral <= {d_tol} km)")
    print("=" * 78)
    pairs, used_a, used_b, mode, off = match_events(new, prod, t_tol, d_tol, calibrate)
    print(f"  distance basis: {mode}")
    if float(np.hypot(*off)) > 1.0:
        print(f"  ! the two runs have DIFFERENT map origins ({np.hypot(*off):.1f} km apart): "
              f"script 17 sets\n    lat_0/lon_0 from the mean of the stations that have picks, and "
              f"the pools differ in\n    station coverage. Raw x/y are not comparable; the offset "
              f"above has been removed.")
    n_common = len(pairs)
    print(f"  common to both      : {n_common}")
    print(f"  unique to new  [{new.name}] : {len(new.ev) - n_common}")
    print(f"  unique to prod [{prod.name}]: {len(prod.ev) - n_common}")
    if len(prod.ev):
        print(f"  recovery of prod catalogue: {n_common/len(prod.ev):.1%}")
    if not pairs:
        return

    ia = [p[0] for p in pairs]
    ib = [p[1] for p in pairs]
    m = pd.DataFrame({
        "dt_s": [p[2] for p in pairs],
        "dist_km": [p[3] for p in pairs],
        "new_key": new.ev.key.to_numpy()[ia],
        "prod_key": prod.ev.key.to_numpy()[ib],
        "sta_new": new.ev.n_sta.to_numpy()[ia],
        "sta_prod": prod.ev.n_sta.to_numpy()[ib],
        "picks_new": new.ev.n_picks.to_numpy()[ia],
        "picks_prod": prod.ev.n_picks.to_numpy()[ib],
        "rms_new": new.ev.rms.to_numpy()[ia],
        "rms_prod": prod.ev.rms.to_numpy()[ib],
    })
    print(f"\n  matched-pair separation: median dt {med(m.dt_s):.3f}s, "
          f"median dist {fmt(med(m.dist_km), 2)} km")

    print("\n  did COMMON events get better constrained?")
    rows = []
    for label, cn, cp, nd in [("stations/event", "sta_new", "sta_prod", 1),
                              ("picks/event", "picks_new", "picks_prod", 1),
                              ("post-fit RMS s", "rms_new", "rms_prod", 3)]:
        d = (m[cn] - m[cp]).dropna()
        if not len(d):
            continue
        rows.append([label, fmt(med(m[cp]), nd), fmt(med(m[cn]), nd),
                     fmt(float(d.median()), nd),
                     f"{(d > 0).sum()} up / {(d == 0).sum()} same / {(d < 0).sum()} down"])
    if rows:
        print("  " + table(rows, ["metric", "prod med", "new med", "median delta",
                                  "per-event direction"]).replace("\n", "\n  "))

    # where do the unique events sit?
    if new.has_assoc and prod.has_assoc:
        uniq_a = new.ev.loc[~new.ev.index.isin(used_a)]
        uniq_b = prod.ev.loc[~prod.ev.index.isin(used_b)]
        print("\n  station counts of the UNIQUE events (are the extras marginal?)")
        rows = []
        for nm, d in (("unique to new", uniq_a), ("unique to prod", uniq_b)):
            rows.append([nm, len(d), fmt(med(d.n_sta), 1),
                         int(class_mask(d.n_sta, wc_min, np.inf).sum()),
                         fmt(med(d.rms), 3)])
        print("  " + table(rows, ["set", "n", "median sta",
                                  f">={wc_min} sta", "median RMS"]).replace("\n", "\n  "))

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "matched_pairs.csv"
        m.to_csv(p, index=False)
        print(f"\n  wrote {p}")


# --------------------------------------------------------------------------
def resolve(label, explicit, kind):
    if explicit:
        return explicit
    if label:
        return REPO / "catalogs" / f"{kind}_{label}.csv"
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Compare two pick pools at the catalogue level (runbook step 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every metric is broken down by station-count class; NLLoc sigma is "
               "reported on the well-constrained subset, because a totals-only "
               "comparison hides marginal-detection inflation.")
    ap.add_argument("--new-label", help="label of the candidate pool, e.g. newpool_2020_01")
    ap.add_argument("--prod-label", help="label of the baseline pool, e.g. prodpool_2020_01")
    ap.add_argument("--new-events"), ap.add_argument("--new-picks"), ap.add_argument("--new-nlloc")
    ap.add_argument("--prod-events"), ap.add_argument("--prod-picks"), ap.add_argument("--prod-nlloc")
    ap.add_argument("--station-classes", default="3-5,6-7,8+",
                    help="station-count classes, default '3-5,6-7,8+'")
    ap.add_argument("--well-constrained-min", type=int, default=8,
                    help="stations required to count as well constrained (default 8)")
    ap.add_argument("--match-time-tol", type=float, default=3.0,
                    help="origin-time tolerance in s for matching (default 3.0)")
    ap.add_argument("--match-dist-tol", type=float, default=15.0,
                    help="epicentral tolerance in km for matching (default 15.0). pyocto grid "
                         "locations scatter ~5 km between pools even for identical events.")
    ap.add_argument("--no-origin-calibration", action="store_true",
                    help="do not de-bias the x/y map-origin difference between the two runs "
                         "(see match_events docstring; you almost always want the de-biasing)")
    ap.add_argument("--out-dir", help="write matched_pairs.csv / per-event tables here")
    args = ap.parse_args()

    assert_nanosecond_sanity()

    if not (args.new_events or args.new_label) or not (args.prod_events or args.prod_label):
        ap.error("give --new-label/--prod-label, or --new-events/--prod-events")

    print("loading ...")
    new = Pool(args.new_label or Path(args.new_events).stem,
               resolve(args.new_label, args.new_events, "pyocto_events"),
               resolve(args.new_label, args.new_picks, "pyocto_picks"),
               resolve(args.new_label, args.new_nlloc, "nlloc"))
    prod = Pool(args.prod_label or Path(args.prod_events).stem,
                resolve(args.prod_label, args.prod_events, "pyocto_events"),
                resolve(args.prod_label, args.prod_picks, "pyocto_picks"),
                resolve(args.prod_label, args.prod_nlloc, "nlloc"))

    classes = parse_classes(args.station_classes)
    wc = args.well_constrained_min

    span = pd.to_datetime(
        [min(new.ev.t.min(), prod.ev.t.min()), max(new.ev.t.max(), prod.ev.t.max())],
        unit="s", utc=True)
    print(f"\nPOOL COMPARISON   new=[{new.name}]  vs  prod=[{prod.name}]")
    print(f"time span {span[0]:%Y-%m-%d %H:%M} .. {span[1]:%Y-%m-%d %H:%M} UTC")

    section_headline(new, prod, wc)
    section_by_class(new, prod, classes)
    section_nlloc(new, prod, classes, wc)
    section_match(new, prod, args.match_time_tol, args.match_dist_tol, wc, args.out_dir,
                  calibrate=not args.no_origin_calibration)

    print("\n" + "=" * 78)
    print("runbook criteria: events up-or-flat-with-better-constraint | picks/event up |")
    print("RMS flat-or-lower | stations/event up | NLLoc sigma DOWN on the >=%d-station" % wc)
    print("subset (section 3) - that last row is the one that decides the paper.")
    print("=" * 78)


if __name__ == "__main__":
    main()
