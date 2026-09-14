"""Example-event waveform figures for the Methods / Phase-picking section.

Five PNGs plus a caption file, all written to `notes/figures/final/`:

    G16_examples_overview.png          4 events side by side, nearest 6 stations
    G16a..G16d_<eventtag>.png          one full-size record section per event

    PYTHONPATH=src python3 scripts/72_example_waveforms.py

The figure belongs BEFORE association in the narrative, so every trace carries
four layers of pick information, in this order of visual weight:

  1. RAW picker output -- every pick the two production pickers emitted inside
     the plotted window, straight from the per-station daily CSVs
     (`catalogs/picks_pn_diting/<NET.STA>/<YYYY-DDD>.csv` and
     `catalogs/picks_pnlight_obs/...`, the pool `scripts/17f_pyocto_year_newpool.sh`
     associated: `picks_pn_diting:PS,picks_pnlight_obs:PS`).  Drawn as faded
     dotted ticks coloured by picker; P occupies the upper half of the trace
     lane and S the lower half, so both phases are visible on both panels
     without a second colour axis.  These are NOT filtered by probability --
     they are the pool exactly as pyocto sees it, including the doubles that
     the two pickers make of the same arrival.
  2. ASSOCIATED picks -- the subset pyocto attached to this event
     (`catalogs/pyocto_picks_year_newpool_no_shots.csv`), solid, P blue / S orange.
  3. MANUAL picks -- the analyst's own arrivals for the same event
     (`catalogs/manual_pick_recall_year_newpool.csv`), black dashed.
  4. NLLoc PREDICTED arrivals from the final v6 location (`nlloc/output/year_v6/
     shard_*/loc.*.grid0.loc.hyp`), thin grey, deliberately secondary.

Each event's raw-in-window vs associated counts are printed and repeated in the
caption, which is the number the Methods text quotes.

Layout per event: left column = vertical component (ELZ / HHZ) with the P
layers, right column = one horizontal (SL1/SL2 or HH1/HH2, whichever has the
larger amplitude in the S window) with the S layers.  Traces are sorted by
epicentral distance, nearest at the top, and the window runs from 1 s before the
first associated P to 2 s after the last associated S.

Clock and timing conventions
----------------------------
* 3-20 Hz zero-phase Butterworth via `bransfield_eq.xcfilter.bandpass`, applied
  once to a padded segment (>= 20 s each side) and then trimmed, so no filter
  transient reaches the plotted window.
* ZX.BRA05's OBS clock ran 0.167 s fast for the whole deployment.  Every pick
  pool on disk is already corrected (`scripts/apply_S1_corrections.py`, markers
  `.bra05_clock_corrected`) and the manual `t` column is corrected in
  `scripts/54_manual_pick_recall.py`, but the WAVEFORMS are not -- so BRA05
  traces are read 0.167 s late in file time and relabelled to true UTC here.
* Pick times are read as epoch floats where the column already is one, and via
  `bransfield_eq.timeutil.epoch_seconds` where it is an ISO string.  No
  `.astype("int64")` on a datetime anywhere (pandas 3 unit trap).

Nothing outside `notes/figures/final/` is written.
"""
from __future__ import annotations

import os

# <= 4 cores (pod is CPU-capped; see memory note jupyterhub_pod_memory_limit)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import datetime as dt
import importlib.util
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bransfield_eq import xcfilter                                   # noqa: E402
from bransfield_eq.timeutil import (assert_nanosecond_sanity,        # noqa: E402
                                    epoch_seconds)

# Reuse the channel chooser from the cross-correlation stage rather than a
# second copy: the two OBS families do not share a band code (ELZ 200 Hz +
# SL1/SL2 100 Hz vs HHZ + HH1/HH2 100 Hz), and an earlier local re-derivation of
# this rule silently dropped every horizontal on 14 stations (see the docstring
# of scripts/63_xcorr_dtcc.py::choose_channels).
_spec = importlib.util.spec_from_file_location(
    "_xcorr63", REPO / "scripts" / "63_xcorr_dtcc.py")
_xc63 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_xc63)
choose_channels = _xc63.choose_channels

OUT = REPO / "notes" / "figures" / "final"
WAVE = REPO / "data" / "waveforms"
CAT_CSV = REPO / "catalogs" / "nlloc_year_v6_strict.csv"
ASSOC_CSV = REPO / "catalogs" / "pyocto_picks_year_newpool_no_shots.csv"
MANUAL_CSV = REPO / "catalogs" / "manual_pick_recall_year_newpool.csv"
STA_CSV = REPO / "catalogs" / "station_geometry.csv"
HYP_DIR = REPO / "nlloc" / "output" / "year_v6"

# The associated pool, in pyocto's own order (scripts/17f_pyocto_year_newpool.sh)
RAW_POOLS = [("picks_pn_diting", "PhaseNet-DiTing"),
             ("picks_pnlight_obs", "PickBlue-PhaseNetLight")]

# ZX.BRA05 clock ran +0.167 s fast; waveforms on disk are NOT corrected.
BRA05_CLOCK_S = 0.167

CALDERA = (-62.4413, -58.44)     # NLLoc TRANS SIMPLE origin, as in scripts/69
TRUSTED_MANUAL_FROM = "2019-10-01"   # analyst-trusted window (scripts/54)
MATCH_OT_S = 3.0                 # manual event <-> catalogue origin tolerance
PICK_MATCH_S = 0.5               # auto <-> manual pairing tolerance (scripts/54)

# Okabe-Ito, colour-blind safe (same palette as scripts/69_final_figures.py)
C = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
         yellow="#F0E442", blue="#0072B2", vermillion="#D55E00",
         purple="#CC79A7", grey="#9a9a9a")
POOL_COLOR = {"picks_pn_diting": C["green"], "picks_pnlight_obs": C["purple"]}
PHASE_COLOR = {"P": C["blue"], "S": C["orange"]}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": 0.9, "axes.grid": True, "grid.alpha": 0.18,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
})

KM_PER_DEG_LAT = 111.195


# ------------------------------------------------------------------ helpers
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def utc(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc)


def daykeys(t0: float, t1: float):
    """UTC (year, doy) pairs spanned by [t0, t1], so a window can cross midnight."""
    out, d = [], utc(t0).replace(hour=0, minute=0, second=0, microsecond=0)
    end = utc(t1)
    while d <= end:
        out.append((d.year, d.timetuple().tm_yday))
        d += dt.timedelta(days=1)
    return out


def clock_shift_s(net: str, sta: str) -> float:
    """File time minus true UTC for this station (BRA05 only)."""
    return BRA05_CLOCK_S if (net, sta) == ("ZX", "BRA05") else 0.0


# ------------------------------------------------------------------ catalogue
def load_tables():
    assert_nanosecond_sanity()
    cat = pd.read_csv(CAT_CSV)
    cat["ot"] = epoch_seconds(cat["origin_time"]).to_numpy()

    assoc = pd.read_csv(ASSOC_CSV, usecols=["event_idx", "station", "time",
                                            "phase", "prob"])
    # `time` is already epoch float64 in this CSV; fall back for safety only.
    if not pd.api.types.is_numeric_dtype(assoc["time"]):
        assoc["time"] = epoch_seconds(assoc["time"])
    assoc["time"] = assoc["time"].astype(float)

    man = pd.read_csv(MANUAL_CSV)
    man["t"] = man["t"].astype(float)          # epoch s, BRA05 already corrected

    sta = pd.read_csv(STA_CSV)
    sta["key"] = sta.network + "." + sta.station
    return cat, assoc, man, sta


def match_manual_events(cat: pd.DataFrame, assoc: pd.DataFrame,
                        man: pd.DataFrame) -> pd.DataFrame:
    """Manual event_id -> catalogue event_idx: origin time within MATCH_OT_S and
    at least 2 stations picked by both the analyst and pyocto."""
    mev = (man.groupby("event_id")
              .agg(ot_iso=("origin_time", "first"), n_man=("pick_time", "size"))
              .reset_index())
    mev["ot"] = epoch_seconds(mev["ot_iso"]).to_numpy()
    sta_auto = assoc.groupby("event_idx")["station"].apply(set)
    sta_man = man.groupby("event_id")["sta_key"].apply(set)

    c = cat.sort_values("ot").reset_index(drop=True)
    cot = c["ot"].to_numpy()
    rows = []
    for r in mev.itertuples():
        j0 = int(np.searchsorted(cot, r.ot))
        best = None
        for j in (j0 - 1, j0, j0 + 1):
            if 0 <= j < len(c):
                d = abs(cot[j] - r.ot)
                if d <= MATCH_OT_S and (best is None or d < best[0]):
                    best = (d, j)
        if best is None:
            continue
        eidx = int(c.event_idx.values[best[1]])
        common = len(sta_auto.get(eidx, set()) & sta_man[r.event_id])
        if common >= 2:
            rows.append(dict(event_idx=eidx, event_id=r.event_id,
                             dt_ot_s=best[0], n_man=int(r.n_man),
                             n_common_sta=common))
    return pd.DataFrame(rows)


def select_events(cat, assoc, man, sta, verbose=True) -> dict:
    """Deterministic four-event selection, one per tier of the brief.

    (a) shallow caldera, (b) deeper under the array, (c) small / few phases with
    a near station, (d) analyst-trusted window with many manual picks.  Every
    tier requires a manual-pick match, and (a), (b) and (d) additionally require
    the auto and manual picks to disagree by > 0.1 s on at least one S, so the
    figure has something to show.  Tier (d) is forced shallower than 4 km so it
    does not duplicate tier (b)'s character.
    """
    mt = match_manual_events(cat, assoc, man)
    mt = mt.merge(cat, on="event_idx")

    gm = {r.key: (r.latitude, r.longitude) for r in sta.itertuples()}
    sta_auto = assoc.groupby("event_idx")["station"].apply(set)

    # largest |auto - manual| on an S pick, from the raw-pool residual column
    mS = man[(man.phase == "S") & man.hit_raw]
    max_s = mS.groupby("event_id").res_raw.apply(lambda x: float(np.abs(x).max()))
    mt["max_s_diff_s"] = mt.event_id.map(max_s)

    mt["d_caldera_km"] = haversine_km(mt.lat.values, mt.lon.values, *CALDERA)
    near = []
    for r in mt.itertuples():
        d = [haversine_km(r.lat, r.lon, *gm[s])
             for s in sta_auto.get(r.event_idx, set()) if s in gm]
        near.append(min(d) if d else np.nan)
    mt["nearest_sta_km"] = near
    mt["ot_ts"] = pd.to_datetime(mt.origin_time, utc=True, format="ISO8601")

    inform = mt.max_s_diff_s > 0.1
    chosen, used = {}, set()

    def take(letter, sub, by, asc, label):
        sub = sub[~sub.event_idx.isin(used)]
        if sub.empty:
            raise SystemExit(f"selection tier {letter} ({label}) is empty")
        row = sub.sort_values(by, ascending=asc).iloc[0]
        used.add(int(row.event_idx))
        chosen[letter] = row
        if verbose:
            print(f"  ({letter}) {label}: event_idx {int(row.event_idx)}  "
                  f"{row.origin_time}  depth_bsf {row.depth_bsf_km:.2f} km  "
                  f"Nphs {int(row.n_phases)}  rms {row.rms_s:.3f} s  "
                  f"manual {int(row.n_man)}  maxS|auto-man| "
                  f"{row.max_s_diff_s:.3f} s")

    take("a", mt[inform & mt.depth_bsf_km.between(0.5, 1.5)
                 & (mt.n_phases >= 8) & (mt.d_caldera_km <= 3.0)],
         ["n_man", "n_phases"], [False, False], "shallow caldera event")
    take("b", mt[inform & mt.depth_bsf_km.between(4.0, 7.0)
                 & (mt.n_phases >= 8) & (mt.d_caldera_km <= 10.0)],
         ["n_man", "n_phases"], [False, False], "deeper event under the array")
    take("c", mt[(mt.max_s_diff_s > 0.1) & mt.n_phases.between(6, 8)
                 & (mt.nearest_sta_km < 8.0)],
         ["nearest_sta_km"], [True], "small event, few phases, near station")
    take("d", mt[(mt.ot_ts >= TRUSTED_MANUAL_FROM) & (mt.n_man >= 10)
                 & inform & (mt.depth_bsf_km < 4.0)],
         ["n_man", "n_common_sta"], [False, False],
         f"analyst-trusted window (>= {TRUSTED_MANUAL_FROM}), many manual picks")
    return chosen


# ------------------------------------------------------------------ NLLoc hyp
GEO_RE = re.compile(
    r"GEOGRAPHIC\s+OT\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)")


def index_hyps(root: Path) -> dict:
    """{'YYYYMMDD.HHMMSS': [paths]} over every shard (~80k files, ~0.2 s)."""
    idx = defaultdict(list)
    for shard in sorted(p for p in root.iterdir() if p.is_dir()):
        for e in os.scandir(shard):
            if e.name.startswith("loc.20") and e.name.endswith("grid0.loc.hyp"):
                idx[e.name[4:19]].append(Path(e.path))
    return idx


def parse_hyp(path: Path):
    """(origin epoch s, [phase dicts]) from one NLLoc .hyp file.

    PHASE columns (v6.04 header): 0 station, 4 phase, 6 date, 7 hrmn, 8 sec,
    15 TTpred, 16 Res, 21 SDist, 26 Tcorr.  The predicted arrival in the
    observation frame is OT + TTpred + Tcorr, because NLLoc applies the LOCDELAY
    station term as obs - delay (verified against the printed Res).
    """
    text = path.read_text()
    g = GEO_RE.search(text)
    if g is None:
        return None, []
    yr, mo, dy, hr, mn = (int(g.group(i)) for i in range(1, 6))
    # NLLoc can print a negative or >= 60 seconds field; add it as a Timedelta.
    ot = (pd.Timestamp(yr, mo, dy, hr, mn, tz="UTC")
          + pd.Timedelta(seconds=float(g.group(6)))).timestamp()
    phases, inblock = [], False
    for line in text.split("\n"):
        if line.startswith("PHASE ID"):
            inblock = True
            continue
        if line.startswith("END_PHASE"):
            break
        if not inblock:
            continue
        f = line.split()
        if len(f) < 27:
            continue
        try:
            obs = (pd.Timestamp(f[6][:4] + "-" + f[6][4:6] + "-" + f[6][6:8]
                                + "T" + f[7][:2] + ":" + f[7][2:4], tz="UTC")
                   + pd.Timedelta(seconds=float(f[8]))).timestamp()
            phases.append(dict(sta=f[0], phase=f[4], obs=obs,
                               tt=float(f[15]), res=float(f[16]),
                               dist=float(f[21]), tcorr=float(f[26]),
                               pred=ot + float(f[15]) + float(f[26])))
        except (ValueError, IndexError):
            continue
    return ot, phases


def find_hyp(hyp_idx: dict, ot: float, apicks: pd.DataFrame):
    """The .hyp of this event: origin time within 1 s of the catalogue's and the
    most picks in common with the associated set (station+phase within 50 ms).

    Matching on content rather than position is deliberate -- the positional
    join is what scrambled the v2 catalogue (see scripts/31 docstring)."""
    want = {(s.split(".")[-1], p, round(t, 1))
            for s, p, t in zip(apicks.station, apicks.phase, apicks.time)}
    best = (0, None, None, None)
    for off in range(-90, 91):
        k = utc(ot + off).strftime("%Y%m%d.%H%M%S")
        for path in hyp_idx.get(k, ()):
            h_ot, ph = parse_hyp(path)
            if h_ot is None or abs(h_ot - ot) > 1.0:
                continue
            n = sum(1 for p in ph
                    if (p["sta"], p["phase"], round(p["obs"], 1)) in want)
            if n > best[0]:
                best = (n, path, h_ot, ph)
    return best          # (n_common, path, origin, phases)


# ------------------------------------------------------------------ raw picks
def load_raw_picks(net_sta: str, t0: float, t1: float) -> pd.DataFrame:
    """Every pick both production pickers emitted for one station in [t0, t1].

    Read straight from the per-station daily CSVs with no probability cut, i.e.
    the pool exactly as `17_pyocto_associate.py::load_picks_for_pyocto` sees it
    before association (and before its cross-picker dedup).
    """
    rows = []
    for pool, _label in RAW_POOLS:
        for yr, doy in daykeys(t0, t1):
            f = REPO / "catalogs" / pool / net_sta / f"{yr}-{doy:03d}.csv"
            if not f.exists():
                continue
            try:
                d = pd.read_csv(f, usecols=["time", "phase", "prob"])
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
                continue
            if d.empty:
                continue
            d["t"] = epoch_seconds(d["time"]).to_numpy()
            d = d[(d.t >= t0) & (d.t <= t1)]
            if d.empty:
                continue
            d = d.assign(pool=pool, phase=d.phase.str.upper().str[0])
            rows.append(d[["pool", "t", "phase", "prob"]])
    if not rows:
        return pd.DataFrame(columns=["pool", "t", "phase", "prob"])
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------ waveforms
def load_window(task: dict):
    """Worker: one station's Z + horizontals over [t0, t1], 3-20 Hz, trimmed.

    The band-pass is applied to a padded segment and the pad is then cut away,
    so the plotted window contains no filter transient (xcfilter's rule).
    """
    import obspy

    net, sta = task["net"], task["sta"]
    t0, t1, pad = task["t0"], task["t1"], task["pad"]
    lo, hi = task["band"]
    shift = clock_shift_s(net, sta)      # file time = true time + shift
    sdir = WAVE / net / sta
    files = [sdir / f"{net}.{sta}.{yr}.{doy:03d}.mseed"
             for yr, doy in daykeys(t0 + shift, t1 + shift)]
    files = [f for f in files if f.exists() and f.stat().st_size > 0]
    if not files:
        return dict(key=task["key"], err="no waveform file", chans={})

    # channels from the header of the first day that has one
    chan_rate = {}
    for f in files:
        try:
            st = obspy.read(str(f), headonly=True)
        except Exception:
            continue
        chan_rate.update({tr.stats.channel: float(tr.stats.sampling_rate)
                          for tr in st})
        if chan_rate:
            break
    zc, hcs = choose_channels(chan_rate)
    codes = ([zc] if zc else []) + list(hcs)
    if not codes:
        return dict(key=task["key"], err=f"no usable channel in {sorted(chan_rate)}",
                    chans={})

    a0 = obspy.UTCDateTime(t0 + shift - pad)
    a1 = obspy.UTCDateTime(t1 + shift + pad)
    out, missing = {}, []
    for code in codes:
        tr = None
        try:
            st = obspy.Stream()
            for f in files:
                st += obspy.read(str(f), sourcename="*.*.*." + code,
                                 starttime=a0, endtime=a1)
            st = obspy.Stream([t for t in st if t.stats.channel == code])
            if len(st) == 0:
                missing.append(code)
                continue
            st.merge(method=1, fill_value=0, interpolation_samples=0)
            st.trim(a0, a1, pad=True, fill_value=0)
            tr = st[0]
        except Exception as e:                       # noqa: BLE001
            missing.append(f"{code} ({type(e).__name__})")
            continue
        d = np.asarray(tr.data)
        if np.ma.isMaskedArray(d):
            d = d.filled(0.0)
        d = d.astype(np.float64)
        if d.size < 32 or not np.isfinite(d).all() or np.allclose(d, 0.0):
            missing.append(f"{code} (flat/empty)")
            continue
        fs = float(tr.stats.sampling_rate)
        d = xcfilter.bandpass(d - d.mean(), lo, hi, fs)
        # absolute start of the array in TRUE UTC, then cut the pad away
        t_start = float(tr.stats.starttime.timestamp) - shift
        i0 = max(int(round((t0 - t_start) * fs)), 0)
        i1 = min(int(round((t1 - t_start) * fs)) + 1, d.size)
        if i1 - i0 < 8:
            missing.append(f"{code} (window outside data)")
            continue
        out[code] = dict(data=np.asarray(d[i0:i1], dtype=np.float32),
                         fs=fs, t0=t_start + i0 / fs,
                         role="Z" if code == zc else "H")
    return dict(key=task["key"], err=None if out else "no channel decoded",
                chans=out, missing=missing)


# ------------------------------------------------------------------ assembly
def build_event(letter, row, assoc, man, mt_row, sta_df, hyp_idx, args):
    """Everything one panel needs: traces, four pick layers, counts."""
    eidx = int(row.event_idx)
    ot = float(row.ot)
    ap = assoc[assoc.event_idx == eidx].copy()
    mp = man[man.event_id == mt_row.event_id].copy() if mt_row is not None \
        else man.iloc[0:0].copy()

    # ---- window: 1 s before the first associated P, 2 s after the last S
    p_t = ap.time[ap.phase == "P"]
    s_t = ap.time[ap.phase == "S"]
    t0 = float(p_t.min() if len(p_t) else ap.time.min()) - 1.0
    t1 = float(s_t.max() if len(s_t) else ap.time.max()) + 2.0
    if not np.isfinite(t1 - t0) or t1 - t0 < 4.0:
        t1 = t0 + 4.0

    # ---- NLLoc predicted arrivals
    n_hyp, hyp_path, hyp_ot, hyp_ph = find_hyp(hyp_idx, ot, ap)
    pred = defaultdict(dict)
    for p in (hyp_ph or []):
        pred[p["sta"]][p["phase"]] = p

    # ---- stations: everything with an associated or a manual pick
    gm = {r.key: (r.latitude, r.longitude) for r in sta_df.itertuples()}
    keys = sorted(set(ap.station) | set(mp.sta_key))
    rows = []
    for k in keys:
        if k not in gm:
            continue
        net, bare = k.split(".")
        rows.append(dict(key=k, net=net, sta=bare,
                         dist_km=haversine_km(row.lat, row.lon, *gm[k])))
    rows.sort(key=lambda r: r["dist_km"])
    # A single far station would stretch the time axis until the near traces are
    # unreadable (event 30153 has one at 33 km, which makes a 15 s window for an
    # event whose first 8 arrivals are inside 3 s).  Drop those, and say so.
    far = [r for r in rows if r["dist_km"] > args.max_dist_km]
    rows = [r for r in rows if r["dist_km"] <= args.max_dist_km][:args.max_stations]

    for r in rows:
        r["assoc"] = {ph: float(g.time.iloc[0])
                      for ph, g in ap[ap.station == r["key"]].groupby("phase")}
        r["manual"] = {ph: float(g.t.iloc[0])
                       for ph, g in mp[mp.sta_key == r["key"]].groupby("phase")}
        r["pred"] = {ph: v["pred"] for ph, v in pred.get(r["sta"], {}).items()}
        r["raw"] = load_raw_picks(r["key"], t0, t1)

    # window from the picks of the stations actually shown
    shown = set(r["key"] for r in rows)
    ap_shown = ap[ap.station.isin(shown)]
    if len(ap_shown):
        p_t = ap_shown.time[ap_shown.phase == "P"]
        s_t = ap_shown.time[ap_shown.phase == "S"]
        t0 = float(p_t.min() if len(p_t) else ap_shown.time.min()) - 1.0
        t1 = float(s_t.max() if len(s_t) else ap_shown.time.max()) + 2.0
        for r in rows:
            r["raw"] = load_raw_picks(r["key"], t0, t1)

    # associated picks ON THE TRACES SHOWN -- the honest denominator for the
    # raw-vs-associated statement, since the total includes far stations that
    # the distance cut removed from the figure
    ap_sh = ap[ap.station.isin(shown) & ap.time.between(t0, t1)]

    return dict(letter=letter, event_idx=eidx, row=row, ot=ot, t0=t0, t1=t1,
                far=[(r["key"], r["dist_km"]) for r in far],
                n_assoc_shown=len(ap_sh),
                n_assoc_shown_p=int((ap_sh.phase == "P").sum()),
                n_assoc_shown_s=int((ap_sh.phase == "S").sum()),
                stations=rows, n_assoc=len(ap), n_manual=len(mp),
                n_assoc_p=int((ap.phase == "P").sum()),
                n_assoc_s=int((ap.phase == "S").sum()),
                hyp=str(hyp_path.relative_to(REPO)) if hyp_path else None,
                n_hyp_common=n_hyp, hyp_ot=hyp_ot, n_pred=len(hyp_ph or []),
                manual_event_id=(mt_row.event_id if mt_row is not None else None))


def attach_waveforms(events, workers: int, band):
    """Load every (event, station) window in a small process pool."""
    tasks, problems = [], []
    for ev in events:
        for i, s in enumerate(ev["stations"]):
            tasks.append(dict(key=(ev["event_idx"], i), net=s["net"],
                              sta=s["sta"], t0=ev["t0"] - 0.5, t1=ev["t1"] + 0.5,
                              pad=20.0, band=band))
    got = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(load_window, tasks, chunksize=1):
            got[res["key"]] = res
    for ev in events:
        for i, s in enumerate(ev["stations"]):
            r = got.get((ev["event_idx"], i), {})
            s["chans"] = r.get("chans", {})
            if r.get("err"):
                problems.append(f"{ev['event_idx']} {s['key']}: {r['err']}")
            for m in r.get("missing", []):
                problems.append(f"{ev['event_idx']} {s['key']}: channel {m} "
                                f"not read")
    return problems


def pick_channels(s, ev):
    """(Z code, horizontal code) for one station: the vertical, and whichever
    horizontal has the larger amplitude inside the S window."""
    z = next((c for c, v in s["chans"].items() if v["role"] == "Z"), None)
    hs = [c for c, v in s["chans"].items() if v["role"] == "H"]
    if not hs:
        return z, None
    t_s = s["assoc"].get("S") or s["manual"].get("S") or s["pred"].get("S")
    best, bestamp = hs[0], -1.0
    for c in hs:
        v = s["chans"][c]
        d = v["data"]
        if t_s is not None:
            i0 = int((t_s - 0.5 - v["t0"]) * v["fs"])
            i1 = int((t_s + 1.5 - v["t0"]) * v["fs"])
            d = v["data"][max(i0, 0):max(i1, 1)]
        amp = float(np.max(np.abs(d))) if d.size else 0.0
        if amp > bestamp:
            best, bestamp = c, amp
    return z, best


# ------------------------------------------------------------------ drawing
def draw_trace(ax, s, code, ev, y, scale=0.40, color="#333333", lw=0.6):
    v = s["chans"].get(code)
    if v is None:
        ax.text(0.01, y, "no data", transform=ax.get_yaxis_transform(),
                fontsize=7, color=C["vermillion"], va="center")
        return False
    t = v["t0"] - ev["ot"] + np.arange(v["data"].size) / v["fs"]
    d = v["data"].astype(float)
    m = float(np.max(np.abs(d)))
    if m > 0:
        d = d / m
    ax.plot(t, y - d * scale, color=color, lw=lw, zorder=3)
    return True


def draw_picks(ax, s, ev, phase, y, h=0.40, raw=True):
    """The four layers for one trace lane, in increasing visual weight."""
    t_ref = ev["ot"]
    if raw and len(s["raw"]):
        for r in s["raw"].itertuples():
            col = POOL_COLOR.get(r.pool, C["grey"])
            # P in the upper half of the lane, S in the lower half
            y0, y1 = (y - h, y) if r.phase == "P" else (y, y + h)
            ax.plot([r.t - t_ref] * 2, [y0, y1], color=col, lw=1.1, ls=":",
                    alpha=0.6, zorder=2)
    tp = s["pred"].get(phase)
    if tp is not None:
        ax.plot([tp - t_ref] * 2, [y - h * 0.55, y + h * 0.55], color=C["grey"],
                lw=0.9, alpha=0.85, zorder=4)
    tm = s["manual"].get(phase)
    if tm is not None:
        ax.plot([tm - t_ref] * 2, [y - h, y + h], color="black", lw=1.3,
                ls=(0, (3, 2)), zorder=6)
    ta = s["assoc"].get(phase)
    if ta is not None:
        ax.plot([ta - t_ref] * 2, [y - h, y + h], color=PHASE_COLOR[phase],
                lw=1.7, zorder=5)
    if ta is not None and tm is not None and abs(ta - tm) <= PICK_MATCH_S:
        return (ta - tm) * 1000.0
    return None


def legend_handles(compact=False):
    h = [Line2D([], [], color=PHASE_COLOR["P"], lw=1.7, label="associated P"),
         Line2D([], [], color=PHASE_COLOR["S"], lw=1.7, label="associated S"),
         Line2D([], [], color="black", lw=1.3, ls=(0, (3, 2)),
                label="analyst manual pick"),
         Line2D([], [], color=C["grey"], lw=0.9, label="NLLoc predicted")]
    for pool, label in RAW_POOLS:
        h.append(Line2D([], [], color=POOL_COLOR[pool], lw=1.1, ls=":",
                        alpha=0.7, label=f"raw pool: {label}"))
    if not compact:
        h.append(Line2D([], [], color="none",
                        label="raw ticks: P above the trace, S below"))
    return h


def event_title(ev):
    r = ev["row"]
    return (f"event {ev['event_idx']}   {utc(ev['ot']).strftime('%Y-%m-%d %H:%M:%S')} UTC   "
            f"{r.lat:.4f}/{r.lon:.4f}   {r.depth_bsf_km:.2f} km below seafloor   "
            f"Nphs {int(r.n_phases)}   rms {r.rms_s:.3f} s")


def raw_counts(ev):
    """{(pool, phase): n} of raw picker picks inside the plotted window."""
    n = defaultdict(int)
    for s in ev["stations"]:
        for r in s["raw"].itertuples():
            n[(r.pool, r.phase)] += 1
    return n


def raw_total(rc) -> int:
    return int(sum(rc.values()))


def fig_event(ev, args):
    rows = ev["stations"]
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, max(4.6, 0.52 * n + 2.6)),
                             sharex=True, sharey=True)
    span = ev["t1"] - ev["t0"]
    labels, deltas = [], []
    for i, s in enumerate(rows):
        z, hcode = pick_channels(s, ev)
        for ax, code, phase in ((axes[0], z, "P"), (axes[1], hcode, "S")):
            ok = draw_trace(ax, s, code, ev, i)
            d = draw_picks(ax, s, ev, phase, i)
            if ok and d is not None:
                ax.text(0.995, i - 0.30, f"{d:+.0f} ms", ha="right", va="center",
                        transform=ax.get_yaxis_transform(), fontsize=7.5,
                        color=C["vermillion"], zorder=7)
            if ax is axes[1]:
                ax.text(0.005, i + 0.34, code or "-", transform=ax.get_yaxis_transform(),
                        fontsize=7, color="#666666", va="center")
            else:
                ax.text(0.005, i + 0.34, code or "-", transform=ax.get_yaxis_transform(),
                        fontsize=7, color="#666666", va="center")
        labels.append(f"{s['key']}  {s['dist_km']:.1f} km")
        deltas.append(s)
    for ax, t in zip(axes, ("vertical component - P", "horizontal component - S")):
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("time after origin (s)")
        ax.set_xlim(ev["t0"] - ev["ot"] - 0.02 * span, ev["t1"] - ev["ot"] + 0.02 * span)
        ax.axvline(0.0, color=C["sky"], lw=0.8, alpha=0.6, zorder=1)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels, fontsize=8.5)
    axes[0].set_ylim(n - 0.55, -0.9)
    rc = raw_counts(ev)
    # Place the two header lines and the legend in INCHES, not figure fractions:
    # the figure height varies with the number of traces, and a fixed fraction
    # made them overlap on the 5-trace event.
    H = fig.get_figheight()
    fig.suptitle(event_title(ev), fontsize=11.5, y=1 - 0.22 / H)
    fig.text(0.5, 1 - 0.52 / H,
             f"{raw_total(rc)} raw picker picks in this window on the {n} traces "
             f"shown -> {ev['n_assoc_shown']} of them associated to this event "
             f"({ev['n_assoc_shown_p']} P / {ev['n_assoc_shown_s']} S of the "
             f"event's {ev['n_assoc']} phases); "
             f"{ev['n_manual']} analyst picks; 3-20 Hz zero-phase",
             ha="center", fontsize=9, color="#444444")
    # second header line, so a long station list cannot stretch the figure
    top_pad = 0.72
    if ev["far"]:
        fig.text(0.5, 1 - 0.74 / H,
                 f"{len(ev['far'])} associated station(s) beyond "
                 f"{args.max_dist_km:g} km not shown: "
                 + ", ".join(f"{k} {d:.1f} km" for k, d in ev["far"]),
                 ha="center", fontsize=8, color="#666666")
        top_pad = 0.92
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.004))
    fig.tight_layout(rect=(0, 0.62 / H, 1, 1 - top_pad / H))
    tag = f"G16{ev['letter']}_ev{ev['event_idx']}_" \
          f"{utc(ev['ot']).strftime('%Y%m%dT%H%M')}.png"
    p = OUT / tag
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_overview(events, args):
    nmax = args.overview_stations
    fig, axes = plt.subplots(1, len(events), figsize=(15.0, 6.4), sharex=False)
    for ax, ev in zip(np.atleast_1d(axes), events):
        rows = ev["stations"][:nmax]
        # the compact panel shows fewer stations, so it gets its own (shorter)
        # window: 1 s before the first P and 2 s after the last S OF THESE ROWS
        tt = [t for r in rows for t in r["assoc"].values()]
        if tt:
            p_all = [r["assoc"]["P"] for r in rows if "P" in r["assoc"]]
            s_all = [r["assoc"]["S"] for r in rows if "S" in r["assoc"]]
            w0 = (min(p_all) if p_all else min(tt)) - 1.0
            w1 = (max(s_all) if s_all else max(tt)) + 2.0
        else:
            w0, w1 = ev["t0"], ev["t1"]
        labels = []
        for i, s in enumerate(rows):
            z, _ = pick_channels(s, ev)
            draw_trace(ax, s, z, ev, i, scale=0.40)
            draw_picks(ax, s, ev, "P", i)
            draw_picks(ax, s, ev, "S", i, raw=False)
            labels.append(f"{s['key'].split('.')[-1]} {s['dist_km']:.1f}")
        ax.set_xlim(w0 - ev["ot"] - 0.2, w1 - ev["ot"] + 0.2)
        ax.set_ylim(len(rows) - 0.4, -0.9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("time after origin (s)")
        r = ev["row"]
        ax.set_title(f"({ev['letter']}) event {ev['event_idx']}\n"
                     f"{utc(ev['ot']).strftime('%Y-%m-%d %H:%M')} UTC, "
                     f"{r.depth_bsf_km:.2f} km bsf\n"
                     f"Nphs {int(r.n_phases)}, rms {r.rms_s:.3f} s, "
                     f"{ev['n_manual']} manual", fontsize=9.5)
        ax.axvline(0.0, color=C["sky"], lw=0.8, alpha=0.6, zorder=1)
    fig.suptitle("Example events - vertical component, nearest "
                 f"{nmax} stations (3-20 Hz)", fontsize=12.5)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    p = OUT / "G16_examples_overview.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ captions
def write_captions(events, paths, problems):
    lines = ["# Figure captions - example events (Methods, phase picking)",
             "",
             "Generated by `scripts/72_example_waveforms.py`. Counts are "
             "recomputed from the pick pools at build time.", ""]
    tot_raw = sum(raw_total(raw_counts(ev)) for ev in events)
    tot_assoc = sum(ev["n_assoc_shown"] for ev in events)
    over = ("**G16 - Example events: raw picker output, associated picks and "
            "analyst picks (overview).** Vertical-component record sections of "
            "four representative earthquakes, nearest "
            f"{paths['overview_stations']} stations each, 3-20 Hz zero-phase "
            "(`src/bransfield_eq/xcfilter.py`), traces sorted by epicentral "
            "distance (station code and distance in km on the left axis) and "
            "individually normalised. Time is relative to the NLLoc v6 origin. "
            "Faded dotted ticks are every pick the two production pickers "
            "(PhaseNet-DiTing, PickBlue-PhaseNetLight) emitted inside the "
            "window, with no probability cut - P drawn above the trace, S "
            "below; solid blue/orange ticks are the subset pyocto associated to "
            "the event; black dashed ticks are the analyst's manual picks; thin "
            "grey ticks are the arrival times predicted by the final NLLoc v6 "
            "location. The compact panels use their own window (1 s before the "
            "first P and 2 s after the last S of the traces shown). "
            "Across the four events "
            f"{tot_raw} raw picks fall in the plotted windows and {tot_assoc} "
            "of them are associated to the event - the rest are the pickers' "
            "doubles of the same arrival (both pickers run on every station), "
            "picks of other phases, and false or unassociated detections. "
            "Panels (a)-(d) are shown full size, with the horizontal component "
            "and the S picks, in G16a-G16d.")
    lines += [over, ""]
    for ev in events:
        r = ev["row"]
        rc = raw_counts(ev)
        d_pool = " / ".join(
            f"{label} {rc[(pool, 'P')] + rc[(pool, 'S')]}"
            for pool, label in RAW_POOLS)
        deltas = []
        for s in ev["stations"]:
            for ph in ("P", "S"):
                a, m = s["assoc"].get(ph), s["manual"].get(ph)
                if a is not None and m is not None and abs(a - m) <= PICK_MATCH_S:
                    deltas.append(abs(a - m) * 1000)
        dtxt = (f"The {len(deltas)} arrivals picked both automatically and by "
                f"hand differ by a median of {np.median(deltas):.0f} ms "
                f"(largest {max(deltas):.0f} ms). "
                if deltas else "")
        lines.append(
            f"**G16{ev['letter']} - {paths['labels'][ev['letter']]} "
            f"(event {ev['event_idx']}).** "
            f"{utc(ev['ot']).strftime('%Y-%m-%d %H:%M:%S')} UTC, "
            f"{r.lat:.4f}/{r.lon:.4f}, {r.depth_km:.2f} km below sea level "
            f"({r.depth_bsf_km:.2f} km below the local seafloor), "
            f"Nphs {int(r.n_phases)}, rms {r.rms_s:.3f} s, "
            f"sigma_z {r.sigma_z_km:.2f} km, azimuthal gap {r.gap_deg:.0f} deg. "
            f"Left: vertical component (ELZ or HHZ) with the P layers; right: "
            f"the horizontal (SL1/SL2 or HH1/HH2) with the larger S amplitude, "
            f"with the S layers; the channel actually plotted is printed above "
            f"each trace. Markers as in G16. "
            f"{raw_total(rc)} raw picker picks fall in the "
            f"{ev['t1'] - ev['t0']:.1f} s window on the "
            f"{len(ev['stations'])} traces shown ({d_pool}); pyocto associated "
            f"{ev['n_assoc_shown']} of them to this event "
            f"({ev['n_assoc_shown_p']} P, {ev['n_assoc_shown_s']} S) out of the "
            f"{ev['n_assoc']} phases it associated in total, and the analyst "
            f"picked {ev['n_manual']} arrivals on this event. {dtxt}"
            + (f"{len(ev['far'])} associated station(s) beyond "
               f"{paths['max_dist_km']:g} km are not plotted ("
               + ", ".join(f"{k} at {d:.1f} km" for k, d in ev["far"])
               + "), so the near traces are not squeezed into the first "
                 "seconds of a long axis. " if ev["far"] else "") +
            f"Vermillion labels give the associated-minus-manual difference in "
            f"ms where both exist. NLLoc predicted arrivals are from "
            f"`{ev['hyp']}` ({ev['n_pred']} phases).")
        lines.append("")
    if problems:
        lines += ["**Traces that could not be read.** " + "; ".join(problems) + ".", ""]
    p = OUT / "captions_3.md"
    p.write_text("\n".join(lines))
    return p


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=None,
                    help="override the automatic choice, e.g. "
                         "a=82625,b=30153,c=66766,d=97120")
    ap.add_argument("--max-stations", type=int, default=14,
                    help="traces per full-size figure (nearest first)")
    ap.add_argument("--overview-stations", type=int, default=6)
    ap.add_argument("--max-dist-km", type=float, default=25.0,
                    help="drop stations further than this from the epicentre; "
                         "one 33 km trace otherwise sets a 15 s time axis")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--select-only", action="store_true",
                    help="print the event choice and stop")
    xcfilter.add_cli(ap)
    args = ap.parse_args()
    if args.workers > 4:
        sys.exit("refusing more than 4 workers (32-CPU pod quota, shared node)")
    band = xcfilter.band_from_args(args) or xcfilter.DEFAULT_BAND
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading catalogues ...")
    cat, assoc, man, sta = load_tables()
    print(f"  {len(cat):,} strict-tier events, {len(assoc):,} associated picks, "
          f"{len(man):,} manual picks, {len(sta)} stations")

    print("selecting events ...")
    chosen = select_events(cat, assoc, man, sta)
    mt = match_manual_events(cat, assoc, man).set_index("event_idx")
    labels = {"a": "shallow caldera event",
              "b": "deeper event under the array",
              "c": "small event with few phases",
              "d": "event in the analyst-trusted window"}
    if args.events:
        chosen = {}
        for tok in args.events.split(","):
            k, v = tok.split("=")
            chosen[k.strip()] = cat[cat.event_idx == int(v)].iloc[0]
        print("  overridden: " + ", ".join(f"{k}={int(v.event_idx)}"
                                           for k, v in chosen.items()))
    if args.select_only:
        return 0

    print("indexing NLLoc .hyp files ...")
    hyp_idx = index_hyps(HYP_DIR)
    print(f"  {sum(len(v) for v in hyp_idx.values()):,} hyp files")

    events = []
    for letter in sorted(chosen):
        row = chosen[letter]
        mrow = mt.loc[int(row.event_idx)] if int(row.event_idx) in mt.index else None
        ev = build_event(letter, row, assoc, man, mrow, sta, hyp_idx, args)
        events.append(ev)
        print(f"  ({letter}) event {ev['event_idx']}: {len(ev['stations'])} stations, "
              f"window {ev['t1'] - ev['t0']:.1f} s, hyp {ev['hyp']} "
              f"({ev['n_hyp_common']} picks in common, "
              f"OT offset {1000 * (ev['hyp_ot'] - ev['ot']):+.0f} ms)"
              if ev["hyp"] else
              f"  ({letter}) event {ev['event_idx']}: NO NLLoc hyp matched")

    print(f"reading waveforms ({args.workers} workers, "
          f"{band[0]:g}-{band[1]:g} Hz) ...")
    problems = attach_waveforms(events, args.workers, band)
    for p in problems:
        print(f"  ! {p}")

    print("counts per event (raw pool in window -> associated):")
    for ev in events:
        rc = raw_counts(ev)
        per = ", ".join(
            f"{label} P{rc[(pool, 'P')]}/S{rc[(pool, 'S')]}"
            for pool, label in RAW_POOLS)
        print(f"  ({ev['letter']}) {ev['event_idx']}: raw {raw_total(rc)} [{per}]"
              f" -> associated on the traces shown {ev['n_assoc_shown']} "
              f"({ev['n_assoc_shown_p']} P / {ev['n_assoc_shown_s']} S) "
              f"of {ev['n_assoc']} total, "
              f"manual {ev['n_manual']}, NLLoc predicted {ev['n_pred']}")

    paths = {"overview_stations": args.overview_stations, "labels": labels,
             "max_dist_km": args.max_dist_km}
    out = [fig_overview(events, args)]
    for ev in events:
        out.append(fig_event(ev, args))
    out.append(write_captions(events, paths, problems))
    print("\nwrote:")
    for p in out:
        print(f"  {p.relative_to(REPO)}  {p.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
