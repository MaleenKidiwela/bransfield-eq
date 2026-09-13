"""Waveform cross-correlation -> hypoDD `dt.cc`, streamed station by station.

Scope
-----
Fresh implementation (2026-09-13).  Nothing under `growclust/` and no earlier
`dt.cc` / `pick_windows*.npy` is read.  The event set, the station/phase list and
the pair list all come from an existing hypoDD run directory's `dt.ct`, so every
row we produce lines up 1:1 with a catalogue row that hypoDD already uses.

Sign convention (hypoDD2 User Guide, B.3.2, p.22)
------------------------------------------------
    "#, ID1, ID2, OTC" then "STA, DT, WGHT, PHA"
    OTC  Origin time correction relative to the event origin time reported in
         the catalog data.  Set to 0.0 if cross-correlation and catalog origin
         times are identical, or if only cross-correlation data is used.
    DT   Differential time (s) between event 1 and event 2 at station STA.
         DT = T1-T2.
`getdata.f:402` implements it as `dt_dt(i) = dt_dt(i) - otc`, and the resulting
`dt_dt` array is the same one filled from `dt.ct` by `dt_dt(i) = dt1 - dt2`
(`getdata.f:488`), i.e. a difference of *travel times*.  So with OTC = 0.0 the
DT column must be a differential TRAVEL time in the catalogue's own origin-time
frame -- not a differential arrival time.

We therefore write

    DT = (TT1 - TT2)_from_dt.ct  +  (off1 - off2)  -  L

with L the cross-correlation lag defined below.  Derivation: let a_i be the true
arrival of event i, p_i the pick, e_i = p_i - a_i the pick error, and off_i the
sub-sample rounding of the window start (see `_extract`).  Windows are cut at
p_i + off_i.  The normalised sliding correlation peaks at the lag
L = (e_1 + off_1) - (e_2 + off_2), hence

    a_1 - a_2 = (p_1 - p_2) + (off_1 - off_2) - L
    TT_1 - TT_2 (refined) = (TT_1 - TT_2)_catalogue + (off_1 - off_2) - L

because TT_i = p_i - OT_i and the OT_i are unchanged (OTC = 0.0).  The catalogue
term is read straight out of `dt.ct`, so `dt.cc - dt.ct = (off1-off2) - L` is
exactly the CC correction and is the quantity checked in the validation block.
The self-test (`--self-test`) imposes a known fractional delay on a real trace
and asserts the recovered L to < 5 ms.

Why a *sliding* normalised correlation and not `correlate(..., "full")`
----------------------------------------------------------------------
`src/bransfield_eq/xcfilter.py` documents that normalising each window once and
taking `mode="full"` applies a triangular taper ~(n-|L|)/n to the correlation,
which drags the apparent peak toward lag 0.  Here the master window (event 1) is
short and the slave window (event 2) is padded by +-MAXLAG, the correlation is
taken in `"valid"` mode, and the denominator uses the *moving* mean/variance of
the slave under the window.  Every lag is then normalised over exactly n
samples, so there is no taper and cc is a true Pearson coefficient.

Memory
------
An earlier XC prep OOM-killed the whole pod by pre-loading every station-day
(see memory note `jupyterhub_pod_memory_limit`).  Here one worker owns one
station and holds at most ONE station-day of waveform at a time (~100-140 MB for
a 4-channel 100 Hz OBS day) plus that station's cut windows (a few tens of MB).
With 8 workers the resident set stays around 2-4 GB.  The filter is applied once
per station-day (xcfilter's recommendation) so no window contains a transient.

Usage
-----
    PYTHONPATH=src python3 scripts/63_xcorr_dtcc.py --self-test
    PYTHONPATH=src python3 scripts/63_xcorr_dtcc.py --label year_v5_3d \
        --out hypodd/year_v5_3d_xc/dt.cc --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bransfield_eq import xcfilter  # noqa: E402

# ---------------------------------------------------------------- correlation


def master_prep(a: np.ndarray):
    """Demeaned master window and its L2 norm, or None if degenerate."""
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean()
    sa = float(np.sqrt(a @ a))
    if not np.isfinite(sa) or sa <= 0.0:
        return None
    return a, sa


def slave_prep(b: np.ndarray, n: int):
    """Moving L2 norm of the demeaned length-n windows of slave `b`.

    Cached per window because one window is the slave of ~80 pairs on average;
    only the numerator then has to be recomputed per pair.
    """
    b = np.asarray(b, dtype=np.float64)
    c1 = np.concatenate(([0.0], np.cumsum(b)))
    c2 = np.concatenate(([0.0], np.cumsum(b * b)))
    s1 = c1[n:] - c1[:-n]
    s2 = c2[n:] - c2[:-n]
    var = s2 - s1 * s1 / n
    np.maximum(var, 0.0, out=var)
    sb = np.sqrt(var)
    if not np.any(sb > 0.0):
        return None
    return b, sb


def sliding_ncc(a: np.ndarray, b: np.ndarray):
    """Pearson correlation of master `a` (n samples) against every length-n
    window of slave `b` (n + 2M samples).  Returns an array of 2M+1 values, or
    None if either side is degenerate.

    cc[j] = corr(a, b[j:j+n]) with the moving mean/variance of b removed, so
    every lag is normalised over exactly n samples (no triangular taper).
    """
    if b.size - a.size + 1 < 1:
        return None
    mp = master_prep(a)
    sp = slave_prep(b, a.size)
    if mp is None or sp is None:
        return None
    return sliding_ncc_prepped(mp[0], mp[1], sp[0], sp[1])


def sliding_ncc_prepped(a_dem, sa, b, sb):
    # np.correlate(b, a, "valid")[j] = sum_k b[j+k] * a[k]; a is demeaned so the
    # moving mean of b drops out of the numerator.
    num = np.correlate(b, a_dem, mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = num / (sa * sb)
    cc[~np.isfinite(cc)] = 0.0
    return cc


def peak_parabolic(cc: np.ndarray):
    """(index, sub-sample offset, cc at the peak) with a 3-point parabolic fit.

    The fitted value is clamped to the sampled peak so a near-degenerate fit can
    never report cc > 1.
    """
    j = int(np.argmax(cc))
    d = 0.0
    if 0 < j < cc.size - 1:
        ym, y0, yp = float(cc[j - 1]), float(cc[j]), float(cc[j + 1])
        den = ym - 2.0 * y0 + yp
        if den < 0.0:                       # a real maximum
            d = 0.5 * (ym - yp) / den
            if abs(d) > 0.5:
                d = 0.0
    return j, d, float(cc[j])


# ---------------------------------------------------------------- dt.ct input


def parse_dtct(path: Path):
    """dt.ct -> (pair_id1, pair_id2, row_pair, sta, pha, tt1, tt2).

    row_pair[i] is the index of the pair header that row i belongs to, so the
    output file can be written back in exactly the input pair order.
    """
    p1, p2 = [], []
    r_pair, sta, pha = [], [], []
    tt1, tt2 = [], []
    cur = -1
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                _, a, b = line.split()
                p1.append(int(a))
                p2.append(int(b))
                cur += 1
                continue
            s = line.split()
            r_pair.append(cur)
            sta.append(s[0])
            tt1.append(s[1])
            tt2.append(s[2])
            pha.append(s[4])
    return (np.asarray(p1, dtype=np.int32), np.asarray(p2, dtype=np.int32),
            np.asarray(r_pair, dtype=np.int32), np.asarray(sta),
            np.asarray(pha), np.asarray(tt1, dtype=np.float64),
            np.asarray(tt2, dtype=np.float64))


def load_picks(csv_path: Path):
    """(event_idx, bare station, phase) -> pick epoch seconds, plus sta -> net.

    `time` in this CSV is already float64 epoch seconds, so no datetime round
    trip happens here at all (pandas 3 `.astype("int64")` unit trap avoided by
    construction; `timeutil.epoch_seconds` is used only if the column is not
    numeric).
    """
    import pandas as pd

    pk = pd.read_csv(csv_path)
    if not pd.api.types.is_numeric_dtype(pk["time"]):
        from bransfield_eq.timeutil import epoch_seconds, assert_nanosecond_sanity
        assert_nanosecond_sanity()
        t = epoch_seconds(pk["time"]).to_numpy()
    else:
        t = pk["time"].to_numpy(dtype=np.float64)
    net = pk["station"].str.split(".").str[0].to_numpy()
    bare = pk["station"].str.split(".").str[-1].to_numpy()
    ev = pk["event_idx"].to_numpy(dtype=np.int64)
    ph = pk["phase"].to_numpy()
    pmap = {}
    netmap = {}
    for e, b, p, tt, nn in zip(ev, bare, ph, t, net):
        pmap[(int(e), b, p)] = float(tt)
        netmap[b] = nn
    return pmap, netmap


# ---------------------------------------------------------------- waveform IO

def choose_channels(chan_rate: dict):
    """(vertical code, [horizontal codes]) from a {channel: sampling rate} map.

    The Bransfield deployment mixes two OBS families and they do NOT share a
    band code:
        BRA02/03/04/05/08/09/10/11 -> HHZ + HH1/HH2 (+HDH), all 100 Hz
        BRA13..BRA27               -> ELZ at 200 Hz vertical, SL1/SL2 at 100 Hz
                                      horizontals (+EDH 200 Hz, LDH 1 Hz)
    An earlier version of this function required the vertical and the
    horizontals to share a band code; on the SL/EL family it found ELZ, no
    horizontals, and silently produced ZERO S measurements for the 14 stations
    that carry 97% of the dt.ct rows.  So: pick the vertical and the horizontal
    band independently, each by highest sampling rate.  P and S are correlated
    on different channels anyway, and hydrophones (…DH) are excluded because
    they end in neither Z nor 1/2/N/E.
    """
    zs = sorted(((fs, c) for c, fs in chan_rate.items() if c[-1] == "Z"),
                reverse=True)
    zc = zs[0][1] if zs else None
    bands = defaultdict(list)
    for c, fs in chan_rate.items():
        if c[-1] in ("1", "2", "N", "E"):
            bands[c[:2]].append((fs, c))
    hcs = []
    if bands:
        best = max(bands.items(), key=lambda kv: (max(f for f, _ in kv[1]), kv[0]))
        hcs = sorted(c for _, c in best[1])
    return zc, hcs


def scan_channels(path: Path) -> dict:
    """{channel: sampling rate} from a header-only read (~10 ms)."""
    import obspy

    try:
        st = obspy.read(str(path), headonly=True)
    except Exception:
        return {}
    return {tr.stats.channel: float(tr.stats.sampling_rate) for tr in st}


def load_station_day(path: Path, lo, hi, codes):
    """Read only `codes` from one station-day, merge gaps to zero, bandpass each.

    Returns {code: (float32 array, fs, t0, npts)}.  Channels are read one at a
    time with libmseed's `sourcename` selection: on a 54 MB SL/EL day that is
    0.17 s of decode instead of 1.24 s, because the 200 Hz hydrophone is never
    unpacked.  Each channel keeps its OWN fs/t0/npts (ELZ 200 Hz and SL1/SL2
    100 Hz live in the same file).  The bandpass runs once per station-day, as
    xcfilter recommends, so no window can contain a filter transient.
    """
    import obspy

    out = {}
    for code in codes:
        try:
            st = obspy.read(str(path), sourcename="*.*.*." + code)
        except Exception:
            continue
        if len(st) == 0:
            continue
        try:
            st.merge(method=1, fill_value=0, interpolation_samples=0)
        except Exception:
            try:
                st.merge(method=0, fill_value=0)
            except Exception:
                continue
        st = [tr for tr in st if tr.stats.channel == code]
        if len(st) != 1:
            continue
        tr = st[0]
        fs = float(tr.stats.sampling_rate)
        d = np.asarray(tr.data)
        if np.ma.isMaskedArray(d):
            d = d.filled(0.0)
        d = d.astype(np.float64)
        d -= d.mean()
        if lo:
            d = xcfilter.bandpass(d, lo, hi, fs)
        else:
            d = np.ascontiguousarray(d, dtype=np.float32)
        out[code] = (d, fs, float(tr.stats.starttime.timestamp), int(tr.stats.npts))
    return out


# ---------------------------------------------------------------- per-station


def run_station(task):
    """Worker: cut every window this station needs, then correlate its dt.ct rows.

    `task` carries only plain arrays so the pickle stays small.  Returns the
    accepted rows as (global row index, dt, cc, lag).
    """
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    import datetime as _dt

    sta = task["sta"]
    net = task["net"]
    lo, hi = task["band"] if task["band"] else (None, None)
    fs_maxlag = task["max_lag"]
    wins = task["windows"]          # (hypodd_id, phase, pick_time) arrays
    rows = task["rows"]             # dict of arrays
    wave_dir = Path(task["wave_dir"])
    t_start = time.time()

    w_id = wins["id"]
    w_ph = wins["pha"]
    w_t = wins["t"]
    nw = w_id.size

    # group windows by UTC day of the pick
    day_of = np.empty(nw, dtype=object)
    for i in range(nw):
        d = _dt.datetime.fromtimestamp(w_t[i], _dt.timezone.utc)
        day_of[i] = (d.year, d.timetuple().tm_yday)
    by_day = defaultdict(list)
    for i in range(nw):
        by_day[day_of[i]].append(i)

    # data[i] = {chan: (a_dem, sa, b, sb)} -- master core and slave, both already
    # prepared, so a pair costs one `np.correlate` and one divide.
    data = [None] * nw
    off = np.zeros(nw, dtype=np.float64)
    fs_w = np.zeros(nw, dtype=np.float64)
    n_days_read = 0
    n_edge = 0
    n_nofile = 0
    n_nodata = 0
    bytes_read = 0

    # Decide this station's channels once, from the header of up to 5 spread-out
    # days (a single day can be missing channels), so the same physical
    # component is correlated for every event at this station.
    day_list = sorted(by_day)
    paths = {d: wave_dir / net / sta / f"{net}.{sta}.{d[0]}.{d[1]:03d}.mseed"
             for d in day_list}
    chan_rate = {}
    probe = day_list[:: max(len(day_list) // 5, 1)][:5] or day_list[:1]
    for d in probe:
        p = paths[d]
        if p.exists() and p.stat().st_size > 0:
            chan_rate.update(scan_channels(p))
    zc, hcs = choose_channels(chan_rate)
    codes = ([zc] if zc else []) + list(hcs)
    if not codes:
        return dict(sta=sta, elapsed=time.time() - t_start, n_days=0,
                    n_windows=nw, n_win_ok=0, n_rows=int(rows["w1"].size),
                    n_ok=0, bytes=0, n_edge=0, n_nofile=nw, n_nodata=0,
                    n_missing=int(rows["w1"].size), n_lowcc=0, n_biglag=0,
                    chan_Z=None, chan_H=[],
                    idx=np.zeros(0, np.int64), dt=np.zeros(0), cc=np.zeros(0, np.float32),
                    lag=np.zeros(0))

    for day in day_list:
        f = paths[day]
        if (not f.exists()) or f.stat().st_size == 0:
            n_nofile += len(by_day[day])
            continue
        bytes_read += f.stat().st_size
        d = load_station_day(f, lo, hi, codes)
        n_days_read += 1
        if not d:
            n_nodata += len(by_day[day])
            continue
        for i in by_day[day]:
            is_p = w_ph[i] == "P"
            want = [zc] if is_p else hcs
            pre = task["p_pre"] if is_p else task["s_pre"]
            wlen = task["p_len"] if is_p else task["s_len"]
            prepped = {}
            wfs = 0.0
            woff = 0.0
            n_seen = n_rng = 0
            for code in want:
                got = d.get(code)
                if got is None:
                    continue
                n_seen += 1
                arr, fs, t0c, npts = got
                M = int(round(fs_maxlag * fs))
                n = int(round(wlen * fs))
                need = n + 2 * M
                wstart = w_t[i] - pre - M / fs      # nominal padded-window start
                idx = int(round((wstart - t0c) * fs))
                if idx < 0 or idx + need > npts:
                    n_rng += 1
                    continue
                seg = np.asarray(arr[idx:idx + need], dtype=np.float64)
                mp = master_prep(seg[M:need - M])
                sp = slave_prep(seg, n)
                if mp is None or sp is None:
                    continue
                prepped[code] = (mp[0], mp[1], sp[0], sp[1])
                # realised window start minus nominal: the sub-sample rounding
                # that the dt correction below removes.  All components of one
                # window share fs and t0, so one value is enough.
                wfs = fs
                woff = idx / fs + t0c - wstart
            if not prepped:
                if n_seen and n_rng == n_seen:
                    n_edge += 1          # window runs past the end of the day file
                else:
                    n_nodata += 1        # channel absent, or flat/zero-filled gap
                continue
            off[i] = woff
            fs_w[i] = wfs
            data[i] = prepped
        del d

    # ---- correlate
    wi1 = rows["w1"]
    wi2 = rows["w2"]
    gidx = rows["gidx"]
    dct = rows["dct"]
    out_i, out_dt, out_cc, out_lag = [], [], [], []
    n_missing = 0
    n_lowcc = 0
    n_biglag = 0
    for k in range(wi1.size):
        i1 = int(wi1[k])
        i2 = int(wi2[k])
        A = data[i1]
        B = data[i2]
        if A is None or B is None:
            n_missing += 1
            continue
        fs = fs_w[i1]
        if fs <= 0 or fs_w[i2] != fs:
            n_missing += 1
            continue
        M = int(round(fs_maxlag * fs))
        acc = None
        used = 0
        for code, (a_dem, sa, _, _) in A.items():
            sl = B.get(code)
            if sl is None:
                continue
            cc = sliding_ncc_prepped(a_dem, sa, sl[2], sl[3])
            acc = cc if acc is None else acc + cc
            used += 1
        if used == 0:
            n_missing += 1
            continue
        if used > 1:
            acc /= used
        j, frac, peak = peak_parabolic(acc)
        if peak < task["cc_min"]:
            n_lowcc += 1
            continue
        lag = (j + frac - M) / fs
        if abs(lag) > fs_maxlag:
            n_biglag += 1
            continue
        out_i.append(gidx[k])
        out_dt.append(dct[k] + (off[i1] - off[i2]) - lag)
        out_cc.append(peak)
        out_lag.append(lag)

    return dict(
        sta=sta, elapsed=time.time() - t_start, n_days=n_days_read,
        n_windows=nw, n_win_ok=int(sum(x is not None for x in data)),
        n_rows=int(wi1.size), n_ok=len(out_i), bytes=bytes_read,
        n_edge=n_edge, n_nofile=n_nofile, n_nodata=n_nodata,
        n_missing=n_missing, n_lowcc=n_lowcc, n_biglag=n_biglag,
        chan_Z=zc, chan_H=list(hcs),
        idx=np.asarray(out_i, dtype=np.int64),
        dt=np.asarray(out_dt, dtype=np.float64),
        cc=np.asarray(out_cc, dtype=np.float32),
        lag=np.asarray(out_lag, dtype=np.float64),
    )


# ---------------------------------------------------------------- self-test


def self_test(band):
    """Correlate a real trace with fractionally shifted copies of itself.

    x2(t) = x1(t - D): event 2 arrives D later, so the sliding correlation must
    peak at lag L = +D and the differential travel time must come out as -D.
    """
    import obspy

    print("self-test: fractional-shift recovery")
    cand = sorted((REPO / "data" / "waveforms" / "ZX" / "BRA21").glob("*.mseed"))
    cand = [c for c in cand if c.stat().st_size > 0]
    tr = obspy.read(str(cand[len(cand) // 2]))[0]
    fs = float(tr.stats.sampling_rate)
    x = np.asarray(tr.data, dtype=np.float64)
    seg = x[3_000_000:3_000_000 + 65536]
    lo, hi = band
    seg = xcfilter.bandpass(seg - seg.mean(), lo, hi, fs).astype(np.float64)

    def shift(sig, d_sec):
        """Exact fractional delay by a Fourier phase ramp: y(t) = sig(t - d)."""
        n = sig.size
        f = np.fft.rfftfreq(n, d=1.0 / fs)
        return np.fft.irfft(np.fft.rfft(sig) * np.exp(-2j * np.pi * f * d_sec), n)

    maxlag = 0.5
    M = int(round(maxlag * fs))
    n = int(round(1.0 * fs))
    i0 = 30000
    worst = 0.0
    rows = []
    for d_sec in (0.0, 0.004, -0.004, 0.013, -0.027, 0.111, -0.250, 0.437):
        y = shift(seg, d_sec)
        a = seg[i0:i0 + n]
        b = y[i0 - M:i0 + n + M]
        cc = sliding_ncc(a, b)
        j, frac, peak = peak_parabolic(cc)
        lag = (j + frac - M) / fs
        err = lag - d_sec
        worst = max(worst, abs(err))
        rows.append((d_sec, lag, err, peak))
        print(f"  imposed {d_sec:+7.4f} s  recovered {lag:+7.4f} s  "
              f"err {err*1000:+7.3f} ms  cc {peak:.4f}")
    print(f"  worst |error| = {worst*1000:.3f} ms  "
          f"({'PASS' if worst < 0.005 else 'FAIL'}; threshold 5 ms)")
    # integer-shift sanity on the sign, independent of the FFT resampler
    s = 17
    a = seg[i0:i0 + n]
    b = seg[i0 - M - s:i0 - M - s + n + 2 * M]   # slave delayed by s samples
    j, frac, peak = peak_parabolic(sliding_ncc(a, b))
    lag = (j + frac - M) / fs
    print(f"  integer check: slave delayed {s} samples ({s/fs:+.3f} s) -> "
          f"lag {lag:+.4f} s, cc {peak:.4f} "
          f"({'PASS' if abs(lag - s / fs) < 1e-4 else 'FAIL'})")
    ok = worst < 0.005 and abs(lag - s / fs) < 1e-4
    print(f"  SELF-TEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="year_v5_3d",
                    help="hypoDD run dir supplying dt.ct / event.sel")
    ap.add_argument("--picks",
                    default="catalogs/pyocto_picks_year_newpool_no_shots.csv")
    ap.add_argument("--out", default=None,
                    help="dt.cc path (default hypodd/<label>_xc/dt.cc)")
    ap.add_argument("--wave-dir", default="data/waveforms")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cc-min", type=float, default=0.7)
    ap.add_argument("--max-lag", type=float, default=0.5)
    ap.add_argument("--p-win", nargs=2, type=float, default=(-0.3, 0.7),
                    metavar=("PRE", "POST"))
    ap.add_argument("--s-win", nargs=2, type=float, default=(-0.4, 1.0),
                    metavar=("PRE", "POST"))
    ap.add_argument("--stations", default=None,
                    help="comma-separated bare codes (smoke test)")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="only use the first N pairs of dt.ct (smoke test)")
    ap.add_argument("--self-test", action="store_true")
    xcfilter.add_cli(ap)
    args = ap.parse_args()

    band = xcfilter.band_from_args(args)
    if args.self_test:
        return self_test(band or xcfilter.DEFAULT_BAND)
    if args.workers > 8:
        sys.exit("refusing more than 8 workers (32-CPU pod quota)")

    run_dir = REPO / "hypodd" / args.label
    out_path = Path(args.out) if args.out else REPO / "hypodd" / f"{args.label}_xc" / "dt.cc"
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    print(f"dt.ct: {run_dir/'dt.ct'}")
    p1, p2, r_pair, sta, pha, tt1, tt2 = parse_dtct(run_dir / "dt.ct")
    n_pairs = p1.size
    print(f"  {n_pairs:,} pairs, {sta.size:,} rows "
          f"(P {int((pha=='P').sum()):,} / S {int((pha=='S').sum()):,})")
    if args.max_pairs:
        keep = r_pair < args.max_pairs
        r_pair, sta, pha, tt1, tt2 = (r_pair[keep], sta[keep], pha[keep],
                                      tt1[keep], tt2[keep])
        n_pairs = args.max_pairs
        p1, p2 = p1[:n_pairs], p2[:n_pairs]
        print(f"  SMOKE: restricted to {n_pairs:,} pairs, {sta.size:,} rows")

    id1 = p1[r_pair]
    id2 = p2[r_pair]
    if args.stations:
        keep = np.isin(sta, [s.strip() for s in args.stations.split(",")])
        idx_keep = np.nonzero(keep)[0]
        print(f"  SMOKE: stations {args.stations} -> {idx_keep.size:,} rows")
    else:
        idx_keep = np.arange(sta.size)

    print(f"picks: {args.picks}")
    pmap, netmap = load_picks(REPO / args.picks)
    print(f"  {len(pmap):,} picks indexed")

    # ---- build per-station window tables and row tables
    tasks = []
    n_nopick = 0
    for s in sorted(set(sta[idx_keep].tolist())):
        sel = idx_keep[sta[idx_keep] == s]
        wkey = {}
        w_id, w_ph, w_t = [], [], []

        def win(hid, ph):
            k = (hid, ph)
            j = wkey.get(k)
            if j is None:
                t = pmap.get((hid - 1, s, ph))
                if t is None:
                    return -1
                j = len(w_id)
                wkey[k] = j
                w_id.append(hid)
                w_ph.append(ph)
                w_t.append(t)
            return j

        rw1, rw2, rg, rdct, risP = [], [], [], [], []
        miss = 0
        for r in sel:
            a = win(int(id1[r]), pha[r])
            b = win(int(id2[r]), pha[r])
            if a < 0 or b < 0:
                miss += 1
                continue
            rw1.append(a)
            rw2.append(b)
            rg.append(r)
            rdct.append(tt1[r] - tt2[r])
            risP.append(pha[r] == "P")
        n_nopick += miss
        if not rg:
            continue
        tasks.append(dict(
            sta=s, net=str(netmap.get(s, "ZX")), band=band,
            max_lag=args.max_lag, cc_min=args.cc_min,
            p_pre=-args.p_win[0], p_len=args.p_win[1] - args.p_win[0],
            s_pre=-args.s_win[0], s_len=args.s_win[1] - args.s_win[0],
            wave_dir=str(REPO / args.wave_dir),
            windows=dict(id=np.asarray(w_id, dtype=np.int32),
                         pha=np.asarray(w_ph),
                         t=np.asarray(w_t, dtype=np.float64)),
            rows=dict(w1=np.asarray(rw1, dtype=np.int32),
                      w2=np.asarray(rw2, dtype=np.int32),
                      gidx=np.asarray(rg, dtype=np.int64),
                      dct=np.asarray(rdct, dtype=np.float64),
                      isP=np.asarray(risP, dtype=bool)),
        ))
    tasks.sort(key=lambda t: -t["rows"]["w1"].size)       # longest first
    n_win = sum(t["windows"]["id"].size for t in tasks)
    print(f"tasks: {len(tasks)} stations, {n_win:,} unique windows, "
          f"{sum(t['rows']['w1'].size for t in tasks):,} rows "
          f"({n_nopick:,} rows dropped: no pick in the CSV)")
    print(f"band {band[0]}-{band[1]} Hz" if band else "band: NONE (unfiltered)")
    print(f"P window {args.p_win[0]:+.2f}/{args.p_win[1]:+.2f} s, "
          f"S window {args.s_win[0]:+.2f}/{args.s_win[1]:+.2f} s, "
          f"max lag {args.max_lag} s, cc >= {args.cc_min}")

    # ---- run
    n_rows_all = sta.size
    dt_out = np.full(n_rows_all, np.nan)
    cc_out = np.full(n_rows_all, np.nan, dtype=np.float32)
    lag_out = np.full(n_rows_all, np.nan)
    stats = []
    done_rows = 0
    done_bytes = 0
    from concurrent.futures import ProcessPoolExecutor

    print(f"\nlaunching {args.workers} workers ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run_station, tasks):
            dt_out[r["idx"]] = r["dt"]
            cc_out[r["idx"]] = r["cc"]
            lag_out[r["idx"]] = r["lag"]
            done_rows += r["n_rows"]
            done_bytes += r["bytes"]
            el = time.time() - t_all
            print(f"  {r['sta']:<7s} [{r['chan_Z']}|{'+'.join(r['chan_H'])}] "
                  f"{r['n_days']:4d} d  {r['bytes']/1e9:6.1f} GB  "
                  f"win {r['n_win_ok']:6d}/{r['n_windows']:6d}  "
                  f"rows {r['n_ok']:7d}/{r['n_rows']:7d} "
                  f"({100*r['n_ok']/max(r['n_rows'],1):5.1f}%)  "
                  f"{r['elapsed']:7.1f} s  | wall {el/60:6.1f} min, "
                  f"{done_bytes/1e9/max(el,1)*60:5.1f} GB/min", flush=True)
            r.pop("idx"); r.pop("dt"); r.pop("cc"); r.pop("lag")
            stats.append(r)
            try:
                cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
                print(f"          cgroup memory.current {cur/2**30:.1f} GiB", flush=True)
            except Exception:
                pass

    # ---- write dt.cc in dt.ct pair order
    ok = np.isfinite(dt_out)
    order = np.lexsort((np.arange(n_rows_all), r_pair))
    n_written = 0
    n_pairs_written = 0
    with open(out_path, "w") as f:
        cur = -1
        buf = []
        for r in order:
            if not ok[r]:
                continue
            if r_pair[r] != cur:
                if buf:
                    f.write("".join(buf))
                    buf = []
                cur = int(r_pair[r])
                f.write(f"# {p1[cur]:9d} {p2[cur]:9d} 0.0\n")
                n_pairs_written += 1
            buf.append(f"{sta[r]:<7s} {dt_out[r]:9.5f} {cc_out[r]:7.4f} {pha[r]}\n")
            n_written += 1
        if buf:
            f.write("".join(buf))
    print(f"\nwrote {out_path}: {n_written:,} rows in {n_pairs_written:,} pairs")

    # ---- validation
    kept = ok
    d = dt_out[kept] - (tt1[kept] - tt2[kept])
    cc = cc_out[kept]
    lag = lag_out[kept]
    isP = pha[kept] == "P"
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    qs = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nVALIDATION")
    print(f"  dt.ct rows {n_rows_all:,} -> dt.cc rows {n_written:,} "
          f"({100*n_written/n_rows_all:.1f}%)   "
          f"P {int(isP.sum()):,} ({100*isP.sum()/max((pha=='P').sum(),1):.1f}%)  "
          f"S {int((~isP).sum()):,} ({100*(~isP).sum()/max((pha=='S').sum(),1):.1f}%)")
    print(f"  pairs {n_pairs:,} -> {n_pairs_written:,} "
          f"({100*n_pairs_written/max(n_pairs,1):.1f}%)")
    print(f"  cc      p{qs} = " + " / ".join(f"{v:.3f}" for v in np.percentile(cc, qs)))
    print(f"  |lag| s p{qs} = " + " / ".join(f"{v:.4f}" for v in np.percentile(np.abs(lag), qs)))
    print(f"  lag   s p{qs} = " + " / ".join(f"{v:+.4f}" for v in np.percentile(lag, qs)))
    print(f"  dt.cc - dt.ct : median {med:+.5f} s   MAD {mad:.5f} s   "
          f"p99 |d| {np.percentile(np.abs(d),99):.4f} s   "
          f"(should be << 0.1 s)")
    for name, m in (("P", isP), ("S", ~isP)):
        if m.sum():
            dd = d[m]
            mm = float(np.median(dd))
            print(f"    {name}: n {int(m.sum()):,}  cc p50 {np.median(cc[m]):.3f}  "
                  f"median {mm:+.5f} s  MAD {np.median(np.abs(dd-mm)):.5f} s")

    summary = dict(
        out=str(out_path), label=args.label,
        band=list(band) if band else None, cc_min=args.cc_min,
        max_lag=args.max_lag, p_win=list(args.p_win), s_win=list(args.s_win),
        n_pairs_ct=int(n_pairs), n_rows_ct=int(n_rows_all),
        n_pairs_cc=int(n_pairs_written), n_rows_cc=int(n_written),
        frac_rows=float(n_written / n_rows_all),
        cc_pct={str(q): float(v) for q, v in zip(qs, np.percentile(cc, qs))},
        lag_abs_pct={str(q): float(v) for q, v in zip(qs, np.percentile(np.abs(lag), qs))},
        d_median=med, d_mad=mad,
        wall_s=time.time() - t_all, gbytes=done_bytes / 1e9,
        stations=stats,
    )
    js = out_path.parent / "dtcc_summary.json"
    js.write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(out_path.parent / "dtcc_measurements.npz",
                        gidx=np.nonzero(kept)[0].astype(np.int64),
                        cc=cc.astype(np.float32), lag=lag.astype(np.float32),
                        d=d.astype(np.float32), isP=isP)
    print(f"wrote {js}")
    print(f"total wall {(time.time()-t_all)/60:.1f} min, {done_bytes/1e9:.0f} GB read")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
