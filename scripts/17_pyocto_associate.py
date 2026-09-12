"""
PyOcto association driver — turn picker outputs into associated events.

Uses:
  - PhaseNet `instance` P picks (catalogs/picks/, P only)
  - OBSTransformer obst2024 P + S picks (catalogs/picks_obst_01/)
  - (Optional) manual mag07 picks as a "control" run

Standard event quality thresholds (Wilcock-style for OBS):
  - min stations:    5
  - min P picks:     3
  - min S picks:     2
  - min total picks: 6
  - pick uncertainty: 0.5 s
  - association window (time_slicing): 1200 s (default)

Outputs:
  catalogs/pyocto_events_<run_label>.csv     — one row per associated event
  catalogs/pyocto_picks_<run_label>.csv      — one row per associated pick (joined to event_idx)

Run-label conventions:
  picker_only  — PhN P + OBST P+S
  with_manual  — picker_only + mag07 manual P+S (sanity check)

Usage:
    python scripts/17_pyocto_associate.py \
        --start 2019-01-01 --end 2020-03-01 \
        --velocity-model configs/velocity_model.csv \
        --label picker_only

The script is intentionally agnostic to the exact velocity-model file format —
it sniffs the input and adapts. Supported formats:
  - CSV with columns {depth, vp, vs} (any units; common headers detected)
  - Whitespace-separated 3-col text file
  - PyOcto pickle (if you've already serialized one)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from bransfield_eq import geo  # noqa: E402
from bransfield_eq.timeutil import epoch_seconds, assert_nanosecond_sanity  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VEL_DELTA_KM = 0.1    # must be <= the thinnest layer we care about (see _resample_velocity)
VEL_ZDIST_KM = 50.0

# Defaults — tuned for OBS / Bransfield basin, "Standard" event quality from notes
DEFAULTS = dict(
    min_stations=5, min_p=3, min_s=2, min_total=6,
    pick_match_tolerance=0.5,
    edt_pick_std=0.5,
    time_slicing=1200.0,
    min_node_size=10.0,
    z_min_km=0.0, z_max_km=40.0,  # local + regional in basin
    refinement_iterations=3,
    n_threads=8,
)


def load_velocity_model(path: Path, vpvs_ratio: float = 1.78, xdist: float = 200.0):
    """Sniff and load a 1D velocity model. Returns pyocto.VelocityModel1D.

    Accepts:
      - 3-col with Vs:    depth | Vp | Vs
      - 2-col P-only:     depth | Vp        (Vs derived as Vp / vpvs_ratio)
      - Auto units: depth in m or km; Vp in m/s or km/s.
    """
    import pyocto
    p = Path(path)
    if not p.exists():
        # Was a [warn] + homogeneous half-space: a typo in --velocity-model produced a
        # full catalogue from a constant-velocity model, with the warning buried in a log.
        raise SystemExit(
            f"velocity model not found: {p}\n"
            f"  Refusing to fall back to a homogeneous half-space.")
        return pyocto.VelocityModel0D(p_velocity=5.5, s_velocity=3.1, tolerance=2.0,
                                       location_p_velocity=5.5,
                                       location_s_velocity=3.1)

    depth = vp = vs = None
    # CSV with header first
    try:
        df = pd.read_csv(p, sep=None, engine="python")
        cols = {c.lower().strip(): c for c in df.columns}
        depth_col = next((cols[k] for k in ("depth_km", "depth_m", "depth", "z") if k in cols), None)
        vp_col = next((cols[k] for k in ("vp", "vp_kms", "vp_ms", "p", "p_velocity") if k in cols), None)
        vs_col = next((cols[k] for k in ("vs", "vs_kms", "vs_ms", "s", "s_velocity") if k in cols), None)
        if depth_col and vp_col:
            depth = df[depth_col].values.astype(float)
            vp = df[vp_col].values.astype(float)
            vs = df[vs_col].values.astype(float) if vs_col else None
    except Exception as e:
        print(f"  [warn] CSV parse failed: {e}; trying whitespace")
    # Whitespace-delimited fallback
    if vp is None:
        try:
            arr = np.loadtxt(p)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                depth = arr[:, 0]
                vp = arr[:, 1]
                vs = arr[:, 2] if arr.shape[1] >= 3 else None
        except Exception as e:
            raise SystemExit(f"Could not parse velocity model {p}: {e}")
    if vp is None:
        raise SystemExit(f"No Vp column found in {p}")

    # Unit detection
    if depth.max() > 100:
        depth = depth / 1000.0  # m → km
    if vp.max() > 100:
        vp = vp / 1000.0  # m/s → km/s
        if vs is not None:
            vs = vs / 1000.0
    # Derive Vs if missing
    if vs is None:
        vs = vp / vpvs_ratio
        print(f"  ⚠  No Vs column in {p}; derived Vs = Vp / {vpvs_ratio} (assumed Vp/Vs)")

    print(f"  loaded 1D velocity model from {p}: {len(depth)} layers")
    print(f"    depth: {depth.min():.2f} → {depth.max():.2f} km")
    print(f"    Vp:    {vp.min():.2f} → {vp.max():.2f} km/s")
    print(f"    Vs:    {vs.min():.2f} → {vs.max():.2f} km/s  (Vp/Vs ≈ {(vp/vs).mean():.2f})")

    # pyocto's VelocityModel1D loads a pre-computed cache file, not in-memory arrays.
    # Build the cache (depth/Vp/Vs DataFrame → create_model) next to the source CSV.
    # pyocto's create_model assigns each layer with p_speeds[:, int(d1/delta):int(d2/delta)],
    # so ANY layer thinner than `delta` produces an empty slice and is DISCARDED, and each
    # surviving cell takes the velocity of its deepest sample. At the previous delta=1.0 km
    # only 15 of 68 layers survived: the whole 1.3 km water column and every shallow
    # gradient vanished, and S-P came out ~1.4-1.5 s too small at all distances -- which
    # maps straight into epicentral distance and depth. Resample onto the delta grid first
    # so no layer is thinner than delta, and keep delta fine enough to hold the structure.
    # No ray in this network crosses seawater: sources are sub-seafloor, OBS sit ON the
    # seafloor (pyocto places them at their true depth, below the water column), and land
    # stations sit above it. But the CSV gives the water column vs=0.5 km/s, so any ray to
    # a LAND station -- which pyocto places at z~0, i.e. at the TOP of the water column --
    # accumulates ~2.6 s of fictitious S delay through 1.3 km of "water". That was masked
    # while delta=1.0 discarded the layer; at delta=0.1 it becomes real. Replace the water
    # column with the first rock velocity: OBS are unaffected (rays never enter it) and
    # land paths become rock, which is what they physically are.
    depth, vp, vs = _fill_water_with_rock(depth, vp, vs)
    model_df = _resample_velocity(depth, vp, vs, VEL_DELTA_KM)
    key = hashlib.md5(
        (model_df.round(6).to_csv(index=False)
         + f"|{VEL_DELTA_KM}|{xdist}|{VEL_ZDIST_KM}").encode()
    ).hexdigest()[:10]
    # Cache key covers the model AND the build parameters: keying on mtime alone meant a
    # changed delta/xdist silently reused the old table.
    cache = p.with_suffix(f".{key}.pyocto")
    if not cache.exists():
        print(f"  building pyocto velocity cache → {cache.name} "
              f"(delta={VEL_DELTA_KM} km, xdist={xdist:.0f} km, {len(model_df)} layers)")
        tmp = cache.with_suffix(".tmp")
        pyocto.VelocityModel1D.create_model(
            model=model_df, delta=VEL_DELTA_KM, xdist=xdist, zdist=VEL_ZDIST_KM, path=tmp,
        )
        import os as _os
        _os.replace(tmp, cache)   # atomic: concurrent chunks can't read a partial table
    else:
        print(f"  reusing velocity cache {cache.name}")
    return pyocto.VelocityModel1D(path=cache, tolerance=2.0)


def _fill_water_with_rock(depth, vp, vs, water_vp_max=1.6):
    """Replace the seawater column with the shallowest rock velocity.

    Returns arrays, unchanged if no water column is present."""
    depth = np.asarray(depth, float).copy()
    vp = np.asarray(vp, float).copy()
    vs = np.asarray(vs, float).copy()
    is_water = vp <= water_vp_max
    if not is_water.any():
        return depth, vp, vs
    first_rock = int(np.argmax(~is_water))
    if first_rock == 0:
        return depth, vp, vs
    vp[:first_rock] = vp[first_rock]
    vs[:first_rock] = vs[first_rock]
    print(f"    water column ({is_water.sum()} layers to "
          f"{depth[first_rock-1]:.3f} km) filled with rock "
          f"Vp={vp[first_rock]:.3f} Vs={vs[first_rock]:.3f} km/s "
          f"(no ray in this network crosses water)")
    return depth, vp, vs


def _resample_velocity(depth, vp, vs, delta):
    """Put the model on a uniform `delta` grid so create_model cannot drop layers.

    Uses slowness averaging within each cell (travel time is the integral of slowness,
    so averaging slowness preserves vertical travel time; averaging velocity does not)."""
    depth = np.asarray(depth, float); vp = np.asarray(vp, float); vs = np.asarray(vs, float)
    zmax = float(depth.max())
    edges = np.arange(0.0, zmax + delta, delta)
    fine = np.arange(0.0, zmax, min(delta / 20.0, 0.005))
    vp_f = np.interp(fine, depth, vp)
    vs_f = np.interp(fine, depth, vs)
    rows = []
    for i in range(len(edges) - 1):
        m = (fine >= edges[i]) & (fine < edges[i + 1])
        if not m.any():
            m = np.array([np.argmin(np.abs(fine - edges[i]))])
        rows.append((edges[i],
                     1.0 / np.mean(1.0 / vp_f[m]),
                     1.0 / np.mean(1.0 / vs_f[m])))
    return pd.DataFrame(rows, columns=["depth", "vp", "vs"])


def load_picks_for_pyocto(start_pd, end_pd, picker_pool: str, dedup_tol: float = 0.0,
                          sources=(("picks", "P"), ("picks_obst_01", "PS")),
                          prob_min: float = 0.0) -> pd.DataFrame:
    """Build the unified pick dataframe in PyOcto format:
        columns = station, time (UTCDateTime), phase ('P' or 'S'), prob (optional)
    """
    # Pre-filter CSVs by filename ("YYYY-DDD.csv") so we don't pandas-parse
    # 28k CSVs to keep one day's worth. CSV-per-station-day naming.
    import datetime as _dt
    _wanted_names = set()
    _d = start_pd.to_pydatetime().replace(tzinfo=None)
    _e = end_pd.to_pydatetime().replace(tzinfo=None)
    while _d < _e + _dt.timedelta(days=1):  # +1 day to cover boundary picks
        _wanted_names.add(f"{_d.year}-{_d.timetuple().tm_yday:03d}.csv")
        _d += _dt.timedelta(days=1)

    def _candidate_csvs(sta_dir):
        # Fast path: only globbing names we actually want.
        for n in sorted(_wanted_names):
            p = sta_dir / n
            if p.exists():
                yield p

    rows = []
    if picker_pool != "manual_only":   # always load picker output unless explicitly manual-only
        for subdir, phases in sources:
            sdir = REPO / "catalogs" / subdir
            if not sdir.exists():
                raise SystemExit(f"--pick-sources: catalogs/{subdir} does not exist")
            want = [c for c in phases.upper() if c in ("P", "S")]
            n_before = sum(len(r) for r in rows)
            for sd in sorted(sdir.iterdir()):
                if not sd.is_dir(): continue
                try: net, sta = sd.name.split(".")
                except ValueError: continue
                for csv in _candidate_csvs(sd):
                    try: d = pd.read_csv(csv)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError): continue
                    if d.empty: continue
                    d = d.copy()
                    d["t"] = pd.to_datetime(d.time, utc=True, format="ISO8601")
                    d = d[(d.t >= start_pd) & (d.t < end_pd)]
                    d["phase"] = d.phase.str.upper().str[0]
                    d = d[d.phase.isin(want)]
                    if prob_min > 0:
                        d = d[d.prob >= prob_min]
                    if d.empty: continue
                    d["station"] = f"{net}.{sta}"
                    rows.append(d[["station", "t", "phase", "prob"]])
            got = sum(len(r) for r in rows) - n_before
            print(f"  pick source catalogs/{subdir} ({'+'.join(want)}): {got:,} picks",
                  flush=True)
    if picker_pool == "with_manual":
        # Add manual mag07 picks
        m = pd.read_csv(REPO / "catalogs" / "manual_picks.csv", parse_dates=["pick_time"])
        m = m[m.source_file == "nllmaleen_mag07_202210.out"].copy()
        m["t"] = pd.to_datetime(m.pick_time, utc=True)
        m = m[(m.t >= start_pd) & (m.t < end_pd)].copy()
        m["station"] = m.network.astype(str) + "." + m.station.astype(str)
        m["phase"] = m.phase.str.upper().str[0]
        m["prob"] = 1.0
        m = m[m.phase.isin(["P", "S"])]
        rows.append(m[["station", "t", "phase", "prob"]])

    if not rows:
        return pd.DataFrame(columns=["station", "t", "phase", "prob"])
    df = pd.concat(rows, ignore_index=True)
    df = df.sort_values("t").reset_index(drop=True)

    # Two pickers over the same stations emit the SAME arrival twice. pyocto's
    # thresholds (n_picks, n_p_picks, n_s_picks) count PICKS, not stations, and
    # --min-stations is a dead flag, so a genuine 3-station detection whose picks are
    # doubled reaches n_picks=6 and is emitted as a 5-station event. Measured on
    # 2019-011: ~30% of diting picks have a pnlight twin within the match tolerance.
    if dedup_tol and dedup_tol > 0 and len(df):
        before = len(df)
        # df["t"] is datetime64 -- go through timeutil, never a raw cast (pandas 3
        # makes .astype("int64") on datetimes silently 1000x wrong).
        bucket = (epoch_seconds(df["t"]) / dedup_tol).round().astype("int64")
        df = (df.assign(_b=bucket)
                .sort_values("prob", ascending=False)
                .drop_duplicates(subset=["station", "phase", "_b"], keep="first")
                .drop(columns="_b")
                .sort_values("t").reset_index(drop=True))
        print(f"  deduplicated cross-picker doubles: {before-len(df):,} dropped "
              f"({len(df):,} remain, tol={dedup_tol}s)", flush=True)
    return df


def build_stations_df():
    sg = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    sg["station"] = sg.network.astype(str) + "." + sg.station.astype(str)
    sg = sg[["station", "latitude", "longitude", "elevation_m"]].copy()
    sg = sg.rename(columns={"elevation_m": "elevation"})
    sg["elevation"] = sg.elevation.fillna(0)
    sg = sg.drop_duplicates(subset=["station"])
    return sg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--velocity-model", default=str(REPO / "configs" / "velocity_model.csv"))
    ap.add_argument("--vpvs", type=float, default=1.78,
                    help="Vp/Vs ratio used to derive Vs if velocity model is P-only")
    ap.add_argument("--label", default="picker_only",
                    help="output label; 'picker_only' loads PN+OBST, 'with_manual' adds "
                         "manual catalog; any other label uses the picker_only sources")
    # NOTE: pyocto OctoAssociator has no n_stations parameter, so this was silently
    # ignored. Kept only to not break existing callers; warns if set.
    ap.add_argument("--min-stations", type=int, default=DEFAULTS["min_stations"],
                    help="DEAD FLAG - pyocto has no such parameter; has no effect.")
    ap.add_argument("--min-p", type=int, default=DEFAULTS["min_p"])
    ap.add_argument("--min-s", type=int, default=DEFAULTS["min_s"])
    ap.add_argument("--min-total", type=int, default=DEFAULTS["min_total"])
    ap.add_argument("--pick-tol", type=float, default=DEFAULTS["pick_match_tolerance"])
    ap.add_argument("--edt-std", type=float, default=DEFAULTS["edt_pick_std"])
    ap.add_argument("--z-max-km", type=float, default=DEFAULTS["z_max_km"])
    ap.add_argument("--dedup-tol", type=float, default=0.25,
                    help="collapse picks of the same station+phase within this many "
                         "seconds to one (keeping highest prob). Two pickers over the "
                         "same stations otherwise double-count arrivals, and pyocto's "
                         "thresholds count picks not stations. 0 disables.")
    ap.add_argument("--time-before", type=float, default=180.0,
                    help="pyocto time-slice overlap, seconds. Must exceed the maximum S "
                         "travel time across the search domain or events spanning a slice "
                         "boundary lose their late arrivals. Was min_node_size*4 = 40 s, "
                         "which is shorter than S across this network.")
    ap.add_argument("--n-threads", type=int, default=DEFAULTS["n_threads"])
    ap.add_argument("--exclude-stations", default="",
                    help="comma-separated NET.STA to drop before association, e.g. "
                         "'5M.DCP,5M.LVN'. Filters existing picks; never requires re-picking.")
    ap.add_argument("--legacy-origin", action="store_true",
                    help="derive the map origin from the mean position of stations that "
                         "have picks, as before 2026-09-12. Reproduces older runs. NOT safe "
                         "for chunked runs: the origin moves with station availability, so "
                         "chunks land on different coordinate frames.")
    ap.add_argument("--pick-sources", default="picks:P,picks_obst_01:PS",
                    help="comma-separated catalogs/<subdir>:<phases> specs deciding which "
                         "picker output feeds association. Default reproduces the published "
                         "catalogue (PhaseNet instance P + OBSTransformer P/S). Benchmarked "
                         "best pool: 'picks_pn_diting:PS,picks_pnlight_obs:PS'")
    ap.add_argument("--pick-prob-min", type=float, default=0.0,
                    help="drop picks below this probability before association (0 = keep all)")
    ap.add_argument("--margin-seconds", type=float, default=0.0,
                    help="when chunking, expand the pick-load window by this many "
                         "seconds on each side so events near the boundary still "
                         "get all their late-arriving picks. Output events are then "
                         "filtered back to the strict [start, end) origin-time window. "
                         "Set to >= max P travel time across the bbox (~120 s for ours).")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    import pyocto
    from pyproj import CRS
    import time as _time
    import sys as _sys
    import traceback as _tb

    # Force unbuffered prints so progress is visible in real time when piped to a log
    _sys.stdout.reconfigure(line_buffering=True)
    _sys.stderr.reconfigure(line_buffering=True)
    def _log(msg):
        ts = _time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    _t_start = _time.time()
    _log(f"=== PyOcto association  label={args.label}  {args.start} → {args.end} ===")
    start_pd = pd.Timestamp(args.start, tz="UTC")
    end_pd = pd.Timestamp(args.end, tz="UTC")
    # Expanded pick-load window so boundary events see all their late picks
    load_start = start_pd - pd.Timedelta(seconds=args.margin_seconds)
    load_end   = end_pd   + pd.Timedelta(seconds=args.margin_seconds)
    if args.margin_seconds > 0:
        _log(f"chunk strict window: [{start_pd}, {end_pd}); "
             f"load window: [{load_start}, {load_end}) (+{args.margin_seconds:.0f}s margin)")

    _t = _time.time()
    _log("Loading picks ...")
    _sources = []
    for spec in args.pick_sources.split(","):
        spec = spec.strip()
        if not spec: continue
        subdir, _, phases = spec.partition(":")
        _sources.append((subdir, phases or "PS"))
    assert_nanosecond_sanity()
    picks = load_picks_for_pyocto(load_start, load_end, args.label,
                                  dedup_tol=args.dedup_tol,
                                  sources=tuple(_sources),
                                  prob_min=args.pick_prob_min)
    _log(f"  total picks: {len(picks):,}  "
         f"({(picks.phase=='P').sum():,} P + {(picks.phase=='S').sum():,} S)  "
         f"({_time.time()-_t:.1f}s)")
    if picks.empty:
        raise SystemExit("No picks loaded.")
    _log(f"  pick time range: {picks.t.min()} -> {picks.t.max()}")
    _log(f"  stations represented: {picks.station.nunique()}")

    excl = {x.strip() for x in args.exclude_stations.split(",") if x.strip()}
    if excl:
        known = set(picks.station.unique())
        missing = excl - known
        if missing:
            _log(f"  [warn] --exclude-stations lists absent stations: {sorted(missing)}")
        before = len(picks)
        picks = picks[~picks.station.isin(excl)].reset_index(drop=True)
        _log(f"  excluded {sorted(excl & known)}: dropped {before-len(picks):,} picks "
             f"({len(picks):,} remain, {picks.station.nunique()} stations)")
        if picks.empty:
            raise SystemExit("all picks excluded")

    _t = _time.time()
    _log("Loading stations ...")
    stations_all = build_stations_df()          # full network: fixes bbox
    stations = stations_all.copy()
    pick_stas = set(picks.station.unique())
    stations = stations[stations.station.isin(pick_stas)].reset_index(drop=True)
    _log(f"  stations with picks: {len(stations)} of {len(stations_all)} "
         f"({_time.time()-_t:.1f}s)")

    _t = _time.time()
    _log(f"Loading velocity model from {args.velocity_model} ...")
    vel = None   # built after the bbox is known -- the table must span the domain diagonal
    _log(f"  velocity model ready ({_time.time()-_t:.1f}s)")

    # Frozen projection origin (bransfield_eq.geo). NEVER derive this from the
    # stations that happen to have picks: that moved the origin between runs
    # (19.9 km between two pools; ~5 km between two months) and silently put
    # chunks of a year on different coordinate frames.
    geo.assert_roundtrip()
    if args.legacy_origin:
        from pyproj import CRS as _CRS
        _lat0 = stations.latitude.mean(); _lon0 = stations.longitude.mean()
        crs = _CRS.from_proj4(f"+proj=tmerc +lat_0={_lat0} +lon_0={_lon0} +ellps=WGS84")
        _log(f"  [WARN] --legacy-origin: origin DERIVED FROM DATA "
             f"(lat_0={_lat0:.4f} lon_0={_lon0:.4f}) over the {len(stations)} stations "
             f"with picks. It moves if station availability changes; do not use for "
             f"chunked runs that will be merged.")
    else:
        crs = geo.make_crs()
        _log(f"  projection origin (frozen): lat_0={geo.ORIGIN_LAT} lon_0={geo.ORIGIN_LON}")
    # Bounds from the FULL network so the search area is identical in every
    # chunk, regardless of which instruments were recording.
    sx, sy = geo.to_xy(stations_all.latitude.values, stations_all.longitude.values)
    sx, sy = sx * 1e3, sy * 1e3
    xmin, xmax = (sx.min() - 50e3) / 1e3, (sx.max() + 50e3) / 1e3
    ymin, ymax = (sy.min() - 50e3) / 1e3, (sy.max() + 50e3) / 1e3
    _log(f"  bbox: x [{xmin:.1f}, {xmax:.1f}] km, y [{ymin:.1f}, {ymax:.1f}] km, z [0, {args.z_max_km}] km")

    # The travel-time table must span the search-domain diagonal, or pyocto hits
    # minimisation errors near the corners. The old literal xdist=200 km covered a
    # domain whose diagonal is ~407 km.
    import math as _math
    _diag = _math.hypot(xmax - xmin, ymax - ymin)
    _xdist = float(_math.ceil(_math.hypot(_diag, args.z_max_km) + 25.0))
    _log(f"  travel-time table xdist={_xdist:.0f} km (domain diagonal {_diag:.0f} km)")
    _t = _time.time()
    _log(f"Loading velocity model from {args.velocity_model} ...")
    vel = load_velocity_model(Path(args.velocity_model), vpvs_ratio=args.vpvs, xdist=_xdist)
    _log(f"  velocity model ready ({_time.time()-_t:.1f}s)")

    _log(f"Thresholds: min_stations={args.min_stations}, min_P={args.min_p}, "
         f"min_S={args.min_s}, min_total={args.min_total}, pick_tol={args.pick_tol}s, "
         f"n_threads={args.n_threads}")

    _t = _time.time()
    _log("Building associator (Eikonal travel-time tables) ...")
    associator = pyocto.OctoAssociator(
        xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(0.0, args.z_max_km),
        velocity_model=vel,
        time_before=args.time_before,   # >= max S travel time across the domain
        min_node_size=DEFAULTS["min_node_size"],
        min_node_size_location=1.5,
        pick_match_tolerance=args.pick_tol,
        edt_pick_std=args.edt_std,
        n_picks=args.min_total,
        n_p_picks=args.min_p,
        n_s_picks=args.min_s,
        n_p_and_s_picks=args.min_p,  # min stations with both phases
        n_threads=args.n_threads,
        crs=crs,
    )

    _log(f"  associator built ({_time.time()-_t:.1f}s)")

    _t = _time.time()
    stations["id"] = stations["station"]
    associator.transform_stations(stations)
    _log(f"  station z (km below sea level): "
         f"min={stations.z.min():.2f}  max={stations.z.max():.2f}  "
         f"mean={stations.z.mean():.2f}  ({_time.time()-_t:.1f}s)")

    picks_in = picks.rename(columns={"t": "time"}).copy()
    picks_in["time"] = epoch_seconds(picks_in["time"])
    _log(f"Running association on {len(picks_in):,} picks (this is the long call) ...")
    _t = _time.time()
    try:
        events, assoc = associator.associate(picks_in, stations)
    except Exception as e:
        _log(f"!!! associate() raised: {type(e).__name__}: {e}")
        _tb.print_exc()
        raise
    _log(f"  associator wall: {_time.time()-_t:.1f}s")
    _log(f"  events: {len(events):,}")
    _log(f"  associated picks: {len(assoc):,}")

    # If we loaded with a margin, drop events whose origin time falls outside the
    # strict [start, end) window so that adjacent chunks don't double-count them.
    if args.margin_seconds > 0 and not events.empty and "time" in events.columns:
        strict_start = start_pd.timestamp()
        strict_end   = end_pd.timestamp()
        before = len(events)
        in_window = (events.time >= strict_start) & (events.time < strict_end)
        events = events[in_window].reset_index(drop=True)
        # filter assoc to the surviving events
        kept_idx = set(events["idx"].tolist()) if "idx" in events.columns else None
        if kept_idx is not None and "event_idx" in assoc.columns:
            assoc = assoc[assoc.event_idx.isin(kept_idx)].reset_index(drop=True)
        elif kept_idx is not None and "idx" in assoc.columns:
            assoc = assoc[assoc.idx.isin(kept_idx)].reset_index(drop=True)
        _log(f"  margin filter: kept {len(events)}/{before} events in strict window "
             f"[{start_pd}, {end_pd})")

    out_events = REPO / "catalogs" / f"pyocto_events_{args.label}.csv"
    out_picks = REPO / "catalogs" / f"pyocto_picks_{args.label}.csv"
    # Self-describing output: x/y are metres-from-origin internally, but the
    # catalogue also carries true lat/lon so nothing downstream has to know or
    # guess the projection.
    if {"x", "y"}.issubset(events.columns) and len(events):
        if args.legacy_origin:
            from pyproj import Transformer as _T
            _inv = _T.from_crs(crs, "EPSG:4326", always_xy=True)
            _lon, _lat = _inv.transform(events["x"].values * 1e3, events["y"].values * 1e3)
        else:
            _lat, _lon = geo.to_latlon(events["x"].values, events["y"].values)
        events["latitude"] = _lat
        events["longitude"] = _lon
        if "z" in events.columns and "depth" not in events.columns:
            events["depth"] = events["z"]
    events.to_csv(out_events, index=False)
    assoc.to_csv(out_picks, index=False)
    import json as _json
    (REPO / "catalogs" / f"pyocto_origin_{args.label}.json").write_text(
        _json.dumps({**geo.origin_metadata(), "legacy_origin": bool(args.legacy_origin),
                     "proj4_used": crs.to_proj4(),
                     "excluded_stations": sorted(excl)}, indent=2))
    print(f"\n  wrote {out_events}")
    print(f"  wrote {out_picks}")

    # Summary
    if not events.empty:
        print(f"\n=== Event catalog summary ({args.label}) ===")
        if "n_p_picks" in events.columns:
            print(f"  median picks/event:  {events.n_picks.median():.0f}  "
                  f"(P: {events.n_p_picks.median():.0f}, S: {events.n_s_picks.median():.0f})")
        if "rms_residual" in events.columns:
            print(f"  median RMS residual: {events.rms_residual.median():.3f}s")
        if "depth" in events.columns:
            print(f"  depth range:         {events.depth.min():.1f} - {events.depth.max():.1f} km")


if __name__ == "__main__":
    main()
