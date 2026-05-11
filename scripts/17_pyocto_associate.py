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
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

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


def load_velocity_model(path: Path, vpvs_ratio: float = 1.78):
    """Sniff and load a 1D velocity model. Returns pyocto.VelocityModel1D.

    Accepts:
      - 3-col with Vs:    depth | Vp | Vs
      - 2-col P-only:     depth | Vp        (Vs derived as Vp / vpvs_ratio)
      - Auto units: depth in m or km; Vp in m/s or km/s.
    """
    import pyocto
    p = Path(path)
    if not p.exists():
        print(f"  [warn] {p} does not exist; using homogeneous Vp=5.5/Vs=3.1 km/s "
              f"(Vp/Vs={5.5/3.1:.2f})")
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
    cache = p.with_suffix(".pyocto")
    if not cache.exists() or cache.stat().st_mtime < p.stat().st_mtime:
        print(f"  building pyocto velocity cache → {cache.name}")
        model_df = pd.DataFrame({"depth": depth, "vp": vp, "vs": vs})
        # Grid covers Bransfield network footprint with comfortable margin.
        pyocto.VelocityModel1D.create_model(
            model=model_df, delta=1.0, xdist=200.0, zdist=50.0, path=cache,
        )
    return pyocto.VelocityModel1D(path=cache, tolerance=2.0)


def load_picks_for_pyocto(start_pd, end_pd, picker_pool: str) -> pd.DataFrame:
    """Build the unified pick dataframe in PyOcto format:
        columns = station, time (UTCDateTime), phase ('P' or 'S'), prob (optional)
    """
    rows = []
    if picker_pool in ("picker_only", "with_manual"):
        # PhaseNet P only
        pn_dir = REPO / "catalogs" / "picks"
        if pn_dir.exists():
            for sd in sorted(pn_dir.iterdir()):
                if not sd.is_dir(): continue
                try: net, sta = sd.name.split(".")
                except ValueError: continue
                for csv in sorted(sd.glob("*.csv")):
                    try: d = pd.read_csv(csv)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError): continue
                    if d.empty: continue
                    d = d.copy()
                    d["t"] = pd.to_datetime(d.time, utc=True, format="ISO8601")
                    d = d[(d.t >= start_pd) & (d.t < end_pd)]
                    d = d[d.phase.str.upper().str[0] == "P"]   # PhaseNet P only
                    if d.empty: continue
                    d["station"] = f"{net}.{sta}"
                    d["phase"] = "P"
                    rows.append(d[["station", "t", "phase", "prob"]])
        # OBSTransformer P+S
        ob_dir = REPO / "catalogs" / "picks_obst_01"
        if ob_dir.exists():
            for sd in sorted(ob_dir.iterdir()):
                if not sd.is_dir(): continue
                try: net, sta = sd.name.split(".")
                except ValueError: continue
                for csv in sorted(sd.glob("*.csv")):
                    try: d = pd.read_csv(csv)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError): continue
                    if d.empty: continue
                    d = d.copy()
                    d["t"] = pd.to_datetime(d.time, utc=True, format="ISO8601")
                    d = d[(d.t >= start_pd) & (d.t < end_pd)]
                    d["phase"] = d.phase.str.upper().str[0]
                    d = d[d.phase.isin(["P", "S"])]
                    if d.empty: continue
                    d["station"] = f"{net}.{sta}"
                    rows.append(d[["station", "t", "phase", "prob"]])
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
                    choices=["picker_only", "with_manual"])
    ap.add_argument("--min-stations", type=int, default=DEFAULTS["min_stations"])
    ap.add_argument("--min-p", type=int, default=DEFAULTS["min_p"])
    ap.add_argument("--min-s", type=int, default=DEFAULTS["min_s"])
    ap.add_argument("--min-total", type=int, default=DEFAULTS["min_total"])
    ap.add_argument("--pick-tol", type=float, default=DEFAULTS["pick_match_tolerance"])
    ap.add_argument("--edt-std", type=float, default=DEFAULTS["edt_pick_std"])
    ap.add_argument("--z-max-km", type=float, default=DEFAULTS["z_max_km"])
    ap.add_argument("--n-threads", type=int, default=DEFAULTS["n_threads"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    import pyocto
    from pyproj import CRS

    print(f"=== PyOcto association  label={args.label}  {args.start} → {args.end} ===")
    start_pd = pd.Timestamp(args.start, tz="UTC")
    end_pd = pd.Timestamp(args.end, tz="UTC")

    print("\nLoading picks ...")
    picks = load_picks_for_pyocto(start_pd, end_pd, args.label)
    print(f"  total picks: {len(picks):,}  ({(picks.phase=='P').sum():,} P + {(picks.phase=='S').sum():,} S)")
    if picks.empty:
        raise SystemExit("No picks loaded.")

    print("\nLoading stations ...")
    stations = build_stations_df()
    # Restrict to stations present in picks
    pick_stas = set(picks.station.unique())
    stations = stations[stations.station.isin(pick_stas)].reset_index(drop=True)
    print(f"  stations with picks: {len(stations)}")

    print(f"\nLoading velocity model from {args.velocity_model} ...")
    vel = load_velocity_model(Path(args.velocity_model), vpvs_ratio=args.vpvs)

    # Compute spatial extent in km from station bounding box
    lat0 = stations.latitude.mean()
    lon0 = stations.longitude.mean()
    crs = CRS.from_proj4(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +ellps=WGS84")
    # Compute bounds w/ margin
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    sx, sy = tx.transform(stations.longitude.values, stations.latitude.values)
    xmin, xmax = (sx.min() - 50e3) / 1e3, (sx.max() + 50e3) / 1e3
    ymin, ymax = (sy.min() - 50e3) / 1e3, (sy.max() + 50e3) / 1e3
    print(f"  bbox: x [{xmin:.1f}, {xmax:.1f}] km, y [{ymin:.1f}, {ymax:.1f}] km, z [0, {args.z_max_km}] km")

    print(f"\nThresholds: min_stations={args.min_stations}, min_P={args.min_p}, "
          f"min_S={args.min_s}, min_total={args.min_total}, pick_tol={args.pick_tol}s")

    # Build associator
    associator = pyocto.OctoAssociator(
        xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(0.0, args.z_max_km),
        velocity_model=vel,
        time_before=DEFAULTS["min_node_size"] * 4,
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

    # PyOcto >=0.6 requires an 'id' column on stations
    stations["id"] = stations["station"]
    # PyOcto requires x/y/z columns on stations -- project lat/lon/elevation
    # via the associator's CRS. elevation is in metres in the inventory; pyocto
    # interprets it as positive-up, so OBS (negative elev) end up at positive z.
    associator.transform_stations(stations)
    print(f"  station z (km below sea level): "
          f"min={stations.z.min():.2f}  max={stations.z.max():.2f}  "
          f"mean={stations.z.mean():.2f}")

    # PyOcto pick format: DataFrame with cols 'station', 'time' (datetime64), 'phase'
    picks_in = picks.rename(columns={"t": "time"}).copy()
    # pyocto's Pick constructor expects float epoch seconds, not pd.Timestamp
    picks_in["time"] = picks_in["time"].astype("int64") / 1e9
    print(f"\nRunning association on {len(picks_in):,} picks ...")
    import time as _time
    t0 = _time.time()
    events, assoc = associator.associate(picks_in, stations)
    dt = _time.time() - t0
    print(f"  associator wall: {dt:.1f}s")
    print(f"  events: {len(events):,}")
    print(f"  associated picks: {len(assoc):,}")

    out_events = REPO / "catalogs" / f"pyocto_events_{args.label}.csv"
    out_picks = REPO / "catalogs" / f"pyocto_picks_{args.label}.csv"
    events.to_csv(out_events, index=False)
    assoc.to_csv(out_picks, index=False)
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
