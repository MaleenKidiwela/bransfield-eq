"""Predict P/S arrivals for a pyocto event using NLLoc v2 travel-time grids.

For a given event (default 14341), interpolates the per-station P travel-time
grid at the event hypocenter, multiplies by Vp/Vs=1.78 to get S, and prints
a table comparing predictions to existing pyocto picks. Optionally overlays
predicted arrivals as dashed vertical lines on the record section produced
by `42_plot_event_picker_probs.py`.

The NLLoc v2 grids encode the canonical 1D velocity model + station geometry
used by Stage 5, so predictions are self-consistent with the v2 location.

Usage:
    python scripts/43_predict_arrivals.py
    python scripts/43_predict_arrivals.py --event-idx 26819
    python scripts/43_predict_arrivals.py --plot
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime

REPO = Path(__file__).resolve().parent.parent
GRID_DIR = REPO / "nlloc" / "time"
GRID_TAG = "ORCA_v2"
EV_CSV = REPO / "catalogs" / "pyocto_events_picker_only_no_shots.csv"
PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv"
ST_CSV = REPO / "catalogs" / "station_geometry.csv"
VPVS = 1.78


@dataclass
class GridHeader:
    nx: int; ny: int; nz: int
    x0: float; y0: float; z0: float
    dx: float; dy: float; dz: float
    sta_x: float; sta_y: float; sta_z: float
    lat_orig: float; lon_orig: float; rot_cw: float


def read_grid_header(hdr_path: Path) -> GridHeader:
    lines = hdr_path.read_text().strip().splitlines()
    # Line 1: "nx ny nz  x0 y0 z0  dx dy dz  TIME FLOAT"
    p = lines[0].split()
    nx, ny, nz = int(p[0]), int(p[1]), int(p[2])
    x0, y0, z0 = float(p[3]), float(p[4]), float(p[5])
    dx, dy, dz = float(p[6]), float(p[7]), float(p[8])
    # Line 2: "<STA> sta_x sta_y sta_z"
    p = lines[1].split()
    sta_x, sta_y, sta_z = float(p[1]), float(p[2]), float(p[3])
    # Line 3: "TRANSFORM SIMPLE LatOrig <lat> LongOrig <lon> RotCW <deg>"
    p = lines[2].split()
    lat_orig = float(p[p.index("LatOrig") + 1])
    lon_orig = float(p[p.index("LongOrig") + 1])
    rot_cw = float(p[p.index("RotCW") + 1])
    return GridHeader(nx, ny, nz, x0, y0, z0, dx, dy, dz,
                      sta_x, sta_y, sta_z, lat_orig, lon_orig, rot_cw)


def latlon_to_grid_xy(lat: float, lon: float, hdr: GridHeader) -> tuple[float, float]:
    """NLLoc TRANS SIMPLE projection: equirectangular then rotate CW by rot_cw."""
    dx_eq = (lon - hdr.lon_orig) * 111.111 * np.cos(np.deg2rad(hdr.lat_orig))
    dy_eq = (lat - hdr.lat_orig) * 111.111
    th = np.deg2rad(hdr.rot_cw)
    x = dx_eq * np.cos(th) + dy_eq * np.sin(th)
    y = -dx_eq * np.sin(th) + dy_eq * np.cos(th)
    return x, y


def trilinear_interp_buf(buf_path: Path, hdr: GridHeader,
                         x: float, y: float, z: float) -> float:
    """Trilinear interp of P travel time at grid coords (x, y, z) in km.

    NLLoc storage order: iz fastest, then iy, then ix.
    """
    fx = (x - hdr.x0) / hdr.dx
    fy = (y - hdr.y0) / hdr.dy
    fz = (z - hdr.z0) / hdr.dz
    if not (0 <= fx <= hdr.nx - 1 and 0 <= fy <= hdr.ny - 1
            and 0 <= fz <= hdr.nz - 1):
        return float("nan")
    ix, iy, iz = int(fx), int(fy), int(fz)
    ix = min(ix, hdr.nx - 2); iy = min(iy, hdr.ny - 2); iz = min(iz, hdr.nz - 2)
    rx, ry, rz = fx - ix, fy - iy, fz - iz

    arr = np.memmap(buf_path, dtype="<f4", mode="r",
                    shape=(hdr.nx, hdr.ny, hdr.nz))
    c000 = arr[ix, iy, iz];   c001 = arr[ix, iy, iz + 1]
    c010 = arr[ix, iy + 1, iz];   c011 = arr[ix, iy + 1, iz + 1]
    c100 = arr[ix + 1, iy, iz];   c101 = arr[ix + 1, iy, iz + 1]
    c110 = arr[ix + 1, iy + 1, iz];   c111 = arr[ix + 1, iy + 1, iz + 1]
    c00 = c000 * (1 - rx) + c100 * rx
    c01 = c001 * (1 - rx) + c101 * rx
    c10 = c010 * (1 - rx) + c110 * rx
    c11 = c011 * (1 - rx) + c111 * rx
    c0 = c00 * (1 - ry) + c10 * ry
    c1 = c01 * (1 - ry) + c11 * ry
    return float(c0 * (1 - rz) + c1 * rz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-idx", type=int, default=14341)
    ap.add_argument("--plot", action="store_true",
                    help="Overlay predictions on a record section")
    args = ap.parse_args()

    ev = pd.read_csv(EV_CSV)
    row = ev[ev.event_idx == args.event_idx]
    if row.empty:
        raise SystemExit(f"event_idx {args.event_idx} not found")
    row = row.iloc[0]
    ot = UTCDateTime(float(row.time))
    print(f"event {args.event_idx} @ {ot.isoformat()[:19]}  "
          f"lat={row.latitude:.4f} lon={row.longitude:.4f} "
          f"depth={row.depth:.3f} km bsf")

    # Pyocto picks for this event (for comparison)
    pk = pd.read_csv(PK_CSV)
    ev_pk = pk[pk.event_idx == args.event_idx].copy()
    ev_pk["dt_s"] = ev_pk["time"].astype(float) - ot.timestamp
    pyocto_lookup: dict[str, dict[str, float]] = {}
    for _, r in ev_pk.iterrows():
        pyocto_lookup.setdefault(r.station, {})[r.phase] = float(r["dt_s"])

    # Discover stations from grid files
    hdrs = sorted(GRID_DIR.glob(f"{GRID_TAG}.P.*.time.hdr"))
    hdrs = [h for h in hdrs if ".mod." not in h.name]
    print(f"grid stations available: {len(hdrs)}\n")

    # Load one header to do the lat/lon → grid xy projection
    ref = read_grid_header(hdrs[0])
    ex, ey = latlon_to_grid_xy(row.latitude, row.longitude, ref)
    ez = float(row.depth)
    print(f"event grid coords: x={ex:.3f}  y={ey:.3f}  z={ez:.3f} km\n")

    # Stations from station_geometry to attach network and dist
    st = pd.read_csv(ST_CSV)
    sta_to_net = dict(zip(st.station, st.network))

    R = 6371.0
    def hav(lat1, lon1, lat2, lon2):
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = p2 - p1
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    sta_ll = dict(zip(st.station, zip(st.latitude, st.longitude)))

    rows_out = []
    for hp in hdrs:
        sta = hp.name.split(".")[2]
        hdr = read_grid_header(hp)
        bp = hp.with_suffix(".buf")
        tp = trilinear_interp_buf(bp, hdr, ex, ey, ez)
        if np.isnan(tp):
            continue
        ts = tp * VPVS
        net = sta_to_net.get(sta, "?")
        full = f"{net}.{sta}"
        d_km = hav(row.latitude, row.longitude, *sta_ll[sta]) if sta in sta_ll else np.nan
        p_obs = pyocto_lookup.get(full, {}).get("P")
        s_obs = pyocto_lookup.get(full, {}).get("S")
        rows_out.append({
            "station": full, "dist_km": d_km,
            "tP_pred": tp, "tS_pred": ts,
            "tP_obs": p_obs, "tS_obs": s_obs,
            "P_res": (p_obs - tp) if p_obs is not None else None,
            "S_res": (s_obs - ts) if s_obs is not None else None,
        })

    df = pd.DataFrame(rows_out).sort_values("dist_km").reset_index(drop=True)
    pd.set_option("display.float_format", lambda v: f"{v:7.3f}" if v is not None else "    nan")
    print(df.to_string(index=False))

    # Summary
    print()
    print(f"  stations with grid prediction: {len(df)}")
    have_p = df.tP_obs.notna().sum()
    have_s = df.tS_obs.notna().sum()
    print(f"  pyocto already has P on:        {have_p}")
    print(f"  pyocto already has S on:        {have_s}")
    print(f"  stations w/o pyocto P (candidates for refinement): {len(df) - have_p}")
    print(f"  stations w/o pyocto S:                              {len(df) - have_s}")
    if df.P_res.notna().any():
        print(f"  P residual (obs−pred): med={df.P_res.median():+.3f} s  "
              f"|med|={df.P_res.abs().median():.3f} s  N={df.P_res.notna().sum()}")
    if df.S_res.notna().any():
        print(f"  S residual (obs−pred): med={df.S_res.median():+.3f} s  "
              f"|med|={df.S_res.abs().median():.3f} s  N={df.S_res.notna().sum()}")

    if args.plot:
        out_csv = REPO / "notes" / "figures" / "associations" / \
                  f"event_{args.event_idx:06d}_predictions.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
