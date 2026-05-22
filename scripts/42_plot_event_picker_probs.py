"""Record-section plot with PhaseNet + OBSTransformer pick-probability overlays.

Same waveform layout as `41_plot_event_record_sections.py`, but overlays each
picker's *saved* picks within the event window as colored, prob-scaled blocks
spanning the `start`-to-`end` window stored in the per-station-day pick CSVs.

This uses ONLY information already on disk (no model re-run): each block's
height is the saved `prob`, the block spans the `[start, end]` interval, and
the peak time is marked with a thin tick. Sub-threshold candidates are not
visible because they were never written out — only picks above the picking
threshold are plotted.

Default: event_idx 14341 at the picker_only_no_shots configuration.

Usage:
    python scripts/42_plot_event_picker_probs.py
    python scripts/42_plot_event_picker_probs.py --event-idx 26819
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from obspy import UTCDateTime, read, Stream

REPO = Path(__file__).resolve().parent.parent
WF_DIR = REPO / "data" / "waveforms"
ST_CSV = REPO / "catalogs" / "station_geometry.csv"
EV_CSV = REPO / "catalogs" / "pyocto_events_picker_only_no_shots.csv"
PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv"
NLLOC_CSV = REPO / "catalogs" / "nlloc_picker_only_no_shots_v2.csv"
GRID_DIR = REPO / "nlloc" / "time"
GRID_TAG = "ORCA_v2"
VPVS = 1.78

# Picker pick directories (per-station-day CSVs of thresholded picks).
PICKER_DIRS = {
    "PhaseNet":       REPO / "catalogs" / "picks",
    "OBSTransformer": REPO / "catalogs" / "picks_obst_01",
}

# (phase, picker) -> color
PICK_COLORS = {
    ("P", "PhaseNet"):       "#d62728",   # red
    ("S", "PhaseNet"):       "#1f77b4",   # blue
    ("P", "OBSTransformer"): "#ff8c00",   # orange
    ("S", "OBSTransformer"): "#17becf",   # cyan
}

OUT_DIR = REPO / "notes" / "figures" / "associations"


def _read_grid_header(hdr_path: Path):
    lines = hdr_path.read_text().strip().splitlines()
    p = lines[0].split()
    nx, ny, nz = int(p[0]), int(p[1]), int(p[2])
    x0, y0, z0 = float(p[3]), float(p[4]), float(p[5])
    dx, dy, dz = float(p[6]), float(p[7]), float(p[8])
    p = lines[2].split()
    lat_orig = float(p[p.index("LatOrig") + 1])
    lon_orig = float(p[p.index("LongOrig") + 1])
    rot_cw = float(p[p.index("RotCW") + 1])
    return dict(nx=nx, ny=ny, nz=nz, x0=x0, y0=y0, z0=z0,
                dx=dx, dy=dy, dz=dz,
                lat_orig=lat_orig, lon_orig=lon_orig, rot_cw=rot_cw)


def _latlon_to_grid_xy(lat, lon, h):
    dx_eq = (lon - h["lon_orig"]) * 111.111 * np.cos(np.deg2rad(h["lat_orig"]))
    dy_eq = (lat - h["lat_orig"]) * 111.111
    th = np.deg2rad(h["rot_cw"])
    return (dx_eq * np.cos(th) + dy_eq * np.sin(th),
            -dx_eq * np.sin(th) + dy_eq * np.cos(th))


def _trilinear(buf_path, h, x, y, z):
    fx = (x - h["x0"]) / h["dx"]
    fy = (y - h["y0"]) / h["dy"]
    fz = (z - h["z0"]) / h["dz"]
    if not (0 <= fx <= h["nx"] - 1 and 0 <= fy <= h["ny"] - 1
            and 0 <= fz <= h["nz"] - 1):
        return float("nan")
    ix, iy, iz = int(fx), int(fy), int(fz)
    ix = min(ix, h["nx"] - 2); iy = min(iy, h["ny"] - 2); iz = min(iz, h["nz"] - 2)
    rx, ry, rz = fx - ix, fy - iy, fz - iz
    arr = np.memmap(buf_path, dtype="<f4", mode="r",
                    shape=(h["nx"], h["ny"], h["nz"]))
    c00 = arr[ix, iy, iz] * (1 - rx) + arr[ix + 1, iy, iz] * rx
    c01 = arr[ix, iy, iz + 1] * (1 - rx) + arr[ix + 1, iy, iz + 1] * rx
    c10 = arr[ix, iy + 1, iz] * (1 - rx) + arr[ix + 1, iy + 1, iz] * rx
    c11 = arr[ix, iy + 1, iz + 1] * (1 - rx) + arr[ix + 1, iy + 1, iz + 1] * rx
    c0 = c00 * (1 - ry) + c10 * ry
    c1 = c01 * (1 - ry) + c11 * ry
    return float(c0 * (1 - rz) + c1 * rz)


def predict_arrivals(ev_lat, ev_lon, ev_depth_km):
    """Return {f'{NET}.{STA}': (tP, tS)} from NLLoc v2 P grids; S = tP*VPVS."""
    hdrs = sorted(GRID_DIR.glob(f"{GRID_TAG}.P.*.time.hdr"))
    hdrs = [p for p in hdrs if ".mod." not in p.name]
    if not hdrs:
        return {}
    ref = _read_grid_header(hdrs[0])
    ex, ey = _latlon_to_grid_xy(ev_lat, ev_lon, ref)
    out: dict[str, tuple[float, float]] = {}
    for hp in hdrs:
        sta = hp.name.split(".")[2]
        h = _read_grid_header(hp)
        tp = _trilinear(hp.with_suffix(".buf"), h, ex, ey, ev_depth_km)
        if np.isnan(tp):
            continue
        out[sta] = (tp, tp * VPVS)
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_z_trace(net: str, sta: str, t0: UTCDateTime, t1: UTCDateTime,
                 band_low: float, band_high: float):
    sta_dir = WF_DIR / net / sta
    if not sta_dir.exists():
        return None
    days = sorted({(t0.year, t0.julday), (t1.year, t1.julday)})
    streams = []
    for yr, doy in days:
        fname = sta_dir / f"{net}.{sta}.{yr}.{doy:03d}.mseed"
        if not fname.exists():
            continue
        try:
            streams.extend(read(str(fname)))
        except Exception:
            continue
    if not streams:
        return None
    stream = Stream(streams)
    z = stream.select(component="Z")
    if len(z) == 0:
        return None
    try:
        z = z.merge(fill_value=0)
        z.trim(t0, t1, pad=True, fill_value=0)
        tr = z[0]
        if tr.stats.npts < 10:
            return None
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=band_low, freqmax=band_high,
                  corners=4, zerophase=True)
        return tr
    except Exception:
        return None


def load_picker_picks(picker_dir: Path, net: str, sta: str,
                      t0: UTCDateTime, t1: UTCDateTime) -> pd.DataFrame:
    """Load a station's picks in [t0, t1] from per-day CSVs; may span days."""
    sta_key = f"{net}.{sta}"
    pdir = picker_dir / sta_key
    if not pdir.exists():
        return pd.DataFrame()
    days = sorted({(t0.year, t0.julday), (t1.year, t1.julday)})
    frames = []
    for yr, doy in days:
        fp = pdir / f"{yr}-{doy:03d}.csv"
        if not fp.exists():
            continue
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    # ObsPy parses ISO8601 with trailing Z; pandas can too.
    df["time_obj"] = pd.to_datetime(df["time"], utc=True)
    df["start_obj"] = pd.to_datetime(df["start"], utc=True)
    df["end_obj"] = pd.to_datetime(df["end"], utc=True)
    ts0 = pd.Timestamp(t0.datetime, tz="UTC")
    ts1 = pd.Timestamp(t1.datetime, tz="UTC")
    mask = (df["time_obj"] >= ts0) & (df["time_obj"] <= ts1)
    return df.loc[mask].reset_index(drop=True)


def plot_event(ev_row, picks_df, stations_df, args) -> Path:
    eidx = int(ev_row.event_idx)
    ot = UTCDateTime(float(ev_row.time))
    t0 = ot - args.window_half_s
    t1 = ot + args.window_half_s

    sd = stations_df.copy()
    sd["dist_km"] = haversine_km(ev_row.latitude, ev_row.longitude,
                                 sd.latitude.values, sd.longitude.values)
    sd = sd.sort_values("dist_km").reset_index(drop=True)

    ev_pyocto = picks_df[picks_df.event_idx == eidx]
    pyocto_lookup: dict[str, dict[str, float]] = {}
    for _, pk in ev_pyocto.iterrows():
        d = pyocto_lookup.setdefault(pk.station, {})
        d[pk.phase] = float(pk.time) - ot.timestamp

    # NLLoc v2 grid predictions. Use the NLLoc-relocated hypocenter (self-
    # consistent with the v2 grids); fall back to pyocto's if not in v2 cat.
    pred_lat, pred_lon, pred_depth = (ev_row.latitude, ev_row.longitude,
                                      float(ev_row.depth))
    pred_src = "pyocto"
    if NLLOC_CSV.exists():
        nl = pd.read_csv(NLLOC_CSV)
        m = nl[nl.event_idx == eidx]
        if not m.empty:
            r = m.iloc[0]
            pred_lat, pred_lon, pred_depth = float(r.lat), float(r.lon), float(r.depth_km)
            pred_src = "NLLoc v2"
    print(f"  prediction hypocenter ({pred_src}): "
          f"lat={pred_lat:.4f} lon={pred_lon:.4f} depth={pred_depth:.3f} km")
    pred = predict_arrivals(pred_lat, pred_lon, pred_depth)

    # Load traces + picker picks per station
    rows = []
    for _, sr in sd.iterrows():
        tr = load_z_trace(sr.network, sr.station, t0, t1,
                          args.band_low, args.band_high)
        if tr is None:
            continue
        maxabs = float(np.max(np.abs(tr.data))) or 1.0
        picker_pks = {
            name: load_picker_picks(pdir, sr.network, sr.station, t0, t1)
            for name, pdir in PICKER_DIRS.items()
        }
        rows.append((sr, tr, maxabs, picker_pks))

    n = len(rows)
    fig_h = max(5.0, 0.42 * n + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    yticks, ylabels = [], []

    # Half-height of a station "row" available for prob blocks
    row_half = 0.45

    for i, (sr, tr, maxabs, picker_pks) in enumerate(rows):
        t_rel = tr.times() - args.window_half_s
        y = tr.data / (2.4 * maxabs)
        y_offset = -i
        full_label = f"{sr.network}.{sr.station}"
        has_pyocto = full_label in pyocto_lookup
        wf_color = "0.15" if has_pyocto else "0.55"
        ax.plot(t_rel, y + y_offset, color=wf_color, linewidth=0.5, zorder=4)
        yticks.append(y_offset)
        ylabels.append(f"{full_label}  {sr.dist_km:6.1f} km")

        # Picker probability blocks. Stack each picker on a different baseline
        # within the row: PhaseNet just above the trace mid, OBST just below.
        picker_baselines = {
            "PhaseNet":       y_offset + 0.05,
            "OBSTransformer": y_offset - 0.05,
        }
        picker_signs = {
            "PhaseNet":       +1.0,   # grow upward
            "OBSTransformer": -1.0,   # grow downward
        }
        for picker_name, df in picker_pks.items():
            if df.empty:
                continue
            base = picker_baselines[picker_name]
            sign = picker_signs[picker_name]
            for _, pk in df.iterrows():
                color = PICK_COLORS.get((pk.phase, picker_name), "gray")
                ts = (UTCDateTime(pk["start"]) - ot)
                te = (UTCDateTime(pk["end"]) - ot)
                tp = (UTCDateTime(pk["time"]) - ot)
                h = float(pk["prob"]) * row_half * 0.9  # scale to row_half max
                rect = Rectangle(
                    (ts, base),
                    te - ts,
                    sign * h,
                    facecolor=color, edgecolor=color,
                    alpha=0.35, linewidth=0.6, zorder=5,
                )
                ax.add_patch(rect)
                # Peak time tick
                ax.plot([tp, tp], [base, base + sign * h],
                        color=color, linewidth=1.0, alpha=0.95, zorder=6)

        # Predicted P/S from NLLoc v2 grid (dashed, full row height)
        if sr.station in pred:
            tp_pred, ts_pred = pred[sr.station]
            ax.plot([tp_pred, tp_pred],
                    [y_offset - 0.55, y_offset + 0.55],
                    color="#00cc00", linewidth=2.0, linestyle="--",
                    alpha=1.0, zorder=20)
            ax.plot([ts_pred, ts_pred],
                    [y_offset - 0.55, y_offset + 0.55],
                    color="#ff00ff", linewidth=2.0, linestyle="--",
                    alpha=1.0, zorder=20)

        # Overlay pyocto picks as solid vertical lines
        if has_pyocto:
            for phase, dt in pyocto_lookup[full_label].items():
                col = "black" if phase == "P" else "0.3"
                ls = "-" if phase == "P" else "--"
                ax.plot([dt, dt], [y_offset - 0.48, y_offset + 0.48],
                        color=col, linewidth=1.2, linestyle=ls, zorder=7)

    ax.axvline(0, color="goldenrod", linewidth=1.0, linestyle="--",
               zorder=3, label="origin time")
    ax.set_xlim(-args.window_half_s, args.window_half_s)
    ax.set_ylim(-(n - 1) - 0.9, 0.9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("seconds from pyocto origin")

    # Legend
    legend_items = [
        plt.Line2D([0], [0], color=PICK_COLORS[("P", "PhaseNet")], lw=6,
                   alpha=0.5, label="PhaseNet P (upward)"),
        plt.Line2D([0], [0], color=PICK_COLORS[("S", "PhaseNet")], lw=6,
                   alpha=0.5, label="PhaseNet S (upward)"),
        plt.Line2D([0], [0], color=PICK_COLORS[("P", "OBSTransformer")], lw=6,
                   alpha=0.5, label="OBST P (downward)"),
        plt.Line2D([0], [0], color=PICK_COLORS[("S", "OBSTransformer")], lw=6,
                   alpha=0.5, label="OBST S (downward)"),
        plt.Line2D([0], [0], color="black", lw=1.4, label="pyocto P"),
        plt.Line2D([0], [0], color="0.3", lw=1.4, linestyle="--",
                   label="pyocto S"),
        plt.Line2D([0], [0], color="#00cc00", lw=2.0,
                   linestyle="--", label="predicted P (NLLoc v2)"),
        plt.Line2D([0], [0], color="#ff00ff", lw=2.0,
                   linestyle="--", label="predicted S (NLLoc v2)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=7,
              ncol=3, framealpha=0.9)

    ax.set_title(
        f"event {eidx} — {ot.isoformat()[:19]}  "
        f"({ev_row.latitude:.3f}, {ev_row.longitude:.3f}, "
        f"{ev_row.depth:.1f} km bsf)  npicks={int(ev_row.picks)}, "
        f"traces shown={n}\n"
        f"box span = pick start→end, box height = saved prob, tick = peak time"
    )
    ax.grid(alpha=0.3, axis="x")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ot.strftime("%Y%m%dT%H%M%S")
    out_path = OUT_DIR / f"event_{eidx:06d}_{stamp}_pickerprobs.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"event {eidx}: {n} traces -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-idx", type=int, default=14341)
    ap.add_argument("--window-half-s", type=float, default=30.0)
    ap.add_argument("--band-low", type=float, default=2.0)
    ap.add_argument("--band-high", type=float, default=20.0)
    args = ap.parse_args()

    ev = pd.read_csv(EV_CSV)
    pk = pd.read_csv(PK_CSV)
    st = pd.read_csv(ST_CSV)

    row = ev[ev.event_idx == args.event_idx]
    if row.empty:
        raise SystemExit(f"event_idx {args.event_idx} not in {EV_CSV.name}")
    plot_event(row.iloc[0], pk, st, args)


if __name__ == "__main__":
    main()
