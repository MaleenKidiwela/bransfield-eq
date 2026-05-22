"""Record-section plots for sampled pyocto associations.

For each selected event, draws all available station Z-component traces
centered on the event origin time, ordered by epicentral distance, with
pyocto P/S picks overlaid. Stations *without* picks are still plotted
so it's easy to spot un-picked but visible arrivals.

Default selection: 10 random events (seed=42) from
`pyocto_events_picker_only_no_shots.csv` filtered to picks >= 15.

Usage:
    python scripts/41_plot_event_record_sections.py
    python scripts/41_plot_event_record_sections.py --event-idx 12345
    python scripts/41_plot_event_record_sections.py --n-events 5 --window-half-s 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
WF_DIR = REPO / "data" / "waveforms"
ST_CSV = REPO / "catalogs" / "station_geometry.csv"
EV_CSV = REPO / "catalogs" / "pyocto_events_picker_only_no_shots.csv"
PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv"
OUT_DIR = REPO / "notes" / "figures" / "associations"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_station_trace(net: str, sta: str, t0: UTCDateTime, t1: UTCDateTime,
                       band_low: float, band_high: float):
    """Read the Z component for one station in [t0, t1]. Handles day-crossing."""
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
            st = read(str(fname))
        except Exception:
            continue
        streams.extend(st)
    if not streams:
        return None
    from obspy import Stream
    stream = Stream(streams)
    # Pick the Z component (variants: BHZ, HHZ, EHZ, SHZ, ZH on OBS)
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


def plot_event(ev_row, picks_df, stations_df, args) -> Path | None:
    eidx = int(ev_row.event_idx)
    ot = UTCDateTime(float(ev_row.time))   # unix epoch seconds, robust
    t0 = ot - args.window_half_s
    t1 = ot + args.window_half_s

    # Distances per station
    sd = stations_df.copy()
    sd["dist_km"] = haversine_km(ev_row.latitude, ev_row.longitude,
                                 sd.latitude.values, sd.longitude.values)
    sd = sd.sort_values("dist_km").reset_index(drop=True)

    ev_picks = picks_df[picks_df.event_idx == eidx]
    pick_lookup: dict[str, dict[str, float]] = {}
    for _, pk in ev_picks.iterrows():
        d = pick_lookup.setdefault(pk.station, {})
        # absolute pick time in seconds since origin
        d[pk.phase] = float(pk.time) - ot.timestamp

    # Read traces
    traces = []
    for _, sr in sd.iterrows():
        tr = load_station_trace(sr.network, sr.station, t0, t1,
                                args.band_low, args.band_high)
        if tr is None:
            continue
        # Normalize per trace
        maxabs = float(np.max(np.abs(tr.data))) or 1.0
        traces.append((sr, tr, maxabs))

    if not traces:
        print(f"  event {eidx}: no traces found, skipping")
        return None

    n = len(traces)
    fig_h = max(4.5, 0.32 * n + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    yticks, ylabels = [], []

    for i, (sr, tr, maxabs) in enumerate(traces):
        t_rel = tr.times() - args.window_half_s   # so 0 = origin
        y = tr.data / (2.2 * maxabs)              # leaves headroom
        y_offset = -i
        full_label = f"{sr.network}.{sr.station}"
        has_picks = full_label in pick_lookup
        color = "0.15" if has_picks else "0.55"
        ax.plot(t_rel, y + y_offset, color=color, linewidth=0.6, zorder=4)
        yticks.append(y_offset)
        ylabels.append(f"{full_label}  {sr.dist_km:6.1f} km")
        # Overlay picks
        if has_picks:
            for phase, dt in pick_lookup[full_label].items():
                col = "red" if phase == "P" else "royalblue"
                ax.plot([dt, dt], [y_offset - 0.45, y_offset + 0.45],
                        color=col, linewidth=1.4, zorder=6)

    ax.axvline(0, color="goldenrod", linewidth=1.0, linestyle="--",
               zorder=3, label="origin time")
    ax.set_xlim(-args.window_half_s, args.window_half_s)
    ax.set_ylim(-(n - 1) - 0.7, 0.7)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("seconds from pyocto origin")
    ax.set_title(
        f"event {eidx} — {ot.isoformat()[:19]}  "
        f"({ev_row.latitude:.3f}, {ev_row.longitude:.3f}, {ev_row.depth:.1f} km bsf)  "
        f"npicks={int(ev_row.picks)}, traces shown={n}\n"
        f"red=P pick, blue=S pick, dashed yellow=pyocto origin time, "
        f"gray label=station with no pick"
    )
    ax.grid(alpha=0.3, axis="x")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ot.strftime("%Y%m%dT%H%M%S")
    out_path = OUT_DIR / f"event_{eidx:06d}_{stamp}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  event {eidx}: {n} traces -> {out_path.name}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=10)
    ap.add_argument("--event-idx", type=int, default=None,
                    help="Plot a specific event_idx instead of sampling.")
    ap.add_argument("--min-picks", type=int, default=15)
    ap.add_argument("--window-half-s", type=float, default=30.0)
    ap.add_argument("--band-low", type=float, default=2.0)
    ap.add_argument("--band-high", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"loading {EV_CSV.name}...")
    ev = pd.read_csv(EV_CSV)
    pk = pd.read_csv(PK_CSV)
    st = pd.read_csv(ST_CSV)

    if args.event_idx is not None:
        chosen = ev[ev.event_idx == args.event_idx]
        if chosen.empty:
            raise SystemExit(f"event_idx {args.event_idx} not in {EV_CSV.name}")
    else:
        pool = ev[ev.picks >= args.min_picks].reset_index(drop=True)
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(len(pool), size=min(args.n_events, len(pool)),
                         replace=False)
        chosen = pool.iloc[sorted(sel)]
    print(f"chosen events: {len(chosen)}")

    for _, row in chosen.iterrows():
        plot_event(row, pk, st, args)


if __name__ == "__main__":
    main()
