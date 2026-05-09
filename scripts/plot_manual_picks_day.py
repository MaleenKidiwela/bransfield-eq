"""
Plot waveforms with manual + PhaseNet pick markers for one day.

For each event in the manual catalog on the target day, draws one figure:
  - one panel per station with manual picks for that event
  - 3-component traces stacked
  - P picks marked red, S picks marked blue
  - PhaseNet picks (if found) marked with dashed lines for comparison

Output: figures/picks_<YYYY-MM-DD>_<event-id>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bransfield_eq.config import mseed_path, pick_csv_path  # noqa: E402

MANUAL_CSV = REPO / "catalogs" / "manual_picks.csv"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--source-file", default="nllmaleen_mag07_202210.out",
                   help="restrict to one manual pick source file")
    p.add_argument("--pre", type=float, default=10.0,
                   help="seconds before earliest pick in the window")
    p.add_argument("--post", type=float, default=20.0,
                   help="seconds after latest pick in the window")
    p.add_argument("--bandpass", default="2,15",
                   help="bandpass filter low,high in Hz (set to '' to disable)")
    p.add_argument("--center-time", default=None,
                   help="UTC ISO time. If set, plot a single window around this "
                        "time using PhaseNet picks only (no manual grouping). "
                        "Useful for events found by PhaseNet that aren't in "
                        "the manual catalog.")
    p.add_argument("--half-window", type=float, default=15.0,
                   help="seconds on each side of --center-time")
    p.add_argument("--label", default=None,
                   help="filename label for --center-time mode "
                        "(default: HHMMSS of center)")
    return p.parse_args()


def load_picks(date: str, source_file: str) -> pd.DataFrame:
    mp = pd.read_csv(MANUAL_CSV)
    mp = mp[mp.source_file == source_file]
    mp["t"] = pd.to_datetime(mp.pick_time, utc=True)
    day = mp[(mp.t >= f"{date}T00:00:00") & (mp.t < f"{date}T23:59:59.999")]
    return day


def load_phasenet_picks(date: str) -> pd.DataFrame:
    """Best-effort load of PhaseNet pick CSVs for the day."""
    d = UTCDateTime(date)
    rows = []
    for sta_dir in sorted((REPO / "catalogs" / "picks").glob("*/")):
        f = sta_dir / f"{d.year}-{d.julday:03d}.csv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) == 0:
            continue
        df["station"] = sta_dir.name.split(".")[1]
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["time", "phase", "prob", "station"])
    out = pd.concat(rows, ignore_index=True)
    out["t"] = pd.to_datetime(out.time, utc=True)
    return out


def load_waveform_window(net: str, sta: str, t0: UTCDateTime, t1: UTCDateTime,
                         bandpass: tuple[float, float] | None):
    """Return a Stream trimmed to [t0, t1] (or None if no data)."""
    day = UTCDateTime(t0.date)
    p = mseed_path(net, sta, day)
    if not p.exists() or p.stat().st_size == 0:
        return None
    st = read(str(p))
    # only seismic 3C — drop hydrophone for plotting
    st.traces = [tr for tr in st if tr.stats.channel[1] in "HLN"
                 and tr.stats.channel[2] in "ZNE12"]
    if len(st) == 0:
        return None
    st = st.trim(t0, t1)
    if len(st) == 0:
        return None
    st.detrend("demean")
    st.taper(0.05)
    if bandpass:
        st.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1],
                  corners=4, zerophase=True)
    return st


def plot_event(event_id: str, picks: pd.DataFrame, ml_picks: pd.DataFrame,
               args: argparse.Namespace) -> Path | None:
    """Plot one event with all its station panels."""
    t_min = picks.t.min()
    t_max = picks.t.max()
    t0 = UTCDateTime(t_min.to_pydatetime()) - args.pre
    t1 = UTCDateTime(t_max.to_pydatetime()) + args.post

    bp = None
    if args.bandpass.strip():
        a, b = [float(x) for x in args.bandpass.split(",")]
        bp = (a, b)

    stations = sorted(picks.station.unique())
    n = len(stations)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 1, figsize=(12, 1.6 * n + 1), sharex=True,
                             squeeze=False)
    axes = axes[:, 0]

    for ax, sta in zip(axes, stations):
        # network — assume ZX since manual picks here are all OBS
        net = picks[picks.station == sta].network.iloc[0]
        if pd.isna(net):
            net = "ZX"
        st = load_waveform_window(net, sta, t0, t1, bp)
        if st is None or len(st) == 0:
            ax.text(0.5, 0.5, f"no waveform for {net}.{sta}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="grey")
            ax.set_yticks([])
            continue

        # stack 3 components vertically with offset
        for i, tr in enumerate(sorted(st, key=lambda t: t.stats.channel[-1])):
            data = tr.data / max(abs(tr.data).max(), 1e-12)
            tt = tr.times(reftime=t0)
            ax.plot(tt, data + i * 2.5, lw=0.5, color="black")
            ax.text(0.005, i * 2.5, tr.stats.channel, transform=ax.get_yaxis_transform(),
                    fontsize=7, color="grey", va="center")

        # manual picks
        for _, p in picks[picks.station == sta].iterrows():
            x = (UTCDateTime(p.t.to_pydatetime()) - t0)
            color = "red" if p.phase == "P" else "royalblue"
            ax.axvline(x, color=color, lw=1.4, alpha=0.85)
            ax.text(x, ax.get_ylim()[1] * 0.95, p.phase, color=color,
                    fontsize=9, fontweight="bold", ha="center")

        # ml picks (dashed)
        if len(ml_picks):
            mlsta = ml_picks[(ml_picks.station == sta)
                             & (ml_picks.t >= t_min - pd.Timedelta(seconds=2))
                             & (ml_picks.t <= t_max + pd.Timedelta(seconds=2))]
            for _, p in mlsta.iterrows():
                x = (UTCDateTime(p.t.to_pydatetime()) - t0)
                color = "red" if p.phase == "P" else "royalblue"
                ax.axvline(x, color=color, lw=1.0, ls="--", alpha=0.6)
                ax.text(x, ax.get_ylim()[0] * 0.95, f"{p.phase}({p.prob:.2f})",
                        color=color, fontsize=7, ha="center", va="bottom")

        ax.set_ylabel(f"{net}.{sta}", fontsize=9)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel(f"seconds after {t0.strftime('%Y-%m-%dT%H:%M:%S')}")
    fig.suptitle(f"Event {event_id}  —  manual (solid)  vs  PhaseNet (dashed)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = FIG_DIR / f"picks_{t0.strftime('%Y-%m-%d')}_{event_id}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_time_window(center: pd.Timestamp, ml: pd.DataFrame,
                     args: argparse.Namespace) -> Path | None:
    """PhaseNet-only plot: all stations with picks within ±half_window of center."""
    if ml.empty:
        print("  No PhaseNet picks to plot.")
        return None
    win_ml = ml[(ml.t >= center - pd.Timedelta(seconds=args.half_window))
              & (ml.t <= center + pd.Timedelta(seconds=args.half_window))]
    if win_ml.empty:
        print(f"  No PhaseNet picks within ±{args.half_window}s of {center}.")
        return None

    bp = None
    if args.bandpass.strip():
        a, b = [float(x) for x in args.bandpass.split(",")]
        bp = (a, b)

    t0 = UTCDateTime(center.to_pydatetime()) - args.half_window
    t1 = UTCDateTime(center.to_pydatetime()) + args.half_window
    stations = sorted(win_ml.station.unique())

    fig, axes = plt.subplots(len(stations), 1, figsize=(12, 1.6 * len(stations) + 1),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    geo = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    sta_to_net = dict(zip(geo.station, geo.network))

    for ax, sta in zip(axes, stations):
        net = sta_to_net.get(sta, "ZX")
        st = load_waveform_window(net, sta, t0, t1, bp)
        if st is None or len(st) == 0:
            ax.text(0.5, 0.5, f"no waveform for {net}.{sta}",
                    transform=ax.transAxes, ha="center", va="center", color="grey")
            ax.set_yticks([])
            continue
        for i, tr in enumerate(sorted(st, key=lambda t: t.stats.channel[-1])):
            data = tr.data / max(abs(tr.data).max(), 1e-12)
            tt = tr.times(reftime=t0)
            ax.plot(tt, data + i * 2.5, lw=0.5, color="black")
            ax.text(0.005, i * 2.5, tr.stats.channel,
                    transform=ax.get_yaxis_transform(),
                    fontsize=7, color="grey", va="center")
        for _, p in win_ml[win_ml.station == sta].iterrows():
            x = (UTCDateTime(p.t.to_pydatetime()) - t0)
            color = "red" if p.phase == "P" else "royalblue"
            ax.axvline(x, color=color, lw=1.0, ls="--", alpha=0.7)
            ax.text(x, ax.get_ylim()[1] * 0.95,
                    f"{p.phase}({p.prob:.2f})",
                    color=color, fontsize=8, ha="center", fontweight="bold")
        ax.set_ylabel(f"{net}.{sta}", fontsize=9)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel(f"seconds after {t0.strftime('%Y-%m-%dT%H:%M:%S')}")
    label = args.label or center.strftime("%H%M%S")
    fig.suptitle(f"PhaseNet-only event @ {center.strftime('%Y-%m-%dT%H:%M:%S')} UTC",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG_DIR / f"picks_{center.strftime('%Y-%m-%d')}_phasenet_{label}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    ml = load_phasenet_picks(args.date)
    print(f"{len(ml)} PhaseNet picks loaded for {args.date}.")

    if args.center_time:
        center = pd.to_datetime(args.center_time, utc=True)
        out = plot_time_window(center, ml, args)
        if out:
            print(f"  → {out.relative_to(REPO)}")
        return

    picks = load_picks(args.date, args.source_file)
    if picks.empty:
        print(f"No manual picks on {args.date} from {args.source_file}.")
        return
    print(f"{len(picks)} manual picks across {picks.event_id.nunique()} events.")
    for eid, group in picks.groupby("event_id"):
        out = plot_event(eid, group, ml, args)
        print(f"  {eid}: {len(group)} picks across {group.station.nunique()} "
              f"stations  →  {out.relative_to(REPO) if out else 'skipped'}")


if __name__ == "__main__":
    main()
