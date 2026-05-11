"""
Per-event picker-comparison plot.

For a given (event_id) + day, render one figure per event:
  - one row per station that recorded the event
  - waveform (Z component) trimmed to [origin - 30s, origin + 90s]
  - vertical lines at manual P (red) and manual S (blue)
  - small markers at each ML picker's P/S picks within the window

Plus a summary bar chart of P/S recall per picker for the day.

Usage:
    python scripts/plot_picker_comparison.py --day 2019-12-26
    python scripts/plot_picker_comparison.py --day 2019-12-26 \
        --pickers picks picks_eqt picks_pn_obs picks_eqt_obs picks_obst \
        --manual-source mag07
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import mseed_path

PICKER_LABELS = {
    "picks":            ("PhaseNet (instance)",      "#1f77b4"),
    "picks_eqt":        ("EQT (instance)",           "#ff7f0e"),
    "picks_pn_obs":     ("PhaseNet (obs/PickBlue)",  "#2ca02c"),
    "picks_pn_obs_03":  ("PhaseNet obs @ 0.3",       "#2ca02c"),
    "picks_eqt_obs":    ("EQT (obs/PickBlue)",       "#d62728"),
    "picks_eqt_obs_03": ("EQT obs @ 0.3",            "#d62728"),
    "picks_obst":       ("OBSTransformer @ 0.1",     "#9467bd"),
    "picks_obst_03":    ("OBSTransformer @ 0.3",     "#8c564b"),
    "picks_obst_05":    ("OBSTransformer @ 0.5",     "#9467bd"),
    "picks_obst_07":    ("OBSTransformer @ 0.7",     "#7f7f7f"),
}


def load_picker_picks(picker_subdir: str, network: str, station: str,
                      day: UTCDateTime) -> pd.DataFrame:
    f = REPO / "catalogs" / picker_subdir / f"{network}.{station}" / \
        f"{day.year}-{day.julday:03d}.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["time"], utc=True)
    df["phase"] = df["phase"].str.upper().str[0]
    return df[["t", "phase", "prob"]]


def plot_event(event_id: str, manual_event: pd.DataFrame, day: UTCDateTime,
               pickers: list[str], window_pre: float, window_post: float,
               highpass_hz: float, out_path: Path) -> None:
    origin = UTCDateTime(manual_event["origin_time"].iloc[0])
    t0, t1 = origin - window_pre, origin + window_post

    stations = sorted(manual_event[["network", "station"]]
                      .drop_duplicates().itertuples(index=False, name=None))
    n = len(stations)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.8 * n + 1.0), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (net, sta) in zip(axes, stations):
        mp = mseed_path(net, sta, day)
        if not mp.exists():
            ax.text(0.5, 0.5, f"{net}.{sta} — no mseed", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            ax.set_yticks([]); ax.set_xticks([])
            continue
        st = read(str(mp), starttime=t0, endtime=t1)
        # prefer Z component; fall back to first available
        ztr = next((tr for tr in st if tr.stats.channel.endswith("Z")), None)
        if ztr is None and len(st):
            ztr = st[0]
        if ztr is None:
            ax.text(0.5, 0.5, f"{net}.{sta} — no Z trace", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            continue
        tr = ztr.copy().detrend("demean")
        if highpass_hz > 0:
            tr.filter("highpass", freq=highpass_hz, zerophase=True)
        times = tr.times(reftime=origin)
        amp = tr.data / (np.max(np.abs(tr.data)) + 1e-12)
        ax.plot(times, amp, color="0.2", lw=0.5)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([])
        ax.set_ylabel(f"{net}.{sta}\n{tr.stats.channel}",
                      rotation=0, ha="right", va="center", fontsize=8)

        # Manual picks (solid vertical lines)
        sta_picks = manual_event[manual_event.station == sta]
        for _, pk in sta_picks.iterrows():
            t_rel = (UTCDateTime(pk["pick_time"]) - origin)
            color = "red" if pk["phase"] == "P" else "blue"
            ax.axvline(t_rel, color=color, lw=1.5, alpha=0.9, zorder=3)

        # Picker picks (top edge markers)
        for i, sd in enumerate(pickers):
            color = PICKER_LABELS.get(sd, (sd, f"C{i}"))[1]
            df = load_picker_picks(sd, net, sta, day)
            if df.empty:
                continue
            in_win = df[(df.t >= pd.Timestamp(t0.datetime, tz="UTC")) &
                        (df.t <= pd.Timestamp(t1.datetime, tz="UTC"))]
            for _, pk in in_win.iterrows():
                t_rel = (UTCDateTime(pk["t"].to_pydatetime()) - origin)
                marker = "v" if pk["phase"] == "P" else "^"
                y = 1.05 + i * 0.08
                ax.plot(t_rel, y, marker=marker, color=color,
                        markersize=4, clip_on=False, zorder=4)
        ax.axvline(0, color="black", lw=0.5, ls=":", alpha=0.5)

    axes[-1].set_xlabel(f"Seconds after origin ({origin.isoformat()})")
    axes[0].set_title(
        f"Event {event_id}  |  origin {origin.isoformat()}  "
        f"|  {len(stations)} stations  |  hp {highpass_hz} Hz"
    )

    # Legend
    legend_handles = [plt.Line2D([0], [0], color="red", lw=1.5, label="manual P"),
                      plt.Line2D([0], [0], color="blue", lw=1.5, label="manual S")]
    for sd in pickers:
        lab, col = PICKER_LABELS.get(sd, (sd, "gray"))
        legend_handles.append(plt.Line2D([0], [0], marker="v", color=col,
                                         markersize=6, lw=0, label=f"{lab} P"))
        legend_handles.append(plt.Line2D([0], [0], marker="^", color=col,
                                         markersize=6, lw=0, label=f"{lab} S"))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(6, len(legend_handles)), fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_recall_bars(stats: dict, out_path: Path) -> None:
    """Stats: { picker_subdir: {'P_recall': x, 'S_recall': y, 'total': n} }."""
    pickers = list(stats.keys())
    labels = [PICKER_LABELS.get(p, (p, "gray"))[0] for p in pickers]
    p_rec = [stats[p]["P_recall"] for p in pickers]
    s_rec = [stats[p]["S_recall"] for p in pickers]
    totals = [stats[p]["total"] for p in pickers]

    x = np.arange(len(pickers))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.bar(x - 0.18, p_rec, width=0.36, color="red", alpha=0.7, label="P recall")
    ax1.bar(x + 0.18, s_rec, width=0.36, color="blue", alpha=0.7, label="S recall")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Recall vs mag07 manual picks")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.set_title("Per-picker recall on 2019-12-26")
    for i, (p, s) in enumerate(zip(p_rec, s_rec)):
        ax1.text(i - 0.18, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax1.text(i + 0.18, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)

    ax2.bar(x, totals, color="0.4")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Total picks (day-26)")
    ax2.set_title("Pick volume — proxy for FP rate at thresh 0.1")
    for i, t in enumerate(totals):
        ax2.text(i, t * 1.02, f"{t:,}", ha="center", fontsize=8)
    ax2.set_yscale("log")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--pickers", nargs="+",
                    default=["picks", "picks_eqt", "picks_pn_obs",
                             "picks_eqt_obs", "picks_obst"])
    ap.add_argument("--manual-source", default="mag07")
    ap.add_argument("--window-pre", type=float, default=10.0,
                    help="seconds before origin")
    ap.add_argument("--window-post", type=float, default=60.0,
                    help="seconds after origin")
    ap.add_argument("--highpass", type=float, default=2.0,
                    help="Hz; 0 to disable")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    day = UTCDateTime(args.day)
    outdir = REPO / args.outdir / f"picker_comparison_{day.date}"

    # Load manual picks for the day, filtered to mag07
    m = pd.read_csv(REPO / "catalogs" / "manual_picks.csv",
                    parse_dates=["origin_time", "pick_time"])
    if args.manual_source != "all":
        m = m[m.source_file.str.contains(args.manual_source, case=False, na=False)]
    day_pd = pd.Timestamp(day.datetime, tz="UTC")
    end_pd = day_pd + pd.Timedelta(days=1)
    day_picks = m[(m.pick_time >= day_pd) & (m.pick_time < end_pd)].copy()
    day_picks["phase"] = day_picks.phase.str.upper().str[0]

    if day_picks.empty:
        print(f"No manual picks on {day.date} (source={args.manual_source}).")
        return

    print(f"Manual picks: {len(day_picks)}  events: {day_picks.event_id.nunique()}  "
          f"stations: {day_picks.station.nunique()}")

    # Per-event waveform plots
    for eid, ev in day_picks.groupby("event_id"):
        out = outdir / f"event_{eid}.png"
        plot_event(eid, ev, day, args.pickers,
                   args.window_pre, args.window_post, args.highpass, out)

    # Summary recall bar chart — counts P/S TP via simple ±tolerance match
    P_TOL, S_TOL = 0.5, 1.0
    stats = {}
    for sd in args.pickers:
        # Aggregate counts across the day
        total = 0
        ptp = stp = pfn = sfn = 0
        for (net, sta), grp in day_picks.groupby(["network", "station"]):
            ml = load_picker_picks(sd, net, sta, day)
            total += len(ml)
            for _, pk in grp.iterrows():
                phase = pk["phase"]
                tol = P_TOL if phase == "P" else S_TOL
                t_pk = pd.Timestamp(pk["pick_time"])
                if t_pk.tzinfo is None:
                    t_pk = t_pk.tz_localize("UTC")
                else:
                    t_pk = t_pk.tz_convert("UTC")
                if ml.empty:
                    if phase == "P": pfn += 1
                    else:            sfn += 1
                    continue
                same_phase = ml[ml.phase == phase]
                if same_phase.empty or \
                   (same_phase.t - t_pk).abs().min() > pd.Timedelta(seconds=tol):
                    if phase == "P": pfn += 1
                    else:            sfn += 1
                else:
                    if phase == "P": ptp += 1
                    else:            stp += 1
        n_p = (day_picks.phase == "P").sum()
        n_s = (day_picks.phase == "S").sum()
        stats[sd] = {
            "P_recall": ptp / n_p if n_p else 0.0,
            "S_recall": stp / n_s if n_s else 0.0,
            "total": total,
        }
    plot_recall_bars(stats, outdir / "summary_recall.png")

    print(f"\nAll figures in: {outdir}")


if __name__ == "__main__":
    main()
