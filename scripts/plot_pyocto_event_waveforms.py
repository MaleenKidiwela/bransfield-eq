"""Plot waveforms with picks for one pyocto event (event_idx=151 on 2019-01-17)."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime

REPO = Path(__file__).resolve().parent.parent
DAY = "2019-01-28"
EVENT_IDX = 148
EV_FILE = REPO / "catalogs/pyocto_daily" / f"events_{DAY}.csv"
PK_FILE = REPO / "catalogs/pyocto_daily" / f"picks_{DAY}.csv"
OUT = REPO / "notes/figures" / f"pyocto_event_{DAY}_idx{EVENT_IDX}_waveforms.png"

events = pd.read_csv(EV_FILE)
picks = pd.read_csv(PK_FILE)

ev = events[events.idx == EVENT_IDX].iloc[0]
evp = picks[picks.event_idx == EVENT_IDX].copy()
origin_time = UTCDateTime(ev.time)
print(f"event {EVENT_IDX}: origin={origin_time}, x={ev.x:.1f} y={ev.y:.1f} z={ev.z:.1f} km, "
      f"{len(evp)} picks across {evp.station.nunique()} stations")

# Window: 5s before earliest pick, 25s after latest pick
t_min = evp.time.min() - origin_time.timestamp - 5
t_max = evp.time.max() - origin_time.timestamp + 25

# Stations to plot — sort by ascending distance (proxied by pick time order)
sta_order = evp.groupby("station").time.min().sort_values().index.tolist()

# Load waveforms (vertical Z component) for each station
DOY = origin_time.julday
fig, axes = plt.subplots(len(sta_order), 1, figsize=(13, 0.55 * len(sta_order)),
                         sharex=True)
for ax, sta in zip(axes, sta_order):
    net, st = sta.split(".")
    f = REPO / "data/waveforms" / net / st / f"{net}.{st}.{origin_time.year}.{DOY:03d}.mseed"
    if not f.exists():
        ax.text(0.5, 0.5, f"{sta}  waveform missing", transform=ax.transAxes,
                ha="center", va="center", color="red")
        ax.set_yticks([])
        continue
    try:
        st_data = read(str(f)).select(channel="*H*Z") or read(str(f)).select(channel="*Z")
        if len(st_data) == 0:
            st_data = read(str(f))
            st_data = st_data.select(channel=st_data[0].stats.channel)  # any channel
        tr = st_data[0]
        tr.trim(origin_time + t_min, origin_time + t_max, pad=True, fill_value=0)
        tr.detrend("demean")
        tr.filter("bandpass", freqmin=2.0, freqmax=20.0, corners=4, zerophase=True)
        t_axis = tr.times() + t_min
        d = tr.data / np.max(np.abs(tr.data)) if np.max(np.abs(tr.data)) > 0 else tr.data
        ax.plot(t_axis, d, "k", lw=0.5)
    except Exception as e:
        ax.text(0.5, 0.5, f"{sta}  read failed: {e}", transform=ax.transAxes,
                ha="center", va="center", color="red", fontsize=8)

    # picks for this station
    sta_picks = evp[evp.station == sta]
    for _, pk in sta_picks.iterrows():
        rel_t = pk.time - origin_time.timestamp
        col = "C3" if pk.phase == "P" else "C0"
        ax.axvline(rel_t, color=col, lw=1.5, alpha=0.9, zorder=10)
        ax.annotate(f"{pk.phase}", (rel_t, 0.9), xycoords=('data','axes fraction'),
                    color=col, fontsize=8, fontweight="bold",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=col, lw=0.5, alpha=0.9))

    ax.set_ylabel(sta, fontsize=8, rotation=0, ha="right", va="center")
    ax.set_yticks([])
    ax.axvline(0, color="orange", lw=0.7, alpha=0.5, zorder=1)
    ax.set_xlim(t_min, t_max)

axes[0].set_title(
    f"Event {EVENT_IDX} — {origin_time}  ({len(evp)} picks, {evp.station.nunique()} stations, "
    f"RMS={np.sqrt((evp.residual**2).mean()):.3f}s)\n"
    f"local pos x={ev.x:.1f} km, y={ev.y:.1f} km, z={ev.z:.1f} km   |   "
    f"red=P pick, blue=S pick, orange=origin time"
)
axes[-1].set_xlabel("time relative to origin (s)")
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
