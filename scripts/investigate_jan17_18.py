"""Diagnose the Jan 17-18 event cluster that the user flagged as suspicious.

For events in Jan 17-18 (a window outside the documented shot window of Jan
21 - Feb 4), compare:
  - their mean spectrum to the known-shot mean and known-EQ mean
  - their spatial distribution vs the swarm core and survey tracks
  - their inter-event time distribution (regular = airgun-like; bursty = EQ-like)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from netCDF4 import Dataset

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notes" / "figures" / "comparison" / "jan17_18_diagnostic.png"

# Load data
ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag_v2.csv",
                 low_memory=False)
ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True, format="mixed", errors="coerce")
ev = ev.dropna(subset=["origin_time"]).reset_index(drop=True)

spectra = np.load(REPO / "catalogs" / "event_spectra.npy")
meta = pd.read_parquet(REPO / "catalogs" / "event_spectra_meta.parquet")
eid_to_row = {int(eid): i for i, eid in enumerate(meta["event_idx"].values)}

# Subset masks
jan17_18 = (ev.origin_time >= "2019-01-17") & (ev.origin_time < "2019-01-19")
jan_clean = (ev.origin_time >= "2019-01-01") & (ev.origin_time < "2019-01-17")  # known clean pre-spike
known_shot = ev["flag_shot"].astype(bool)
known_eq_outside = (
    ((ev.origin_time < "2019-01-21") | (ev.origin_time >= "2019-02-05"))
    & ~ev["flag_shot_v2"].astype(bool)
)

def mean_spectrum(mask):
    rows = [eid_to_row.get(int(e), -1) for e in ev[mask]["event_idx"]]
    rows = [r for r in rows if r >= 0]
    if not rows:
        return None
    return spectra[rows].mean(axis=0), len(rows)

freq = np.linspace(0, 50, spectra.shape[1])
spec_jan17_18, n_jan17_18 = mean_spectrum(jan17_18)
spec_jan_clean, n_jan_clean = mean_spectrum(jan_clean)
spec_shot, n_shot = mean_spectrum(known_shot)
spec_eq, n_eq = mean_spectrum(known_eq_outside)

# --- Figure ---
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.25)

# (1) Mean spectrum comparison
ax = fig.add_subplot(gs[0, 0])
ax.plot(freq, spec_jan17_18, color="red", linewidth=2,
        label=f"Jan 17-18 ({n_jan17_18:,})")
ax.plot(freq, spec_jan_clean, color="purple", linewidth=1.5,
        label=f"Jan 1-16 (pre-spike) ({n_jan_clean:,})")
ax.plot(freq, spec_shot, color="orange", linewidth=1.5, linestyle="--",
        label=f"known shots ({n_shot:,})")
ax.plot(freq, spec_eq, color="steelblue", linewidth=1.5, linestyle="--",
        label=f"known EQ (outside-window) ({n_eq:,})")
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel("mean log-power (normalized)")
ax.set_title("Mean per-event spectrum — Jan 17-18 vs reference")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (2) Map of Jan 17-18 events
ax = fig.add_subplot(gs[0, 1])
bath = Dataset(REPO / "notes" / "figures" / "Orca_bathymetry.nc")
lat_b = bath.variables["latitude"][:]; lon_b = bath.variables["longitude"][:]
z_b = bath.variables["data"][:]; bath.close()
z_b_plot = np.where(z_b > 2000, np.nan, z_b)
LON, LAT = np.meshgrid(lon_b, lat_b)
levels = np.arange(-2400, 200, 50)
norm = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
ax.contourf(LON, LAT, z_b_plot, levels=levels, cmap=plt.cm.GnBu_r,
            norm=norm, extend="both")
ax.contour(LON, LAT, z_b_plot, levels=[-1000], colors="k",
           linewidths=1.0, zorder=9)
stations = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
ob = stations[stations.network == "ZX"]
ax.scatter(ob.longitude, ob.latitude, marker="^", s=50, c="white",
           edgecolors="k", linewidths=0.7, zorder=8)
sub = ev[jan17_18]
ax.scatter(sub.longitude, sub.latitude, c=sub.origin_time.astype("int64"),
           cmap="plasma", s=8, alpha=0.7, zorder=7)
ax.set_xlim(-60, -57.5); ax.set_ylim(-63.0, -62.0)
ax.set_aspect(1.0 / np.cos(np.radians(-62.5)))
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"Jan 17-18 event locations ({len(sub):,})")

# (3) Hourly count
ax = fig.add_subplot(gs[0, 2])
sub2 = ev[(ev.origin_time >= "2019-01-16") & (ev.origin_time < "2019-01-21")].copy()
sub2["hour"] = sub2.origin_time.dt.floor("1h")
hr = sub2.groupby("hour").size()
ax.plot(hr.index, hr.values, color="red", linewidth=1.0)
ax.fill_between(hr.index, 0, hr.values, color="red", alpha=0.4)
ax.set_ylabel("events / hour")
ax.set_title("Hourly event count Jan 16-20")
ax.grid(alpha=0.3)
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
for lbl in ax.get_xticklabels():
    lbl.set_rotation(30); lbl.set_ha("right")

# (4) Inter-event time distribution
ax = fig.add_subplot(gs[1, 0])
for mask, color, label in [
    (jan17_18, "red", "Jan 17-18"),
    (jan_clean, "purple", "Jan 1-16"),
    (known_shot, "orange", "known shots"),
    (known_eq_outside, "steelblue", "known EQ"),
]:
    times = ev[mask]["origin_time"].sort_values().values
    if len(times) < 2: continue
    dt_sec = np.diff(times) / np.timedelta64(1, 's')
    dt_sec = dt_sec[dt_sec > 0]
    ax.hist(dt_sec[dt_sec < 300], bins=np.logspace(-1, np.log10(300), 50),
            histtype="step", linewidth=1.6, label=label, color=color)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("inter-event time (s)")
ax.set_ylabel("count")
ax.set_title("Inter-event time distribution\n(airgun = sharp peak; EQ = exponential)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

# (5) classifier prob_shot histogram for Jan 17-18 vs others
ax = fig.add_subplot(gs[1, 1])
for mask, color, label in [
    (jan17_18, "red", "Jan 17-18"),
    (jan_clean, "purple", "Jan 1-16"),
    (known_shot, "orange", "known shots"),
    (known_eq_outside, "steelblue", "known EQ"),
]:
    p = ev[mask]["prob_shot"].dropna()
    if len(p) == 0: continue
    ax.hist(p, bins=50, histtype="step", linewidth=1.6, label=label,
            color=color, density=True)
ax.set_xlabel("classifier P(shot)")
ax.set_ylabel("density")
ax.set_title("Classifier shot probability per group")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# (6) Per-minute count zoomed on Jan 17 alone
ax = fig.add_subplot(gs[1, 2])
sub3 = ev[(ev.origin_time >= "2019-01-17") & (ev.origin_time < "2019-01-18")].copy()
sub3["minute"] = sub3.origin_time.dt.floor("1min")
mn = sub3.groupby("minute").size()
ax.bar(mn.index, mn.values, width=1/(24*60), color="red",
       edgecolor="k", linewidth=0.2)
ax.set_ylabel("events / minute")
ax.set_title(f"Per-minute count on Jan 17 (peak day): {len(sub3):,} events")
ax.grid(alpha=0.3, axis="y")
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

fig.suptitle("Jan 17-18 anomaly diagnostic — is this real EQ activity, or undocumented active source?",
             fontsize=13)
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")

# Print numeric summary
print(f"\nJan 17-18: {n_jan17_18:,} events")
p_jan17_18 = ev[jan17_18]["prob_shot"].dropna()
p_jan_clean = ev[jan_clean]["prob_shot"].dropna()
print(f"  Jan 17-18 mean classifier P(shot): {p_jan17_18.mean():.3f}")
print(f"  Jan 1-16   mean classifier P(shot): {p_jan_clean.mean():.3f}")
print(f"  known shots mean P(shot): {ev[known_shot]['prob_shot'].dropna().mean():.3f}")
print(f"  known EQ  mean P(shot): {ev[known_eq_outside]['prob_shot'].dropna().mean():.3f}")
