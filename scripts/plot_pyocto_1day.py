"""Plot the 2019-01-17 pyocto 1-day test catalog: map, depth, time, vs manual."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

REPO = Path(__file__).resolve().parent.parent
EV = REPO / "catalogs" / "pyocto_events_1day_2019-01-17.csv"
ST = REPO / "catalogs" / "station_geometry.csv"
MAN = REPO / "catalogs" / "manual_picks.csv"
OUT = REPO / "notes" / "figures" / "pyocto_1day_2019-01-17.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

stations = pd.read_csv(ST)
lat0 = stations.latitude.mean()
lon0 = stations.longitude.mean()
crs = CRS.from_proj4(f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +ellps=WGS84")
inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

e = pd.read_csv(EV)
e["t"] = pd.to_datetime(e.time, unit="s", utc=True)
# project local x,y (km) -> lat,lon
lons, lats = inv.transform(e.x.values * 1000, e.y.values * 1000)
e["lon"] = lons; e["lat"] = lats

man = pd.read_csv(MAN)
man = man[man.source_file == "nllmaleen_mag07_202210.out"].copy()
man["t"] = pd.to_datetime(man.origin_time, utc=True)
man_day = man[man.t.dt.date == pd.Timestamp("2019-01-17").date()]
man_ev = man_day.drop_duplicates(subset=["event_id"])

print(f"pyocto: {len(e)} events")
print(f"manual: {len(man_ev)} events")

# quality flags
in_net = (e.x.abs() < 30) & (e.y.abs() < 30)
at_floor = e.z >= 35

fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# (a) map
ax = axes[0,0]
ax.scatter(e.lon[~in_net], e.lat[~in_net], s=18, c="lightgray", edgecolors="0.5",
           linewidths=0.3, label=f"pyocto outside-net (n={(~in_net).sum()})")
sc = ax.scatter(e.lon[in_net], e.lat[in_net], s=22, c=e.z[in_net], cmap="viridis_r",
                vmin=0, vmax=20, edgecolors="k", linewidths=0.3,
                label=f"pyocto in-net (n={in_net.sum()})")
ax.scatter(stations.longitude, stations.latitude, marker="^", s=80,
           c="red", edgecolors="k", linewidths=0.5, label="stations", zorder=5)
ax.scatter(stations[stations.network=="ZX"].longitude,
           stations[stations.network=="ZX"].latitude, marker="^", s=80,
           c="blue", edgecolors="k", linewidths=0.5, label="OBS (ZX)", zorder=6)
cb = plt.colorbar(sc, ax=ax, label="depth (km)")
ax.set_xlim(-62, -57.5); ax.set_ylim(-63.7, -62.0)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title(f"pyocto events 2019-01-17 — map view ({len(e)} events)")
ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=8)

# (b) depth histogram
ax = axes[0,1]
ax.hist(e.z, bins=np.arange(0, 41, 2), edgecolor="k", alpha=0.6, label="all")
ax.hist(e.z[in_net], bins=np.arange(0, 41, 2), edgecolor="k", alpha=0.7,
        color="C2", label="in-network")
ax.set_xlabel("depth below sea level (km)")
ax.set_ylabel("event count")
ax.set_title(f"depth histogram (in-net median {e.z[in_net].median():.1f} km)")
ax.axvspan(35, 40, alpha=0.15, color="red", label="model floor")
ax.legend(); ax.grid(alpha=0.3)

# (c) origin time series with manual overlay
ax = axes[1,0]
hours = (e.t - e.t.dt.normalize()).dt.total_seconds() / 3600
man_hours = (man_ev.t - pd.Timestamp("2019-01-17", tz="UTC")).dt.total_seconds() / 3600
ax.hist(hours, bins=np.arange(0, 25, 1), alpha=0.6, edgecolor="k",
        label=f"pyocto ({len(e)})", color="C0")
ax.hist(man_hours, bins=np.arange(0, 25, 1), alpha=0.6, edgecolor="k",
        label=f"manual mag07 ({len(man_ev)})", color="C3")
ax.set_xlabel("hour of day (UTC)")
ax.set_ylabel("event count")
ax.set_title("temporal distribution — 2019-01-17 swarm")
ax.legend(); ax.grid(alpha=0.3)

# (d) recall: manual vs pyocto match within Δt
ax = axes[1,1]
match_thresh = np.arange(1, 31)
recalls = []
pyt = e.t.values
for dt in match_thresh:
    n = 0
    for mt in man_ev.t.values:
        if np.min(np.abs((pyt - mt) / np.timedelta64(1, "s"))) <= dt:
            n += 1
    recalls.append(100 * n / len(man_ev))
ax.plot(match_thresh, recalls, "o-", color="C2")
ax.axhline(100, color="0.6", ls="--", alpha=0.5)
ax.set_xlabel("origin-time match tolerance (s)")
ax.set_ylabel("manual events recovered (%)")
ax.set_title("recall vs origin-time tolerance")
ax.set_ylim(0, 105); ax.grid(alpha=0.3)
for thr, r in zip([5, 10, 30], [recalls[4], recalls[9], recalls[29]]):
    ax.annotate(f"{r:.0f}% @ {thr}s", (thr, r), textcoords="offset points",
                xytext=(5, -10), fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
