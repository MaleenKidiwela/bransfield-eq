"""Distribution of cross-correlation (CC) values over time for events in the
largest GrowClust cluster of the year-long run.

For each event in cid=1, gather every CC value from its dt.cc differential-time
observations (across all pair blocks where the event appears on either side),
then plot the distribution per time bin (daily / weekly).
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent
DTCC = REPO / "growclust" / "picker_only" / "dt.cc"
CAT  = REPO / "catalogs" / "growclust_picker_only.csv"
OUT  = REPO / "notes" / "figures" / "cc_over_time_year_topcluster.png"

COLS = ['year','mo','dy','hr','mn','sec','evid','lat_gc','lon_gc','dep_gc',
        'mag','evid2','cid','nbranch','qID','qNN','qNX','rmsP','rmsS','eh','ez','et',
        'lat_py','lon_py','dep_py']

print("Loading GrowClust catalog ...")
cat = pd.read_csv(CAT, names=COLS, skiprows=1)
base = pd.to_datetime(dict(
    year=cat.year, month=cat.mo, day=cat.dy,
    hour=cat.hr.clip(0,23), minute=cat.mn.clip(0,59)), utc=True, errors="coerce")
cat["t"] = base + pd.to_timedelta(cat.sec, unit="s")
top_cid = cat.groupby("cid").size().idxmax()
clu_ev = cat[cat.cid == top_cid].copy()
clu_evid = set(int(e) for e in clu_ev.evid)
print(f"  top cluster cid={top_cid}: {len(clu_ev):,} events")

print("Parsing dt.cc and collecting CC values per cluster event ...")
# dt.cc header lines: "# id1 id2 0.0"; obs lines: "sta_bare dt cc phase"
event_ccs = {evid: [] for evid in clu_evid}
header_re = re.compile(r"^#\s*(\d+)\s+(\d+)")
cur_a = cur_b = None
n_blocks = 0
n_obs = 0
with open(DTCC) as f:
    for ln in f:
        ln = ln.rstrip()
        if not ln:
            continue
        if ln.startswith("#"):
            m = header_re.match(ln)
            if not m:
                cur_a = cur_b = None
                continue
            cur_a, cur_b = int(m.group(1)), int(m.group(2))
            n_blocks += 1
        else:
            if cur_a is None:
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                cc = float(parts[2])
            except ValueError:
                continue
            n_obs += 1
            if cur_a in event_ccs:
                event_ccs[cur_a].append(cc)
            if cur_b in event_ccs:
                event_ccs[cur_b].append(cc)
print(f"  read {n_blocks:,} pair blocks  {n_obs:,} observations")

# Per-event aggregates (median CC across all obs touching the event)
ev_med = pd.Series({eid: (np.median(ccs) if ccs else np.nan)
                    for eid, ccs in event_ccs.items()})
ev_mean = pd.Series({eid: (np.mean(ccs) if ccs else np.nan)
                     for eid, ccs in event_ccs.items()})
ev_n = pd.Series({eid: len(ccs) for eid, ccs in event_ccs.items()})

clu_ev = clu_ev.set_index(clu_ev.evid.astype(int))
clu_ev["cc_median"] = ev_med
clu_ev["cc_mean"]   = ev_mean
clu_ev["n_obs"]     = ev_n
clu_ev = clu_ev.dropna(subset=["cc_median"]).sort_values("t").reset_index(drop=True)
print(f"  events with at least 1 CC observation: {len(clu_ev):,}")

# Daily aggregate stats
clu_ev["date"] = clu_ev.t.dt.floor("1D")
daily = clu_ev.groupby("date").agg(
    n_ev=("cc_median", "size"),
    cc_p25=("cc_median", lambda x: np.percentile(x, 25)),
    cc_p50=("cc_median", "median"),
    cc_p75=("cc_median", lambda x: np.percentile(x, 75)),
    cc_p95=("cc_median", lambda x: np.percentile(x, 95)),
).reset_index()

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1.0, 1.0], hspace=0.12)

# (a) scatter of per-event median CC vs time, colored by n_obs
ax = fig.add_subplot(gs[0])
sc = ax.scatter(clu_ev.t, clu_ev.cc_median, c=clu_ev.n_obs, s=4,
                cmap="viridis", alpha=0.5,
                vmin=1, vmax=max(2, np.percentile(clu_ev.n_obs, 95)))
ax.fill_between(daily.date, daily.cc_p25, daily.cc_p75,
                color="orange", alpha=0.3, label="daily p25-p75")
ax.plot(daily.date, daily.cc_p50, color="darkorange", linewidth=1.3,
        label="daily median")
ax.axhline(0.6, color="red", linestyle="--", linewidth=0.8, label="rmin=0.6")
ax.set_ylim(0.55, 1.02)
ax.set_ylabel("per-event median CC")
ax.set_title(f"Cross-correlation values over time — top cluster cid={top_cid} "
             f"({len(clu_ev):,} events; {n_obs:,} obs)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.colorbar(sc, ax=ax, label="# CC obs per event", shrink=0.7, pad=0.01)

# (b) daily count of events in the cluster
ax = fig.add_subplot(gs[1], sharex=fig.axes[0])
ax.bar(daily.date, daily.n_ev, width=1.0, color="steelblue",
       edgecolor="k", linewidth=0.2)
ax.set_ylabel("events / day")
ax.grid(alpha=0.3)

# (c) p95 line — captures the high end of CC quality
ax = fig.add_subplot(gs[2], sharex=fig.axes[0])
ax.plot(daily.date, daily.cc_p95, color="green", linewidth=1.2, label="daily p95")
ax.plot(daily.date, daily.cc_p50, color="darkorange", linewidth=1.2, label="daily median")
ax.axhline(0.6, color="red", linestyle="--", linewidth=0.8)
ax.set_ylim(0.55, 1.02)
ax.set_ylabel("CC quantile")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
for lbl in ax.get_xticklabels():
    lbl.set_rotation(30); lbl.set_ha("right")
ax.set_xlabel("date")

plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"wrote {OUT}")
