"""Daily counts: pyocto events vs manual catalog (events + picks) for first 2 months."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notes" / "figures" / "pyocto_vs_manual_2months.png"

START = pd.Timestamp("2019-01-01", tz="UTC")
END = pd.Timestamp("2019-03-01", tz="UTC")

# --- pyocto daily ---
ev_files = sorted(glob.glob(str(REPO / "catalogs" / "pyocto_daily" / "events_*.csv")))
py_per_day = {}
py_picks_per_day = {}
for f in ev_files:
    tag = Path(f).stem.replace("events_", "")
    day = pd.Timestamp(tag, tz="UTC")
    if day < START or day >= END:
        continue
    try:
        df = pd.read_csv(f)
    except (pd.errors.EmptyDataError, Exception):
        py_per_day[day] = 0
        py_picks_per_day[day] = 0
        continue
    py_per_day[day] = len(df)
    pk_f = f.replace("events_", "picks_")
    try:
        pk = pd.read_csv(pk_f)
        py_picks_per_day[day] = len(pk)
    except (pd.errors.EmptyDataError, Exception):
        py_picks_per_day[day] = 0
py = pd.Series(py_per_day).sort_index()
py_pk = pd.Series(py_picks_per_day).sort_index()

# --- manual catalog ---
man = pd.read_csv(REPO / "catalogs" / "manual_picks.csv")
man = man[man.source_file == "nllmaleen_mag07_202210.out"].copy()
man["t"] = pd.to_datetime(man.origin_time, utc=True)
man = man[(man.t >= START) & (man.t < END)]
man_events = man.drop_duplicates(subset=["event_id"])

man_ev_per_day = man_events.groupby(man_events.t.dt.date).size()
man_pk_per_day = man.groupby(man.t.dt.date).size()

# align index to dates
all_dates = pd.date_range(START, END - pd.Timedelta("1d"), freq="D", tz="UTC")
py = py.reindex(all_dates, fill_value=0)
py_pk = py_pk.reindex(all_dates, fill_value=0)
man_ev = pd.Series({pd.Timestamp(d, tz="UTC"): v for d, v in man_ev_per_day.items()})
man_ev = man_ev.reindex(all_dates, fill_value=0)
man_pk = pd.Series({pd.Timestamp(d, tz="UTC"): v for d, v in man_pk_per_day.items()})
man_pk = man_pk.reindex(all_dates, fill_value=0)

# coverage: pyocto only finished days that exist as files
done_dates = set(py_per_day.keys())
finished = py.copy()
finished[:] = [py.loc[d] if d in done_dates else np.nan for d in py.index]

# --- plot ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top: events per day -- pyocto vs manual
ax = axes[0]
ax.bar(py.index - pd.Timedelta("4h"), finished, width=0.35,
       color="steelblue", edgecolor="k", linewidth=0.3,
       label=f"pyocto events ({int(finished.sum()):,})")
ax.bar(man_ev.index + pd.Timedelta("4h"), man_ev.values, width=0.35,
       color="C3", edgecolor="k", linewidth=0.3,
       label=f"manual events ({int(man_ev.sum()):,})")
# gray for not-yet-done pyocto days
not_done = [d for d in py.index if d not in done_dates]
for d in not_done:
    ax.axvspan(d - pd.Timedelta("12h"), d + pd.Timedelta("12h"),
               color="0.85", alpha=0.4)
ax.set_ylabel("events / day")
ax.set_yscale("symlog")
ax.set_title(f"Daily event counts — Jan + Feb 2019  "
             f"(pyocto: {len(done_dates)}/59 days completed)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3, axis="y")
ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))

# Bottom: pyocto associated picks vs manual picks
ax = axes[1]
ax.bar(py_pk.index - pd.Timedelta("4h"), py_pk.values, width=0.35,
       color="steelblue", edgecolor="k", linewidth=0.3,
       label=f"pyocto associated picks ({int(py_pk.sum()):,})")
ax.bar(man_pk.index + pd.Timedelta("4h"), man_pk.values, width=0.35,
       color="C2", edgecolor="k", linewidth=0.3,
       label=f"manual picks ({int(man_pk.sum()):,})")
ax.set_ylabel("picks / day")
ax.set_yscale("symlog")
ax.set_xlabel("date (2019)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3, axis="y")
ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
plt.setp(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(OUT, dpi=140)
print(f"wrote {OUT}")
print(f"pyocto total over completed days: {int(finished.sum()):,}")
print(f"manual events total:              {int(man_ev.sum()):,}")
print(f"manual picks total:               {int(man_pk.sum()):,}")
