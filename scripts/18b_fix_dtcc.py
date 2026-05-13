"""Post-process dt.cc to make the dt values true travel-time differentials.

The current dt.cc has raw (pick_a - pick_b) which is dominated by origin-time
differences (events minutes-hours apart give dt values of seconds-hours).
GrowClust expects dt = (tt_a - tt_b) where tt = pick - origin.

Fix: for each event pair, look up origin_a and origin_b, subtract (oa - ob) from
every dt value under that pair's header.
"""
from pathlib import Path
import pandas as pd
import sys

REPO = Path(__file__).resolve().parent.parent
LABEL = sys.argv[1] if len(sys.argv) > 1 else "partial30days"
GROWDIR = REPO / "growclust" / LABEL
DT_CC_IN = GROWDIR / "dt.cc"
DT_CC_OUT = GROWDIR / "dt.cc"
DT_CC_BAK = GROWDIR / "dt.cc.bak_raw_picktime"

# Load events to get origin times keyed by event_idx (1-based, matches evlist)
ev = pd.read_csv(REPO / "catalogs" / f"pyocto_events_{LABEL}.csv")
ev["origin_time"] = pd.to_datetime(ev["time"], unit="s", utc=True)
# evlist.txt is 1-based; the dt.cc references match what the script wrote:
# we used (anchor + 1, b + 1) keys, where anchor/b were positional indices.
# That should match evlist row order. Use positional index + 1 -> origin.
ev = ev.reset_index(drop=True)
origin_by_id = {i + 1: ev.iloc[i].origin_time.timestamp() for i in range(len(ev))}
print(f"loaded {len(origin_by_id):,} event origins")

# Make a backup of the broken dt.cc
if not DT_CC_BAK.exists():
    DT_CC_BAK.write_bytes(DT_CC_IN.read_bytes())
    print(f"backed up original -> {DT_CC_BAK.name}")

# Stream through dt.cc, rewriting
n_pairs = 0
n_obs = 0
n_dropped = 0
out_lines = []
with open(DT_CC_IN) as fh:
    current_otc = 0.0
    for ln in fh:
        ln = ln.rstrip()
        if ln.startswith("#"):
            parts = ln.split()
            id_a, id_b = int(parts[1]), int(parts[2])
            oa = origin_by_id.get(id_a)
            ob = origin_by_id.get(id_b)
            if oa is None or ob is None:
                current_otc = None  # signal: drop subsequent obs
                continue
            current_otc = oa - ob   # in seconds
            out_lines.append(f"# {id_a:8d} {id_b:8d}   0.0")
            n_pairs += 1
        else:
            if current_otc is None:
                n_dropped += 1
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            sta = parts[0]
            try: dt = float(parts[1])
            except: continue
            cc = parts[2]
            phase = parts[3]
            dt_corrected = dt - current_otc
            out_lines.append(f"{sta:12s} {dt_corrected:8.4f} {cc:>6s} {phase}")
            n_obs += 1

DT_CC_OUT.write_text("\n".join(out_lines) + "\n")
print(f"wrote {DT_CC_OUT}")
print(f"  pairs: {n_pairs:,}")
print(f"  obs:   {n_obs:,}")
if n_dropped:
    print(f"  dropped (missing event in header): {n_dropped:,}")

# sanity check distribution
import numpy as np
vals = []
with open(DT_CC_OUT) as fh:
    for ln in fh:
        if ln.startswith("#"): continue
        parts = ln.split()
        if len(parts) >= 4:
            try: vals.append(float(parts[1]))
            except: pass
vals = np.array(vals)
print(f"\ncorrected |dt| distribution (s):")
print(f"  P1={np.percentile(np.abs(vals),1):.3f}  P50={np.percentile(np.abs(vals),50):.3f}  "
      f"P90={np.percentile(np.abs(vals),90):.3f}  P99={np.percentile(np.abs(vals),99):.3f}  "
      f"max={np.max(np.abs(vals)):.3f}")
