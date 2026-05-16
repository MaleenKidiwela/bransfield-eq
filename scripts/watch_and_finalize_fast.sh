#!/bin/bash
# Wait for the orphaned hypoDD binary (PID 22568) to finish, then parse the
# reloc file into a CSV and plot. We do this because the python wrapper that
# was originally going to handle parsing was killed to escape its 3h timeout.
set -e
cd /home/jovyan/bransfield-eq

LABEL="picker_only_fast"
WATCH_PID=22568
LOG="logs/watch_${LABEL}.log"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== watching PID $WATCH_PID ==="
while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 60
done
log "PID $WATCH_PID exited"

RELOC="hypodd/${LABEL}/hypoDD.reloc"
if [ ! -s "$RELOC" ]; then
    log "!!! $RELOC missing or empty -- hypoDD must have crashed"
    exit 1
fi
log "reloc bytes: $(stat -c %s "$RELOC")"

log "=== parsing + depth-sanity + plotting ==="
.venv/bin/python - <<PY 2>&1 | tee -a "$LOG"
import pandas as pd
cols = ['id','lat','lon','dep','x','y','z','ex','ey','ez','yr','mo','dy','hr','mi','sc','mag','nccp','nccs','nctp','ncts','rcc','rct','cid']
df = pd.read_csv('hypodd/${LABEL}/hypoDD.reloc', names=cols, sep=r'\s+', engine='python')
st = pd.read_csv('catalogs/station_geometry.csv')
floor_km = st[st.network=='ZX'].water_depth_m.median()/1000
df['physical'] = df.dep >= floor_km
df.to_csv('catalogs/hypodd_${LABEL}.csv', index=False)
print(f'wrote catalogs/hypodd_${LABEL}.csv: {len(df):,} events; {df.physical.sum():,} physical, {(~df.physical).sum():,} above seafloor')
print(f'cid: unique={df.cid.nunique()}, largest={df.groupby("cid").size().max()}')
print(f'rms ct = {df.rct.mean():.3f} s,  rms cc = {df.rcc.mean():.3f}')
print(f'depth: median={df.dep.median():.2f}, p5={df.dep.quantile(.05):.2f}, p95={df.dep.quantile(.95):.2f}')
PY

.venv/bin/python scripts/plot_hypodd_relocations.py --label "$LABEL" \
    --title-extra "(year-long FAST, niter_ct=1)" >> "$LOG" 2>&1
log "=== done ==="
