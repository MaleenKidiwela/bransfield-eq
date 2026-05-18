"""Convert pyocto picks/events to a single NLLOC_OBS phase file.

Format per pick line (one per event-station-phase):
    STA  INST  COMP  ON_DATE  PHASE  ?  YYYYMMDD  HHMM  SS.ssss  GAU  err  -1 -1 -1  1.0
Events separated by blank lines.

Filters (Phase 1):
  - Drop picks on stations without a P travel-time grid in nlloc/time/
    (i.e. anything that is not one of the 15 BRA OBS).
  - Drop events with fewer than --min-picks ZX picks remaining.

A sidecar `event_order.csv` lists pyocto event_idx in the same order
events appear in the .obs file -- used by the .hyp parser (script 31) to
re-attach event_idx (NLLoc does not propagate arbitrary IDs through SUM).

Output:
    nlloc/obs/<label>.obs
    nlloc/obs/<label>.event_order.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

REPO_ = Path(__file__).resolve().parent.parent


def _stations_with_grids(tt_dir: Path, prefix: str) -> set[str]:
    """Stations for which a P time grid exists (e.g. ORCA_v2.P.<STA>.time.hdr)."""
    hdrs = tt_dir.glob(f"{prefix}.P.*.time.hdr")
    return {h.name.split(".")[2] for h in hdrs}  # 'ORCA_v2.P.BRA05.time.hdr' -> 'BRA05'


ZX_GRID_STATIONS = {  # legacy default for the original ORCA-prefix run
    "ZX.BRA05", "ZX.BRA13", "ZX.BRA14", "ZX.BRA15", "ZX.BRA16",
    "ZX.BRA18", "ZX.BRA19", "ZX.BRA20", "ZX.BRA21", "ZX.BRA22",
    "ZX.BRA23", "ZX.BRA24", "ZX.BRA25", "ZX.BRA26", "ZX.BRA27",
}


def _fmt_pick(bare_sta: str, dt: pd.Timestamp, phase: str, err_s: float) -> str:
    ymd = dt.strftime("%Y%m%d")
    hm = dt.strftime("%H%M")
    sec = dt.second + dt.microsecond * 1e-6
    return (
        f" {bare_sta:<7s} SP    Z E      {phase} ? {ymd} {hm} "
        f"{sec:7.4f} GAU  {err_s:.2e} -1.00e+00 -1.00e+00 -1.00e+00  1.00e+00"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="picker_only_no_shots")
    p.add_argument("--events-csv",
                   default="catalogs/pyocto_events_picker_only_no_shots.csv")
    p.add_argument("--picks-csv",
                   default="catalogs/pyocto_picks_picker_only_no_shots.csv")
    p.add_argument("--min-picks", type=int, default=4)
    p.add_argument("--p-err-s", type=float, default=0.1)
    p.add_argument("--s-err-s", type=float, default=0.2)
    p.add_argument("--tt-prefix", default="ORCA",
                   help="If set to a prefix with grids for stations beyond the "
                        "default ZX 15, picks on those stations are kept too.")
    args = p.parse_args()

    ev = pd.read_csv(REPO / args.events_csv)
    pk = pd.read_csv(REPO / args.picks_csv)
    tt_dir = REPO / "nlloc" / "time"
    grid_stas = _stations_with_grids(tt_dir, args.tt_prefix)
    if not grid_stas:
        raise SystemExit(f"no time grids found for prefix {args.tt_prefix!r}")
    print(f"keeping picks on {len(grid_stas)} stations with {args.tt_prefix} grids")
    pk["bare_sta"] = pk["station"].str.split(".").str[1]
    pk = pk[pk["bare_sta"].isin(grid_stas)].copy()

    per_ev = pk.groupby("event_idx").size()
    keep = set(per_ev[per_ev >= args.min_picks].index)
    ev_keep = ev[ev["event_idx"].isin(keep)].sort_values("event_idx").reset_index(drop=True)

    pk["dt"] = pd.to_datetime(pk["time"], unit="s", utc=True)
    pk_by_ev = pk.groupby("event_idx", sort=False)

    out_dir = REPO / "nlloc" / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / f"{args.label}.obs"
    order_path = out_dir / f"{args.label}.event_order.csv"

    n_lines = 0
    n_events = 0
    with obs_path.open("w") as fh, order_path.open("w") as ofh:
        ofh.write("obs_order,event_idx\n")
        for order_idx, row in ev_keep.iterrows():
            eidx = int(row["event_idx"])
            grp = pk_by_ev.get_group(eidx).sort_values("dt")
            for _, pr in grp.iterrows():
                err = args.p_err_s if pr["phase"] == "P" else args.s_err_s
                fh.write(_fmt_pick(pr["bare_sta"], pr["dt"], pr["phase"], err) + "\n")
                n_lines += 1
            fh.write("\n")  # event separator
            ofh.write(f"{order_idx},{eidx}\n")
            n_events += 1

    print(f"wrote {n_events} events, {n_lines} pick lines -> {obs_path}")
    print(f"event-order map -> {order_path}")


if __name__ == "__main__":
    main()
