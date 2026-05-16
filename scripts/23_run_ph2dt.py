"""Run the HypoDD `ph2dt` binary to build catalog differential-time pairs
(dt.ct) from phase.dat. Produces, in hypodd/<label>/:
    event.sel   - selected event list with HypoDD-internal ID column
    event.dat   - same as event.sel but full set
    dt.ct       - catalog differential travel times
    ph2dt.log   - ph2dt log
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BIN = Path("/home/jovyan/HypoDD/src/ph2dt/ph2dt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only")
    ap.add_argument("--binary", default=str(DEFAULT_BIN))
    # Tuned to match the GrowClust XC-prep pair selection:
    ap.add_argument("--minwght", type=float, default=0.2,
                    help="Min pick weight kept (ph2dt MINWGHT). Default 0.2 "
                         "matches the lower bound of our calibrated wt; picks "
                         "below this are uninformative for relative TTs.")
    ap.add_argument("--maxdist", type=float, default=100.0,
                    help="Max event-station distance km (ph2dt MAXDIST). "
                         "Bransfield array aperture is ~50 km; 100 km is "
                         "permissive without admitting garbage.")
    ap.add_argument("--maxsep", type=float, default=5.0,
                    help="Max event-event separation km (ph2dt MAXSEP).")
    ap.add_argument("--maxngh", type=int, default=80,
                    help="Max neighbours per event (ph2dt MAXNGH).")
    ap.add_argument("--minlnk", type=int, default=8,
                    help="Min phase links to qualify as neighbour (ph2dt MINLNK).")
    ap.add_argument("--minobs", type=int, default=8,
                    help="Min observations per pair to keep (ph2dt MINOBS).")
    ap.add_argument("--maxobs", type=int, default=80,
                    help="Max observations per pair (ph2dt MAXOBS). 38 stations "
                         "x 2 phases = up to 76 obs/pair, so allow 80.")
    args = ap.parse_args()

    run_dir = REPO / "hypodd" / args.label
    if not (run_dir / "phase.dat").exists():
        sys.exit(f"Missing {run_dir/'phase.dat'} -- run 22_pyocto_to_hypodd_input.py first.")
    if not (run_dir / "station.dat").exists():
        sys.exit(f"Missing {run_dir/'station.dat'}.")

    # ph2dt.inp template
    inp = f"""* ph2dt input control file
* file: station.dat
station.dat
* file: phase.dat
phase.dat
*
*--- (1)MINWGHT (2)MAXDIST (3)MAXSEP (4)MAXNGH (5)MINLNK (6)MINOBS (7)MAXOBS
{args.minwght:6.2f} {args.maxdist:7.1f} {args.maxsep:6.2f} {args.maxngh:5d} {args.minlnk:5d} {args.minobs:5d} {args.maxobs:5d}
"""
    (run_dir / "ph2dt.inp").write_text(inp)
    print(f"Wrote {run_dir/'ph2dt.inp'}")
    print(inp)

    print(f"Running ph2dt in {run_dir} ...")
    proc = subprocess.run(
        [args.binary, "ph2dt.inp"],
        cwd=run_dir,
        capture_output=True,
        timeout=3600,
    )
    log_path = run_dir / "ph2dt.log"
    log_path.write_text(proc.stdout.decode(errors="replace"))
    print(proc.stdout.decode(errors="replace")[-1500:])
    if proc.returncode != 0:
        print("---- stderr ----")
        print(proc.stderr.decode(errors="replace"))
        sys.exit(f"ph2dt returned {proc.returncode}")
    for out in ("event.sel", "event.dat", "dt.ct"):
        p = run_dir / out
        print(f"  {p}: "
              f"{(p.stat().st_size if p.exists() else 0):,} bytes")


if __name__ == "__main__":
    sys.exit(main() or 0)
