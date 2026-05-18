"""Run Grid2Time once per station on the extended P velocity grid.

Writes a Grid2Time control file using TRANS SIMPLE -62.4413 -58.44 36
(matches the existing nlloc/time/ORCA.P.* convention -- station x/y land
at the same Stingray-frame positions). Stations come from
catalogs/station_geometry.csv, filtered to those that have picks in the
no-shots catalog.

Outputs:
    nlloc/time/<prefix>.P.<STA>.time.hdr / .buf  (one per station)

Each Grid2Time invocation takes ~5-15 min depending on grid size. Runs
stations in parallel with multiprocessing.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

CONTROL_TEMPLATE = """CONTROL 1 54321
TRANS SIMPLE -62.4413 -58.44 36

GTFILES {model_root} {tt_root} P
GTMODE GRID3D ANGLES_NO
GT_PLFD 1.0e-3 0
{gtsrce}
"""


def build_gtsrce(station_geom_csv: Path, picks_csv: Path) -> list[tuple[str, float, float, float]]:
    """Pick stations that appear in the picks CSV; return list of
    (label, lat, lon, depth_km_for_GTSRCE)."""
    st = pd.read_csv(station_geom_csv)
    pk = pd.read_csv(picks_csv, usecols=["station"])
    have = set(pk.station.unique())
    out = []
    for _, r in st.iterrows():
        full = f"{r.network}.{r.station}"
        if full not in have:
            continue
        # NLLoc forbids '.' in GTSRCE labels; pick a unique name.
        # Use bare station code if it's unambiguous (no collision across networks),
        # otherwise NET_STA.
        label = r.station
        # depth: stations are at the surface (z=0) by NLLoc convention.
        # NLLoc's elev arg in GTSRCE is positive-up. We use 0.
        out.append((label, float(r.latitude), float(r.longitude), 0.0))
    return out


def run_one_station(args_tuple) -> tuple[str, int]:
    label, ctrl_text, work_dir = args_tuple
    work_dir.mkdir(parents=True, exist_ok=True)
    ctrl = work_dir / f"g2t_{label}.in"
    ctrl.write_text(ctrl_text)
    res = subprocess.run(
        ["Grid2Time", str(ctrl)],
        cwd=str(REPO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return label, res.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ORCA_v2")
    ap.add_argument("--picks-csv",
                    default="catalogs/pyocto_picks_picker_only_no_shots.csv")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not shutil.which("Grid2Time"):
        raise SystemExit("Grid2Time not on PATH")

    model_root = REPO / "nlloc" / "model" / args.prefix
    tt_root = REPO / "nlloc" / "time" / args.prefix

    sta_list = build_gtsrce(
        REPO / "catalogs" / "station_geometry.csv",
        REPO / args.picks_csv,
    )
    print(f"will compute TT grids for {len(sta_list)} stations: "
          f"{[s[0] for s in sta_list]}")

    # One Grid2Time call per station so we can parallelize and recover
    # individually-failed stations
    work_root = REPO / "nlloc" / "run" / "g2t_tmp"
    work_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for label, lat, lon, depth in sta_list:
        gtsrce = f"GTSRCE {label} LATLON {lat:.4f} {lon:.4f} 0.0 {depth:.1f}"
        text = CONTROL_TEMPLATE.format(
            model_root=str(model_root.relative_to(REPO)),
            tt_root=str(tt_root.relative_to(REPO)),
            gtsrce=gtsrce,
        )
        tasks.append((label, text, work_root / label))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one_station, tasks))
    failed = [lbl for lbl, rc in results if rc != 0]
    if failed:
        print(f"FAILED stations: {failed}")
    else:
        print("all stations ok")


if __name__ == "__main__":
    main()
