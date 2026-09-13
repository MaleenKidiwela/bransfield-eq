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
        # TRUE station depth, positive DOWN, on a sea-level datum (ORCA_v4+).
        # OBS sit on the seafloor (785-1943 m); land stations are above sea level
        # so they take a negative depth. Placing everything at 0 flattened the
        # array and made it impossible to locate a source above a deep-water OBS,
        # which jammed Orca's shallow events onto the grid's top face.
        # NB GTSRCE LATLON fields are: lat lon DEPTH ELEV. Putting the depth in
        # the elev slot would place an OBS ~1.4 km INTO THE AIR.
        if bool(r.get("on_seafloor", False)) and float(r.water_depth_m or 0) > 0:
            depth_km = float(r.water_depth_m) / 1000.0
        else:
            # Clamp at 0: the grid starts at sea level, so a station ABOVE it
            # cannot be placed and Grid2Time silently fails for it (this cost 10
            # land stations on the first attempt). Max elevation here is 30 m,
            # i.e. <=10 ms of error on 2% of picks.
            depth_km = max(0.0, -float(r.elevation_m or 0.0) / 1000.0)
        out.append((label, float(r.latitude), float(r.longitude), depth_km))
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
        gtsrce = f"GTSRCE {label} LATLON {lat:.6f} {lon:.6f} {depth:.4f} 0.0"
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
        print(f"FAILED stations (nonzero rc): {failed}")
    # Grid2Time can return 0 and still write nothing (e.g. a station outside the
    # grid), so check the actual files rather than the return code. This silently
    # dropped 10 land stations on the first ORCA_v4 attempt while printing "ok".
    built = {h.name.split(".")[2]
             for h in (REPO / "nlloc" / "time").glob(f"{args.prefix}.P.*.time.hdr")}
    wanted = {t[0] for t in tasks}
    missing = sorted(wanted - built)
    if missing:
        raise SystemExit(f"Grid2Time produced no grid for {len(missing)} "
                         f"station(s): {missing}")
    print(f"all {len(wanted)} stations ok")


if __name__ == "__main__":
    main()
