"""
GrowClust runner — assemble inputs, write control file, invoke binary, parse output.

Takes the dt.cc / evlist.txt / stlist.txt produced by 18_growclust_xc_prep.py and:
  - reformats / copies them into a GrowClust-style IN/ subdir
  - generates a vzmodel.txt from configs/velocity_model.csv
  - writes a control file (.inp)
  - calls the pre-compiled growclust binary at /home/jovyan/GrowClust/SRC/growclust
  - parses OUT/out.growclust_cat → catalogs/growclust_<label>.csv

Usage:
    python scripts/19_run_growclust.py --label picker_only

Assumes the GrowClust binary at the path above. Adjust --binary as needed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = Path("/home/jovyan/GrowClust/SRC/growclust")


def build_vzmodel(velocity_csv: Path, out_path: Path) -> None:
    """Convert configs/velocity_model.csv → GrowClust vzmodel.txt.

    GrowClust expects layered format: one row per layer interface with cumulative depth.
    Our CSV is a continuous depth-vp-vs profile; we treat it as piecewise-constant
    between samples (boundary at each depth, same Vp for [d_i, d_{i+1})).
    Vs column written as 0 if not present; GrowClust will derive from vpvs_factor.
    """
    df = pd.read_csv(velocity_csv)
    depth = df["depth_km"].values
    vp = df["vp_kms"].values
    vs = df["vs_kms"].values if "vs_kms" in df.columns else np.zeros_like(vp)
    # Clip negative depths (Orca grid starts at -0.0 due to float rounding)
    depth = np.clip(depth, 0.0, None)

    # Write each layer as two rows (top, bottom) with the same velocity.
    lines = []
    for i in range(len(depth) - 1):
        lines.append(f"{depth[i]:6.2f} {vp[i]:5.3f} {vs[i]:5.3f}")
        lines.append(f"{depth[i+1]:6.2f} {vp[i]:5.3f} {vs[i]:5.3f}")
    # Final half-space marker
    lines.append(f"{depth[-1]:6.2f} {vp[-1]:5.3f} {vs[-1]:5.3f}")
    out_path.write_text("\n".join(lines) + "\n")


def write_control(run_dir: Path, label: str) -> Path:
    """Write a GrowClust control (.inp) file referencing IN/, TT/, OUT/ inside run_dir."""
    # Conservative defaults tuned for OBS networks (Bransfield ~50 km aperture):
    #   tt_dep: 0–30 km, 0.5 km grid
    #   tt_del: 0–150 km, 1 km grid
    #   rmin (min CC):       0.6 (matches XC prep threshold)
    #   delmax (max sta dist): 100 km
    #   rmsmax: 0.5 s
    ctl = f"""****  GrowClust Control File  ****
**  Run label: {label}  **
*
* evlist_fmt
1
* fin_evlist
IN/evlist.txt
*
* stlist_fmt
1
* fin_stlist
IN/stlist.txt
*
* xcordat_fmt   tdif_fmt (12 = tt1-tt2)
1  12
* fin_xcordat
IN/xcordata.txt
*
* fin_vzmdl
IN/vzmodel.txt
* fout_vzfine
TT/vzfine.txt
* fout_pTT
TT/tt.pg
* fout_sTT
TT/tt.sg
*
* vpvs_factor   rayparam_min
  1.78          0.0
* tt_dep0  tt_dep1  tt_ddep
   0.0      60.0     0.5
* tt_del0  tt_del1  tt_ddel
   0.0      300.0    1.0
*
* rmin  delmax  rmsmax
  0.6    250      0.5
* rpsavgmin  rmincut  ngoodmin  iponly
   0          0         0         0
*
* nboot  nbranch_min
   0       1
* fout_cat
OUT/out.growclust_cat
* fout_clust
OUT/out.growclust_clust
* fout_log
OUT/out.growclust_log
* fout_boot
OUT/out.growclust_boot
"""
    path = run_dir / "control.inp"
    path.write_text(ctl)
    return path


def parse_growclust_cat(cat_path: Path) -> pd.DataFrame:
    """Parse OUT/out.growclust_cat into a DataFrame.

    GrowClust cat columns (whitespace-separated):
      yr mo dy hr mn sec evid lat lon dep mag qid cid n_branch n_pair n_pair_used
      rms_p rms_s eh_avg ez_avg t_uncert lat_orig lon_orig dep_orig
    Different versions have slightly different columns; we read with names=None
    and let the user inspect.
    """
    df = pd.read_csv(cat_path, sep=r"\s+", header=None)
    # Best-effort column naming for the common output
    cols = ["yr","mo","dy","hr","mn","sec","evid","lat","lon","dep","mag",
            "qid","cid","n_branch","n_pair","n_pair_used","rms_p","rms_s",
            "eh_avg","ez_avg","t_uncert","lat_orig","lon_orig","dep_orig"]
    if df.shape[1] == len(cols):
        df.columns = cols
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="picker_only")
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--velocity-csv", default="configs/velocity_model.csv")
    args = ap.parse_args()

    prep_dir = REPO / "growclust" / args.label
    if not prep_dir.exists():
        sys.exit(f"XC prep dir {prep_dir} not found — run 18_growclust_xc_prep.py first.")
    for needed in ("evlist.txt", "stlist.txt", "dt.cc"):
        if not (prep_dir / needed).exists():
            sys.exit(f"Missing {prep_dir/needed}; run XC prep first.")

    run_dir = prep_dir / "run"
    in_dir = run_dir / "IN"
    out_dir = run_dir / "OUT"
    tt_dir = run_dir / "TT"
    for d in (in_dir, out_dir, tt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Stage inputs
    print("Staging GrowClust inputs ...")
    shutil.copy(prep_dir / "evlist.txt", in_dir / "evlist.txt")
    shutil.copy(prep_dir / "stlist.txt", in_dir / "stlist.txt")
    shutil.copy(prep_dir / "dt.cc",      in_dir / "xcordata.txt")
    build_vzmodel(REPO / args.velocity_csv, in_dir / "vzmodel.txt")
    print(f"  evlist:   {(in_dir/'evlist.txt').stat().st_size:,} bytes")
    print(f"  stlist:   {(in_dir/'stlist.txt').stat().st_size:,} bytes")
    print(f"  xcordata: {(in_dir/'xcordata.txt').stat().st_size:,} bytes")
    print(f"  vzmodel:  {(in_dir/'vzmodel.txt').stat().st_size:,} bytes")

    # Control file
    ctl_path = write_control(run_dir, args.label)
    print(f"  control:  {ctl_path}")

    # Run binary (from run_dir so relative paths resolve)
    print(f"\nRunning {args.binary} from {run_dir} ...")
    # GrowClust expects the control file as argv[1], not stdin
    proc = subprocess.run(
        [args.binary, ctl_path.name],
        cwd=run_dir,
        capture_output=True,
        timeout=3600,
    )
    print("---- stdout ----")
    print(proc.stdout.decode(errors="replace"))
    cat_path = out_dir / "out.growclust_cat"
    # GrowClust often errors when writing the bootstrap output even with nboot=0;
    # the main catalog is still valid. Trust cat-file existence over returncode.
    if not cat_path.exists():
        print("---- stderr ----")
        print(proc.stderr.decode(errors="replace"))
        sys.exit(f"GrowClust did not produce {cat_path} (returncode={proc.returncode})")
    if proc.returncode != 0:
        print(f"  [warn] growclust returned {proc.returncode} — cat file present, continuing")

    df = parse_growclust_cat(cat_path)
    out_csv = REPO / "catalogs" / f"growclust_{args.label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df):,} relocated events)")
    if "lat" in df.columns and "lat_orig" in df.columns:
        dlat = (df["lat"] - df["lat_orig"]).abs().mean()
        dlon = (df["lon"] - df["lon_orig"]).abs().mean()
        ddep = (df["dep"] - df["dep_orig"]).abs().mean()
        print(f"  mean |Δlat|={dlat:.4f}°  |Δlon|={dlon:.4f}°  |Δdep|={ddep:.2f} km")
    print(f"\nDone.")


if __name__ == "__main__":
    sys.exit(main() or 0)
