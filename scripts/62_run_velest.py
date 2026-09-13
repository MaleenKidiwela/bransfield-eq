"""Drive one VELEST run: stage inputs into velest/<run>/, write velest.cmn, execute.

VELEST always reads its control file from ./velest.cmn (velest.f:982), so every
run needs its own directory.  Damping is applied through the *thet parameters --
subr. SETUNT scales each parameter class by sqrt(othet/Xthet), so a huge xythet /
zthet freezes the hypocentres and a huge vthet freezes the layers -- and through
the per-layer vdamp column of the .mod file (999 = frozen).

Binary: ~/src/velest/orca/velest, built from github.com/jubenjum/velest
(VELEST 3.1, ETH) with vel_com.f array sizes raised to ieq=1600, ist=60,
inltot=40.  Single-threaded, ~0.2 GB resident.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BIN = Path.home() / "src" / "velest" / "orca" / "velest"


def freeze_model(src: Path, dst: Path) -> None:
    """Copy a .mod with every vdamp set to 999 (all layers held fixed)."""
    out = []
    for ln in src.read_text().splitlines():
        if len(ln) >= 26 and ln[:5].strip().replace(".", "").isdigit() and ln[19:26].strip():
            out.append(ln[:19] + "999.000" + ln[26:])
        else:
            out.append(ln)
    dst.write_text("\n".join(out) + "\n")


def velout_to_mod(velout: Path, freeze: bool, title: str) -> str:
    """VELEST writes velout.mod with the layer count right-padded ("        16"),
    which its own (i3) reader parses as 0.  Rewrite it as a valid input .mod."""
    blocks, cur = [], None
    for ln in velout.read_text().splitlines():
        t = ln.strip()
        if not t or t.lower().startswith("output"):
            continue
        if t.isdigit():
            cur = []
            blocks.append(cur)
            continue
        if cur is not None and len(ln) >= 26:
            cur.append((float(ln[0:5]), float(ln[10:17]), float(ln[19:26])))
    if not blocks:
        raise SystemExit(f"{velout}: no model block found")
    out = [title[:40]]
    for b, label in zip(blocks, ("P-VELOCITY MODEL", "S-VELOCITY MODEL")):
        out.append(f"{len(b):3d}        vel,depth,vdamp,phase (f5.2,5x,f7.2,2x,f7.3,3x,a1)")
        for i, (v, z, d) in enumerate(b):
            dd = 999.0 if freeze else d
            out.append(f"{v:5.2f}     {z:7.2f}  {dd:7.3f}" + (f"            {label}" if i == 0 else ""))
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name -> velest/<run>/")
    ap.add_argument("--input", default="velest/input")
    ap.add_argument("--model", default="orca.mod")
    ap.add_argument("--cnv", default="orca.cnv",
                    help="orca.cnv (all events) or orca_sub.cnv (inversion subset)")
    ap.add_argument("--model-from", default="",
                    help="take the start model from this run's velout.mod (frozen)")
    ap.add_argument("--sta-from", default="",
                    help="take station corrections from this run's final.STA and USE them")
    ap.add_argument("--nsinv", type=int, default=1,
                    help="1 = invert for station corrections, 0 = hold them fixed")
    ap.add_argument("--freeze-hypo", action="store_true",
                    help="hold x, y, z fixed (origin time stays free)")
    ap.add_argument("--freeze-model", action="store_true",
                    help="hold all layer velocities fixed (vdamp -> 999)")
    ap.add_argument("--othet", type=float, default=0.01)
    ap.add_argument("--xythet", type=float, default=0.01)
    ap.add_argument("--zthet", type=float, default=0.01)
    ap.add_argument("--vthet", type=float, default=1.0)
    ap.add_argument("--stathet", type=float, default=0.01)
    ap.add_argument("--swtfac", type=float, default=0.5)
    ap.add_argument("--vpvs", type=float, default=0.0,
                    help="override the control-file Vp/Vs (only used when nsp=3)")
    ap.add_argument("--nsp", type=int, default=2, choices=[2, 3],
                    help="2 = independent S model (Vp/Vs free per layer); "
                         "3 = S tied to P by the fixed control-file Vp/Vs")
    ap.add_argument("--ittmax", type=int, default=9)
    ap.add_argument("--iresolcalc", type=int, default=0)
    ap.add_argument("--lowveloclay", type=int, default=1,
                    help="1 = allow low-velocity layers (the shallow sediment wants one)")
    ap.add_argument("--zmin", type=float, default=0.30)
    ap.add_argument("--n-events", type=int, default=0, help="truncate the .cnv (smoke tests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inp = REPO / args.input
    meta = json.loads((inp / "meta.json").read_text())
    rd = REPO / "velest" / args.run
    rd.mkdir(parents=True, exist_ok=True)

    # ---- phase data -------------------------------------------------------
    lines = (inp / args.cnv).read_text().splitlines()
    neqs = int(meta["neqs_sub"] if args.cnv.endswith("_sub.cnv") else meta["neqs"])
    if args.n_events:
        blocks, cur = [], []
        for ln in lines:
            if ln.strip() == "9999":
                break
            if not ln.strip():
                if cur:
                    blocks.append(cur)
                cur = []
            else:
                cur.append(ln)
        if cur:
            blocks.append(cur)
        blocks = blocks[: args.n_events]
        neqs = len(blocks)
        lines = [l for b in blocks for l in (b + [""])] + ["9999"]
    (rd / "orca.cnv").write_text("\n".join(lines) + "\n")

    if args.sta_from:
        shutil.copy(REPO / "velest" / args.sta_from / "final.STA", rd / "orca.sta")
    else:
        shutil.copy(inp / "orca.sta", rd / "orca.sta")

    if args.model_from:
        src = velout_to_mod(REPO / "velest" / args.model_from / "velout.mod",
                            freeze=args.freeze_model,
                            title=f" model from velest/{args.model_from}")
        (rd / "start.mod").write_text(src)
    elif args.freeze_model:
        freeze_model(inp / args.model, rd / "start.mod")
    else:
        shutil.copy(inp / args.model, rd / "start.mod")

    xythet = 1.0e6 if args.freeze_hypo else args.xythet
    zthet = 1.0e6 if args.freeze_hypo else args.zthet
    vthet = 1.0e6 if args.freeze_model else args.vthet

    cmn = (inp / "cmn_template.txt").read_text().format(
        title=f"ORCA strict tier -- VELEST run {args.run}",
        olat=meta["olat"], olon=meta["olon"], neqs=neqs,
        iresolcalc=args.iresolcalc, dmax=meta["dmax"], zmin=args.zmin,
        lowveloclay=args.lowveloclay,
        swtfac=args.swtfac, vpvs=args.vpvs or meta["vpvs"],
        nsp=args.nsp, nmod=2 if args.nsp == 2 else 1,
        othet=args.othet, xythet=xythet, zthet=zthet,
        vthet=vthet, stathet=args.stathet,
        iusestacorr=1 if args.sta_from else 0, nsinv=args.nsinv,
        ittmax=args.ittmax, modfile="start.mod", stafile="orca.sta",
        cnvfile="orca.cnv")
    (rd / "velest.cmn").write_text(cmn)

    cfg = vars(args) | dict(neqs=neqs, xythet_eff=xythet, zthet_eff=zthet,
                            vthet_eff=vthet, binary=str(BIN))
    (rd / "run_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  {args.run}: neqs={neqs}  model={args.model}"
          f"{' (frozen)' if args.freeze_model else ''}"
          f"  hypo={'FIXED' if args.freeze_hypo else 'free'}  ittmax={args.ittmax}")
    if args.dry_run:
        return

    t0 = time.time()
    with open(rd / "stdout.log", "w") as fh:
        p = subprocess.run([str(BIN)], cwd=rd, stdout=fh, stderr=subprocess.STDOUT)
    print(f"  {args.run}: exit {p.returncode} in {time.time() - t0:.0f} s")
    print((rd / "stdout.log").read_text()[-1200:])


if __name__ == "__main__":
    main()
