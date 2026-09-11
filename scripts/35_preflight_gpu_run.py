"""Readiness check for the full-year re-pick with the benchmarked pool.

Verifies the environment, the model weights, the channel mapping per network,
disk space and resume state, then TIMES a few station-days on whatever device
is present and projects the wall time for the whole run. Run this first; it
changes nothing.

    PYTHONPATH=src python scripts/35_preflight_gpu_run.py
    PYTHONPATH=src python scripts/35_preflight_gpu_run.py --timing-days 5 --workers 6
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# the benchmarked pool: diting everywhere (3C), PickBlue-PhaseNetLight on the OBS net (4C)
CH3 = "EH?,HH?,BH?,SH?,EL?,HL?,BL?,SL?"
CH4 = CH3 + ",EDH,HDH,BDH,SDH"
PLAN = [
    dict(tag="diting", model="PhaseNet",      weights="diting", glob=CH3,
         out="picks_pn_diting",   networks=None, kb=104),
    dict(tag="pnlight", model="PhaseNetLight", weights="obs",    glob=CH4,
         out="picks_pnlight_obs", networks=["ZX"], kb=221),
]

OK, WARN, BAD = "  \033[32mOK\033[0m  ", "  \033[33mWARN\033[0m", "  \033[31mFAIL\033[0m"
results: list[tuple[str, str]] = []


def check(label: str, ok: bool, detail: str = "", warn: bool = False) -> bool:
    tag = OK if ok else (WARN if warn else BAD)
    print(f"{tag}  {label}" + (f" — {detail}" if detail else ""), flush=True)
    results.append(("ok" if ok else ("warn" if warn else "fail"), label))
    return ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timing-days", type=int, default=3,
                   help="station-days to time per model for the projection (0 to skip)")
    p.add_argument("--workers", type=int, default=None,
                   help="worker count the real run will use (default: 6 on GPU, 8 on CPU)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    print("\n=== Full-year re-pick preflight ===\n")

    # 1 — epoch conversion (the pandas-3 bug)
    import pandas as pd
    from bransfield_eq.timeutil import assert_nanosecond_sanity
    try:
        assert_nanosecond_sanity()
        check(f"epoch conversion sane (pandas {pd.__version__})", True)
    except RuntimeError as e:
        check("epoch conversion", False, str(e))

    # 2 — no unguarded datetime->int casts left in the pipeline
    needle = 'astype("' + 'int64")'          # assembled so this file never self-matches
    bad = []
    for f in glob.glob(str(REPO / "scripts" / "*.py")) + glob.glob(str(REPO / "src/bransfield_eq/*.py")):
        if os.path.basename(f) in ("timeutil.py", os.path.basename(__file__)):
            continue
        if needle in open(f).read():
            bad.append(os.path.basename(f))
    check("no unguarded datetime->int64 casts", not bad,
          ", ".join(bad) if bad else "all conversions go through timeutil")

    # 3 — associator accepts the pool
    src = (REPO / "scripts" / "17_pyocto_associate.py").read_text()
    check("associator takes --pick-sources", "--pick-sources" in src)

    # 4 — device
    import torch
    cuda = torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0)
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("GPU usable by torch", True, f"{name}, {gb:.0f} GB")
    else:
        has_dev = bool(glob.glob("/dev/nvidia[0-9]*"))
        if has_dev:
            # a GPU is attached but torch cannot drive it — almost always a
            # torch-build / driver CUDA mismatch, which is fixable
            check("GPU usable by torch", False,
                  f"GPU attached but torch {torch.__version__} (built for CUDA "
                  f"{torch.version.cuda}) cannot initialise it — reinstall torch for the "
                  f"driver's CUDA version")
        else:
            check("GPU attached", False,
                  f"no GPU on this pod; torch {torch.__version__} is built for CUDA "
                  f"{torch.version.cuda}, so the GPU pod's driver must be CUDA "
                  f"{torch.version.cuda} or newer or torch needs reinstalling. "
                  f"CPU run works, ~10x slower", warn=True)
    device = "cuda" if cuda else "cpu"
    workers = a.workers or (6 if cuda else 8)

    # 5 — weights resolvable (cached or downloadable)
    import seisbench.models as sbm
    models = {}
    for step in PLAN:
        try:
            m = getattr(sbm, step["model"]).from_pretrained(step["weights"])
            models[step["tag"]] = m
            check(f"weights {step['model']}/{step['weights']}", True,
                  f"{m.component_order}, {m.in_channels}ch, {m.sampling_rate} Hz")
        except Exception as e:
            check(f"weights {step['model']}/{step['weights']}", False, str(e)[:70])

    # 6 — channel mapping per network
    runner = importlib.util.spec_from_file_location(
        "r", REPO / "scripts" / "03_run_phasenet.py")
    rmod = importlib.util.module_from_spec(runner); runner.loader.exec_module(rmod)
    from obspy import read
    for step in PLAN:
        nets = step["networks"] or ["ZX", "5M", "AI", "AM"]
        need = models.get(step["tag"])
        need_n = need.in_channels if need else 3
        detail = []
        ok_all = True
        for n in nets:
            f = [x for x in sorted(glob.glob(str(REPO / f"data/waveforms/{n}/*/*.mseed")))
                 if os.path.getsize(x) > 0]
            if not f:
                continue
            st = None
            for cand in f[len(f) // 2:] + f[:len(f) // 2]:
                try:
                    st = read(cand, headonly=True); break
                except Exception:
                    continue
            if st is None:
                detail.append(f"{n}:unreadable"); ok_all = False; continue
            comps = sorted({tr.stats.channel[-1] for tr in st
                            if rmod._channel_match(tr.stats.channel, step["glob"])})
            detail.append(f"{n}:{''.join(comps)}")
            if len(comps) < need_n:
                ok_all = False
        check(f"{step['tag']} channel mapping", ok_all, " ".join(detail),
              warn=not ok_all)
        if not ok_all:
            print(f"        {step['tag']} wants {need_n} channels; a network short of that "
                  f"gets the missing one zero-filled by SeisBench")

    # 7 — inventory, resume state, disk
    inv, empty = {}, {}
    for d in sorted(glob.glob(str(REPO / "data/waveforms/*"))):
        n = os.path.basename(d)
        files = glob.glob(f"{d}/*/*.mseed")
        inv[n] = sum(1 for x in files if os.path.getsize(x) > 0)
        empty[n] = len(files) - inv[n]
    total_all = sum(inv.values())
    print(f"\n  station-days with data: " +
          "  ".join(f"{k} {v:,}" for k, v in inv.items()) + f"   total {total_all:,}")
    print(f"  zero-byte placeholders skipped: {sum(empty.values()):,} "
          f"(" + " ".join(f"{k} {v}" for k, v in empty.items() if v) + ")")

    todo_total, bytes_total = 0, 0
    for step in PLAN:
        nets = step["networks"] or list(inv)
        want = sum(inv.get(n, 0) for n in nets)
        done = len(glob.glob(str(REPO / f"catalogs/{step['out']}/*/*.csv")))
        todo = max(want - done, 0)
        todo_total += todo
        bytes_total += todo * step["kb"] * 1024
        print(f"  {step['tag']:8} -> catalogs/{step['out']:18} "
              f"{want:6,} station-days, {done:6,} already done, {todo:6,} to run")
    st = os.statvfs(REPO / "catalogs")
    free_gb = st.f_bavail * st.f_frsize / 1e9
    check("disk space", free_gb > bytes_total / 1e9 * 2,
          f"need ~{bytes_total/1e9:.1f} GB, {free_gb:,.0f} GB free")

    # 8 — throughput, measured not guessed
    if a.timing_days and models:
        print(f"\n  timing {a.timing_days} station-day(s) per model on {device} ...")
        per_day = {}
        for step in PLAN:
            m = models.get(step["tag"])
            if m is None:
                continue
            m.to(device)
            nets = step["networks"] or ["ZX"]
            f = [x for x in sorted(glob.glob(str(REPO / f"data/waveforms/{nets[0]}/*/*.mseed")))
                 if os.path.getsize(x) > 0]
            sample = f[len(f) // 2: len(f) // 2 + a.timing_days]
            t0 = time.time()
            npk, nok = 0, 0
            for path in sample:
                try:
                    df = rmod.pick_one_day(m, Path(path), 100.0, 0.1, 0.1, step["glob"],
                                           batch_size=512 if device == "cuda" else 256)
                except Exception as e:
                    print(f"      skipped {os.path.basename(path)}: {str(e)[:50]}")
                    continue
                npk += len(df); nok += 1
            if not nok:
                print(f"    {step['tag']:8} could not time any station-day"); continue
            dt = (time.time() - t0) / nok
            per_day[step["tag"]] = dt
            print(f"    {step['tag']:8} {dt:6.1f} s/station-day serial "
                  f"({npk/nok:,.0f} picks/day)")
        if per_day:
            wall = 0.0
            for step in PLAN:
                if step["tag"] not in per_day:
                    continue
                nets = step["networks"] or list(inv)
                want = sum(inv.get(n, 0) for n in nets)
                done = len(glob.glob(str(REPO / f"catalogs/{step['out']}/*/*.csv")))
                wall += max(want - done, 0) * per_day[step["tag"]] / max(workers, 1)
            print(f"\n  projected wall time at --workers {workers}: "
                  f"{wall/3600:.1f} h for {todo_total:,} station-days")
            print("  (parallel efficiency is optimistic; treat it as a lower bound)")

    nfail = sum(1 for s, _ in results if s == "fail")
    nwarn = sum(1 for s, _ in results if s == "warn")
    print(f"\n=== {len(results)-nfail-nwarn} ok, {nwarn} warn, {nfail} fail ===")
    if nfail:
        print("Fix the failures before launching scripts/34_run_full_year_picks.sh\n")
        sys.exit(1)
    print("Ready. Launch with: bash scripts/34_run_full_year_picks.sh\n")


if __name__ == "__main__":
    main()
