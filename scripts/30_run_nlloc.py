"""Invoke NLLoc on a labelled obs file.

Single-process by default (NLLoc processes events serially in one .obs file,
~0.2-0.3 s per event => roughly 2-3 h for a 30k-event catalog).

Optional sharding (--shards N): splits the obs file into N chunks, runs
N NLLocs in parallel, and concatenates the per-shard .sum.grid0.loc.hyp
into a single combined file at nlloc/output/<label>/loc.sum.grid0.loc.hyp.
Per-event .hyp files from all shards co-exist in the same output dir.

Usage:
    python scripts/30_run_nlloc.py --label picker_only_no_shots [--shards 8]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def split_obs(obs_path: Path, n: int, work_dir: Path) -> list[Path]:
    text = obs_path.read_text()
    events = [b for b in text.split("\n\n") if b.strip()]
    chunks: list[list[str]] = [[] for _ in range(n)]
    for i, blk in enumerate(events):
        chunks[i % n].append(blk)
    paths = []
    for i, chunk in enumerate(chunks):
        p = work_dir / f"shard_{i:02d}.obs"
        p.write_text("\n\n".join(chunk) + "\n\n")
        paths.append(p)
    return paths


def write_shard_control(template: str, label: str, shard_idx: int,
                        shard_obs: Path, shard_out_dir: Path) -> Path:
    shard_out_dir.mkdir(parents=True, exist_ok=True)
    text = template.replace(f"nlloc/obs/{label}.obs", str(shard_obs.relative_to(REPO)))
    text = text.replace(f"nlloc/output/{label}/loc",
                        str((shard_out_dir / "loc").relative_to(REPO)))
    p = REPO / "nlloc" / "run" / f"{label}_shard_{shard_idx:02d}.in"
    p.write_text(text)
    return p


def run_nlloc(ctrl: Path) -> int:
    print(f"  starting {ctrl.name}", flush=True)
    res = subprocess.run(["NLLoc", str(ctrl)], cwd=str(REPO),
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"  FAILED {ctrl.name}: {res.stderr.decode()[-400:]}", flush=True)
    return res.returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="picker_only_no_shots")
    p.add_argument("--shards", type=int, default=1)
    args = p.parse_args()

    ctrl = REPO / "nlloc" / "run" / f"{args.label}.in"
    out_dir = REPO / "nlloc" / "output" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.shards <= 1:
        rc = run_nlloc(ctrl)
        sys.exit(rc)

    work = REPO / "nlloc" / "obs" / f"{args.label}_shards"
    work.mkdir(parents=True, exist_ok=True)
    obs_path = REPO / "nlloc" / "obs" / f"{args.label}.obs"
    shard_obs = split_obs(obs_path, args.shards, work)
    template = ctrl.read_text()
    ctrls = [
        write_shard_control(template, args.label, i, shard_obs[i],
                            out_dir / f"shard_{i:02d}")
        for i in range(args.shards)
    ]

    with ProcessPoolExecutor(max_workers=args.shards) as ex:
        rcs = list(ex.map(run_nlloc, ctrls))
    if any(rcs):
        print(f"some shards failed: {rcs}")
        sys.exit(1)

    # Concatenate per-shard SUM into one combined .sum
    combined = out_dir / "loc.sum.grid0.loc.hyp"
    with combined.open("w") as fh:
        for i in range(args.shards):
            shard_sum = out_dir / f"shard_{i:02d}" / "loc.sum.grid0.loc.hyp"
            if shard_sum.exists():
                fh.write(shard_sum.read_text())
    print(f"wrote combined SUM -> {combined}")


if __name__ == "__main__":
    main()
