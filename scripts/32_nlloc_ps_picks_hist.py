"""Histogram of P-picks and S-picks per event for an NLLoc catalog.

Counts come from the `Pha` column of each per-event .hyp PHASE block (the
phases actually used in the location, so n_P + n_S == catalog n_phases).

Events are attached to pyocto event_idx by reusing script 31's
round-robin obs_order mapping (block k -> shard k % N), then filtered to
the event_idx present in the chosen catalog CSV (e.g. the *_reliable subset).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def _load_script31():
    path = REPO / "scripts" / "31_nlloc_hyp_to_catalog.py"
    spec = importlib.util.spec_from_file_location("nlloc_hyp_to_catalog", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_ps(hyp_path: Path) -> tuple[int, int]:
    """Return (n_P, n_S) from a per-event .hyp PHASE block."""
    n_p = n_s = 0
    in_block = False
    for line in hyp_path.read_text().splitlines():
        if line.startswith("PHASE ID"):
            in_block = True
            continue
        if line.startswith("END_PHASE"):
            break
        if in_block and line.strip():
            parts = line.split()
            if len(parts) < 5:
                continue
            pha = parts[4]            # Sta Ins Cmp On Pha ...
            if pha.startswith("P"):
                n_p += 1
            elif pha.startswith("S"):
                n_s += 1
    return n_p, n_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-label", default="picker_only_no_shots_v2",
                    help="nlloc/output/<out-label> directory of per-event hyps")
    ap.add_argument("--catalog", default="picker_only_no_shots_v2_reliable",
                    help="catalogs/nlloc_<catalog>.csv defining the event subset")
    args = ap.parse_args()

    s31 = _load_script31()

    cat = pd.read_csv(REPO / "catalogs" / f"nlloc_{args.catalog}.csv")
    want = set(cat["event_idx"].astype(int))
    nphs_lookup = dict(zip(cat["event_idx"].astype(int), cat["n_phases"].astype(int)))
    print(f"catalog nlloc_{args.catalog}.csv: {len(cat)} events")

    out_dir = REPO / "nlloc" / "output" / args.out_label
    ordered = s31._collect_hyps_in_obs_order(out_dir)
    order = pd.read_csv(REPO / "nlloc" / "obs" / f"{args.out_label}.event_order.csv")
    oo_to_idx = dict(zip(order["obs_order"].astype(int), order["event_idx"].astype(int)))

    rows = []
    for oo, h in enumerate(ordered):
        if h is None:
            continue
        eidx = oo_to_idx.get(oo)
        if eidx is None or eidx not in want:
            continue
        n_p, n_s = count_ps(h)
        rows.append((eidx, n_p, n_s))

    df = pd.DataFrame(rows, columns=["event_idx", "n_P", "n_S"])
    df["total"] = df["n_P"] + df["n_S"]
    print(f"matched {len(df)} / {len(cat)} catalog events to hyp files")

    # sanity: P+S must equal catalog n_phases
    df["n_phases_cat"] = df["event_idx"].map(nphs_lookup)
    mism = int((df["total"] != df["n_phases_cat"]).sum())
    print(f"sanity check  n_P + n_S == catalog n_phases : "
          f"{len(df) - mism}/{len(df)} match"
          + (f"  ({mism} MISMATCH)" if mism else "  (all OK)"))

    print("\n            P-picks/event   S-picks/event   total/event")
    for name, col in (("mean", "mean"), ("median", "median"),
                      ("min", "min"), ("max", "max")):
        fn = getattr(df[["n_P", "n_S", "total"]], col)
        vals = fn()
        print(f"  {name:7s}   {vals['n_P']:11.2f}   {vals['n_S']:11.2f}   {vals['total']:9.2f}")
    print(f"\n  events with 0 S-picks: {(df['n_S'] == 0).sum()}"
          f"   |  events with 0 P-picks: {(df['n_P'] == 0).sum()}")

    # ---- histogram --------------------------------------------------------
    pmax = int(df["total"].max())
    bins = np.arange(0, pmax + 2) - 0.5  # integer-centred bins
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(df["n_P"], bins=bins, alpha=0.6, color="#1f77b4",
            label=f"P picks  (mean {df['n_P'].mean():.1f})", edgecolor="white", linewidth=0.4)
    ax.hist(df["n_S"], bins=bins, alpha=0.6, color="#d62728",
            label=f"S picks  (mean {df['n_S'].mean():.1f})", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("picks per event")
    ax.set_ylabel("number of events")
    ax.set_title(f"NLLoc P vs S picks per event — {args.catalog}\n({len(df):,} events)")
    ax.set_xlim(0, pmax + 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_png = REPO / "figures" / f"nlloc_ps_picks_per_event_{args.catalog}.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_png}")

    out_csv = REPO / "catalogs" / f"nlloc_{args.catalog}_ps_counts.csv"
    df[["event_idx", "n_P", "n_S", "total"]].to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
