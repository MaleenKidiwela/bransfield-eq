"""Parse the ablation NLLoc run and compare to the baseline v2 reliable
catalog event-by-event.

Joins refined .hyp files -> pyocto event_idx via the obs_order sidecar,
then merges against catalogs/nlloc_picker_only_no_shots_v2_reliable.csv
on event_idx. Reports per-event delta and population stats for
sigma_x, sigma_y, sigma_z, rms_s, gap_deg, n_phases, and writes a
4-panel scatter (refined vs baseline) figure.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_s31_spec = importlib.util.spec_from_file_location(
    "_s31", REPO / "scripts" / "31_nlloc_hyp_to_catalog.py"
)
_s31 = importlib.util.module_from_spec(_s31_spec)
_s31_spec.loader.exec_module(_s31)
parse_hyp = _s31.parse_hyp

BASE_CSV_DEFAULT = REPO / "catalogs" / "nlloc_picker_only_no_shots_v2.csv"


def parse_label_to_df(label: str) -> pd.DataFrame:
    """Parse all per-event .hyp files for a label (single-shard run only)
    and tag each with the pyocto event_idx via the label's event_order.csv.
    Match by SORTED FILENAME order = NLLoc input order = obs_order, which
    is stable even when NLLoc shifts the computed origin time."""
    out_dir = REPO / "nlloc" / "output" / label
    hyps = sorted([h for h in out_dir.glob("loc.20*.grid0.loc.hyp")
                   if "last" not in h.name])
    if not hyps:
        raise SystemExit(f"no per-event hyp files in {out_dir}")
    recs = []
    for h in hyps:
        r = parse_hyp(h)
        if r is None:
            r = {}
        recs.append(r)
    df = pd.DataFrame.from_records(recs).reset_index(drop=True)
    order_path = REPO / "nlloc" / "obs" / f"{label}.event_order.csv"
    order = pd.read_csv(order_path).sort_values("obs_order").reset_index(drop=True)
    if len(order) != len(df):
        print(f"WARN [{label}]: {len(order)} ordered events vs {len(df)} parsed hyps")
    n = min(len(order), len(df))
    df = df.iloc[:n].reset_index(drop=True)
    df["event_idx"] = order.event_idx.values[:n]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="ablation_refined")
    ap.add_argument("--base-csv", default=str(BASE_CSV_DEFAULT),
                    help="baseline NLLoc catalog (event_idx-keyed)")
    args = ap.parse_args()

    refined = parse_label_to_df(args.label)
    base = pd.read_csv(args.base_csv)
    print(f"refined: {len(refined)} events; baseline pool: {len(base)} events")

    cols = ["event_idx", "lat", "lon", "depth_km",
            "sigma_x_km", "sigma_y_km", "sigma_z_km",
            "rms_s", "n_phases", "gap_deg"]
    m = base[cols].merge(refined[cols], on="event_idx",
                         suffixes=("_base", "_ref"))
    print(f"merged {len(m)} events")

    metrics = ["sigma_x_km", "sigma_y_km", "sigma_z_km",
               "rms_s", "gap_deg", "n_phases"]
    print("\nPopulation medians (baseline -> refined):")
    print(f"{'metric':14s} {'base_med':>10s} {'ref_med':>10s} {'Δ_med':>10s} "
          f"{'n_better':>9s} {'n_worse':>9s}")
    for met in metrics:
        b = m[f"{met}_base"]; r = m[f"{met}_ref"]
        bmed = float(np.median(b))
        rmed = float(np.median(r))
        # For sigma/rms/gap "better" = smaller; for n_phases "better" = larger
        if met == "n_phases":
            better = int((r > b).sum()); worse = int((r < b).sum())
        else:
            better = int((r < b).sum()); worse = int((r > b).sum())
        print(f"{met:14s} {bmed:10.3f} {rmed:10.3f} {rmed-bmed:+10.3f} "
              f"{better:9d} {worse:9d}")

    # Horizontal offset between locations
    R = 6371.0
    lat0 = np.radians((m.lat_base + m.lat_ref) / 2)
    dy = np.radians(m.lat_ref - m.lat_base) * R
    dx = np.radians(m.lon_ref - m.lon_base) * R * np.cos(lat0)
    dz = m.depth_km_ref - m.depth_km_base
    m["dist_xy_km"] = np.sqrt(dx**2 + dy**2)
    m["ddepth_km"] = dz
    print(f"\nlocation drift (refined - baseline):")
    print(f"  horizontal median {m.dist_xy_km.median():.2f} km  "
          f"p95 {m.dist_xy_km.quantile(0.95):.2f} km")
    print(f"  depth      median {m.ddepth_km.median():+.2f} km  "
          f"abs-p95 {m.ddepth_km.abs().quantile(0.95):.2f} km")

    out_csv = REPO / "catalogs" / f"{args.label}_compare.csv"
    m.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # 4-panel scatter
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    panels = [("sigma_x_km", "σ_x [km]"),
              ("sigma_y_km", "σ_y [km]"),
              ("sigma_z_km", "σ_z [km]"),
              ("rms_s",      "RMS [s]")]
    for ax, (met, lbl) in zip(axes.ravel(), panels):
        b = m[f"{met}_base"]; r = m[f"{met}_ref"]
        ax.scatter(b, r, s=25, alpha=0.7, edgecolor="k", linewidth=0.4)
        lim = max(float(b.max()), float(r.max())) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y=x")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(f"baseline {lbl}")
        ax.set_ylabel(f"refined  {lbl}")
        ax.set_title(f"{lbl}  base med={np.median(b):.2f}  "
                     f"ref med={np.median(r):.2f}")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Pick-refinement ablation ({len(m)} reliable events)\n"
        "below diagonal = refined picks tighter than pyocto baseline"
    )
    fig.tight_layout()
    fig_path = REPO / "notes" / "figures" / f"{args.label}_compare.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
