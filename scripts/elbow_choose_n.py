"""Choose K-means N via the elbow method on the Stage A backbone.

For each candidate N in a range, fit K-means and record:
  - inertia (within-cluster sum-of-squared-distances)
  - silhouette score (-1..+1; how tight clusters are vs how separated)

Output:
  - notes/figures/stage_c_elbow.png with inertia + silhouette curves
  - Prints recommended N (max silhouette in the typical range)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def latlon_to_xy_km(lat, lon, lat0, lon0):
    R = 6371.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    return R * dlon * np.cos(np.radians(lat0)), R * dlat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-csv", default="catalogs/hypodd_picker_only_pruned.csv")
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=40)
    ap.add_argument("--out-plot", default="notes/figures/stage_c_elbow.png")
    args = ap.parse_args()

    bb = pd.read_csv(REPO / args.backbone_csv)
    print(f"loaded {len(bb):,} backbone events from {args.backbone_csv}")
    x, y = latlon_to_xy_km(bb.lat.values, bb.lon.values, bb.lat.mean(), bb.lon.mean())
    X = np.c_[x, y]

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Ns = list(range(args.n_min, args.n_max + 1))
    inertias, silhs = [], []
    for n in Ns:
        km = KMeans(n_clusters=n, n_init=10, random_state=42).fit(X)
        inertias.append(km.inertia_)
        s = silhouette_score(X, km.labels_) if n > 1 else np.nan
        silhs.append(s)
        print(f"  N={n:>3d}  inertia={km.inertia_:>10.2f}  silhouette={s:>6.3f}")

    inertias = np.array(inertias)
    silhs = np.array(silhs)

    # Elbow via "kneedle"-style: largest distance from the line connecting
    # first and last point.
    p0 = np.array([Ns[0], inertias[0]])
    p1 = np.array([Ns[-1], inertias[-1]])
    line = p1 - p0
    line_norm = line / np.linalg.norm(line)
    dists = []
    for i, n in enumerate(Ns):
        v = np.array([n, inertias[i]]) - p0
        proj = np.dot(v, line_norm) * line_norm
        perp = v - proj
        dists.append(np.linalg.norm(perp))
    elbow_idx = int(np.argmax(dists))
    N_elbow = Ns[elbow_idx]
    silh_idx = int(np.argmax(silhs))
    N_silh = Ns[silh_idx]

    print(f"\nelbow point: N = {N_elbow} (inertia={inertias[elbow_idx]:.1f})")
    print(f"max silhouette:  N = {N_silh} (silhouette={silhs[silh_idx]:.3f})")
    # Consensus pick
    if N_elbow == N_silh:
        N_star = N_elbow
        note = "elbow == silhouette"
    elif abs(N_elbow - N_silh) <= 2:
        N_star = int(round((N_elbow + N_silh) / 2))
        note = "elbow and silhouette close (avg)"
    else:
        N_star = N_silh
        note = "elbow and silhouette disagree; prefer silhouette"
    print(f"\nrecommended N* = {N_star}  ({note})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(Ns, inertias, "o-", color="steelblue")
    axes[0].axvline(N_elbow, color="orange", linestyle="--", label=f"elbow N={N_elbow}")
    axes[0].axvline(N_silh, color="green", linestyle="--", label=f"silh max N={N_silh}")
    axes[0].set_xlabel("N (number of clusters)")
    axes[0].set_ylabel("within-cluster SSE (km²)")
    axes[0].set_title("Elbow plot")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(Ns, silhs, "o-", color="darkorange")
    axes[1].axvline(N_silh, color="green", linestyle="--", label=f"silh max N={N_silh}")
    axes[1].set_xlabel("N (number of clusters)")
    axes[1].set_ylabel("silhouette score")
    axes[1].set_title("Silhouette")
    axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle(f"K-means N selection on Stage A backbone "
                 f"({len(bb):,} events)  →  recommended N* = {N_star}", fontsize=12)
    plt.tight_layout()
    out = REPO / args.out_plot
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    # Emit N_star to stdout last line for easy capture
    print(f"\nN_STAR={N_star}")


if __name__ == "__main__":
    main()
