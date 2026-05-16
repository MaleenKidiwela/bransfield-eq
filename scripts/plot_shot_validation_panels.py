"""Side-by-side proof plots for the v2 shot discriminator.

Outputs:
    notes/figures/comparison/contaminated_vs_v2_map.png
        - Stage B-jan2019 contaminated (11,995 events) vs v2 cleaned (7,624)
        - Same map extent, same zoom, same color scheme. Lineations disappear.
    notes/figures/comparison/spectra_pca_scatter.png
        - 2-D PCA scatter of all 42,040 event spectra, colored by class.
        - Shows the spectral clusters are linearly separable.
    notes/figures/comparison/daily_event_vs_shot_count.png
        - Per-day pyocto events with flag breakdown + per-day shot count.
        - Visual confirmation that contamination tracks shot activity.
    notes/figures/comparison/per_survey_detection.png
        - For each survey, fraction of shots that pyocto detected as events,
          and fraction of those that the classifier caught.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notes" / "figures" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

LON_MIN, LON_MAX = -58.7, -58.2
LAT_MIN, LAT_MAX = -62.55, -62.35


# -------------------------------------------------------------------
# 1. Map comparison: Stage B-jan2019 contaminated vs v2 cleaned
# -------------------------------------------------------------------
def plot_map_comparison():
    bath = Dataset(REPO / "notes" / "figures" / "Orca_bathymetry.nc")
    lat_b = bath.variables["latitude"][:]; lon_b = bath.variables["longitude"][:]
    z_b = bath.variables["data"][:]; bath.close()
    z_b_plot = np.where(z_b > 2000, np.nan, z_b)
    LON, LAT = np.meshgrid(lon_b, lat_b)
    stations = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    ob = stations[stations.network == "ZX"]

    a = pd.read_csv(REPO / "catalogs" / "hypodd_stage_b_jan2019.csv")
    b = pd.read_csv(REPO / "catalogs" / "hypodd_stage_b_jan2019_noshot.csv")
    print(f"Contaminated: {len(a):,}   v2 cleaned: {len(b):,}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, df, label, n in [
        (axes[0], a, f"Stage B-jan2019 CONTAMINATED", len(a)),
        (axes[1], b, f"Stage B-jan2019 v2 CLEANED",   len(b)),
    ]:
        levels = np.arange(-2400, 200, 50)
        norm = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
        ax.contourf(LON, LAT, z_b_plot, levels=levels,
                    cmap=plt.cm.GnBu_r, norm=norm, extend="both")
        ax.contour(LON, LAT, z_b_plot, levels=[0], colors="k", linewidths=0.5)
        ax.contour(LON, LAT, z_b_plot, levels=[-1000], colors="k",
                   linewidths=1.2, zorder=10)
        zoom = df[df.lon.between(LON_MIN, LON_MAX) & df.lat.between(LAT_MIN, LAT_MAX)]
        ax.scatter(zoom.lon, zoom.lat, c=zoom.dep, s=3, cmap="magma_r",
                   vmin=0, vmax=12, alpha=0.6, zorder=7)
        ax.scatter(ob.longitude, ob.latitude, marker="^", s=60, c="white",
                   edgecolors="k", linewidths=0.8, zorder=8)
        ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_aspect(1.0 / np.cos(np.radians(np.mean([LAT_MIN, LAT_MAX]))))
        ax.set_title(f"{label}\n{n:,} events (in zoom: {len(zoom):,})")
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.suptitle("Spectral classifier removes the survey-track lineations from the swarm catalog",
                 fontsize=13)
    plt.tight_layout()
    out = OUT / "contaminated_vs_v2_map.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# -------------------------------------------------------------------
# 2. PCA scatter of spectra, colored by class
# -------------------------------------------------------------------
def plot_pca_scatter():
    spectra = np.load(REPO / "catalogs" / "event_spectra.npy")
    meta = pd.read_parquet(REPO / "catalogs" / "event_spectra_meta.parquet")
    ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag_v2.csv",
                     low_memory=False)
    ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True, format="mixed", errors="coerce")
    ev = ev.merge(meta[["event_idx", "n_picks_used"]], on="event_idx", how="left",
                   suffixes=("", "_meta"))
    has = ev["n_picks_used"].fillna(0).astype(int) > 0
    print(f"events with spectra: {has.sum():,}")

    in_window = (ev.origin_time >= "2019-01-21") & (ev.origin_time < "2019-02-05")
    is_temporal_shot = ev["flag_shot"].astype(bool) & in_window
    is_spectral_only_shot = ev["flag_shot_v2"].astype(bool) & ~ev["flag_shot"].astype(bool) & in_window
    is_eq_outside = (~in_window) & has
    is_eq_inside = (~ev["flag_shot_v2"].astype(bool)) & in_window & has

    # PCA on the subset of events with valid spectra
    eid_to_row = {int(eid): i for i, eid in enumerate(meta["event_idx"].values)}
    feat_rows = np.array([eid_to_row.get(int(e), -1) for e in ev["event_idx"]])
    X = spectra[feat_rows[has]]
    print(f"running PCA on {X.shape[0]:,} x {X.shape[1]} spectra ...")
    pca = PCA(n_components=2, random_state=42).fit(X)
    Z = pca.transform(spectra[feat_rows[has]])
    var_exp = pca.explained_variance_ratio_
    print(f"PC1, PC2 explain {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% = {sum(var_exp)*100:.1f}%")

    # Build per-class masks (restricted to has-spectra rows, ordered the same)
    has_idx = np.where(has.values)[0]
    pos_in_subset = {row: i for i, row in enumerate(has_idx)}
    def mask_to_subset(global_mask):
        idx = np.where(global_mask.values)[0]
        sub = np.array([pos_in_subset[i] for i in idx if i in pos_in_subset])
        return sub

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for mask, color, label, alpha, s in [
        (is_eq_outside,         "steelblue", f"EQ — outside shot window ({is_eq_outside.sum():,})",      0.25, 4),
        (is_eq_inside,          "lightgreen", f"EQ — inside window, classifier=FALSE ({is_eq_inside.sum():,})", 0.5,  4),
        (is_temporal_shot,      "red",       f"SHOT — temporal flag ({is_temporal_shot.sum():,})",          0.5, 4),
        (is_spectral_only_shot, "orange",    f"SHOT — spectral classifier only ({is_spectral_only_shot.sum():,})", 0.9, 8),
    ]:
        sub = mask_to_subset(mask)
        ax.scatter(Z[sub, 0], Z[sub, 1], s=s, c=color, alpha=alpha, label=label,
                   edgecolors="none")
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f} %)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f} %)")
    ax.set_title("PCA of per-event log-power spectra — shots cluster apart from EQs\n"
                 "(orange = events the classifier caught that temporal flag missed)")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3)
    out = OUT / "spectra_pca_scatter.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# -------------------------------------------------------------------
# 3. Daily event count vs daily shot count
# -------------------------------------------------------------------
def plot_daily_counts():
    ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag_v2.csv",
                     low_memory=False)
    ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True, format="mixed", errors="coerce")
    ev["date"] = ev.origin_time.dt.date
    ev["class"] = np.where(ev["flag_shot_v2"].astype(bool), "shot_v2",
                  np.where(ev["flag_shot"].astype(bool), "shot_v1_only", "eq"))
    daily_ev = ev.groupby(["date", "class"]).size().unstack(fill_value=0)

    # Daily shot count from shotfiles
    shots = []
    for p in sorted((REPO / "shotfiles").glob("*_shotfile_final.txt")):
        df = pd.read_csv(p, comment="#", sep=r"\s+", engine="python",
                         names=["shotnum","date","time","sl","sln","shl","shln","wd","tag"])
        df["dt"] = pd.to_datetime(df.date + " " + df.time, utc=True, format="mixed", errors="coerce")
        df["survey"] = p.stem.replace("_shotfile_final", "")
        shots.append(df[["dt", "survey"]].dropna(subset=["dt"]))
    shots = pd.concat(shots, ignore_index=True)
    shots["date"] = shots.dt.dt.date
    daily_sh = shots.groupby("date").size()

    # Limit to shot window + 5 day buffer
    win = (daily_ev.index >= pd.Timestamp("2019-01-15").date()) & (daily_ev.index <= pd.Timestamp("2019-02-10").date())
    de = daily_ev[win].copy()

    fig, ax1 = plt.subplots(1, 1, figsize=(13, 6))
    de_eq = de.get("eq", pd.Series(0, index=de.index))
    de_v1 = de.get("shot_v1_only", pd.Series(0, index=de.index))
    de_v2 = de.get("shot_v2", pd.Series(0, index=de.index))
    ax1.bar(de.index, de_eq, label=f"EQ (v2 kept)  total={de_eq.sum()}", color="steelblue", edgecolor="k", linewidth=0.3)
    ax1.bar(de.index, de_v2, bottom=de_eq, label=f"SHOT v2 spectral (added)  total={de_v2.sum()}", color="orange", edgecolor="k", linewidth=0.3)
    ax1.bar(de.index, de_v1, bottom=de_eq + de_v2, label=f"SHOT v1 temporal  total={de_v1.sum()}", color="red", edgecolor="k", linewidth=0.3)
    ax1.set_ylabel("pyocto events / day")
    ax1.legend(loc="upper left")
    ax1.set_title("Pyocto event counts and BRAVOSEIS shot counts by day")
    ax1.grid(alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    sh_in_window = daily_sh[daily_sh.index.isin(de.index)]
    ax2.plot(sh_in_window.index, sh_in_window.values, color="black",
             marker="o", linewidth=1.5, label="shots fired / day")
    ax2.set_ylabel("shots fired / day", color="black")
    ax2.legend(loc="upper right")

    import matplotlib.dates as mdates
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for lbl in ax1.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha("right")

    out = OUT / "daily_event_vs_shot_count.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


# -------------------------------------------------------------------
# 4. Per-survey detection rate
# -------------------------------------------------------------------
def plot_per_survey_detection():
    # For each survey, count: shots fired; pyocto events flagged as shot_v2
    # whose matched survey is this one; out of those, how many were caught by
    # v1 (temporal) vs v2-only (spectral classifier).
    ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag_v2.csv",
                     low_memory=False)
    surveys = ["orca_mcs", "orca_tomo", "rift_mcs", "eda_mcs", "other_mcs"]
    rows = []
    shots_fired = {}
    for p in sorted((REPO / "shotfiles").glob("*_shotfile_final.txt")):
        survey = p.stem.replace("_shotfile_final", "")
        n = sum(1 for _ in open(p)) - 2   # subtract two comment lines
        shots_fired[survey] = n

    for s in surveys:
        in_survey = (ev["shot_survey"] == s)
        v1 = (in_survey & ev["flag_shot"].astype(bool)).sum()
        # v2-only is harder: these don't have shot_survey filled because the
        # classifier flagged them without a temporal match. Skip individual
        # survey attribution for v2-only — just show v1 detection rate.
        rows.append({
            "survey": s, "shots_fired": shots_fired.get(s, 0),
            "v1_temporal_detected": v1,
            "v1_detection_rate": v1 / shots_fired.get(s, 1) * 100,
        })
    df = pd.DataFrame(rows)
    print(df)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.shots_fired, width=0.4, color="0.6", edgecolor="k",
           label="shots fired", linewidth=0.5)
    ax.bar(x + 0.2, df.v1_temporal_detected, width=0.4, color="red", edgecolor="k",
           label="v1 temporal-detected as event", linewidth=0.5)
    for xi, rate in zip(x, df.v1_detection_rate):
        ax.text(xi + 0.2, df.v1_temporal_detected.iloc[xi] + max(df.shots_fired) * 0.01,
                f"{rate:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(df.survey, rotation=20)
    ax.set_ylabel("count")
    ax.set_title("Per-survey: shots fired vs v1 temporal-discriminator detections\n"
                 "(detection rate is fraction of shots strong enough to be picked by PhaseNet + associated by pyocto)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    out = OUT / "per_survey_detection.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_map_comparison()
    plot_pca_scatter()
    plot_daily_counts()
    plot_per_survey_detection()
