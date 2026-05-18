"""Side-by-side comparison of the Vp/Vs=1.78 and Vp/Vs=2.10 NLLoc runs.

Produces a single figure with:
  - HQ map for each run on bathymetry
  - depth histogram overlay
  - per-event 3D shift (2.10 - 1.78) histogram
  - RMS scatter (1.78 vs 2.10)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import netcdf_file

REPO = Path(__file__).resolve().parent.parent
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

GX_MIN, GX_MAX = -29.8, 30.2
GY_MIN, GY_MAX = -20.0, 20.0


def load(label: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / "catalogs" / f"nlloc_{label}.csv")
    df["on_boundary"] = (
        (df.nlloc_x_km - GX_MIN < 0.5) | (GX_MAX - df.nlloc_x_km < 0.5) |
        (df.nlloc_y_km - GY_MIN < 0.5) | (GY_MAX - df.nlloc_y_km < 0.5)
    )
    df["hq"] = (~df.on_boundary & (df.gap_deg < 180) &
                (df.rms_s < 0.5) & (df.n_phases >= 6))
    return df


def main() -> None:
    a = load("picker_only_no_shots").set_index("event_idx")
    b = load("picker_only_no_shots_vpvs210").set_index("event_idx")

    common = a.index.intersection(b.index)
    a_c, b_c = a.loc[common], b.loc[common]
    both = a_c.hq & b_c.hq
    print(f"both HQ: {both.sum():,}")

    dlat = (b_c.lat - a_c.lat) * 111.32
    dlon = (b_c.lon - a_c.lon) * np.cos(np.radians(a_c.lat)) * 111.32
    ddep = b_c.depth_km - a_c.depth_km

    stations = pd.read_csv(ST)
    zx = stations[stations.network == "ZX"]
    with netcdf_file(str(BATHY), "r") as ds:
        lat_b = ds.variables["latitude"][:].copy()
        lon_b = ds.variables["longitude"][:].copy()
        z_b = ds.variables["data"][:].copy()
    z_b = np.where(z_b > 2000, np.nan, z_b)
    LON, LAT = np.meshgrid(lon_b, lat_b)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1], hspace=0.28, wspace=0.25)

    def map_panel(ax, df, hq_mask, title):
        levels = np.arange(-2400, 200, 50)
        norm_b = mcolors.TwoSlopeNorm(vmin=-2400, vcenter=0, vmax=200)
        ax.contourf(LON, LAT, z_b, levels=levels, cmap=plt.cm.GnBu_r,
                    norm=norm_b, extend="both")
        ax.contour(LON, LAT, z_b, levels=[0], colors="k", linewidths=0.6)
        ax.contour(LON, LAT, z_b, levels=[-1000], colors="k",
                   linewidths=1.0, zorder=10)
        sub = df[hq_mask]
        ax.scatter(sub.lon, sub.lat, c=sub.depth_km, s=4, cmap="magma_r",
                   vmin=0, vmax=20, alpha=0.55, edgecolors="none", zorder=7)
        ax.scatter(zx.longitude, zx.latitude, marker="^", s=70, c="white",
                   edgecolors="k", linewidths=1.0, zorder=11)
        ax.set_xlim(-58.7, -58.2); ax.set_ylim(-62.55, -62.35)
        ax.set_aspect(1.0 / np.cos(np.radians(-62.45)))
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title(f"{title} — {hq_mask.sum():,} HQ")

    all_mask_a = pd.Series(True, index=a.index)
    all_mask_b = pd.Series(True, index=b.index)
    map_panel(fig.add_subplot(gs[0, 0]), a, all_mask_a, "Vp/Vs = 1.78 (all)")
    map_panel(fig.add_subplot(gs[0, 1]), b, all_mask_b, "Vp/Vs = 2.10 (all)")

    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, 25, 51)
    ax.hist(a.depth_km, bins=bins, alpha=0.6, color="steelblue",
            label=f"1.78  (n={len(a):,})", edgecolor="k")
    ax.hist(b.depth_km, bins=bins, alpha=0.6, color="crimson",
            label=f"2.10  (n={len(b):,})", edgecolor="k")
    ax.set_xlabel("depth (km)"); ax.set_ylabel("event count (all)")
    ax.set_title("Depth distributions (all events)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    horiz_shift = np.sqrt(dlat**2 + dlon**2)[both]
    ax.hist(horiz_shift, bins=np.linspace(0, 5, 51), color="steelblue", edgecolor="k")
    ax.set_xlabel("horizontal shift |2.10 − 1.78| (km)")
    ax.set_ylabel("count")
    ax.set_title(f"horizontal shift  (median {horiz_shift.median():.2f} km)")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(ddep[both], bins=np.linspace(-5, 5, 51), color="crimson", edgecolor="k")
    ax.set_xlabel("depth shift (2.10 − 1.78) (km)"); ax.set_ylabel("count")
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title(f"depth shift  (median {ddep[both].median():.2f} km)")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(a_c.rms_s[both].clip(0, 1), b_c.rms_s[both].clip(0, 1),
               s=4, alpha=0.4, c="black")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("RMS (s)  Vp/Vs=1.78")
    ax.set_ylabel("RMS (s)  Vp/Vs=2.10")
    ax.set_title("per-event RMS comparison (HQ-both)")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "NLLoc Vp/Vs sensitivity: 1.78 vs 2.10  "
        f"({both.sum():,} events HQ in both runs)",
        fontsize=13,
    )
    out = REPO / "notes" / "figures" / "nlloc_vpvs_compare.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
