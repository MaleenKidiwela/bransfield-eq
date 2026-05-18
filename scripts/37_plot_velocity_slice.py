"""Plot a horizontal Vp anomaly slice from nlloc/srModel_3DEQ.mat at a given depth.

Anomaly is (Vp - <Vp>) / <Vp>, where <Vp> is the laterally-averaged Vp at
that depth across all nodes inside the model envelope. Plotted as a
diverging colormap (red = slow, blue = fast). Projects the Stingray-frame
grid to lat/lon via the same TRANS SIMPLE -62.4413 -58.44 36 used by NLLoc.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import netcdf_file

REPO = Path(__file__).resolve().parent.parent
SR_MAT = REPO / "nlloc" / "srModel_3DEQ.mat"
ST = REPO / "catalogs" / "station_geometry.csv"
BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"

LAT0, LON0, ROT_DEG = -62.4413, -58.44, 36.0


def stingray_xy_to_latlon(x_km: np.ndarray, y_km: np.ndarray):
    c, s = np.cos(np.radians(ROT_DEG)), np.sin(np.radians(ROT_DEG))
    dE = x_km * c - y_km * s
    dN = x_km * s + y_km * c
    lat = LAT0 + dN / 111.32
    lon = LON0 + dE / (np.cos(np.radians(LAT0)) * 111.32)
    return lat, lon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-km", type=float, default=2.0)
    ap.add_argument("--anomaly-pct", type=float, default=12.0,
                    help="Colorbar half-range in percent (default ±12%).")
    args = ap.parse_args()

    m = sio.loadmat(SR_MAT, squeeze_me=True, struct_as_record=False)
    sr = m["srModel"]
    u = sr.P.u
    xg, yg, zg = sr.xg, sr.yg, sr.zg
    iz = int(np.argmin(np.abs(zg - (-args.depth_km))))
    z_actual = -zg[iz]
    Vp = 1.0 / u[:, :, iz]
    Vp_mean = float(np.nanmean(Vp))
    dVp_pct = 100.0 * (Vp - Vp_mean) / Vp_mean
    print(f"iz={iz}  z={z_actual:.3f} km  <Vp>={Vp_mean:.3f} km/s")
    print(f"dVp/Vp range: {dVp_pct.min():.1f}% to {dVp_pct.max():.1f}%")

    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
    lat_grid, lon_grid = stingray_xy_to_latlon(Xg, Yg)

    stations = pd.read_csv(ST)
    zx = stations[stations.network == "ZX"]
    with netcdf_file(str(BATHY), "r") as ds:
        lat_b = ds.variables["latitude"][:].copy()
        lon_b = ds.variables["longitude"][:].copy()
        z_b = ds.variables["data"][:].copy()
    z_b = np.where(z_b > 2000, np.nan, z_b)
    LON_B, LAT_B = np.meshgrid(lon_b, lat_b)

    xs = np.array([-29.8, 30.2, 30.2, -29.8, -29.8])
    ys = np.array([-20.0, -20.0, 20.0, 20.0, -20.0])
    g_lat, g_lon = stingray_xy_to_latlon(xs, ys)

    fig, ax = plt.subplots(1, 1, figsize=(11, 9))
    ax.contour(LON_B, LAT_B, z_b, levels=[-2000, -1500, -1000, -500],
               colors="0.55", linewidths=0.5, alpha=0.7)
    ax.contour(LON_B, LAT_B, z_b, levels=[0], colors="k", linewidths=0.6)

    norm = mcolors.TwoSlopeNorm(vmin=-args.anomaly_pct, vcenter=0.0,
                                vmax=args.anomaly_pct)
    pc = ax.pcolormesh(lon_grid, lat_grid, dVp_pct, shading="auto",
                       cmap="RdBu_r", norm=norm, alpha=0.85, zorder=3)
    cb = plt.colorbar(pc, ax=ax, shrink=0.7, pad=0.02, extend="both")
    cb.set_label(f"(Vp − ⟨Vp⟩) / ⟨Vp⟩  at z = {z_actual:.1f} km   [%]\n"
                 f"⟨Vp⟩ = {Vp_mean:.2f} km/s")

    ax.plot(g_lon, g_lat, color="black", linewidth=2, linestyle="--",
            zorder=10, label="srModel / NLLoc grid")
    ax.scatter(zx.longitude, zx.latitude, marker="^", s=80, c="white",
               edgecolors="k", linewidths=1.2, zorder=12, label="ZX OBS")
    for _, r in zx.iterrows():
        ax.annotate(r.station.split(".")[-1], (r.longitude, r.latitude),
                    xytext=(5, 4), textcoords="offset points", fontsize=7,
                    color="white", zorder=13)

    ax.set_xlim(g_lon.min() - 0.05, g_lon.max() + 0.05)
    ax.set_ylim(g_lat.min() - 0.05, g_lat.max() + 0.05)
    ax.set_aspect(1.0 / np.cos(np.radians(LAT0)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"srModel_3DEQ Vp anomaly at z = {z_actual:.1f} km  "
                 f"(slow ↔ red, fast ↔ blue)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)

    out = REPO / "notes" / "figures" / f"velocity_anomaly_z{z_actual:.0f}km.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
