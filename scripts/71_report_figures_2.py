"""The eight remaining Methods & Results figures (Figures 1, 2, 3, 7, 8, 9, 14, 15).

Companion to `scripts/69_final_figures.py`, which produced Figures 4, 5, 6, 10,
11, 12 and 13 (files F1-F8 in the same directory).  Same house style: 200 dpi,
Agg backend, Okabe-Ito colour-blind-safe palette, <= 4 cores, nothing written
outside `notes/figures/final/`.

    PYTHONPATH=src python3 scripts/71_report_figures_2.py [--only G01 G07 ...]

Outputs (notes/figures/final/):
  G01_location_map.png        draft Figure 1
  G02_picker_benchmark.png    draft Figure 2
  G03_velocity_model.png      draft Figure 3
  G07_crosscorrelation.png    draft Figure 7
  G08_uncertainty_budget.png  draft Figure 8
  G09_accounting.png          draft Figure 9
  G14_relative_geometry.png   draft Figure 14
  G15_validation.png          draft Figure 15
  captions_2.md               captions for the eight

Inputs (read-only, every one verified to exist before use):
  catalogs/station_geometry.csv                     38 stations; OBS elevation<0 = water depth
  notes/figures/Orca_bathymetry.nc + GEBCO_2023.nc  bathymetry (same precedence as scripts/41)
  notes/picker_benchmark.html                       `const DATA = {...}` benchmark metrics
  catalogs/manual_pick_recall_year_newpool.csv      analyst picks, res_raw/hit_raw/hit_cat
  nlloc/model/ORCA_v4.P.mod.{hdr,buf}               SLOW_LEN grid, Vp = dx / value
  nlloc/model/ORCA_v4.water_depth_km.npy            seafloor on the same (576,451) grid
  hypodd/year_v6_3d_xc/dtcc_measurements.npz        cc / lag / d / isP / gidx
  hypodd/year_v6_3d_xc/dtcc_summary.json            run parameters and percentiles
  catalogs/nlloc_year_v6_{standard,strict}.csv      absolute catalogue tiers
  catalogs/hypodd_year_v6_3d_{qc,xc_qc}.csv         relative catalogues
  nlloc/delays/v6_it1B.delays                       LOCDELAY station terms

Numbers that are NOT recomputed here are quoted from a named note and are marked
as such on the figure itself:
  * the cross-correlation null test  -> notes/30_xcorr_dtcc.md section D3
  * the station-term perturbation dz -> notes/31_final_catalogue_v6.md section 2
  * the processing accounting        -> notes/31_final_catalogue_v6.md section 3
"""
from __future__ import annotations

import os

# <= 4 cores (pod is CPU-capped; see notes/jupyterhub_pod_memory_limit)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.timeutil import assert_nanosecond_sanity  # noqa: E402

OUT = REPO / "notes" / "figures" / "final"
GEBCO = Path("/home/jovyan/ooi/rsn_cabled/SummerSchool2025/global_ocean_data/GEBCO_2023.nc")
ORCA_BATHY = REPO / "notes" / "figures" / "Orca_bathymetry.nc"
MAX_ELEV_KM = 1.0

# NLLoc control files, line 5: TRANS SIMPLE -62.4413 -58.44 36
CLAT, CLON, TRANS_ROT = -62.4413, -58.44, 36.0
CALDERA_LAT, CALDERA_LON = -62.45, -58.43
KM_PER_DEG_LAT = 111.195

VELMOD = REPO / "nlloc" / "model" / "ORCA_v4.P.mod"
WATER_NPY = REPO / "nlloc" / "model" / "ORCA_v4.water_depth_km.npy"
XC_DIR = REPO / "hypodd" / "year_v6_3d_xc"
TRUSTED_START = pd.Timestamp("2019-10-01", tz="UTC")

# Okabe-Ito, colour-blind safe (identical to scripts/69)
C = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
         yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", purple="#CC79A7",
         grey="#9a9a9a")
TIER_COLOR = {"loose": C["sky"], "standard": C["blue"], "strict": C["vermillion"],
              "hypodd": C["green"]}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.linewidth": 0.9, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
})

CAPTIONS: dict[str, str] = {}


# ----------------------------------------------------------------- helpers
def kmperdeglon(lat: float) -> float:
    return KM_PER_DEG_LAT * np.cos(np.radians(lat))


def ll_to_km(lat, lon, lat0=CLAT, lon0=CLON):
    e = (np.asarray(lon, float) - lon0) * kmperdeglon(lat0)
    n = (np.asarray(lat, float) - lat0) * KM_PER_DEG_LAT
    return e, n


def stingray_xy(lat, lon):
    """lat/lon -> NLLoc/Stingray grid x/y km (TRANS SIMPLE with 36 deg rotation).

    Identical to scripts/41_build_unsheared_velgrid.py::stingray_xy.
    """
    from pyproj import CRS, Transformer
    crs = CRS.from_proj4(f"+proj=tmerc +lat_0={CLAT} +lon_0={CLON} +ellps=WGS84")
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    X, Y = tx.transform(np.asarray(lon, float), np.asarray(lat, float))
    X, Y = X / 1e3, Y / 1e3
    a = np.radians(TRANS_ROT)
    return X * np.cos(a) + Y * np.sin(a), -X * np.sin(a) + Y * np.cos(a)


def grid_xy_to_latlon(x_km, y_km):
    """Inverse of stingray_xy (scripts/37_plot_velocity_slice.py convention)."""
    c, s = np.cos(np.radians(TRANS_ROT)), np.sin(np.radians(TRANS_ROT))
    dE = np.asarray(x_km, float) * c - np.asarray(y_km, float) * s
    dN = np.asarray(x_km, float) * s + np.asarray(y_km, float) * c
    return CLAT + dN / KM_PER_DEG_LAT, CLON + dE / kmperdeglon(CLAT)


_BATHY_CACHE: dict = {}


def bathymetry(lat0, lat1, lon0, lon1, n=700):
    """Water depth (km, +down) on a regular lat/lon grid over the box.

    Same source and precedence as scripts/41_build_unsheared_velgrid.py
    ::sample_bathymetry (30 m Orca multibeam where it covers the point, GEBCO_2023
    elsewhere).  Verbatim from scripts/69_final_figures.py::bathymetry.
    Returns (lat_1d, lon_1d, W[nlat, nlon], point_fn(lat, lon) -> km).
    """
    key = (round(lat0, 4), round(lat1, 4), round(lon0, 4), round(lon1, 4), n)
    if key in _BATHY_CACHE:
        return _BATHY_CACHE[key]
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    pad = 0.02
    g = xr.open_dataset(GEBCO)["elevation"].sel(
        lat=slice(lat0 - pad, lat1 + pad), lon=slice(lon0 - pad, lon1 + pad))
    glat = np.asarray(g.lat.values, float)
    glon = np.asarray(g.lon.values, float)
    ge = np.clip(np.asarray(g.values, float) / 1000.0, -6.0, MAX_ELEV_KM)
    fg = RegularGridInterpolator((glat, glon), ge, bounds_error=False, fill_value=np.nan)

    fo = None
    if ORCA_BATHY.exists():
        o = xr.open_dataset(ORCA_BATHY)
        olat = np.asarray(o.latitude.values, float)
        olon = np.asarray(o.longitude.values, float)
        i0, i1 = np.searchsorted(olat, [lat0 - pad, lat1 + pad])
        j0, j1 = np.searchsorted(olon, [lon0 - pad, lon1 + pad])
        i0, j0 = max(i0 - 1, 0), max(j0 - 1, 0)
        i1, j1 = min(i1 + 1, olat.size), min(j1 + 1, olon.size)
        if i1 - i0 > 4 and j1 - j0 > 4:
            od = np.asarray(o["data"][i0:i1, j0:j1].values, float)
            if od.shape != (i1 - i0, j1 - j0):
                od = od.T
            od = np.where(od > 2000.0, np.nan, od)      # glitch cells in the 30 m grid
            od = np.clip(od / 1000.0, -6.0, MAX_ELEV_KM)
            fo = RegularGridInterpolator((olat[i0:i1], olon[j0:j1]), od,
                                         bounds_error=False, fill_value=np.nan)

    def point_fn(lat, lon):
        lat = np.atleast_1d(np.asarray(lat, float))
        lon = np.atleast_1d(np.asarray(lon, float))
        pts = np.stack([lat, lon], axis=-1)
        e = fg(pts)
        if fo is not None:
            eo = fo(pts)
            e = np.where(np.isfinite(eo), eo, e)
        return np.clip(-e, 0.0, None)

    la = np.linspace(lat0, lat1, n)
    lo = np.linspace(lon0, lon1, n)
    LA, LO = np.meshgrid(la, lo, indexing="ij")
    W = point_fn(LA.ravel(), LO.ravel()).reshape(LA.shape)
    _BATHY_CACHE[key] = (la, lo, W, point_fn)
    return _BATHY_CACHE[key]


def load_stations():
    st = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    st["is_obs"] = st["elevation_m"] < 0
    st["water_km"] = np.where(st.is_obs, -st.elevation_m / 1000.0, 0.0)
    return st


def load_tier(tier):
    d = pd.read_csv(REPO / "catalogs" / f"nlloc_year_v6_{tier}.csv")
    d["ot"] = pd.to_datetime(d["origin_time"], utc=True, format="ISO8601")
    return d


def load_hypodd(name="hypodd_year_v6_3d_qc"):
    h = pd.read_csv(REPO / "catalogs" / f"{name}.csv")
    h["event_idx"] = h["id"].astype(int) - 1          # scripts/22, line 84
    return h


def hypodd_origin(h):
    """yr/mo/dy/hr/mi/sc -> tz-aware datetime (never .astype('int64') on a datetime)."""
    sec = np.asarray(h["sc"], float)
    whole = np.floor(sec).astype(int)
    frac_us = np.rint((sec - whole) * 1e6).astype(int)
    base = pd.to_datetime(dict(year=h.yr.astype(int), month=h.mo.astype(int),
                               day=h.dy.astype(int), hour=h.hr.astype(int),
                               minute=h.mi.astype(int), second=whole), utc=True)
    return base + pd.to_timedelta(frac_us, unit="us")


def load_delays(p=REPO / "nlloc" / "delays" / "v6_it1B.delays"):
    rows = []
    for line in Path(p).read_text().split("\n"):
        f = line.split()
        if len(f) == 5 and f[0] == "LOCDELAY":
            rows.append((f[1], f[2], int(f[3]), float(f[4])))
    return pd.DataFrame(rows, columns=["station", "phase", "n", "delay"])


def read_nlloc_grid(prefix=VELMOD):
    """SLOW_LEN grid -> Vp[nx, ny, nz] km/s, origin and spacing.

    Same reader as scripts/56_build_hypodd_3dmodel.py::read_nlloc_grid.
    """
    hdr = Path(str(prefix) + ".hdr").read_text().split()
    nx, ny, nz = map(int, hdr[:3])
    x0, y0, z0 = map(float, hdr[3:6])
    dx, dy, dz = map(float, hdr[6:9])
    assert hdr[9] == "SLOW_LEN", hdr
    buf = np.fromfile(str(prefix) + ".buf", dtype=np.float32).reshape(nx, ny, nz)
    return dx / buf, (x0, y0, z0), (dx, dy, dz), (nx, ny, nz)


def manual_picks():
    M = pd.read_csv(REPO / "catalogs" / "manual_pick_recall_year_newpool.csv")
    M["day"] = pd.to_datetime(M["day"], utc=True)
    M["network"] = M.sta_key.str.split(".").str[0]
    M["station"] = M.sta_key.str.split(".").str[-1]
    M["trusted"] = M["day"] >= TRUSTED_START
    return M


def manual_sp_table(M=None):
    """(event, station) analyst S-P, with the auto-minus-manual S-P on the same pairs."""
    M = manual_picks() if M is None else M
    piv = M.pivot_table(index=["event_id", "sta_key"], columns="phase",
                        values=["t", "res_raw", "hit_raw"], aggfunc="min")
    piv = piv.dropna(subset=[("t", "P"), ("t", "S")])
    out = pd.DataFrame({
        "sp": piv[("t", "S")] - piv[("t", "P")],
        "d_auto": piv[("res_raw", "S")] - piv[("res_raw", "P")],
        "both_hit": (piv[("hit_raw", "P")] > 0) & (piv[("hit_raw", "S")] > 0),
    }).reset_index()
    out["network"] = out.sta_key.str.split(".").str[0]
    out["station"] = out.sta_key.str.split(".").str[-1]
    return out


def scalebar(ax, km, lat_ref, label=None, loc=(0.06, 0.07), color="k"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dlon = km / kmperdeglon(lat_ref)
    xs = x0 + loc[0] * (x1 - x0)
    ys = y0 + loc[1] * (y1 - y0)
    ax.plot([xs, xs + dlon], [ys, ys], color=color, lw=3, solid_capstyle="butt",
            zorder=8, clip_on=False)
    ax.text(xs + dlon / 2, ys + 0.018 * (y1 - y0), label or f"{km:g} km",
            ha="center", va="bottom", fontsize=9.5, color=color, zorder=8)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    kb = p.stat().st_size / 1024
    print(f"  [{'OK ' if kb > 30 else 'SMALL'}] {p.relative_to(REPO)}  {kb:,.0f} kB")
    return p


# ------------------------------------------------------------------ G01
def g01():
    """Draft Figure 1 - location map."""
    st = load_stations()
    zx = st[st.network == "ZX"]
    obs, land = st[st.is_obs], st[~st.is_obs]

    # panel (c): the dense cluster on the edifice, within 8 km of the projection origin
    _e, _n = ll_to_km(zx.latitude.values, zx.longitude.values)
    dz = zx[np.hypot(_e, _n) <= 9.0]
    dlat0, dlat1 = dz.latitude.min() - 0.020, dz.latitude.max() + 0.020
    dlon0, dlon1 = dz.longitude.min() - 0.050, dz.longitude.max() + 0.050
    dense = set(dz.station)
    # panel (b): the whole ZX ocean-bottom array
    lat0, lat1 = zx.latitude.min() - 0.06, zx.latitude.max() + 0.07
    lon0, lon1 = zx.longitude.min() - 0.12, zx.longitude.max() + 0.12
    # panel (a): the Bransfield Strait
    rlat0, rlat1, rlon0, rlon1 = -63.55, -61.95, -61.30, -57.10

    shades = ["#f2f2ee", "#dceaf6", "#c2dcef", "#a4cbe6", "#84b8dc", "#5f9fcd", "#3d84b8"]
    LEV = [-10, 1, 400, 800, 1200, 1600, 2000, 6000]
    CMAP, VMIN, VMAX = "cividis", 0.7, 2.0

    fig = plt.figure(figsize=(17.4, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.10, 1.0], wspace=0.24)

    # ---------------- (a) regional
    axa = fig.add_subplot(gs[0])
    rla, rlo, RW, _ = bathymetry(rlat0, rlat1, rlon0, rlon1, n=320)
    axa.contourf(rlo, rla, RW * 1000.0, levels=LEV, colors=shades, zorder=0)
    axa.contour(rlo, rla, RW * 1000.0, levels=[1.0], colors="k", linewidths=0.7, zorder=1)
    axa.contour(rlo, rla, RW * 1000.0, levels=[1000.0], colors=C["grey"], linewidths=0.6,
                zorder=1)
    axa.scatter(obs.longitude, obs.latitude, marker="^", s=55, c=C["vermillion"],
                edgecolor="k", lw=0.5, zorder=4, label=f"ZX OBS ({len(obs)})")
    axa.scatter(land.longitude, land.latitude, marker="s", s=42, facecolor="white",
                edgecolor="k", lw=0.8, zorder=4,
                label=f"land / island ({len(land)}): "
                      f"{', '.join(sorted(land.network.unique()))}")
    axa.plot([], [], color=C["blue"], lw=1.8, label="frame of (b)")
    for _, r in land.iterrows():
        axa.annotate(r.station, (r.longitude, r.latitude), textcoords="offset points",
                     xytext=(5, 3), fontsize=6.8, zorder=5,
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    axa.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False,
                            edgecolor=C["blue"], lw=1.8, zorder=6))
    axa.set_xlim(rlon0, rlon1)
    axa.set_ylim(rlat0, rlat1)
    axa.set_aspect(1.0 / np.cos(np.radians(CLAT)))
    axa.set_xlabel("longitude (deg E)")
    axa.set_ylabel("latitude (deg N)")
    axa.set_title(f"(a)  Bransfield Strait, all {len(st)} stations", loc="left")
    axa.legend(loc="lower left", framealpha=0.93, fontsize=8)
    axa.text(0.30, 0.90, "Bransfield Strait", transform=axa.transAxes, ha="center",
             va="top", fontsize=10, style="italic", zorder=7,
             bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
    axa.text(0.80, 0.045, "Antarctic Peninsula", transform=axa.transAxes, ha="center",
             va="bottom", fontsize=8.5, style="italic", color="0.30", zorder=7)

    # ---------------- (b) the ZX array
    axb = fig.add_subplot(gs[1])
    la, lo, W, _ = bathymetry(lat0, lat1, lon0, lon1, n=560)
    axb.contourf(lo, la, W * 1000.0, levels=LEV, colors=shades, zorder=0)
    cl = np.arange(250, 2251, 250)
    cs = axb.contour(lo, la, W * 1000.0, levels=cl, colors=C["grey"], linewidths=0.5, zorder=1)
    axb.clabel(cs, cl[1::2], fmt="%d", fontsize=6.5, inline=True)
    axb.contour(lo, la, W * 1000.0, levels=[1.0], colors="k", linewidths=0.8, zorder=2)
    m = ((obs.latitude.between(lat0, lat1)) & (obs.longitude.between(lon0, lon1)))
    sc = axb.scatter(obs.longitude[m], obs.latitude[m], marker="^", s=150,
                     c=obs.water_km[m], cmap=CMAP, vmin=VMIN, vmax=VMAX,
                     edgecolor="k", lw=0.9, zorder=6)
    ml = ((land.latitude.between(lat0, lat1)) & (land.longitude.between(lon0, lon1)))
    axb.scatter(land.longitude[ml], land.latitude[ml], marker="s", s=80,
                facecolor="white", edgecolor="k", lw=1.2, zorder=6)
    for _, r in pd.concat([obs[m], land[ml]]).iterrows():
        if r.station in dense:                       # labelled in (c) instead
            continue
        axb.annotate(r.station, (r.longitude, r.latitude), textcoords="offset points",
                     xytext=(7, 5), fontsize=7.8, zorder=7,
                     bbox=dict(fc="white", ec="none", alpha=0.68, pad=0.7))
    axb.add_patch(Rectangle((dlon0, dlat0), dlon1 - dlon0, dlat1 - dlat0, fill=False,
                            edgecolor=C["vermillion"], lw=1.8, zorder=8))
    axb.plot(CALDERA_LON, CALDERA_LAT, marker="*", ms=16, color=C["yellow"],
             markeredgecolor="k", markeredgewidth=0.8, zorder=9)
    axb.set_xlim(lon0, lon1)
    axb.set_ylim(lat0, lat1)
    axb.set_aspect(1.0 / np.cos(np.radians(CLAT)))
    axb.set_xlabel("longitude (deg E)")
    axb.set_ylabel("latitude (deg N)")
    axb.set_title("(b)  The ZX ocean-bottom array", loc="left")
    scalebar(axb, 20, CLAT, loc=(0.045, 0.945))
    axb.legend(handles=[
        Line2D([], [], marker="^", ls="", ms=10, color="#8d8a63", markeredgecolor="k",
               label=f"ZX OBS, shaded by water depth ({int(m.sum())})"),
        Line2D([], [], marker="s", ls="", ms=8, markerfacecolor="white",
               markeredgecolor="k", label=f"land station ({int(ml.sum())} in frame)"),
        Line2D([], [], marker="*", ls="", ms=13, color=C["yellow"], markeredgecolor="k",
               label="Orca caldera"),
        Line2D([], [], color=C["grey"], lw=0.8, label="bathymetry, 250 m"),
        Line2D([], [], color=C["vermillion"], lw=1.8, label="frame of (c)"),
    ], loc="lower right", framealpha=0.93, fontsize=8)

    # ---------------- (c) the dense cluster on the edifice
    axc = fig.add_subplot(gs[2])
    dla, dlo, DW, _ = bathymetry(dlat0, dlat1, dlon0, dlon1, n=420)
    axc.contourf(dlo, dla, DW * 1000.0,
                 levels=[-10, 600, 800, 1000, 1200, 1400, 1600, 6000],
                 colors=shades, zorder=0)
    dcl = np.arange(650, 1651, 50)
    cs2 = axc.contour(dlo, dla, DW * 1000.0, levels=dcl, colors=C["grey"],
                      linewidths=0.45, zorder=1)
    axc.clabel(cs2, dcl[1::4], fmt="%d", fontsize=6.2, inline=True)
    axc.contour(dlo, dla, DW * 1000.0, levels=[1000], colors="k", linewidths=1.1, zorder=2)
    sc2 = axc.scatter(dz.longitude, dz.latitude, marker="^", s=150, c=dz.water_km,
                      cmap=CMAP, vmin=VMIN, vmax=VMAX, edgecolor="k", lw=0.8, zorder=4)
    for _, r in dz.iterrows():
        axc.annotate(r.station, (r.longitude, r.latitude), textcoords="offset points",
                     xytext=(6, 5), fontsize=7.6, zorder=6,
                     bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.6))
    axc.plot(CALDERA_LON, CALDERA_LAT, marker="*", ms=19, color=C["yellow"],
             markeredgecolor="k", markeredgewidth=0.8, zorder=5)
    axc.annotate("Orca caldera", (CALDERA_LON, CALDERA_LAT), textcoords="offset points",
                 xytext=(-14, -22), ha="right", fontsize=10, fontweight="bold", zorder=7,
                 bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.5))
    axc.set_xlim(dlon0, dlon1)
    axc.set_ylim(dlat0, dlat1)
    axc.set_aspect(1.0 / np.cos(np.radians(CLAT)))
    axc.set_xlabel("longitude (deg E)")
    axc.set_ylabel("latitude (deg N)")
    axc.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    axc.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    axc.set_title(f"(c)  The Orca edifice, {len(dz)} OBS", loc="left")
    scalebar(axc, 2, CLAT, loc=(0.045, 0.945))
    cb = fig.colorbar(sc2, ax=axc, pad=0.015, shrink=0.86)
    cb.set_label("OBS water depth (km)")
    axc.text(0.985, 0.035, "black line: 1000 m isobath\ngrey lines: 50 m",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(fc="white", ec=C["grey"], lw=0.9, alpha=0.93, pad=3))

    CAPTIONS["G01"] = (
        f"**Figure 1 - location map.** Bathymetry throughout is the 30 m Orca multibeam "
        f"grid inside its footprint and GEBCO_2023 outside it - the same surface and the "
        f"same precedence used to un-shear the velocity grid "
        f"(`scripts/41_build_unsheared_velgrid.py::sample_bathymetry`) - and every station "
        f"position is from `catalogs/station_geometry.csv`. (a) Regional setting of the "
        f"whole {len(st)}-station network: {len(obs)} ZX ocean-bottom seismometers "
        f"(vermillion triangles) and {len(land)} land and island stations of networks "
        f"{'/'.join(sorted(land.network.unique()))} (white squares, labelled), with the "
        f"coast in black and the 1000 m isobath in grey. (b) The ZX array, filled shading "
        f"and grey contours every 250 m; OBS are shaded by water depth (range "
        f"{obs.water_km.min():.3f}-{obs.water_km.max():.3f} km, taken as the negative "
        f"`elevation_m`) and the star marks the Orca caldera. (c) The dense cluster on the "
        f"edifice - the {len(dz)} instruments within 9 km of the NLLoc projection origin "
        f"{CLAT}/{CLON}, a {(dlon1-dlon0)*kmperdeglon(CLAT):.0f} x "
        f"{(dlat1-dlat0)*KM_PER_DEG_LAT:.0f} km footprint that supplies almost all of the "
        f"catalogue - over the 30 m grid contoured every 50 m, with the 1000 m isobath "
        f"picking out the caldera rim. Scale bars 20 km in (b) and 2 km in (c)."
    )
    return save(fig, "G01_location_map")


# ------------------------------------------------------------------ G02
def read_benchmark_data():
    """`const DATA = {...};` embedded in notes/picker_benchmark.html."""
    html = (REPO / "notes" / "picker_benchmark.html").read_text()
    m = re.search(r"const DATA = (\{.*?\});\s*\n", html, flags=re.S)
    if m is None:
        return None
    return json.loads(m.group(1))


def g02():
    """Draft Figure 2 - picker benchmark."""
    D = read_benchmark_data()
    if D is None:
        raise SystemExit("could not find `const DATA` in notes/picker_benchmark.html")
    P = pd.DataFrame(D["pickers"]).sort_values("P_recall", ascending=True).reset_index(drop=True)
    pools = pd.DataFrame(D["pools"])

    fig = plt.figure(figsize=(14.4, 11.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82], hspace=0.30, wspace=0.24)

    # (a) recall bars, 12 models
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(P))
    bh = 0.38
    ax.barh(y + bh / 2, P.P_recall * 100, bh, color=C["blue"], edgecolor="k", lw=0.4,
            label=f"P recall (n = {int(P.P_n.iloc[0]):,} analyst P picks)")
    ax.barh(y - bh / 2, P.S_recall * 100, bh, color=C["orange"], edgecolor="k", lw=0.4,
            label=f"S recall (n = {int(P.S_n.iloc[0]):,} analyst S picks)")
    for yi, (p, s) in enumerate(zip(P.P_recall, P.S_recall)):
        ax.text(p * 100 + 0.9, yi + bh / 2, f"{p*100:.1f}", va="center", fontsize=8)
        ax.text(s * 100 + 0.9, yi - bh / 2, f"{s*100:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(P.label, fontsize=9)
    ax.set_xlim(0, 118)
    ax.set_xlabel("recall against the analyst set (%)")
    ax.set_title("(a)  Recall, 12 models at a common probability floor of 0.1", loc="left")
    for lbl, yy in zip(P.label, y):
        if lbl.startswith("PhaseNet diting") or lbl.startswith("PickBlue PhaseNetLight"):
            ax.axhspan(yy - 0.5, yy + 0.5, color=C["green"], alpha=0.12, zorder=0)
    h, l = ax.get_legend_handles_labels()
    h.append(matplotlib.patches.Patch(color=C["green"], alpha=0.20))
    l.append("green band: one of the two adopted models")
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.105), ncol=3,
              framealpha=0.93, fontsize=8.8)

    # (b) bias +- MAD for the four leading models
    axb = fig.add_subplot(gs[0, 1])
    lead = pd.DataFrame(D["pickers"]).sort_values("P_recall", ascending=False).head(4)
    lead = lead.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(lead))
    for off, ph, col in ((0.16, "P", C["blue"]), (-0.16, "S", C["orange"])):
        axb.errorbar(lead[f"{ph}_bias_ms"], yy + off,
                     xerr=lead[f"{ph}_mad_ms"], fmt="o", ms=8, color=col,
                     ecolor=col, elinewidth=2.2, capsize=5, markeredgecolor="k",
                     markeredgewidth=0.5, label=f"{ph} bias, whiskers = MAD")
        for v, mad, yv in zip(lead[f"{ph}_bias_ms"], lead[f"{ph}_mad_ms"], yy + off):
            axb.annotate(f"{v:+.0f} / {mad:.0f} ms", (v, yv), textcoords="offset points",
                         xytext=(0, 11 if ph == "P" else -15), ha="center", fontsize=8.5)
    axb.axvline(0, color="k", lw=1.1, ls="--")
    axb.set_yticks(yy)
    axb.set_yticklabels(lead.label, fontsize=9.5)
    axb.set_ylim(-0.6, len(lead) - 0.4)
    axb.set_xlabel("pick-time residual, auto - analyst (ms)")
    axb.set_title("(b)  Timing of the four leading models (median bias, MAD)", loc="left")
    axb.legend(loc="upper right", framealpha=0.93)

    # (c) pools: recall vs detection volume
    axc = fig.add_subplot(gs[1, 0])
    axc.scatter(pools.rate, pools["S"] * 100, s=np.where(pools["pick"], 190, 60),
                c=C["orange"], marker="s", edgecolor="k",
                lw=np.where(pools["pick"], 1.8, 0.5), zorder=3, alpha=0.85,
                label="S recall of the pool")
    axc.scatter(pools.rate, pools["P"] * 100, s=np.where(pools["pick"], 190, 60),
                c=C["blue"], marker="o", edgecolor="k",
                lw=np.where(pools["pick"], 1.8, 0.5), zorder=4,
                label="P recall of the pool")
    short = {"Current production (PN-inst P + OBST P/S)": "current production",
             "Best 2-model pool \u2014 diting+PB-PNL": "diting + PB-PNL",
             "Best 3-model pool \u2014 OBST+diting+PB-PNL": "OBST + diting + PB-PNL"}
    for _, r in pools.iterrows():
        nm = short.get(r.pool, r.pool.replace(" alone", ""))
        if r["pick"]:
            continue
        axc.annotate(nm, (r.rate, r.P * 100), textcoords="offset points",
                     xytext=(6, -11), fontsize=7.2, color="0.30", zorder=5)
    adopted = pools[pools["pick"]].iloc[0]
    axc.annotate(f"adopted pool: diting + PB-PNL\n"
                 f"P {adopted.P*100:.1f}%   S {adopted.S*100:.1f}%\n"
                 f"{adopted.rate:,.0f} picks / station-day",
                 (adopted.rate, adopted.P * 100), xycoords="data",
                 xytext=(0.035, 0.955), textcoords="axes fraction",
                 ha="left", va="top", fontsize=9.5, zorder=6,
                 arrowprops=dict(arrowstyle="->", lw=1.4, color=C["green"],
                                 connectionstyle="arc3,rad=-0.2"),
                 bbox=dict(fc="white", ec=C["green"], lw=1.4, alpha=0.95, pad=4))
    axc.set_xscale("log")
    axc.set_xlim(150, 7000)
    axc.set_ylim(22, 104)
    axc.set_xticks([200, 500, 1000, 2000, 5000])
    axc.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axc.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axc.set_xlabel("detection volume (picks per station-day, log scale)")
    axc.set_ylabel("recall against the analyst set (%)")
    axc.set_title(f"(c)  Exhaustive 1-, 2- and 3-model pools ({len(pools)})", loc="left")
    axc.legend(loc="lower right", framealpha=0.93)

    # (d) production timing residuals, from the analyst recall table
    axd = fig.add_subplot(gs[1, 1])
    M = manual_picks()
    hit = M[(M.hit_raw.astype(bool)) & M.res_raw.notna()]
    bins = np.linspace(-0.5, 0.5, 81)
    txt = []
    for ph, col in (("P", C["blue"]), ("S", C["orange"])):
        for trusted, ls, alpha in ((True, "-", 0.55), (False, "--", 0.0)):
            v = hit[(hit.phase == ph) & (hit.trusted == trusted)].res_raw.values
            if v.size == 0:
                continue
            lab = (f"{ph}, from 2019-10-01 (n={v.size:,})" if trusted
                   else f"{ph}, earlier (n={v.size:,})")
            axd.hist(v, bins=bins, histtype="stepfilled" if trusted else "step",
                     color=col, alpha=alpha if trusted else 1.0, lw=1.6, ls=ls,
                     density=True, label=lab, zorder=3 if trusted else 4)
            med = float(np.median(v))
            mad = float(np.median(np.abs(v - med)))
            txt.append(f"{ph} {'late ' if trusted else 'early'}: "
                       f"median {med*1000:+.0f} ms, MAD {mad*1000:.0f} ms")
    axd.axvline(0, color="k", lw=1.0, ls=":")
    axd.set_xlim(-0.5, 0.5)
    axd.set_xlabel("production pick minus analyst pick (s)")
    axd.set_ylabel("density")
    axd.set_title("(d)  Timing residual of the adopted pool over the whole deployment",
                  loc="left")
    axd.legend(loc="upper left", framealpha=0.93, fontsize=8.5)
    axd.text(0.985, 0.97, "\n".join(txt), transform=axd.transAxes, ha="right", va="top",
             fontsize=9, family="monospace",
             bbox=dict(fc="white", ec=C["grey"], lw=1.0, alpha=0.94, pad=4))

    best_p = P.iloc[-1]
    CAPTIONS["G02"] = (
        f"**Figure 2 - picker benchmark.** All of (a)-(c) are the benchmark metrics "
        f"embedded in `notes/picker_benchmark.html` (`const DATA`, generated 2026-09-11 "
        f"from `benchmark_metrics.csv`), i.e. the sweep data ARE on disk and are replotted "
        f"here rather than transcribed. Evaluation set: the ten days from 2019-10-01 "
        f"carrying the most analyst picks - {int(P.P_n.iloc[0]):,} P and "
        f"{int(P.S_n.iloc[0]):,} S picks over 447 events, 20 ZX stations, 220 "
        f"station-days per model - scored by greedy nearest-time matching per station and "
        f"phase (0.5 s for P, 1.0 s for S). (a) P and S recall for the twelve models, all "
        f"run at a common probability floor of 0.1; the best single model on P is "
        f"{best_p.label} at {best_p.P_recall*100:.1f}%, and the best on S is PickBlue "
        f"PhaseNetLight at {pd.DataFrame(D['pickers']).S_recall.max()*100:.1f}%. "
        f"(b) Median signed pick-time residual with MAD whiskers for the four models with "
        f"the highest P recall; a late S bias maps directly into S-P and therefore into "
        f"depth, which is why OBSTransformer's +70 ms S was rejected. (c) Recall of every "
        f"one-, two- and three-model pool against detection volume (log axis); the adopted "
        f"pool, PhaseNet DiTing + PickBlue PhaseNetLight with both contributing P and S, "
        f"is ringed. (d) is NOT from the benchmark: it is the timing residual of the "
        f"delivered production picks against all {len(hit):,} matched analyst picks of "
        f"`catalogs/manual_pick_recall_year_newpool.csv` (`res_raw` = auto - manual on "
        f"`hit_raw` rows), split by phase and by the trusted window (from 2019-10-01) "
        f"against the earlier, less carefully picked part of the analyst set."
    )
    return save(fig, "G02_picker_benchmark")


# ------------------------------------------------------------------ G03
def g03():
    """Draft Figure 3 - velocity model."""
    vp, (x0, y0, z0), (dx, dy, dz), (nx, ny, nz) = read_nlloc_grid()
    water = np.load(WATER_NPY)
    assert water.shape == (nx, ny), (water.shape, (nx, ny))
    gx = x0 + dx * np.arange(nx)
    gy = y0 + dy * np.arange(ny)
    gz = z0 + dz * np.arange(nz)

    cx, cy = [float(v) for v in stingray_xy(CALDERA_LAT, CALDERA_LON)]
    ic, jc = int(round((cx - x0) / dx)), int(round((cy - y0) / dy))
    HX, HY, ZMAX = 32.0, 22.0, 12.0
    i0, i1 = int((-HX - x0) / dx), int((HX - x0) / dx) + 1
    j0, j1 = int((-HY - y0) / dy), int((HY - y0) / dy) + 1
    kz = int(round((2.0 - z0) / dz))
    RESOLVED = 4.0                        # km below the seafloor (draft sec.9)

    st = load_stations()
    sx, sy = stingray_xy(st.latitude.values, st.longitude.values)
    st = st.assign(gx=sx, gy=sy)
    obs = st[st.is_obs]

    # grid azimuths: grid +x is 90 - rot = 54 deg E of N, grid +y is 54 - 90 = -36 deg
    az_x, az_y = 90.0 - TRANS_ROT, -TRANS_ROT
    vmin, vmax = 1.4, 7.0

    fig = plt.figure(figsize=(12.6, 11.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.24, wspace=0.22)

    # (a) map slice at 2 km below sea level
    ax = fig.add_subplot(gs[0, :])
    GX, GY = np.meshgrid(gx[i0:i1], gy[j0:j1], indexing="ij")
    LAT, LON = grid_xy_to_latlon(GX, GY)
    pc = ax.pcolormesh(LON, LAT, vp[i0:i1, j0:j1, kz], cmap="viridis",
                       vmin=vmin, vmax=vmax, shading="auto", zorder=1)
    cb = fig.colorbar(pc, ax=ax, pad=0.012, shrink=0.92)
    cb.set_label("Vp at 2 km below sea level (km/s)")
    csw = ax.contour(LON, LAT, water[i0:i1, j0:j1] * 1000.0, levels=[1000.0],
                     colors="w", linewidths=1.9, zorder=3)
    ax.clabel(csw, fmt="%d m", fontsize=8.5, inline=True)
    ax.scatter(obs.longitude, obs.latitude, marker="^", s=70, c=C["vermillion"],
               edgecolor="k", lw=0.6, zorder=6)
    # the two section traces
    la_b, lo_b = grid_xy_to_latlon(np.array([-HX, HX]), np.array([cy, cy]))
    la_c, lo_c = grid_xy_to_latlon(np.array([cx, cx]), np.array([-HY, HY]))
    ax.plot(lo_b, la_b, color=C["black"], lw=2.0, zorder=7)
    ax.plot(lo_c, la_c, color=C["black"], lw=2.0, ls="--", zorder=7)
    for (lo_e, la_e), tag in (((lo_b, la_b), "b"), ((lo_c, la_c), "c")):
        ax.annotate(f"{tag}", (lo_e[0], la_e[0]), fontsize=11, fontweight="bold",
                    textcoords="offset points", xytext=(-12, -4), zorder=8,
                    bbox=dict(fc="white", ec="k", lw=0.8, alpha=0.9, pad=1.5))
        ax.annotate(f"{tag}'", (lo_e[1], la_e[1]), fontsize=11, fontweight="bold",
                    textcoords="offset points", xytext=(6, -4), zorder=8,
                    bbox=dict(fc="white", ec="k", lw=0.8, alpha=0.9, pad=1.5))
    ax.plot(CALDERA_LON, CALDERA_LAT, marker="*", ms=20, color=C["yellow"],
            markeredgecolor="k", markeredgewidth=0.8, zorder=9)
    cor_lat, cor_lon = grid_xy_to_latlon(np.array([-HX, HX, HX, -HX]),
                                         np.array([-HY, -HY, HY, HY]))
    ax.set_xlim(cor_lon.min(), cor_lon.max())
    ax.set_ylim(cor_lat.min(), cor_lat.max())
    ax.set_aspect(1.0 / np.cos(np.radians(CLAT)))
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_title(f"(a)  Un-sheared model, horizontal slice at {gz[kz]:.1f} km below sea "
                 f"level; white line = 1000 m isobath", loc="left")
    ax.grid(False)
    ax.legend(handles=[
        Line2D([], [], marker="^", ls="", ms=9, color=C["vermillion"],
               markeredgecolor="k", label="ZX OBS"),
        Line2D([], [], color="k", lw=2.0, label=f"section b-b' (grid +x, {az_x:.0f} deg E of N)"),
        Line2D([], [], color="k", lw=2.0, ls="--",
               label=f"section c-c' (grid +y, {az_y:.0f} deg E of N)"),
        Line2D([], [], marker="*", ls="", ms=13, color=C["yellow"], markeredgecolor="k",
               label="Orca caldera"),
    ], loc="lower left", framealpha=0.93, fontsize=9)

    # (b), (c) the two orthogonal sections through the caldera
    for pi, (label, tag) in enumerate((("b", "along grid +x"), ("c", "along grid +y"))):
        axs = fig.add_subplot(gs[1, pi])
        if label == "b":
            d = gx[i0:i1]
            V = vp[i0:i1, jc, :]
            sea = water[i0:i1, jc]
            sel = obs[np.abs(obs.gy - cy) <= 6.0]  # noqa: E501
            sd = sel.gx.values
            xlab = f"distance along b-b' (km, grid +x, {az_x:.0f} deg E of N)"
            lim = HX
        else:
            d = gy[j0:j1]
            V = vp[ic, j0:j1, :]
            sea = water[ic, j0:j1]
            sel = obs[np.abs(obs.gx - cx) <= 6.0]
            sd = sel.gy.values
            xlab = f"distance along c-c' (km, grid +y, {az_y:.0f} deg E of N)"
            lim = HY
        DD, ZZ = np.meshgrid(d, gz, indexing="ij")
        pcs = axs.pcolormesh(DD, ZZ, V, cmap="viridis", vmin=vmin, vmax=vmax,
                             shading="auto", zorder=1)
        axs.fill_between(d, 0, sea, facecolor="white", edgecolor="none", zorder=3)
        axs.fill_between(d, 0, sea, facecolor="none", hatch="////",
                         edgecolor="#8fb4cf", lw=0.0, zorder=4)
        axs.plot(d, sea, color=C["black"], lw=1.8, zorder=6, label="seafloor")
        axs.plot(d, sea + RESOLVED, color=C["vermillion"], lw=1.5, ls="--", zorder=6,
                 label=f"seafloor + {RESOLVED:g} km (resolution limit)")
        axs.fill_between(d, sea, sea + RESOLVED, facecolor="none", hatch="\\",
                         edgecolor="#e8905c", lw=0.0, alpha=0.7, zorder=5)
        sel = sel[np.abs(sd) <= lim]
        sd = sd[np.abs(sd) <= lim]
        if len(sel):
            axs.scatter(sd, sel.water_km.values, marker="^", s=95, c=C["vermillion"],
                        edgecolor="k", lw=0.8, zorder=8, clip_on=False,
                        label=f"OBS within 6 km of the section ({len(sel)})")
        axs.axvline(cx if label == "b" else cy, color=C["yellow"], lw=1.4, zorder=7)
        axs.set_xlim(-lim, lim)
        axs.set_ylim(ZMAX, 0)
        axs.set_xlabel(xlab)
        axs.set_ylabel("depth below sea level (km)")
        axs.set_title(f"({label})  Section {label}-{label}' through the caldera "
                      f"({tag})", loc="left")
        axs.grid(False)
        axs.legend(loc="lower right", framealpha=0.93, fontsize=8)
        if pi == 1:
            cbs = fig.colorbar(pcs, ax=axs, pad=0.012, shrink=0.92)
            cbs.set_label("Vp (km/s)")

    vsl = vp[i0:i1, j0:j1, kz]
    CAPTIONS["G03"] = (
        f"**Figure 3 - velocity model.** The production P model, "
        f"`nlloc/model/ORCA_v4.P.mod` ({nx} x {ny} x {nz} nodes at {dx:g} km, stored as "
        f"SLOW_LEN so Vp = dx / value), in the NLLoc frame `TRANS SIMPLE {CLAT} {CLON} "
        f"{TRANS_ROT:.0f}`; grid +x therefore runs {az_x:.0f} deg east of north "
        f"(along-strike) and grid +y {az_y:.0f} deg (across-strike). (a) Horizontal slice "
        f"at {gz[kz]:.1f} km below sea level over the central {2*HX:.0f} x {2*HY:.0f} km "
        f"of the grid, Vp {vsl.min():.2f}-{vsl.max():.2f} km/s in view; the white line is "
        f"the 1000 m isobath of `ORCA_v4.water_depth_km.npy`, the seafloor sampled on this "
        f"same grid. (b, c) Two orthogonal sections through the Orca caldera (yellow line) "
        f"along the grid axes. The black line is the seafloor from the same array; the "
        f"hatched blue band above it is the water column, which this grid fills with rock "
        f"velocity rather than water (a deliberate choice of the model build, so the slice "
        f"colours there are not a measurement). The vermillion hatch is the resolved "
        f"interval, the {RESOLVED:g} km below the seafloor within which the model is "
        f"resolved (draft sec. 9); below it Vp relaxes toward the 1D starting model. "
        f"Triangles are the ZX OBS lying within 6 km of each section, plotted at their "
        f"true instrument depths - the same values that reproduce the grid's own water "
        f"depth at those nodes to better than 0.04 km."
    )
    return save(fig, "G03_velocity_model")


# ------------------------------------------------------------------ G07
# notes/30_xcorr_dtcc.md section D3: BRA22, juldays 200-225, 2,989 real dt.ct rows
# against the same rows with the second window randomly re-paired.  Chosen window
# row of the table (P -0.3/+0.7 s, S -0.4/+1.0 s).
NULL_TEST = dict(
    station="BRA22", n_rows=2989, juldays="200-225",
    thr=[0.6, 0.7, 0.8],
    true_frac=[0.098, 0.043, 0.019],
    null_frac=[0.047, 0.011, 0.004],
    purity_quoted={0.7: 0.74, 0.8: 0.80},
    median_null_cc=0.40,
    per_phase="P 0.74 / S 0.73 at cc >= 0.7; P 0.79 / S 0.83 at cc >= 0.8",
)


def g07():
    """Draft Figure 7 - cross-correlation."""
    z = np.load(XC_DIR / "dtcc_measurements.npz")
    summ = json.loads((XC_DIR / "dtcc_summary.json").read_text())
    cc, d, isP = z["cc"].astype(float), z["d"].astype(float), z["isP"].astype(bool)
    thr = float(summ["cc_min"])

    fig = plt.figure(figsize=(14.0, 10.0))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

    # (a1) delivered coefficient distribution
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(thr, 1.0, 61)
    ax.hist(cc[isP], bins=bins, histtype="stepfilled", color=C["blue"], alpha=0.55,
            label=f"P ({int(isP.sum()):,} rows, median {np.median(cc[isP]):.3f})")
    ax.hist(cc[~isP], bins=bins, histtype="step", color=C["orange"], lw=2.0,
            label=f"S ({int((~isP).sum()):,} rows, median {np.median(cc[~isP]):.3f})")
    ax.axvline(thr, color=C["vermillion"], lw=2.0,
               label=f"adopted threshold cc >= {thr:g}")
    ax.set_yscale("log")
    ax.set_xlabel("cross-correlation coefficient")
    ax.set_ylabel("measurements per bin (log scale)")
    ax.set_title(f"(a)  Delivered coefficients, {len(cc):,} rows in "
                 f"{summ['n_pairs_cc']:,} pairs", loc="left")
    ax.legend(loc="upper right", framealpha=0.93, fontsize=9)
    ax.text(0.02, 0.04, f"the file is cut at cc >= {thr:g}, so nothing below it exists",
            transform=ax.transAxes, fontsize=8.5, color="0.3")

    # (a2) null test (quoted, not recomputed - scripts/63 has no null-test entry point)
    axn = fig.add_subplot(gs[1, 0])
    t = np.array(NULL_TEST["thr"])
    tf = np.array(NULL_TEST["true_frac"])
    nf = np.array(NULL_TEST["null_frac"])
    bw = 0.028
    axn.bar(t - bw / 2, tf * 100, bw, color=C["green"], edgecolor="k", lw=0.5,
            label="true pairs")
    axn.bar(t + bw / 2, nf * 100, bw, color=C["grey"], edgecolor="k", lw=0.5,
            label="null (randomly re-paired) windows")
    for x_, a_, b_ in zip(t, tf, nf):
        axn.text(x_ - bw / 2, a_ * 100 * 1.06, f"{a_*100:.1f}%", ha="center", fontsize=8.5)
        axn.text(x_ + bw / 2, b_ * 100 * 1.06, f"{b_*100:.1f}%", ha="center", fontsize=8.5)
    axn.set_yscale("log")
    axn.set_ylim(0.2, 40)
    axn.set_xlim(0.55, 0.86)
    axn.set_xticks(list(t))
    axn.set_xlabel("coefficient threshold")
    axn.set_ylabel("fraction of dt.ct rows kept (%, log)")
    axn.axvline(thr, color=C["vermillion"], lw=2.0, zorder=1)
    ax2 = axn.twinx()
    purity = 1.0 - nf / tf
    ax2.plot(t, purity, marker="o", ms=9, lw=2.2, color=C["vermillion"],
             markeredgecolor="k", markeredgewidth=0.5, label="purity = 1 - null/true")
    for x_, p_ in zip(t, purity):
        ax2.annotate(f"{p_:.2f}", (x_, p_), textcoords="offset points", xytext=(6, 6),
                     fontsize=9.5, color=C["vermillion"], fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("purity", color=C["vermillion"])
    ax2.tick_params(axis="y", colors=C["vermillion"])
    ax2.grid(False)
    h1, l1 = axn.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axn.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.93, fontsize=8.5)
    axn.set_title("(a, continued)  Null test that sets the threshold", loc="left")
    axn.text(0.0, -0.245,
             f"quoted from notes/30 sec.D3 - {NULL_TEST['station']}, juldays "
             f"{NULL_TEST['juldays']}, {NULL_TEST['n_rows']:,} real rows;\n"
             f"median cc of UNRELATED windows at this station = "
             f"{NULL_TEST['median_null_cc']:.2f}; per phase {NULL_TEST['per_phase']}.\n"
             f"scripts/63_xcorr_dtcc.py exposes no null-test option, so this was not "
             f"recomputed here.",
             transform=axn.transAxes, fontsize=8.2, va="top", ha="left",
             bbox=dict(fc="white", ec=C["grey"], lw=0.9, alpha=0.94, pad=3))

    # (b) dt.cc - dt.ct against coefficient
    axb = fig.add_subplot(gs[0, 1])
    edges = np.array([0.80, 0.825, 0.85, 0.875, 0.90, 0.95, 1.001])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    for ph, mask, col, off in (("P", isP, C["blue"], -0.003), ("S", ~isP, C["orange"], 0.003)):
        med, mad, ns = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            v = d[mask & (cc >= a) & (cc < b)]
            med.append(np.median(v) * 1000 if v.size else np.nan)
            mad.append(np.median(np.abs(v - np.median(v))) * 1000 if v.size else np.nan)
            ns.append(v.size)
        axb.errorbar(ctr + off, med, yerr=mad, fmt="o-", ms=7, lw=1.8, capsize=4,
                     color=col, markeredgecolor="k", markeredgewidth=0.5,
                     label=f"{ph} (n = {int(mask.sum()):,}); bars = MAD")
        for x_, n_ in zip(ctr + off, ns):
            axb.annotate(f"{int(n_):,}", (x_, 0.975 if ph == "P" else 0.925),
                         xycoords=("data", "axes fraction"), ha="center", va="top",
                         fontsize=7.2, color=col)
    axb.axhline(0, color="k", lw=1.0, ls="--")
    axb.set_xlabel("cross-correlation coefficient (bin)")
    axb.set_ylabel("dt.cc - dt.ct  (ms)")
    axb.set_ylim(-75, 88)
    axb.set_title("(b)  Correction applied to the catalogue differential time", loc="left")
    axb.legend(loc="lower left", framealpha=0.93, fontsize=9)
    axb.text(0.985, 0.04,
             f"all rows: median {np.median(d)*1000:+.1f} ms, "
             f"MAD {np.median(np.abs(d - np.median(d)))*1000:.1f} ms",
             transform=axb.transAxes, fontsize=9, ha="right",
             bbox=dict(fc="white", ec=C["grey"], lw=0.9, alpha=0.94, pad=3))

    # (c) nearest-neighbour separation against correlation link count
    axc = fig.add_subplot(gs[1, 1])
    from scipy.spatial import cKDTree
    h = load_hypodd("hypodd_year_v6_3d_xc_qc")
    e, n_ = ll_to_km(h.lat.values, h.lon.values)
    pts = np.column_stack([e, n_, h.depth_bsl_km.values])
    dist, _ = cKDTree(pts).query(pts, k=2)
    nn = dist[:, 1]
    ncc = (h.nccp + h.nccs).values
    b = np.array([0, 1, 10, 50, 100, 200, 1e9])
    lab = ["0", "1-9", "10-49", "50-99", "100-199", ">= 200"]
    med, p25, p75, cnt = [], [], [], []
    for a_, b_ in zip(b[:-1], b[1:]):
        v = nn[(ncc >= a_) & (ncc < b_)]
        cnt.append(v.size)
        med.append(np.median(v) if v.size else np.nan)
        p25.append(np.percentile(v, 25) if v.size else np.nan)
        p75.append(np.percentile(v, 75) if v.size else np.nan)
    xx = np.arange(len(lab))
    axc.errorbar(xx, med, yerr=[np.array(med) - np.array(p25), np.array(p75) - np.array(med)],
                 fmt="o-", ms=8, lw=2.0, capsize=5, color=C["green"],
                 markeredgecolor="k", markeredgewidth=0.5,
                 label="median nearest-neighbour 3D separation (bars p25-p75)")
    for x_, m_, c_ in zip(xx, med, cnt):
        if np.isfinite(m_):
            axc.annotate(f"n={c_:,}", (x_, m_), textcoords="offset points", xytext=(0, 14),
                         ha="center", fontsize=8)
    axc.axhline(np.median(nn), color=C["vermillion"], lw=1.5, ls="--",
                label=f"all {len(h):,} events: {np.median(nn):.3f} km")
    axc.set_xticks(xx)
    axc.set_xticklabels(lab, fontsize=9)
    axc.set_xlabel("cross-correlation links per event (nccp + nccs)")
    axc.set_ylabel("nearest-neighbour separation (km)")
    axc.set_title("(c)  Cluster tightness against correlation link count", loc="left")
    axc.legend(loc="upper right", framealpha=0.93, fontsize=9)

    CAPTIONS["G07"] = (
        f"**Figure 7 - cross-correlation.** (a, top) Distribution of the "
        f"{len(cc):,} delivered differential times of "
        f"`hypodd/year_v6_3d_xc/dtcc_measurements.npz` "
        f"({int(isP.sum()):,} P at median cc {np.median(cc[isP]):.3f}, "
        f"{int((~isP).sum()):,} S at {np.median(cc[~isP]):.3f}) in "
        f"{summ['n_pairs_cc']:,} event pairs, {summ['frac_rows']*100:.2f}% of the "
        f"{summ['n_rows_ct']:,} catalogue rows; the file is cut at the production "
        f"threshold cc >= {thr:g}, so no true-pair distribution exists below it. "
        f"(a, bottom) The null test that fixed that threshold, quoted from "
        f"notes/30_xcorr_dtcc.md sec. D3 and NOT recomputed - "
        f"`scripts/63_xcorr_dtcc.py` has a `--self-test` but no null-test entry point. "
        f"At station {NULL_TEST['station']} over juldays {NULL_TEST['juldays']}, "
        f"{NULL_TEST['n_rows']:,} real rows were rescored against the same rows with the "
        f"second window randomly re-paired: purity rises from "
        f"{purity[0]:.2f} at cc >= 0.6 through {purity[1]:.2f} at 0.7 to {purity[2]:.2f} "
        f"at the adopted 0.8, and the median coefficient of unrelated windows is already "
        f"{NULL_TEST['median_null_cc']:.2f}. (b) The correction the correlation applies, "
        f"dt.cc - dt.ct, binned by coefficient for P and S with MAD whiskers and the bin "
        f"count above each point; over all rows the median is "
        f"{np.median(d)*1000:+.1f} ms with a MAD of "
        f"{np.median(np.abs(d - np.median(d)))*1000:.1f} ms, a near-zero bias on a "
        f"scatter consistent with a 35-40 ms per-pick error entering a difference of two "
        f"picks. (c) Median nearest-neighbour 3D separation of the "
        f"{len(h):,} QC-pass events of "
        f"`catalogs/hypodd_year_v6_3d_xc_qc.csv`, binned by how many correlation links "
        f"each event ended with (nccp + nccs); the separation falls monotonically with "
        f"link count, from {med[0]:.3f} km for the {cnt[0]:,} events that ended with no "
        f"correlation link to {med[-2]:.3f} km at 100-199 links (n = {cnt[-2]:,}) and "
        f"{med[-1]:.3f} km above 200 (n = {cnt[-1]:,}), against {np.median(nn):.3f} km "
        f"for the catalogue as a whole - the dose-response that says the correlation data "
        f"are doing real work where they are dense."
    )
    return save(fig, "G07_crosscorrelation")


# ------------------------------------------------------------------ G08
# notes/31_final_catalogue_v6.md sec.2 -- per-event |dz| against the production run,
# by reference (it1Bg) depth bin, all located events of the 2,000-event sample.
PERT_BINS = [(0, 1, 758), (1, 2, 292), (2, 4, 375), (4, 6, 190), (6, 9, 116), (9, 15, 54)]
PERT_MED = dict(pert12=[0.00, 0.12, 0.11, 0.23, 0.24, 0.41],
                pertA=[0.03, 0.43, 0.59, 1.00, 1.00, 1.56],
                pertS=[0.04, 0.44, 0.52, 0.95, 0.96, 1.55])
PERT_P90 = dict(pert12=[0.20, 0.30, 0.40, 0.57, 0.72, 0.84],
                pertA=[0.63, 1.30, 1.72, 2.23, 3.86, 3.21],
                pertS=[0.56, 0.88, 1.55, 2.03, 2.68, 3.24])
ENV_MED = [0.10, 0.48, 0.70, 1.16, 1.44, 1.82]
ENV_P90 = [0.77, 1.36, 1.87, 2.36, 4.94, 3.85]
PERT_LABEL = dict(pert12="x1.2 S-P terms (admissible, ratio 0.92)",
                  pertA="corrections-only terms (rejected, 1.19)",
                  pertS="S-P-isolated terms (rejected, 1.12)")


def g08():
    """Draft Figure 8 - depth uncertainty budget."""
    std, strict = load_tier("standard"), load_tier("strict")
    hdd = load_hypodd("hypodd_year_v6_3d_qc")
    j = hdd.merge(std[["event_idx", "depth_km"]], on="event_idx", how="inner")
    ctr = np.array([(a + b) / 2 for a, b, _ in PERT_BINS])
    edges = np.array([0, 1, 2, 4, 6, 9, 15], float)

    fig = plt.figure(figsize=(14.0, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.20)
    ax = fig.add_subplot(gs[0])

    # formal sigma_z of the delivered tiers, on the same bins
    for tier, dat in (("standard", std), ("strict", strict)):
        g = dat.groupby(pd.cut(dat.depth_km, edges), observed=True).sigma_z_km
        med = g.median().values
        lo = g.quantile(0.10).values
        hi = g.quantile(0.90).values
        col = C["blue"] if tier == "standard" else C["green"]
        ax.plot(ctr[:len(med)], med, marker="o", ms=7, lw=2.2, color=col,
                markeredgecolor="k", markeredgewidth=0.5,
                label=f"formal sigma_z, {tier} tier ({len(dat):,} events)")
        ax.fill_between(ctr[:len(med)], lo, hi, color=col, alpha=0.16, lw=0)

    # station-term envelope (notes/31 sec.2)
    ax.plot(ctr, ENV_MED, marker="s", ms=7, lw=2.2, color=C["purple"],
            markeredgecolor="k", markeredgewidth=0.5,
            label="station-term envelope, median (notes/31 sec.2)")
    ax.plot(ctr, ENV_P90, marker="s", ms=6, lw=1.6, ls="--", color=C["purple"],
            label="station-term envelope, p90")
    ax.fill_between(ctr, ENV_MED, ENV_P90, color=C["purple"], alpha=0.12, lw=0)

    # absolute vs relative depth difference, recomputed from the catalogue join
    dz = (j.depth_bsl_km - j.depth_km).values
    gg = pd.DataFrame({"x": j.depth_km.values, "adz": np.abs(dz), "dz": dz}).groupby(
        pd.cut(j.depth_km.values, edges), observed=True)
    madz = gg.adz.median().values
    sdz = gg.dz.median().values
    nn = gg.size().values
    ax.plot(ctr[:len(madz)], madz, marker="D", ms=7, lw=2.4, color=C["vermillion"],
            markeredgecolor="k", markeredgewidth=0.5,
            label=f"|hypoDD - NLLoc| depth, median ({len(j):,} joined events)")
    for x_, y_, n_ in zip(ctr[:len(madz)], madz, nn):
        ax.annotate(f"n={int(n_):,}", (x_, y_), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=7.4, color=C["vermillion"])

    for e in edges[1:-1]:
        ax.axvline(e, color=C["grey"], lw=0.6, ls=":")
    ax.axvline(4.0, color="k", lw=1.6, ls="--", zorder=1)
    ax.text(4.0 / 15.0 + 0.012, 0.035, "model resolved to ~4 km\nbelow the seafloor",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(fc="white", ec="k", lw=0.9, alpha=0.9, pad=3))
    ax.set_yscale("log")
    ax.set_xlim(0, 15)
    ax.set_xlabel("depth below sea level (km)")
    ax.set_ylabel("depth uncertainty contribution (km, log scale)")
    ax.set_title("(a)  The three contributions on one axis", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=2,
              framealpha=0.93, fontsize=8.8)

    # (b) the individual perturbations
    axb = fig.add_subplot(gs[1])
    x = np.arange(len(PERT_BINS))
    bw = 0.26
    for k, (key, col) in enumerate((("pert12", C["green"]), ("pertA", C["orange"]),
                                    ("pertS", C["sky"]))):
        axb.bar(x + (k - 1) * bw, PERT_MED[key], bw, color=col, edgecolor="k", lw=0.4,
                label=PERT_LABEL[key])
        axb.errorbar(x + (k - 1) * bw, PERT_MED[key],
                     yerr=[np.zeros(len(x)), np.array(PERT_P90[key]) - np.array(PERT_MED[key])],
                     fmt="none", ecolor="k", elinewidth=0.9, capsize=3)
        for xi, v, hi in zip(x + (k - 1) * bw, PERT_MED[key], PERT_P90[key]):
            axb.text(xi, hi + 0.07, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=7.2, rotation=90, color=col)
    axb.set_xticks(x)
    axb.set_xticklabels([f"{a:g}-{b:g}\nn={n:,}" for a, b, n in PERT_BINS], fontsize=8.5)
    axb.set_xlabel("reference (it1Bg) depth bin, km below sea level")
    axb.set_ylabel("median |dz| against production (km); whiskers to p90")
    axb.set_title("(b)  Depth response to each perturbed station-term set", loc="left")
    axb.legend(loc="upper left", framealpha=0.93, fontsize=8.5)
    axb.set_ylim(0, 4.5)
    axb.text(0.025, 0.74,
             "2,000-event sample; control files differ from\nthe reference only in the "
             "LOCDELAY block\n(notes/31 sec.2, scripts/68_delay_depth_uncertainty.py)",
             transform=axb.transAxes, ha="left", va="top", fontsize=8.5,
             bbox=dict(fc="white", ec=C["grey"], lw=0.9, alpha=0.94, pad=3))

    CAPTIONS["G08"] = (
        f"**Figure 8 - depth uncertainty budget.** (a) The three contributions to depth "
        f"uncertainty, all as functions of depth below sea level, on one logarithmic axis. "
        f"Blue and vermillion lines with shaded p10-p90 bands: the NLLoc formal marginal "
        f"sigma_z of the delivered standard ({len(std):,} events) and strict "
        f"({len(strict):,}) tiers, binned on the same depth bins used by the perturbation "
        f"study. Purple: the station-term envelope - the largest |dz| per event over the "
        f"three perturbed term sets - quoted from notes/31_final_catalogue_v6.md sec. 2 "
        f"(median {ENV_MED[0]:.2f} km in 0-1 km rising to {ENV_MED[-1]:.2f} km at "
        f"9-15 km, p90 up to {max(ENV_P90):.2f} km). Vermillion diamonds: the "
        f"absolute-versus-relative disagreement, recomputed here from the join of "
        f"`catalogs/nlloc_year_v6_standard.csv` with "
        f"`catalogs/hypodd_year_v6_3d_qc.csv` on event_idx (hypoDD id - 1), "
        f"{len(j):,} events. The three curves cross near 4 km: above it the formal sigma_z "
        f"dominates, below it the station-term envelope and the estimator disagreement "
        f"both exceed it, which is why depths below ~4 km beneath the seafloor are quoted "
        f"as upper bounds. (b) The same perturbation study broken out by term set, median "
        f"|dz| with whiskers to p90; the x1.2 set is the one the S-P spread gate still "
        f"accepts, the two corrections-only sets are rejected by it and serve as an outer "
        f"bound."
    )
    return save(fig, "G08_uncertainty_budget")


# ------------------------------------------------------------------ G09
# notes/31_final_catalogue_v6.md sec.3 / draft sec.10, counted by
# scripts/68_catalogue_accounting.py from the obs blocks, hyp files and catalogue.
ACCOUNT = [
    ("associated events\n(year, new pool)", 98631, "start"),
    ("airgun window\n2019-01-21 - 02-05", -18848, "loss"),
    ("into NLLoc\n(year_v6.obs)", 79783, "total"),
    ("obs events NLLoc\ncould not locate", -50, "loss"),
    (".hyp with no parsable\nGEOGRAPHIC line", -10, "loss"),
    ("rows in\nnlloc_year_v6.csv", 79723, "total"),
    ("nlloc_status\nREJECTED", -220, "loss"),
    ("LOCATED\nsolutions", 79503, "total"),
]
TIERS = [("loose", 15184), ("standard", 12194), ("strict", 3864)]
PICKS = dict(before=818767, dropped=20140, after=798627,
             p=409090, s_before=409677, s_after=389537,
             events_touched=13979, frac_events=0.175)


def g09():
    """Draft Figure 9 - processing accounting."""
    fig = plt.figure(figsize=(14.2, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 0.66],
                          hspace=0.40, wspace=0.30)
    ax = fig.add_subplot(gs[:, 0])

    n = len(ACCOUNT)
    ypos = np.arange(n)[::-1].astype(float)          # first stage at the top
    running = 0
    for i, (lab, val, kind) in enumerate(ACCOUNT):
        y = ypos[i]
        if kind in ("start", "total"):
            ax.barh(y, val, 0.58, left=0,
                    color=C["blue"] if kind == "start" else C["sky"],
                    edgecolor="k", lw=0.6, zorder=3)
            ax.annotate(f"{val:,}", (val, y), textcoords="offset points", xytext=(7, 0),
                        va="center", fontsize=10.5, fontweight="bold", zorder=5)
            running = val
        else:
            bot = running + val                      # val is negative
            ax.barh(y, -val, 0.58, left=bot, color=C["vermillion"], edgecolor="k",
                    lw=0.6, zorder=3)
            ax.plot([bot, bot], [y - 0.29, y + 0.29], color="k", lw=1.0, zorder=4)
            ax.annotate(f"{val:,}  ({100*abs(val)/ACCOUNT[0][1]:.2f}% of associated)",
                        (running, y), textcoords="offset points", xytext=(7, 0),
                        va="center", fontsize=9.5, color=C["vermillion"],
                        fontweight="bold", zorder=5)
            running = bot
        if i:
            ax.plot([running, running], [y - 0.5, y + 0.5], color="0.55", lw=0.8,
                    ls=":", zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab.replace("\n", " ") for lab, _, _ in ACCOUNT], fontsize=9)
    ax.set_ylim(-1.75, n - 0.3)
    ax.set_xlim(0, 152000)
    ax.set_xticks([0, 20000, 40000, 60000, 80000, 100000])
    ax.set_xlabel("events")
    ax.set_title("(a)  From association to located solutions", loc="left")
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", ms=10, color=C["blue"], markeredgecolor="k",
               label="associated"),
        Line2D([], [], marker="s", ls="", ms=10, color=C["sky"], markeredgecolor="k",
               label="surviving total"),
        Line2D([], [], marker="s", ls="", ms=10, color=C["vermillion"],
               markeredgecolor="k", label="removed at this stage"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.085), ncol=3,
        framealpha=0.93, fontsize=9)
    ax.text(0.985, 0.012,
            f"{ACCOUNT[-1][1]:,} LOCATED = "
            f"{100*ACCOUNT[-1][1]/ACCOUNT[2][1]:.1f}% of the {ACCOUNT[2][1]:,}\n"
            f"events offered to NLLoc; the three losses after\n"
            f"association total {abs(ACCOUNT[3][1] + ACCOUNT[4][1] + ACCOUNT[6][1]):,} "
            f"events and are too small to see",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(fc="white", ec=C["green"], lw=1.2, alpha=0.94, pad=4))

    # (b) tiers
    axb = fig.add_subplot(gs[0, 1])
    loc = ACCOUNT[-1][1]
    names = [t for t, _ in TIERS]
    vals = [v for _, v in TIERS]
    axb.bar(names, vals, 0.6, color=[TIER_COLOR[t] for t in names], edgecolor="k", lw=0.6)
    for i, (t, v) in enumerate(TIERS):
        axb.annotate(f"{v:,}\n{100*v/loc:.1f}% of LOCATED", (i, v),
                     textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9.5)
    axb.set_ylim(0, max(vals) * 1.40)
    axb.set_ylabel("events")
    axb.set_title("(b)  Delivered quality tiers", loc="left")

    # (c) picks
    axc = fig.add_subplot(gs[1, 1])
    axc.barh(1, PICKS["p"], 0.5, color=C["blue"], edgecolor="k", lw=0.5, label="P picks")
    axc.barh(0, PICKS["s_after"], 0.5, color=C["orange"], edgecolor="k", lw=0.5,
             label="S kept")
    axc.barh(0, PICKS["dropped"], 0.5, left=PICKS["s_after"], color=C["vermillion"],
             edgecolor="k", lw=0.5, label="S dropped by the guard")
    axc.annotate("dropped", (PICKS["s_before"], 0.36), textcoords="offset points",
                 xytext=(0, 2), ha="center", va="bottom", fontsize=7.5,
                 color=C["vermillion"])
    axc.annotate(f"{PICKS['p']:,}", (PICKS["p"], 1), textcoords="offset points",
                 xytext=(6, 0), va="center", fontsize=9)
    axc.annotate(f"{PICKS['s_after']:,}  +{PICKS['dropped']:,} dropped",
                 (PICKS["s_before"], 0), textcoords="offset points", xytext=(6, 0),
                 va="center", fontsize=9)
    axc.set_yticks([0, 1])
    axc.set_yticklabels(["S", "P"])
    axc.set_xlim(0, 640000)
    axc.set_xlabel("picks offered to NLLoc")
    axc.set_title(f"(c)  Picks: {PICKS['dropped']:,} S "
                  f"({100*PICKS['dropped']/PICKS['s_before']:.1f}% of all S) removed by "
                  f"the S-before-P guard\n     in {PICKS['events_touched']:,} events "
                  f"({100*PICKS['frac_events']:.1f}%), deleting no events",
                  loc="left", fontsize=9.5)

    CAPTIONS["G09"] = (
        f"**Figure 9 - processing accounting.** Waterfall of the full chain, every number "
        f"counted by `scripts/68_catalogue_accounting.py` from the obs blocks, the per-event "
        f"`.hyp` files and the catalogue itself, and reproduced in "
        f"notes/31_final_catalogue_v6.md sec. 3. (a) {ACCOUNT[0][1]:,} associated events "
        f"lose {abs(ACCOUNT[1][1]):,} to the airgun exclusion window "
        f"(2019-01-21 to 2019-02-05), leaving {ACCOUNT[2][1]:,} offered to NLLoc; "
        f"{abs(ACCOUNT[3][1])} obs events produce no solution and "
        f"{abs(ACCOUNT[4][1])} `.hyp` files carry no parsable GEOGRAPHIC line, giving the "
        f"{ACCOUNT[5][1]:,} rows of `catalogs/nlloc_year_v6.csv`, of which "
        f"{abs(ACCOUNT[6][1])} are REJECTED - so {ACCOUNT[7][1]:,} LOCATED solutions, "
        f"{100*ACCOUNT[7][1]/ACCOUNT[2][1]:.1f}% of the events offered (the last three "
        f"cuts remove "
        f"{abs(ACCOUNT[3][1] + ACCOUNT[4][1] + ACCOUNT[6][1]):,} events in total and are "
        f"too small to render at this scale). (b) The three delivered tiers cut from those "
        f"solutions: loose {TIERS[0][1]:,}, standard {TIERS[1][1]:,} and strict "
        f"{TIERS[2][1]:,} events ({100*TIERS[0][1]/loc:.1f}%, "
        f"{100*TIERS[1][1]/loc:.1f}% and {100*TIERS[2][1]/loc:.1f}% of LOCATED). The "
        f"binding constraints are azimuthal gap and grid-face pinning, not pick count or "
        f"rms. (c) The pick side of the same accounting: of {PICKS['s_before']:,} S picks, "
        f"{PICKS['dropped']:,} ({100*PICKS['dropped']/PICKS['s_before']:.1f}% of all S, "
        f"{100*PICKS['dropped']/PICKS['before']:.1f}% of all picks) were removed by the "
        f"S-before-P guard in {PICKS['events_touched']:,} events "
        f"({100*PICKS['frac_events']:.1f}%), deleting no events; the "
        f"{PICKS['p']:,} P picks are untouched."
    )
    return save(fig, "G09_accounting")


# ------------------------------------------------------------------ G14
def g14():
    """Draft Figure 14 - relative geometry of the cross-correlation-refined catalogue."""
    h = load_hypodd("hypodd_year_v6_3d_xc_qc")
    h["ot"] = hypodd_origin(h)
    t_days = (h.ot - h.ot.min()) / pd.Timedelta(days=1)
    e, n_ = ll_to_km(h.lat.values, h.lon.values)
    z = h.depth_bsl_km.values

    st = load_stations()
    se, sn = ll_to_km(st.latitude.values, st.longitude.values)
    st = st.assign(e=se, n=sn)
    obs = st[st.is_obs]

    HALF = 8.0
    keep = (np.abs(e) <= HALF) & (np.abs(n_) <= HALF)
    ZMAX = 10.0
    cmap, vmin, vmax = "plasma", float(t_days.min()), float(t_days.max())

    fig = plt.figure(figsize=(10.8, 11.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0], hspace=0.22, wspace=0.24)

    # (a) map
    ax = fig.add_subplot(gs[0, :])
    lat0, lat1 = CLAT - HALF / KM_PER_DEG_LAT, CLAT + HALF / KM_PER_DEG_LAT
    lon0, lon1 = CLON - HALF / kmperdeglon(CLAT), CLON + HALF / kmperdeglon(CLAT)
    la, lo, W, pf = bathymetry(lat0, lat1, lon0, lon1, n=420)
    cl = np.arange(600, 1801, 100)
    cs = ax.contour(lo, la, W * 1000.0, levels=cl, colors=C["grey"], linewidths=0.5, zorder=1)
    ax.clabel(cs, cl[2::4], fmt="%d", fontsize=7, inline=True)
    sc = ax.scatter(h.lon[keep], h.lat[keep], s=3.6, c=t_days[keep], cmap=cmap,
                    vmin=vmin, vmax=vmax, lw=0, zorder=3)
    ax.scatter(h.lon[~keep], h.lat[~keep], s=2.0, c=C["grey"], alpha=0.4, lw=0, zorder=2)
    mo = (obs.e.abs() <= HALF) & (obs.n.abs() <= HALF)
    ax.scatter(obs.longitude[mo], obs.latitude[mo], marker="^", s=110, c="white",
               edgecolor="k", lw=1.1, zorder=6)
    for _, r in obs[mo].iterrows():
        ax.annotate(r.station, (r.longitude, r.latitude), textcoords="offset points",
                    xytext=(6, 5), fontsize=7, zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.6))
    ax.axhline(CLAT, color="k", lw=1.2, ls="--", zorder=4)
    ax.axvline(CLON, color="k", lw=1.2, ls=":", zorder=4)
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / np.cos(np.radians(CLAT)))
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_title(f"(a)  Cross-correlation-refined catalogue, {int(keep.sum()):,} of "
                 f"{len(h):,} events inside +/-{HALF:g} km of "
                 f"{CLAT}/{CLON}", loc="left")
    cb = fig.colorbar(sc, ax=ax, pad=0.012, shrink=0.88)
    cb.set_label(f"days since {h.ot.min():%Y-%m-%d}")
    scalebar(ax, 2, CLAT)
    ax.legend(handles=[
        Line2D([], [], marker="^", ls="", ms=10, color="white", markeredgecolor="k",
               label="ZX OBS"),
        Line2D([], [], color="k", lw=1.2, ls="--", label="section b-b' (W-E)"),
        Line2D([], [], color="k", lw=1.2, ls=":", label="section c-c' (S-N)"),
        Line2D([], [], color=C["grey"], lw=0.8, label="bathymetry, 100 m"),
    ], loc="upper left", framealpha=0.93, fontsize=8.5)

    # (b) W-E and (c) S-N sections
    for pi, tag in enumerate(("b", "c")):
        axs = fig.add_subplot(gs[1, pi])
        if tag == "b":
            along, across, name = e, n_, "W-E"
            prof_lat = np.full(400, CLAT)
            prof_lon = CLON + np.linspace(-HALF, HALF, 400) / kmperdeglon(CLAT)
            s_along, s_across = obs.e.values, obs.n.values
        else:
            along, across, name = n_, e, "S-N"
            prof_lat = CLAT + np.linspace(-HALF, HALF, 400) / KM_PER_DEG_LAT
            prof_lon = np.full(400, CLON)
            s_along, s_across = obs.n.values, obs.e.values
        m = np.abs(across) <= HALF
        axs.scatter(along[m], z[m], s=3.6, c=t_days[m], cmap=cmap, vmin=vmin, vmax=vmax,
                    lw=0, zorder=3)
        prof = pf(prof_lat, prof_lon)
        dline = np.linspace(-HALF, HALF, 400)
        axs.fill_between(dline, 0, prof, facecolor="#dceaf6", zorder=1)
        axs.plot(dline, prof, color=C["black"], lw=1.8, zorder=5, label="seafloor")
        ms = np.abs(s_across) <= HALF
        axs.scatter(s_along[ms], obs.water_km.values[ms], marker="^", s=95, c="white",
                    edgecolor="k", lw=0.9, zorder=6, clip_on=False,
                    label=f"ZX OBS ({int(ms.sum())})")
        axs.axvline(0, color=C["grey"], lw=1.0, ls="--", zorder=2)
        axs.set_xlim(-HALF, HALF)
        axs.set_ylim(ZMAX, 0)
        axs.set_xlabel(f"distance along {name} through {CLAT}/{CLON} (km)")
        axs.set_ylabel("depth below sea level (km)")
        axs.set_title(f"({tag})  {name} section, |offset| <= {HALF:g} km "
                      f"({int(m.sum()):,} events)", loc="left")
        axs.legend(loc="lower right", framealpha=0.93, fontsize=8.5)

    from scipy.spatial import cKDTree
    pts = np.column_stack([e, n_, z])
    dd, _ = cKDTree(pts).query(pts, k=2)
    nnm = float(np.median(dd[:, 1]))
    CAPTIONS["G14"] = (
        f"**Figure 14 - relative geometry.** The cross-correlation-refined relative "
        f"catalogue `catalogs/hypodd_year_v6_3d_xc_qc.csv` ({len(h):,} QC-pass events) at "
        f"full resolution, every event coloured by origin time (built from the yr/mo/dy/"
        f"hr/mi/sc columns) over {(h.ot.max() - h.ot.min()).days} days from "
        f"{h.ot.min():%Y-%m-%d} to {h.ot.max():%Y-%m-%d}. (a) Map of the "
        f"{int(keep.sum()):,} events within +/-{HALF:g} km of the NLLoc projection origin "
        f"{CLAT}/{CLON} (grey dots: the {int((~keep).sum()):,} outside the frame), over "
        f"the 30 m Orca bathymetry contoured every 100 m, with the ZX OBS and the traces "
        f"of the two sections. (b) W-E and (c) S-N sections through the same origin, each "
        f"taking events within +/-{HALF:g} km of the section line, with the seafloor "
        f"profile sampled from the same bathymetry and the OBS at their true depths. The "
        f"median nearest-neighbour 3D separation of this catalogue is {nnm:.3f} km, the "
        f"scale at which the lineations and sub-clusters in these panels are resolved. "
        f"A few events plot above the seafloor line in (b) and (c): quality control "
        f"removes events above their OWN local seafloor, while the line drawn is the "
        f"profile along the section, so an event projected in from off-line can sit above "
        f"it. "
        f"Depths are below sea level and inherit the absolute frame, so the panels show "
        f"relative geometry, not absolute depth accuracy."
    )
    return save(fig, "G14_relative_geometry")


# ------------------------------------------------------------------ G15
def g15():
    """Draft Figure 15 - validation."""
    M = manual_picks()
    sp = manual_sp_table(M)
    dl = load_delays()
    w = dl.pivot_table(index="station", columns="phase", values="delay")
    spd = (w["S"] - w["P"]).rename("sp_delay")

    zx = sp[sp.network == "ZX"].copy()
    cnt = zx.groupby("station").size()
    keep_st = cnt[cnt >= 20].index
    zx = zx[zx.station.isin(keep_st)]
    order = spd.reindex(sorted(keep_st)).sort_values()
    order = order[order.notna()]
    stations = list(order.index)

    fig = plt.figure(figsize=(14.6, 11.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.0, 0.9], hspace=0.42)

    # (a) per-station analyst S-P with the station S-P term marked
    ax = fig.add_subplot(gs[0])
    data = [zx.sp[zx.station == s].values for s in stations]
    bp = ax.boxplot(data, positions=np.arange(len(stations)), widths=0.62,
                    showfliers=False, patch_artist=True, whis=(5, 95))
    for b in bp["boxes"]:
        b.set(facecolor=C["sky"], alpha=0.55, edgecolor="k", linewidth=0.7)
    for k in ("whiskers", "caps"):
        for b in bp[k]:
            b.set(color="k", linewidth=0.8)
    for b in bp["medians"]:
        b.set(color=C["blue"], linewidth=1.8)
    ax.plot(np.arange(len(stations)), order.values, marker="D", ms=8, ls="none",
            color=C["vermillion"], markeredgecolor="k", markeredgewidth=0.6, zorder=6,
            label="station S-P term (S delay - P delay), v6_it1B")
    below = [(zx.sp[zx.station == s] < order[s]).mean() for s in stations]
    nbel = [int((zx.sp[zx.station == s] < order[s]).sum()) for s in stations]
    for i, (s, f, nb) in enumerate(zip(stations, below, nbel)):
        ax.annotate(f"{nb} ({f*100:.1f}%)", (i, order[s]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=7, color=C["vermillion"])
    ax.set_xticks(np.arange(len(stations)))
    ax.set_xticklabels([f"{s}\nn={cnt[s]:,}" for s in stations], fontsize=7.6)
    ax.set_ylim(0, 3.6)
    ax.set_ylabel("analyst-measured S-P (s)")
    ax.set_title(f"(a)  Analyst S-P against the station S-P term: {len(zx):,} ZX "
                 f"event/station pairs, {len(stations)} stations with n >= 20 "
                 f"(box p25-p75, whiskers p5-p95; axis clipped at 3.6 s)", loc="left")
    ax.legend(loc="upper left", framealpha=0.93)

    # (b) automatic minus analyst S-P
    axb = fig.add_subplot(gs[1])
    hit = zx[zx.both_hit & zx.d_auto.notna()]
    bins = np.linspace(-1.0, 1.0, 101)
    axb.hist(hit.d_auto, bins=bins, color=C["green"], alpha=0.6, edgecolor="k", lw=0.3,
             label=f"all matched pairs (n = {len(hit):,})")
    tr = hit[hit.sta_key.isin(M.sta_key[M.trusted].unique())]
    axb.axvline(0, color="k", lw=1.2, ls="--")
    med = float(np.median(hit.d_auto))
    mad = float(np.median(np.abs(hit.d_auto - med)))
    axb.axvline(med, color=C["vermillion"], lw=1.8,
                label=f"median {med*1000:+.0f} ms (MAD {mad*1000:.0f} ms)")
    axb.set_xlim(-1.0, 1.0)
    axb.set_xlabel("automatic minus analyst S-P (s)")
    axb.set_ylabel("event/station pairs")
    axb.set_title("(b)  Does the production pool reproduce the analyst's S-P?", loc="left")
    axb.legend(loc="upper left", framealpha=0.93)
    axins = axb.inset_axes([0.615, 0.20, 0.365, 0.68])
    per = hit.groupby("station").d_auto.agg(n="size", med="median").reindex(stations).dropna()
    axins.barh(np.arange(len(per)), per["med"] * 1000, 0.7, color=C["green"],
               edgecolor="k", lw=0.4)
    axins.axvline(0, color="k", lw=1.0)
    axins.set_yticks(np.arange(len(per)))
    axins.set_yticklabels(per.index, fontsize=6.4)
    axins.set_xlabel("median per station (ms)", fontsize=8)
    axins.tick_params(axis="x", labelsize=7)
    axins.set_title("by station", fontsize=8.5)

    # (c) raw and catalogue recall per station in the trusted window
    axc = fig.add_subplot(gs[2])
    T = M[M.trusted & (M.network == "ZX")]
    r = T.groupby(["station", "phase"]).agg(n=("hit_raw", "size"),
                                            raw=("hit_raw", "mean"),
                                            cat=("hit_cat", "mean")).reset_index()
    stc = [s for s in stations if s in set(r.station)]
    xx = np.arange(len(stc))
    bw = 0.2
    for k, (ph, key, col, hatch) in enumerate((("P", "raw", C["blue"], ""),
                                               ("S", "raw", C["orange"], ""),
                                               ("P", "cat", C["blue"], "///"),
                                               ("S", "cat", C["orange"], "///"))):
        v = [float(r[(r.station == s) & (r.phase == ph)][key].iloc[0]) * 100
             if len(r[(r.station == s) & (r.phase == ph)]) else np.nan for s in stc]
        axc.bar(xx + (k - 1.5) * bw, v, bw, color=col, edgecolor="k", lw=0.4,
                hatch=hatch, alpha=1.0 if key == "raw" else 0.55,
                label=f"{ph} {'raw pick pool' if key == 'raw' else 'catalogue (associated)'}")
    axc.set_xticks(xx)
    axc.set_xticklabels(stc, rotation=90, fontsize=7.6)
    axc.set_ylim(0, 128)
    axc.set_ylabel("recall of analyst picks (%)")
    axc.set_title(f"(c)  Recall per station in the trusted window "
                  f"(from {TRUSTED_START:%Y-%m-%d}; {len(T):,} analyst picks): "
                  f"raw pool {T.hit_raw.mean()*100:.1f}%, catalogue "
                  f"{T.hit_cat.mean()*100:.1f}%", loc="left")
    axc.legend(loc="upper center", ncol=4, framealpha=0.93, fontsize=8.5)

    n_below = int(sum((zx.sp[zx.station == s] < order[s]).sum() for s in stations))
    CAPTIONS["G15"] = (
        f"**Figure 15 - validation.** (a) Per-station distribution of the analyst's own "
        f"S-P measurements ({len(zx):,} ZX event/station pairs with both a P and an S hand "
        f"pick, {len(stations)} stations with n >= 20; box p25-p75, median in blue, "
        f"whiskers p5-p95) against that station's S-P term from "
        f"`nlloc/delays/v6_it1B.delays` (vermillion diamond, S delay minus P delay), "
        f"stations ordered by the term; the axis is clipped at 3.6 s, above which a "
        f"handful of mis-paired analyst measurements run to {zx.sp.max():.0f} s. The terms "
        f"played no part in the analyst picks, so this is an out-of-sample test: only "
        f"{n_below} of {len(zx):,} pairs "
        f"({100*n_below/len(zx):.1f}%) measure an S-P smaller than their own station's "
        f"term (the count and percentage above each diamond, matching the 83 of 17,347 of "
        f"notes/31 sec. 1), i.e. the terms never exceed what the "
        f"analyst actually measured. (b) Automatic minus analyst S-P on the "
        f"{len(hit):,} pairs where the production pool matched both phases "
        f"(`res_raw` of `catalogs/manual_pick_recall_year_newpool.csv`, S residual minus "
        f"P residual), median {med*1000:+.0f} ms with a MAD of {mad*1000:.0f} ms; the "
        f"inset gives the per-station median. (c) Recall of the analyst picks per station "
        f"in the trusted window (from {TRUSTED_START:%Y-%m-%d}, {len(T):,} picks), for the "
        f"raw pick pool (solid, {T.hit_raw.mean()*100:.1f}% overall) and for the picks "
        f"that survive into the associated catalogue (hatched, "
        f"{T.hit_cat.mean()*100:.1f}%) - the gap is what association and the S-before-P "
        f"guard remove, not what the picker missed."
    )
    return save(fig, "G15_validation")


FIGS = {"G01": g01, "G02": g02, "G03": g03, "G07": g07,
        "G08": g08, "G09": g09, "G14": g14, "G15": g15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. --only G01 G07")
    a = ap.parse_args()
    assert_nanosecond_sanity()
    OUT.mkdir(parents=True, exist_ok=True)
    names = a.only or list(FIGS)
    made = []
    for nm in names:
        print(f"[{nm}]")
        made.append(FIGS[nm]())
    print("\n--- output ---")
    bad = []
    for p in made:
        kb = p.stat().st_size / 1024
        print(f"{p.relative_to(REPO)}  {kb:,.1f} kB")
        if kb <= 30:
            bad.append(p.name)
    if CAPTIONS:
        cap = OUT / "captions_2.md"
        old = {}
        if cap.exists() and a.only:
            for blk in cap.read_text().split("\n\n"):
                mm = re.match(r"\*\*Figure (\d+)", blk.strip())
                if mm:
                    old[f"G{int(mm.group(1)):02d}"] = blk.strip()
        old.update(CAPTIONS)
        body = ["# Figure captions - Methods & Results draft, Figures 1, 2, 3, 7, 8, 9, 14, 15",
                "",
                "Generated by `scripts/71_report_figures_2.py` (companion to "
                "`scripts/69_final_figures.py`, which produced Figures 4, 5, 6, 10-13 as "
                "F1-F8). Numbers are recomputed from the catalogues, grids and measurement "
                "files at build time except where the caption names the note they are "
                "quoted from.",
                ""]
        for k in sorted(old):
            body += [old[k], ""]
        cap.write_text("\n".join(body))
        print(f"{cap.relative_to(REPO)}  {cap.stat().st_size / 1024:,.1f} kB")
    if bad:
        raise SystemExit(f"FAILED size check (<= 30 kB): {bad}")


if __name__ == "__main__":
    main()
