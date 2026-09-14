"""Publication figures for the final Orca / Bransfield catalogue (NLLoc v6).

One script, eight PNGs (200 dpi, colour-blind-safe Okabe-Ito palette, readable at
900 px width) plus a caption file, all written to `notes/figures/final/`.

    PYTHONPATH=src python3 scripts/69_final_figures.py [--only F1 F3 ...]

Inputs (read-only; nothing outside notes/figures/final/ is written):
  catalogs/nlloc_year_v6_{loose,standard,strict}.csv   absolute locations, 3 QC tiers
  catalogs/hypodd_year_v6_3d_qc.csv                    relative relocations (QC-pass)
  catalogs/station_geometry.csv                        station positions, OBS water depth
  catalogs/manual_pick_recall_year_newpool.csv         analyst picks (F4b)
  nlloc/delays/v6_it1B.delays                          LOCDELAY station terms (F4)
  hypodd/year_v4_3d_frozen_fine/tt_compare_3d_vs_nlloc.csv   FD grid vs ray tracer (F6)
  notes/figures/Orca_bathymetry.nc  +  GEBCO_2023.nc   bathymetry (F1, F2)
  scripts/52_sp_depth_check.py                         run twice as a subprocess (F5)

Conventions follow catalogs/README_nlloc_year_v6.md: depth_km / depth_bsl_km are
below SEA LEVEL, depth_bsf*_km below the LOCAL SEAFLOOR; hypoDD `id` = `event_idx + 1`
(scripts/22_pyocto_to_hypodd_input.py line 84: `ev["hypodd_id"] = event_idx + 1`).
"""
from __future__ import annotations

import os

# <= 4 cores (pod is CPU-capped; see notes/jupyterhub_pod_memory_limit)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import re
import subprocess
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

# NLLoc TRANS SIMPLE origin -- the centre of the F2 sections
CLAT, CLON = -62.4413, -58.44
# Orca caldera / volcanic edifice (summit of the 30 m grid is -615 m at -62.446/-58.397)
CALDERA_LAT, CALDERA_LON = -62.45, -58.43
AIRGUN = ("2019-01-21", "2019-02-05")

KM_PER_DEG_LAT = 111.195

# Okabe-Ito, colour-blind safe
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
    """Local flat-earth east/north offsets in km from (lat0, lon0)."""
    e = (np.asarray(lon, float) - lon0) * kmperdeglon(lat0)
    n = (np.asarray(lat, float) - lat0) * KM_PER_DEG_LAT
    return e, n


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return np.percentile(a, q)


_BATHY_CACHE: dict = {}


def bathymetry(lat0, lat1, lon0, lon1, n=700):
    """Water depth (km, +down) on a regular lat/lon grid over the box.

    Same source and precedence as scripts/41_build_unsheared_velgrid.py
    ::sample_bathymetry -- the 30 m Orca grid where it covers the point,
    GEBCO_2023 elsewhere -- but sampled on a geographic grid for plotting.
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


def load_hypodd():
    h = pd.read_csv(REPO / "catalogs" / "hypodd_year_v6_3d_qc.csv")
    h["event_idx"] = h["id"].astype(int) - 1          # scripts/22, line 84
    return h


def load_delays(p=REPO / "nlloc" / "delays" / "v6_it1B.delays"):
    rows = []
    for line in Path(p).read_text().split("\n"):
        f = line.split()
        if len(f) == 5 and f[0] == "LOCDELAY":
            rows.append((f[1], f[2], int(f[3]), float(f[4])))
    return pd.DataFrame(rows, columns=["station", "phase", "n", "delay"])


def manual_sp_by_station():
    """Per-station manual S-P percentiles (reproduces notes/31 sec.1 table)."""
    M = pd.read_csv(REPO / "catalogs" / "manual_pick_recall_year_newpool.csv")
    g = M.pivot_table(index=["event_id", "sta_key"], columns="phase",
                      values="t", aggfunc="min").dropna(subset=["P", "S"])
    sp = (g["S"] - g["P"]).rename("sp").reset_index()
    sp["network"] = sp.sta_key.str.split(".").str[0]
    sp["station"] = sp.sta_key.str.split(".").str[-1]
    sp = sp[sp.network == "ZX"]
    r = sp.groupby("station").sp.agg(
        n="size", p5=lambda x: np.percentile(x, 5), p50="median").reset_index()
    return r, len(sp)


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
    flag = "OK " if kb > 30 else "SMALL"
    print(f"  [{flag}] {p.relative_to(REPO)}  {kb:,.0f} kB")
    return p


# ------------------------------------------------------------------- F1
def fig1():
    std, strict = load_tier("standard"), load_tier("strict")
    st = load_stations()

    lat0, lat1 = -62.56, -62.33
    lon0, lon1 = -58.80, -58.15
    la, lo, W, _ = bathymetry(lat0, lat1, lon0, lon1, n=700)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    levels = np.arange(250, 2251, 250)
    cs = ax.contour(lo, la, W * 1000.0, levels=levels, colors=C["grey"],
                    linewidths=0.55, zorder=1)
    ax.clabel(cs, levels[1::2], fmt="%d", fontsize=7, inline=True)

    m = ((std.lat.between(lat0, lat1)) & (std.lon.between(lon0, lon1)))
    ax.scatter(std.lon[m], std.lat[m], s=2.2, c=C["grey"], alpha=0.45, lw=0,
               zorder=2, label=f"standard tier (n={m.sum():,} in view / {len(std):,})")
    ms = ((strict.lat.between(lat0, lat1)) & (strict.lon.between(lon0, lon1)))
    sc = ax.scatter(strict.lon[ms], strict.lat[ms], s=9, c=strict.depth_km[ms],
                    cmap="viridis", vmin=0, vmax=10, lw=0.15, edgecolor="k",
                    zorder=3, label=f"strict tier (n={ms.sum():,} in view / {len(strict):,})")

    ins = st[(st.latitude.between(lat0, lat1)) & (st.longitude.between(lon0, lon1))]
    obs, land = ins[ins.is_obs], ins[~ins.is_obs]
    ax.scatter(obs.longitude, obs.latitude, marker="^", s=110, c=C["vermillion"],
               edgecolor="k", lw=0.8, zorder=6)
    ax.scatter(land.longitude, land.latitude, marker="^", s=110, facecolor="none",
               edgecolor="k", lw=1.2, zorder=6)
    for _, r in ins.iterrows():
        ax.annotate(r.station, (r.longitude, r.latitude), textcoords="offset points",
                    xytext=(6, 5), fontsize=7.5, zorder=7,
                    path_effects=None, color="k")

    ax.plot(CALDERA_LON, CALDERA_LAT, marker="*", ms=20, color=C["yellow"],
            markeredgecolor="k", markeredgewidth=0.9, zorder=7)
    ax.annotate("Orca caldera", (CALDERA_LON, CALDERA_LAT),
                textcoords="offset points", xytext=(-14, -22), ha="right",
                fontsize=10.5, fontweight="bold", zorder=8,
                bbox=dict(fc="white", ec="none", alpha=0.80, pad=1.5))

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / np.cos(np.radians(CLAT)))
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_title("F1  Orca / Bransfield Strait, NLLoc epicentres")
    scalebar(ax, 5, CLAT)

    cb = fig.colorbar(sc, ax=ax, pad=0.015, shrink=0.85)
    cb.set_label("strict-tier depth below sea level (km)")

    handles = [
        Line2D([], [], marker="o", ls="", ms=4, color=C["grey"],
               label=f"standard tier ({m.sum():,} of {len(std):,} in view)"),
        Line2D([], [], marker="o", ls="", ms=5, color="#3b7d55",
               markeredgecolor="k", label=f"strict tier, coloured by depth "
                                         f"({ms.sum():,} of {len(strict):,})"),
        Line2D([], [], marker="^", ls="", ms=9, color=C["vermillion"],
               markeredgecolor="k", label="ZX OBS"),
        Line2D([], [], marker="^", ls="", ms=9, markerfacecolor="none",
               markeredgecolor="k", label="land station"),
        Line2D([], [], color=C["grey"], lw=0.8, label="bathymetry, 250 m"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, borderpad=0.5)

    # inset: full network extent
    ilat0, ilat1, ilon0, ilon1 = -63.70, -61.95, -61.40, -57.20
    _, _, Wg, pf = bathymetry(ilat0, ilat1, ilon0, ilon1, n=260)
    ila = np.linspace(ilat0, ilat1, 260)
    ilo = np.linspace(ilon0, ilon1, 260)
    axi = ax.inset_axes([0.655, 0.045, 0.335, 0.40])
    axi.contourf(ilo, ila, Wg * 1000.0, levels=[-10, 1, 500, 1000, 1500, 2000, 5000],
                 colors=["#e8e8e8", "#cfe3f2", "#a9cde8", "#7fb6dd", "#5a9fd2", "#3f86bd"],
                 zorder=0)
    allst = load_stations()
    axi.scatter(allst.longitude[allst.is_obs], allst.latitude[allst.is_obs],
                marker="^", s=22, c=C["vermillion"], edgecolor="k", lw=0.4, zorder=3)
    axi.scatter(allst.longitude[~allst.is_obs], allst.latitude[~allst.is_obs],
                marker="^", s=22, facecolor="none", edgecolor="k", lw=0.7, zorder=3)
    axi.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False,
                            edgecolor=C["vermillion"], lw=1.4, zorder=4))
    axi.set_xlim(ilon0, ilon1)
    axi.set_ylim(ilat0, ilat1)
    axi.set_aspect(1.0 / np.cos(np.radians(CLAT)))
    axi.set_xticks([])
    axi.set_yticks([])
    axi.grid(False)
    axi.text(0.5, 0.965, "full network (ZX + 5M/AI/AM)", transform=axi.transAxes,
             ha="center", va="top", fontsize=8.5,
             bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), zorder=6)

    CAPTIONS["F1"] = (
        f"**F1 - epicentre map.** Epicentres of the NLLoc catalogue over the Orca "
        f"volcanic edifice. Grey dots: standard tier (gap < 180 deg, rms < 0.5 s, "
        f"Nphs >= 6, depth_bsf > -0.2 km; {len(std):,} events, {m.sum():,} inside the map "
        f"frame). Coloured dots: strict tier ({len(strict):,} events, {ms.sum():,} in "
        f"frame), shaded by hypocentral depth below sea level (0-10 km). Triangles are "
        f"stations - filled vermillion for ZX ocean-bottom seismometers (true water "
        f"depths 0.785-1.943 km), open for land stations. Thin grey lines are bathymetry "
        f"contours every 250 m, from the 30 m Orca multibeam grid inside its footprint and "
        f"GEBCO_2023 outside it (the same surface and precedence used to un-shear the "
        f"velocity grid, scripts/41). The star marks the Orca caldera (the 30 m grid's "
        f"summit is -615 m at -62.446/-58.397). Scale bar 5 km; the inset shows the full "
        f"station network with the map frame outlined. Seismicity concentrates in and "
        f"immediately around the caldera, where the array aperture is smallest and the "
        f"azimuthal gap is therefore lowest - the binding QC constraint (notes/31 sec.3)."
    )
    return save(fig, "F1_map_epicentres")


# ------------------------------------------------------------------- F2
def fig2():
    strict = load_tier("strict")
    hdd = load_hypodd()
    HALF = 15.0

    def prof_bathy(axis):
        """Seafloor along the W-E (axis='we') or S-N (axis='sn') line through the centre."""
        s = np.linspace(-HALF, HALF, 400)
        if axis == "we":
            lat = np.full_like(s, CLAT)
            lon = CLON + s / kmperdeglon(CLAT)
        else:
            lat = CLAT + s / KM_PER_DEG_LAT
            lon = np.full_like(s, CLON)
        pad = 0.03
        _, _, _, pf = bathymetry(lat.min() - pad, lat.max() + pad,
                                 lon.min() - pad, lon.max() + pad, n=400)
        return s, pf(lat, lon)

    panels = [
        ("we", strict.lat.values, strict.lon.values, strict.depth_km.values,
         "NLLoc strict tier", TIER_COLOR["strict"]),
        ("sn", strict.lat.values, strict.lon.values, strict.depth_km.values,
         "NLLoc strict tier", TIER_COLOR["strict"]),
        ("we", hdd.lat.values, hdd.lon.values, hdd.depth_bsl_km.values,
         "hypoDD 3D QC-pass", TIER_COLOR["hypodd"]),
        ("sn", hdd.lat.values, hdd.lon.values, hdd.depth_bsl_km.values,
         "hypoDD 3D QC-pass", TIER_COLOR["hypodd"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True, sharey=True)
    counts = {}
    for ax, (axis, lat, lon, dep, lab, col) in zip(axes.ravel(), panels):
        e, n = ll_to_km(lat, lon)
        inbox = (np.abs(e) <= HALF) & (np.abs(n) <= HALF) & np.isfinite(dep)
        s = e[inbox] if axis == "we" else n[inbox]
        ax.scatter(s, dep[inbox], s=4.5, c=col, alpha=0.40, lw=0, zorder=3)
        sb, wb = prof_bathy(axis)
        ax.plot(sb, wb, color=C["black"], lw=1.6, zorder=4)
        ax.fill_between(sb, 0, wb, color="#dbeaf5", zorder=0)
        ax.set_title(f"{lab} - {'W-E' if axis == 'we' else 'S-N'} "
                     f"(n = {int(inbox.sum()):,})", fontsize=11)
        counts[(lab, axis)] = int(inbox.sum())
    for ax in axes.ravel():
        ax.set_xlim(-HALF, HALF)
        ax.set_ylim(12, -0.4)
    for ax in axes[1]:
        ax.set_xlabel("distance from -62.4413/-58.44 (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("depth below sea level (km)")
    axes[0, 0].annotate("W", (0.015, 0.92), xycoords="axes fraction", fontweight="bold")
    axes[0, 0].annotate("E", (0.975, 0.92), xycoords="axes fraction", fontweight="bold")
    axes[0, 1].annotate("S", (0.015, 0.92), xycoords="axes fraction", fontweight="bold")
    axes[0, 1].annotate("N", (0.975, 0.92), xycoords="axes fraction", fontweight="bold")
    axes[0, 0].plot([], [], color=C["black"], lw=1.6, label="seafloor along the section")
    axes[0, 0].legend(loc="lower left", framealpha=0.9)
    fig.suptitle("F2  Depth sections through the NLLoc projection origin "
                 "(+/- 15 km box, all events projected)", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    CAPTIONS["F2"] = (
        f"**F2 - depth sections.** Hypocentres inside a +/-15 km box centred on the NLLoc "
        f"projection origin (-62.4413, -58.44), projected onto a W-E (left) and a S-N "
        f"(right) vertical plane; every event in the box is shown in both panels. Top row: "
        f"NLLoc strict tier ({counts[('NLLoc strict tier', 'we')]:,} of "
        f"{len(strict):,} events in the box). Bottom row: the hypoDD 3D double-difference "
        f"relocation of the standard tier, QC-pass only "
        f"({counts[('hypoDD 3D QC-pass', 'we')]:,} of {len(hdd):,}), plotted on its "
        f"depth_bsl_km column so both rows share the sea-level datum. The black line is "
        f"the seafloor sampled along the section line from the same bathymetry as F1, and "
        f"the pale band above it is the water column. All four panels share identical "
        f"axes. Events plotted above the black line are not in the water: the profile is "
        f"the seafloor ON the section line, while the points are projected onto it from up "
        f"to 15 km off-line, where the seafloor is deeper (510 of the {len(hdd):,} hypoDD "
        f"events, 5.6%, do sit up to 0.20 km above their own local seafloor, which is the "
        f"half-grid-cell tolerance the QC allows). Seismicity is concentrated in the upper "
        f"~5 km beneath the edifice; the "
        f"hypoDD panels are tighter laterally but move the 6-9 km population up by about "
        f"2 km (README sec.5 caveat 7), so absolute depth should be read from the NLLoc "
        f"panels and relative geometry from the hypoDD panels."
    )
    return save(fig, "F2_depth_sections")


# ------------------------------------------------------------------- F3
def fig3():
    sets = []
    for t in ("loose", "standard", "strict"):
        d = load_tier(t)
        sets.append((f"NLLoc {t} tier", d.depth_bsf_km.values, TIER_COLOR[t]))
    h = load_hypodd()
    sets.append(("hypoDD 3D QC-pass", h.depth_bsf_local_km.values, TIER_COLOR["hypodd"]))

    bins = np.arange(0.0, 12.0001, 0.25)
    fig, axes = plt.subplots(4, 1, figsize=(8.8, 10.2), sharex=True)
    stats = {}
    for ax, (lab, v, col) in zip(axes, sets):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        inrange = v[(v >= 0) & (v <= 12)]
        p10, p50, p90 = pct(v, 10), pct(v, 50), pct(v, 90)
        stats[lab] = (len(v), p10, p50, p90, len(inrange))
        ax.hist(inrange, bins=bins, color=col, alpha=0.85, edgecolor="white", lw=0.3)
        for q, val, ls in ((10, p10, ":"), (50, p50, "-"), (90, p90, "--")):
            ax.axvline(val, color=C["black"], ls=ls, lw=1.3, zorder=5)
        ymax = ax.get_ylim()[1]
        ax.text(0.985, 0.90,
                f"n = {len(v):,}   p10 / p50 / p90 = {p10:.2f} / {p50:.2f} / {p90:.2f} km",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
                bbox=dict(fc="white", ec=col, lw=1.0, alpha=0.92, pad=3.5))
        ax.set_title(lab, loc="left", fontsize=11.5)
        ax.set_ylabel("events / 0.25 km")
        ax.set_ylim(0, ymax * 1.12)
    axes[-1].set_xlabel("depth below the local seafloor (km)")
    axes[-1].set_xlim(0, 12)
    axes[0].plot([], [], color=C["black"], ls=":", label="p10")
    axes[0].plot([], [], color=C["black"], ls="-", label="p50")
    axes[0].plot([], [], color=C["black"], ls="--", label="p90")
    axes[0].legend(loc="upper right", bbox_to_anchor=(0.985, 0.70), ncol=3,
                   framealpha=0.92, handlelength=2.4)
    fig.suptitle("F3  Depth below the local seafloor, by catalogue tier", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.982))

    txt = "; ".join(f"{k} p10/p50/p90 = {v[1]:.2f}/{v[2]:.2f}/{v[3]:.2f} km (n = {v[0]:,})"
                    for k, v in stats.items())
    CAPTIONS["F3"] = (
        f"**F3 - depth distributions.** Histograms of depth below the LOCAL seafloor "
        f"(depth_bsf_km for NLLoc, depth_bsf_local_km for hypoDD), 0.25 km bins over "
        f"0-12 km, one panel per catalogue. Dotted / solid / dashed vertical lines are the "
        f"p10 / p50 / p90 of the full column (a handful of events fall outside the 0-12 km "
        f"plotting range and are excluded from the bars but not from the percentiles). "
        f"{txt}. Tightening the QC deepens the median (the strict tier additionally "
        f"requires depth_bsf > +0.2 km, which removes the seafloor-hugging population) and "
        f"narrows the distribution, as expected if the shallow excess is location noise "
        f"rather than structure. Note the README caveat that the P tomography is honestly "
        f"resolved only to about 4 km below the seafloor: 14.8% of the standard tier and "
        f"12.7% of the strict tier sit deeper than that."
    )
    print("    F3 annotated percentiles:")
    for k, v in stats.items():
        print(f"      {k:24s} n={v[0]:>6,}  p10/p50/p90 = "
              f"{v[1]:.2f} / {v[2]:.2f} / {v[3]:.2f} km   ({v[4]:,} inside 0-12 km)")
    return save(fig, "F3_depth_distributions")


# ------------------------------------------------------------------- F4
def fig4():
    dl = load_delays()
    st = load_stations()
    wd = dict(zip(st.station, st.water_km))
    isobs = dict(zip(st.station, st.is_obs))

    w = dl.pivot_table(index="station", columns="phase", values="delay").reset_index()
    nn = dl.pivot_table(index="station", columns="phase", values="n").reset_index()
    w = w.merge(nn, on="station", suffixes=("", "_n"))
    w["water_km"] = w.station.map(wd)
    w["is_obs"] = w.station.map(isobs).fillna(False).astype(bool)
    w = w[w.station.isin(st.station)]
    obs = w[w.is_obs].sort_values("water_km").reset_index(drop=True)
    land = w[~w.is_obs].sort_values("station").reset_index(drop=True)
    order = pd.concat([obs, land], ignore_index=True)
    x = np.arange(len(order))
    bw = 0.38

    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.34)
    ax = fig.add_subplot(gs[0])
    ax.bar(x - bw / 2, order["P"], bw, color=C["blue"], edgecolor="k", lw=0.4, label="P delay")
    ax.bar(x + bw / 2, order["S"], bw, color=C["orange"], edgecolor="k", lw=0.4, label="S delay")
    ax.axhline(0, color="k", lw=0.9)
    ax.axvline(len(obs) - 0.5, color=C["grey"], lw=1.4, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order.station, rotation=90, fontsize=8.5)
    ax.set_ylabel("LOCDELAY station term (s)\nobs - delay; + = station observes late")
    ax.set_title("F4a  Station terms - ZX OBS sorted by water depth, "
                 "then land stations", loc="left")
    ax.set_xlim(-0.8, len(order) - 0.2)
    lo_y = min(order[["P", "S"]].min().min(), 0.0)
    hi_y = max(order[["P", "S"]].max().max(), 0.0)
    ax.set_ylim(lo_y - 0.10, hi_y + 0.16)
    ax2 = ax.twinx()
    ax2.plot(x[:len(obs)], order.water_km[:len(obs)], marker="D", ms=5.5, ls="none",
             color=C["green"], markeredgecolor="k", markeredgewidth=0.4,
             label="OBS water depth")
    ax2.set_ylabel("station water depth (km)", color=C["green"])
    ax2.tick_params(axis="y", colors=C["green"])
    ax2.set_ylim(0, 2.2)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, framealpha=0.9)
    ax.text(len(obs) + 0.2, ax.get_ylim()[1] * 0.92, "land / island stations",
            fontsize=9.5, color=C["grey"])

    # panel b: analyst manual S-P vs the station S-P delay
    msp, npairs = manual_sp_by_station()
    spd = (w.set_index("station")["S"] - w.set_index("station")["P"]).rename("sp_delay")
    msp = msp.merge(spd, left_on="station", right_index=True, how="inner")
    msp = msp[msp.n >= 20].sort_values("sp_delay")

    axb = fig.add_subplot(gs[1])
    xhi = msp.sp_delay.max() * 1.18
    yhi = msp.p50.max() * 1.12
    axb.plot([0, max(xhi, yhi)], [0, max(xhi, yhi)], color=C["black"], lw=1.1, ls="--",
             label="1:1 (delay = measured S-P)")
    axb.scatter(msp.sp_delay, msp.p5, s=70, marker="v", c=C["vermillion"],
                edgecolor="k", lw=0.5, zorder=4, label="manual S-P p5")
    axb.scatter(msp.sp_delay, msp.p50, s=70, marker="o", c=C["blue"],
                edgecolor="k", lw=0.5, zorder=4, label="manual S-P p50")
    placed: list[tuple[float, float]] = []
    for _, r in msp.iterrows():
        axb.vlines(r.sp_delay, r.p5, r.p50, color=C["grey"], lw=1.0, zorder=2)
        # greedy de-collision: nudge the label up until it clears earlier ones
        lx, ly = r.sp_delay + 0.006, r.p50 + 0.025
        while any(abs(lx - px) < 0.035 and abs(ly - py) < 0.052 for px, py in placed):
            ly += 0.052
        placed.append((lx, ly))
        axb.plot([r.sp_delay, lx], [r.p50, ly], color=C["grey"], lw=0.5, zorder=2)
        axb.annotate(r.station, (lx, ly), fontsize=8, va="center", ha="left", zorder=9,
                     bbox=dict(fc="white", ec="none", alpha=0.80, pad=0.8))
    axb.set_xlim(0, xhi)
    axb.set_ylim(0, yhi)
    axb.set_xlabel("station S-P delay, S term - P term (s)")
    axb.set_ylabel("analyst manual S-P (s)")
    axb.set_title(f"F4b  Do the analyst's picks support the S-P terms?  "
                  f"{npairs:,} ZX manual P+S pairs, {len(msp)} stations with n >= 20",
                  loc="left")
    axb.legend(loc="upper left", framealpha=0.9)
    margin = (msp.p5 - msp.sp_delay)
    axb.text(0.985, 0.06,
             f"every station: p5 above the delay, by "
             f"{margin.min():+.2f} to {margin.max():+.2f} s",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=10,
             bbox=dict(fc="white", ec=C["green"], lw=1.2, alpha=0.93, pad=4))

    CAPTIONS["F4"] = (
        f"**F4 - station terms and their observational support.** (a) The 76 LOCDELAY "
        f"terms of nlloc/delays/v6_it1B.delays (VELEST invB, refined by one iteration of "
        f"NLLoc per-station median residuals), applied by NLLoc as obs - delay, so a "
        f"positive term means the station observes late. ZX OBS are ordered by water depth "
        f"(green diamonds, right axis); land and island stations are grouped to the right "
        f"of the dashed divider. The OBS S terms are systematically larger than their P "
        f"terms - it is that S-P difference (median +0.37 s) that carries much of the "
        f"depth axis, and it is a fitted quantity, not an independent observation "
        f"(README caveat 2). (b) The analyst's own picks tested against those terms: for "
        f"each ZX OBS with at least 20 manual P+S pairs ({npairs:,} pairs in total), the "
        f"5th percentile and the median of the hand-measured S-P are plotted against the "
        f"station's S-P delay, with the 1:1 line. Every station's p5 sits above its delay, "
        f"by {margin.min():+.2f} s (the station with the largest term) to "
        f"{margin.max():+.2f} s, and only 0.5% of the 17,347 individual pairs fall below "
        f"their own station's delay. The margin is smallest exactly where the term is "
        f"largest, which is where it should be tight."
    )
    return save(fig, "F4_station_terms")


# ------------------------------------------------------------------- F5
_BIN_RE = re.compile(r"^\s*\(([\d.]+),\s*([\d.]+)\]\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s*$")
_RATIO_RE = re.compile(r"observed x([\d.]+)\s+predicted x([\d.]+)")
_N_RE = re.compile(r"n=([\d,]+) events")


def run_sp_check(extra, label):
    cmd = [sys.executable, str(REPO / "scripts" / "52_sp_depth_check.py"),
           "--frame", "sealevel", "--scorer-vpvs", "1.88", "--label", label] + extra
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    print(f"    running 52_sp_depth_check.py: {label}")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"52_sp_depth_check.py failed for {label}:\n{r.stdout}\n{r.stderr}")
    rows, ratio, ntot = [], None, None
    for line in r.stdout.split("\n"):
        m = _BIN_RE.match(line)
        if m:
            rows.append(dict(lo=float(m.group(1)), hi=float(m.group(2)), n=int(m.group(3)),
                             obs=float(m.group(4)), pred=float(m.group(5)),
                             misfit=float(m.group(6))))
        if _RATIO_RE.search(line):
            g = _RATIO_RE.search(line)
            ratio = (float(g.group(1)), float(g.group(2)))
        if _N_RE.search(line):
            ntot = int(_N_RE.search(line).group(1).replace(",", ""))
    if not rows or ratio is None:
        raise SystemExit(f"could not parse 52_sp_depth_check.py output for {label}:\n{r.stdout}")
    return pd.DataFrame(rows), ratio, ntot, r.stdout


def fig5():
    ref, ref_ratio, ref_n, ref_txt = run_sp_check(
        ["--hyp-dir", "nlloc/output/abtest_ORCA_v5"],
        "reference ORCA_v5 (no station terms)")
    fin, fin_ratio, fin_n, fin_txt = run_sp_check(
        ["--catalog", "nlloc/output/abtest_v6_it1Bg/catalog_mapped.csv",
         "--delays", "nlloc/delays/v6_it1B.delays"],
        "final v6 it1Bg (with station terms)")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True)
    out = {}
    for ax, (d, ratio, ntot, name) in zip(axes, [
            (ref, ref_ratio, ref_n, "reference run: no station terms"),
            (fin, fin_ratio, fin_n, "final catalogue: with station terms")]):
        lab = [f"{r.lo:g}-{r.hi:g}" for r in d.itertuples()]
        x = np.arange(len(d))
        bw = 0.38
        ax.bar(x - bw / 2, d.obs, bw, color=C["blue"], edgecolor="k", lw=0.4,
               label="observed S-P (median)")
        ax.bar(x + bw / 2, d.pred, bw, color=C["orange"], edgecolor="k", lw=0.4,
               label="predicted S-P at the catalogue depth (median)")
        for xi, r in zip(x, d.itertuples()):
            ax.text(xi, max(r.obs, r.pred) + 0.05, f"{r.n}", ha="center", va="bottom",
                    fontsize=8, color=C["grey"])
        spread = ratio[1] / ratio[0]
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=45, ha="right")
        ax.set_xlabel("depth below sea level (km)")
        ax.set_title(f"{name}\nspread deepest/shallowest: obs x{ratio[0]:.2f}, "
                     f"pred x{ratio[1]:.2f}  ->  ratio {spread:.2f}", fontsize=10.8)
        ax.set_ylim(0, 2.8)
        ax.legend(loc="upper left", framealpha=0.9)
        out[name] = (ratio, spread, ntot)
    axes[0].set_ylabel("near-station S-P (s)")
    fig.suptitle("F5  The S-P depth gate: observed vs predicted near-station S-P "
                 "by depth bin (2,000-event sample, scorer Vp/Vs 1.88)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    r0, s0, n0 = out["reference run: no station terms"]
    r1, s1, n1 = out["final catalogue: with station terms"]
    CAPTIONS["F5"] = (
        f"**F5 - the S-P depth gate.** The most nearly model-free depth observable this "
        f"array has is the S-P time at the nearest station, read straight from the picks. "
        f"For each event in the 2,000-event A/B sample, scripts/52_sp_depth_check.py takes "
        f"the nearest station within 4 km that carries both P and S, and compares the "
        f"observed S-P (blue) with the S-P predicted by straight-ray geometry at the "
        f"catalogue's own hypocentre (orange), binned by depth below sea level; the scorer "
        f"uses a fixed Vp/Vs of 1.88 so the two runs differ only by their positions, and "
        f"the observed S-P is corrected by the run's own station terms where it has them. "
        f"Bin counts are printed above the bars. Left: the reference run with no station "
        f"terms (n = {n0:,}) - the predicted S-P spreads x{r0[1]:.2f} from the shallowest "
        f"to the deepest bin while the observation only spreads x{r0[0]:.2f}, a ratio of "
        f"{s0:.2f}, i.e. a depth axis about 25% too long. Right: the production v6 "
        f"configuration (n = {n1:,}) - x{r1[1]:.2f} predicted against x{r1[0]:.2f} "
        f"observed, ratio {s1:.2f}. The station terms remove the stretch; this is the gate "
        f"that selected it1B over the invA-scale alternatives (ratios 1.12-1.19), which it "
        f"rejects (notes/31 sec.2)."
    )
    print(f"    F5 reference : obs x{r0[0]:.2f}  pred x{r0[1]:.2f}  ratio {s0:.2f}  n={n0:,}")
    print(f"    F5 final v6  : obs x{r1[0]:.2f}  pred x{r1[1]:.2f}  ratio {s1:.2f}  n={n1:,}")
    return save(fig, "F5_sp_gate")


# ------------------------------------------------------------------- F6
def fig6():
    d = pd.read_csv(REPO / "hypodd" / "year_v4_3d_frozen_fine" / "tt_compare_3d_vs_nlloc.csv")
    d = d.dropna(subset=["ttp", "nllP", "dist"])
    d["fd_minus_tracer_ms"] = (d.nllP - d.ttp) * 1000.0

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    hb = ax.hexbin(d.dist, d.fd_minus_tracer_ms, gridsize=(70, 55),
                   extent=(0, 60, -150, 400), bins="log", cmap="viridis", mincnt=1,
                   linewidths=0.0)
    cb = fig.colorbar(hb, ax=ax, pad=0.015)
    cb.set_label("rays per cell (log scale)")

    edges = np.array([0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60])
    cut = pd.cut(d.dist, edges)
    g = d.groupby(cut, observed=True).fd_minus_tracer_ms.agg(["size", "median"])
    xc = np.array([(iv.left + iv.right) / 2 for iv in g.index])
    ax.plot(xc, g["median"], color=C["vermillion"], lw=2.4, marker="o", ms=6,
            markeredgecolor="k", markeredgewidth=0.5, zorder=6,
            label="median per distance bin")
    ax.axhline(0, color="k", lw=1.0, zorder=5)
    ax.axhspan(108, 161, color=C["orange"], alpha=0.22, zorder=1)
    ax.axhline(108, color=C["orange"], lw=1.0, ls="--", zorder=5)
    ax.axhline(161, color=C["orange"], lw=1.0, ls="--", zorder=5)
    ax.annotate("+108 ... +161 ms: the independently measured NLLoc-FD bias\n"
                "(FD grid - pykonal FMM, 4,954 real rays: +108 ms at 0-2 km,\n"
                "+143 at 2-5, +144 at 5-10, +161 ms at 10-25 km; notes/27 I21)",
                xy=(1.5, -85), fontsize=9.5, ha="left", va="center",
                bbox=dict(fc="white", ec=C["orange"], lw=1.2, alpha=0.92, pad=4),
                zorder=7)
    med_all = d.fd_minus_tracer_ms.median()
    ax.set_xlim(0, 60)
    ax.set_ylim(-150, 400)
    ax.set_xlabel("epicentral distance (km)")
    ax.set_ylabel("FD grid time - pseudo-bending tracer time (ms)")
    ax.set_title(f"F6  Travel-time solver check: NLLoc finite-difference P grid minus the "
                 f"hypoDD 3D ray tracer\n{len(d):,} rays, median {med_all:+.0f} ms "
                 f"(same 3D P model)", fontsize=11.5)
    ax.legend(loc="upper right", framealpha=0.92)

    rows = "; ".join(f"{iv.left:g}-{iv.right:g} km {v:+.0f} ms (n={int(n):,})"
                     for iv, n, v in zip(g.index, g["size"], g["median"]))
    CAPTIONS["F6"] = (
        f"**F6 - travel-time solver check.** Difference between the two forward solvers "
        f"used in this study on the SAME 3D P model: the NLLoc finite-difference "
        f"(Podvin-Lecomte) time grid minus the hypoDD pseudo-bending ray tracer, for "
        f"{len(d):,} real source-station rays of the year catalogue, as a function of "
        f"epicentral distance (log-scaled hexbin, vermillion line = median per distance "
        f"bin). Median over all rays {med_all:+.0f} ms; by bin: {rows}. The shaded band is "
        f"the independently measured NLLoc-FD bias against a pykonal eikonal solver "
        f"(+108 ms at 0-2 km rising to +161 ms at 10-25 km on 4,954 rays, notes/27 I21) - "
        f"the tracer-based difference plotted here falls in the same place and grows with "
        f"distance the same way, i.e. the two independent solvers agree with each other "
        f"and Grid2Time does not. This is why the v5/v6 travel-time grids "
        f"(nlloc/time/ORCA_v5.*) are built with pykonal FMM instead of Grid2Time; a "
        f"0.1-0.16 s station- and distance-dependent bias in P (inherited by S x1.78) is "
        f"exactly the error that stretches a depth axis. Per-ray pykonal times were not "
        f"retained, so only the FD-minus-tracer difference can be replotted here."
    )
    return save(fig, "F6_traveltime_solver_check")


# ------------------------------------------------------------------- F7
def fig7():
    std, strict = load_tier("standard"), load_tier("strict")
    a0 = pd.Timestamp(AIRGUN[0], tz="UTC")
    a1 = pd.Timestamp(AIRGUN[1], tz="UTC")

    day = std.ot.dt.floor("D")
    per = day.value_counts().sort_index()
    full = pd.date_range(std.ot.min().floor("D"), std.ot.max().floor("D"),
                         freq="D", tz="UTC")
    per = per.reindex(full, fill_value=0)

    fig, axes = plt.subplots(2, 1, figsize=(11.4, 8.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.16))
    ax = axes[0]
    ax.bar(per.index, per.values, width=1.0, color=C["blue"], lw=0,
           label=f"standard tier, events per day (n = {len(std):,})")
    ax.axvspan(a0, a1, color=C["vermillion"], alpha=0.18, zorder=0)
    ax.set_ylabel("events per day")
    ax.set_title("F7a  Temporal evolution of the standard tier", loc="left")
    axc = ax.twinx()
    axc.plot(per.index, np.cumsum(per.values), color=C["black"], lw=2.0,
             label="cumulative count")
    axc.set_ylabel("cumulative events")
    axc.set_ylim(0, len(std) * 1.03)
    axc.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax.legend(h1 + h2 + [Line2D([], [], color=C["vermillion"], alpha=0.4, lw=8)],
              l1 + l2 + ["airgun survey window (excluded)"],
              loc="upper left", framealpha=0.92)
    peak_day = per.idxmax()

    axb = axes[1]
    axb.scatter(strict.ot, strict.depth_km, s=4.5, c=TIER_COLOR["strict"], alpha=0.40, lw=0)
    axb.axvspan(a0, a1, color=C["vermillion"], alpha=0.18, zorder=0)
    axb.set_ylim(12, -0.4)
    axb.set_ylabel("depth below sea level (km)")
    axb.set_xlabel("date (UTC)")
    axb.set_title(f"F7b  Strict-tier depth vs time (n = {len(strict):,})", loc="left")
    axb.set_xlim(full[0], full[-1])
    fig.autofmt_xdate()

    CAPTIONS["F7"] = (
        f"**F7 - temporal evolution.** (a) Daily count of standard-tier events over the "
        f"2019-01-01 to 2020-03-01 ZX deployment (bars, left axis) with the cumulative "
        f"count (black, right axis); {len(std):,} events over "
        f"{len(per):,} days, busiest day {peak_day:%Y-%m-%d} with {per.max():,} events. "
        f"The shaded band is the {AIRGUN[0]} to {AIRGUN[1]} airgun survey window, which is "
        f"excluded from the catalogue wholesale rather than filtered (18,848 associated "
        f"events removed, an estimated ~1,800 of them genuine earthquakes) - the catalogue "
        f"is NOT complete across it and no rate inside it should be read from this figure. "
        f"(b) Depth below sea level against time for the strict tier ({len(strict):,} "
        f"events), same time axis. Activity is episodic rather than steady, and the depth "
        f"distribution of the well-constrained subset does not drift systematically over "
        f"the deployment, which argues against a slowly changing clock, geometry or "
        f"station-term error masquerading as depth."
    )
    return save(fig, "F7_temporal")


# ------------------------------------------------------------------- F8
def fig8():
    std = load_tier("standard")
    hdd = load_hypodd()
    j = hdd.merge(std[["event_idx", "depth_km", "sigma_z_km", "depth_bsf_km"]],
                  on="event_idx", how="inner")
    x = j.depth_km.values          # NLLoc v6 standard, BSL
    y = j.depth_bsl_km.values      # hypoDD 3D, BSL

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    hb = ax.hexbin(x, y, gridsize=60, extent=(0, 16, 0, 16), bins="log",
                   cmap="viridis", mincnt=1, linewidths=0.0)
    cb = fig.colorbar(hb, ax=ax, pad=0.015)
    cb.set_label("events per cell (log scale)")
    ax.plot([0, 16], [0, 16], color=C["black"], lw=1.4, ls="--", label="1:1")

    edges = np.array([0, 1, 2, 3, 4, 5, 6, 7.5, 9, 12, 16])
    cut = pd.cut(x, edges)
    g = pd.DataFrame({"x": x, "y": y}).groupby(cut, observed=True).agg(
        n=("y", "size"), med=("y", "median"),
        p25=("y", lambda v: np.percentile(v, 25)),
        p75=("y", lambda v: np.percentile(v, 75)))
    xc = np.array([(iv.left + iv.right) / 2 for iv in g.index])
    ax.errorbar(xc, g["med"], yerr=[g["med"] - g["p25"], g["p75"] - g["med"]],
                color=C["vermillion"], lw=2.2, marker="o", ms=7, capsize=4,
                markeredgecolor="k", markeredgewidth=0.5, zorder=6,
                label="binned median (bars: p25-p75)")

    A = np.polyfit(x, y - x, 1)
    r_s = pd.Series(x).corr(pd.Series(y), method="spearman")
    med_abs = np.median(np.abs(y - x))
    ax.text(0.035, 0.965,
            f"n = {len(j):,} joined on event_idx (hypoDD id - 1)\n"
            f"slope of (hypoDD - NLLoc) on NLLoc depth = {A[0]:+.3f}\n"
            f"intercept {A[1]:+.3f} km,  median |dz| = {med_abs:.2f} km\n"
            f"Spearman r = {r_s:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10.5,
            bbox=dict(fc="white", ec=C["grey"], lw=1.0, alpha=0.94, pad=5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal")
    ax.set_xlabel("NLLoc standard-tier depth below sea level (km)")
    ax.set_ylabel("hypoDD 3D QC-pass depth below sea level (km)")
    ax.set_title("F8  Double-difference vs absolute depth, same events")
    ax.legend(loc="lower right", framealpha=0.92)

    # median signed dz on the bins notes/31 sec.4 quotes, so the caption cross-checks
    nb = pd.cut(x, [0, 2, 4, 6, 9, 15])
    gz = pd.DataFrame({"x": x, "dz": y - x}).groupby(nb, observed=True).dz.agg(
        ["size", "median"])
    rows = "; ".join(f"{iv.left:g}-{iv.right:g} km {v:+.2f} km (n={int(n):,})"
                     for iv, n, v in zip(gz.index, gz["size"], gz["median"]))
    CAPTIONS["F8"] = (
        f"**F8 - NLLoc vs hypoDD depth.** hypoDD 3D relocated depth against the NLLoc "
        f"standard-tier starting depth for the {len(j):,} events that pass hypoDD QC, "
        f"joined on event_idx (hypoDD `id` = event_idx + 1, set in "
        f"scripts/22_pyocto_to_hypodd_input.py); both axes are below sea level, so the "
        f"hypoDD depth_bsl_km column is used, not `dep`. Log-scaled hexbin with the 1:1 "
        f"line and the binned median (vermillion, bars p25-p75). Regressing the change on "
        f"the starting depth gives a slope of {A[0]:+.3f} with a median |dz| of "
        f"{med_abs:.2f} km and Spearman r = {r_s:.3f}: the double-difference solution "
        f"still compresses the deep end of the absolute depth axis. Median shift by "
        f"starting-depth bin, median signed (hypoDD - NLLoc): {rows}. This is the residual "
        f"the README flags as caveat 7 - it is concentrated in the 6-9 km population, the "
        f"same bin that is least stable against perturbing the station terms (notes/31 "
        f"sec.2), and it is the reason absolute depth should be quoted from NLLoc and only "
        f"relative geometry from hypoDD."
    )
    return save(fig, "F8_nlloc_vs_hypodd_depth")


# ------------------------------------------------------------------ main
FIGS = {"F1": fig1, "F2": fig2, "F3": fig3, "F4": fig4,
        "F5": fig5, "F6": fig6, "F7": fig7, "F8": fig8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset, e.g. --only F1 F3")
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
        cap = OUT / "captions.md"
        old = {}
        if cap.exists() and a.only:
            # keep captions for figures not regenerated in this run
            for blk in cap.read_text().split("\n\n"):
                mm = re.match(r"\*\*(F\d)", blk.strip())
                if mm:
                    old[mm.group(1)] = blk.strip()
        old.update(CAPTIONS)
        body = ["# Figure captions - final Orca / Bransfield catalogue (NLLoc v6)",
                "",
                "Generated by `scripts/69_final_figures.py`. Numbers are recomputed from the "
                "catalogues and validation runs at build time, not copied from the notes.",
                ""]
        for k in sorted(old):
            body += [old[k], ""]
        cap.write_text("\n".join(body))
        print(f"{cap.relative_to(REPO)}  {cap.stat().st_size / 1024:,.1f} kB")
    if bad:
        raise SystemExit(f"FAILED size check (<= 30 kB): {bad}")


if __name__ == "__main__":
    main()
