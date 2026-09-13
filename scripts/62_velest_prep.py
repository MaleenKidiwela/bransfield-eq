"""Build VELEST inputs (.cnv / .sta / .mod / velest.cmn) for a joint 1D
velocity + hypocentre + station-correction inversion on the ORCA strict tier.

Why this exists
---------------
Every single-parameter velocity lever tried so far (Vp/Vs 1.78 vs 1.90, top-2-km
x0.9 and x1.15, the Grid2Time -> pykonal forward-model fix) left the NLLoc depth
axis stretched: predicted / observed near-station S-P spread stays 1.25-1.31
(notes/27 I13, I15, I24, I27).  The remaining lever is to stop holding the model
fixed while relocating: invert for layer velocities, hypocentres and station
corrections together (VELEST "minimum 1D model").

Frame
-----
Datum = SEA LEVEL, stations at their TRUE depth.  VELEST does NOT force stations
onto a flat surface: subr. RAYPATH sets d(1)=zr and thk(1)=d(2)-zr, i.e. the top
layer is truncated at the receiver depth, and only requires zr >= d(1).  So the
rigid 1.0 km seafloor shift used for hypoDD (scripts/22 --datum-shift-km 1.0,
notes/27 I1) is NOT needed here and is not used; station elevations go in as the
negative numbers from catalogs/station_geometry.csv (OBS -785 .. -1943 m, land
0 .. +30 m).  The model top is put at -0.05 km BSL, 20 m above the highest land
station, so every station satisfies zr >= d(1).

The 1D column is ROCK ONLY.  configs/velocity_model.csv is on a sea-level datum
and carries a 1.3 km water column at 1.4558 km/s (see 24_run_hypodd.py::
write_velocity); OBS sit ON the seafloor, so no ray in this data set crosses the
water column, and the interval above the model's rock top is filled with the
shallowest rock velocity -- the same convention as scripts/17 (notes/27 B4).

S waves: nsp=2, i.e. an INDEPENDENT S model and independent S station
corrections, started at Vp/1.78.  The resulting Vp/Vs(z) is an output, which is
exactly the "depth-dependent Vp/Vs" lever Merlin asked for (notes/27 I23).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.timeutil import epoch_seconds, assert_nanosecond_sanity  # noqa: E402

R_EARTH = 6371.0

# Layer tops in km below SEA LEVEL.  -0.05 is above the highest land station
# (+30 m); 0.78 is the shallowest OBS (BRA03, 785 m); 1.30 is the rock top of
# configs/velocity_model.csv.  Below that the spacing follows the strict tier's
# depth distribution (p10/50/90 ~ 1.5 / 3.8 / 8.8 km BSL).
LAYER_TOPS = [-0.05, 0.78, 1.30, 1.70, 2.10, 2.50, 3.00, 3.50, 4.00,
              5.00, 6.50, 8.00, 11.00, 16.00, 21.00, 31.00]

# VELEST station names are character*4 (vel_com.f).  ZX.BRAnn is 5 characters
# and AM.R4DE2 is 5, so both need a unique 4-character alias.
def station_alias(net: str, sta: str) -> str:
    if sta.startswith("BRA"):
        return "B" + sta[3:]          # BRA13 -> B13
    return sta[:4]                    # DCP, JUBA, R4DE2 -> R4DE


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    return 2 * R_EARTH * np.arcsin(np.sqrt(
        np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


# --------------------------------------------------------------------------- #
# catalogue
# --------------------------------------------------------------------------- #
STRICT = dict(gap=120.0, rms=0.3, nph=8, sx=1.0, sy=1.0, sz=2.0, bsf=0.2)


def load_catalog(path: Path, standard_path: Path) -> pd.DataFrame:
    """Strict tier.  If the strict CSV is absent, derive it from the standard
    tier -- strict is a strict subset of standard in 40_filter_nlloc_reliable.py
    and the standard CSV already carries every column the strict cuts need."""
    if path.exists():
        c = pd.read_csv(path)
        print(f"  catalogue: {path.name}  {len(c):,} events")
        return c
    d = pd.read_csv(standard_path)
    need = ["on_boundary", "in_hull", "gap_deg", "rms_s", "n_phases",
            "sigma_x_km", "sigma_y_km", "sigma_z_km", "depth_bsf_km"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise SystemExit(f"{standard_path} lacks {missing}; cannot derive the strict tier")
    m = (~d.on_boundary & d.in_hull &
         (d.gap_deg < STRICT["gap"]) & (d.rms_s < STRICT["rms"]) &
         (d.n_phases >= STRICT["nph"]) &
         (d.sigma_x_km < STRICT["sx"]) & (d.sigma_y_km < STRICT["sy"]) &
         (d.sigma_z_km < STRICT["sz"]) & (d.depth_bsf_km > STRICT["bsf"]))
    c = d[m].copy()
    print(f"  catalogue: {path.name} ABSENT -> derived strict tier from "
          f"{standard_path.name}: {len(c):,} of {len(d):,} standard events")
    return c


# --------------------------------------------------------------------------- #
# velocity model
# --------------------------------------------------------------------------- #
def layer_velocities(vm: pd.DataFrame, tops, col: str) -> list[float]:
    """Slowness-averaged (travel-time preserving) velocity of the rock column
    over each layer.  Above the model's rock top the first rock value is used."""
    rock = vm[vm.vp_kms > 1.6].sort_values("depth_km")
    zg = rock.depth_km.to_numpy(float)
    vg = rock[col].to_numpy(float)
    out = []
    for i, z0 in enumerate(tops):
        z1 = tops[i + 1] if i + 1 < len(tops) else zg[-1]
        a, b = max(z0, zg[0]), max(z1, zg[0] + 1e-6)
        if b <= a:                       # entirely above the rock top
            out.append(float(vg[0]))
            continue
        zs = np.linspace(a, b, 64)
        out.append(float((b - a) / np.trapezoid(1.0 / np.interp(zs, zg, vg), zs)))
    return out


def write_model(path: Path, tops, vp, vs, vdamp_p, vdamp_s, title: str) -> None:
    def block(vel, damp, label):
        lines = [f"{len(tops):3d}        vel,depth,vdamp,phase (f5.2,5x,f7.2,2x,f7.3,3x,a1)"]
        for i, (v, z, d) in enumerate(zip(vel, tops, damp)):
            tail = f"            {label}" if i == 0 else ""
            lines.append(f"{v:5.2f}     {z:7.2f}  {d:7.3f}{tail}")
        return lines
    txt = [title[:40]]
    txt += block(vp, vdamp_p, "P-VELOCITY MODEL")
    txt += block(vs, vdamp_s, "S-VELOCITY MODEL")
    path.write_text("\n".join(txt) + "\n")


# --------------------------------------------------------------------------- #
# control file
# --------------------------------------------------------------------------- #
CMN = """******* CONTROL-FILE FOR PROGRAM  V E L E S T  *******
***
*** next line contains a title (printed on output):
{title}
***  olat       olon   icoordsystem      zshift   itrial ztrial    ised
{olat:10.4f} {olon:10.4f}      0            0.000      0     0.00       0
***
*** neqs   nshot   rotate
    {neqs:4d}      0      0.0
***
*** isingle   iresolcalc
       0          {iresolcalc}
***
*** dmax    itopo    zmin     veladj    zadj   lowveloclay
   {dmax:6.1f}     0     {zmin:6.2f}     0.20    5.00       {lowveloclay}
***
*** nsp    swtfac   vpvs       nmod
     {nsp}      {swtfac:4.2f}    {vpvs:5.3f}        {nmod}
***
***   othet   xythet    zthet    vthet   stathet
     {othet:.5g}    {xythet:.5g}      {zthet:.5g}     {vthet:.5g}    {stathet:.5g}
***
*** nsinv   nshcor   nshfix     iuseelev    iusestacorr
       {nsinv}       0       0           1            {iusestacorr}
***
*** iturbo    icnvout   istaout   ismpout
       1         1         2         0
***
*** irayout   idrvout   ialeout   idspout   irflout   irfrout   iresout
       0         0         0         0         0         0         0
***
*** delmin   ittmax   invertratio
    0.010      {ittmax:02d}          1
***
*** Modelfile:
{modfile}
***
*** Stationfile:
{stafile}
***
*** Seismofile:

***
*** File with region names:

***
*** File with region coordinates:

***
*** File #1 with topo data:

***
*** File #2 with topo data:

***
*** DATA INPUT files:
***
*** File with Earthquake data:
{cnvfile}
***
*** File with Shot data:

***
*** OUTPUT files:
***
*** Main print output file:
velest.OUT
***
*** File with single event locations:

***
*** File with final hypocenters in *.cnv format:
final.CNV
***
*** File with new station corrections:
final.STA
***
******* END OF THE CONTROL-FILE FOR PROGRAM  V E L E S T  *******
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalogs/nlloc_year_v5_strict.csv")
    ap.add_argument("--standard", default="catalogs/nlloc_year_v5_standard.csv")
    ap.add_argument("--picks", default="catalogs/pyocto_picks_year_newpool_no_shots.csv")
    ap.add_argument("--out", default="velest/input")
    ap.add_argument("--dmax", type=float, default=150.0, help="max epicentral distance, km")
    ap.add_argument("--vpvs", type=float, default=1.78)
    ap.add_argument("--prob-good", type=float, default=0.30,
                    help="picks at or above this probability get VELEST weight class 0, else 1")
    ap.add_argument("--max-obs", type=int, default=100)
    ap.add_argument("--subset-n", type=int, default=400,
                    help="events in the SIMULTANEOUS-inversion subset (Kissling recipe)")
    ap.add_argument("--subset-cell-km", type=float, default=1.5)
    ap.add_argument("--subset-per-cell", type=int, default=6)
    args = ap.parse_args()

    assert_nanosecond_sanity()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- catalogue ----------------
    cat = load_catalog(REPO / args.catalog, REPO / args.standard)
    if str(cat.depth_datum.iloc[0]) != "sealevel":
        raise SystemExit(f"expected a sealevel datum, got {cat.depth_datum.iloc[0]!r}")
    cat = cat[["event_idx", "origin_time", "lat", "lon", "depth_km"]].copy()
    cat["ot"] = epoch_seconds(cat.origin_time)
    cat = cat.sort_values("ot").reset_index(drop=True)

    # ---------------- stations ----------------
    st = pd.read_csv(REPO / "catalogs" / "station_geometry.csv")
    st["key"] = st.network + "." + st.station
    st["alias"] = [station_alias(n, s) for n, s in zip(st.network, st.station)]
    if st.alias.duplicated().any():
        raise SystemExit(f"station alias collision: {st[st.alias.duplicated(keep=False)].alias.tolist()}")
    # true depth below sea level; OBS carry negative elevation_m
    st["z_bsl_km"] = np.where(st.on_seafloor, st.water_depth_m / 1000.0,
                              -st.elevation_m / 1000.0)
    if st.z_bsl_km.min() < LAYER_TOPS[0]:
        raise SystemExit(f"station above the model top: {st.z_bsl_km.min()} < {LAYER_TOPS[0]}")

    # ---------------- picks ----------------
    pk = pd.read_csv(REPO / args.picks)
    pk = pk[pk.event_idx.isin(set(cat.event_idx))]
    pk = pk[pk.phase.isin(["P", "S"])]
    pk = pk.drop_duplicates(["event_idx", "station", "phase"])
    m = pk.merge(cat[["event_idx", "lat", "lon", "ot"]], on="event_idx") \
          .merge(st[["key", "alias", "latitude", "longitude"]],
                 left_on="station", right_on="key")
    m["tt"] = m.time - m.ot
    m["epi_km"] = great_circle_km(m.lat, m.lon, m.latitude, m.longitude)
    n0 = len(m)
    m = m[(m.epi_km <= args.dmax) & (m.tt > -2.0) & (m.tt < 400.0)]
    print(f"  picks: {n0:,} -> {len(m):,} after dmax {args.dmax} km and travel-time sanity")
    m["iwt"] = np.where(m.prob >= args.prob_good, 0, 1)
    m = m.sort_values(["event_idx", "epi_km", "phase"])
    m = m.groupby("event_idx").head(args.max_obs)

    keep = m.groupby("event_idx").size()
    keep = set(keep[keep >= 4].index)
    cat = cat[cat.event_idx.isin(keep)].reset_index(drop=True)
    m = m[m.event_idx.isin(keep)]
    print(f"  events with >=4 usable observations: {len(cat):,}   observations {len(m):,} "
          f"(P {(m.phase == 'P').sum():,} / S {(m.phase == 'S').sum():,})")

    # ---------------- inversion subset ----------------
    # VELEST's simultaneous mode solves one single-precision normal system of
    # 4*neqs + nlayers + nstacorr unknowns.  At 1,576 events (6,374 unknowns) the
    # solve went unstable (NaN trial steps, "4 TIMES BACKUP MADE", ludecp
    # ier=129).  Kissling's recipe is to derive the minimum 1D model from a
    # well-recorded, well-distributed SUBSET and then relocate the full
    # catalogue with it -- which is what we do.
    nobs = m.groupby("event_idx").size().rename("n_obs")
    nsta_ps = m.groupby(["event_idx", "alias"]).phase.nunique()
    n_ps = (nsta_ps == 2).groupby("event_idx").sum().rename("n_ps_sta")
    sel = cat.merge(nobs, on="event_idx").merge(n_ps, on="event_idx")
    sel = sel[(sel.n_obs >= 12) & (sel.n_ps_sta >= 4)]
    ykm = (sel.lat - cat.lat.median()) * 111.2
    xkm = (sel.lon - cat.lon.median()) * 111.2 * np.cos(np.radians(cat.lat.median()))
    sel = sel.assign(
        cx=np.floor(xkm / args.subset_cell_km), cy=np.floor(ykm / args.subset_cell_km),
        cz=np.floor(sel.depth_km / args.subset_cell_km))
    sel = (sel.sort_values(["n_ps_sta", "n_obs"], ascending=False)
              .groupby(["cx", "cy", "cz"]).head(args.subset_per_cell))
    # Depth-stratified: events with many observations are systematically the
    # deeper ones, so a straight "best N" subset would leave the shallow layers
    # -- the ones under test -- unsampled.  Match the strict tier's own depth
    # histogram instead.
    dbins = [0, 2, 3, 4, 6, 9, 15, 100]
    want = (pd.cut(cat.depth_km, dbins).value_counts(normalize=True) * args.subset_n)
    sel = sel.assign(dbin=pd.cut(sel.depth_km, dbins))
    sel = (sel.sort_values(["n_ps_sta", "n_obs"], ascending=False)
              .groupby("dbin", observed=True, group_keys=False)
              .apply(lambda g: g.head(int(round(want.get(g.name, 0))))))
    sub_ids = set(sel.event_idx)
    print(f"  inversion subset: {len(sub_ids):,} events "
          f"(n_obs >= 12 and >= 4 stations with both P and S, "
          f"<= {args.subset_per_cell} per {args.subset_cell_km} km cell); "
          f"median n_obs {sel.n_obs.median():.0f}, depth p10/50/90 "
          f"{'/'.join(f'{v:.2f}' for v in np.percentile(sel.depth_km, [10, 50, 90]))} km BSL")

    # ---------------- .cnv ----------------
    obs = {k: v for k, v in m.groupby("event_idx")}
    lines, order = [], []
    sub_lines, sub_order = [], []
    for _, ev in cat.iterrows():
        t = pd.Timestamp(ev.ot, unit="s", tz="UTC")
        sec = t.second + t.microsecond / 1e6
        la, lo = abs(ev.lat), abs(ev.lon)
        lines.append(f"{t.year % 100:02d}{t.month:2d}{t.day:2d} {t.hour:2d}{t.minute:2d} "
                     f"{sec:5.2f} {la:7.4f}{'S' if ev.lat < 0 else 'N'} "
                     f"{lo:8.4f}{'W' if ev.lon < 0 else 'E'} {ev.depth_km:7.2f}   0.00")
        g = obs[ev.event_idx]
        cells = [f"{a:<4s}{p:1s}{w:1d}{tt:6.2f}"
                 for a, p, w, tt in zip(g.alias, g.phase, g.iwt, g.tt)]
        block = [lines[-1]] + ["".join(cells[i:i + 6]) for i in range(0, len(cells), 6)] + [""]
        lines.extend(block[1:])
        order.append(ev.event_idx)
        if ev.event_idx in sub_ids:
            sub_lines.extend(block)
            sub_order.append(ev.event_idx)
    lines.append("9999")
    sub_lines.append("9999")
    (out / "orca.cnv").write_text("\n".join(lines) + "\n")
    (out / "orca_sub.cnv").write_text("\n".join(sub_lines) + "\n")
    pd.DataFrame({"event_idx": order}).to_csv(out / "event_order.csv", index=False)
    pd.DataFrame({"event_idx": sub_order}).to_csv(out / "event_order_sub.csv", index=False)

    # ---------------- .sta ----------------
    # Reference station (icc = nsta) has its P correction held fixed; pick the
    # OBS with the most observations so the datum is the best-determined station.
    counts = m.alias.value_counts()
    used = st[st.alias.isin(counts.index)].copy()
    used["n"] = used.alias.map(counts)
    ref = used.sort_values("n", ascending=False).alias.iloc[0]
    print(f"  reference station (P correction fixed at 0): {ref}  ({counts[ref]:,} obs)")
    ordered = [a for a in used.alias if a != ref] + [ref]
    fmt = "(a4,f7.4,a1,1x,f8.4,a1,1x,i5,1x,i1,1x,i3,1x,f5.2,2x,f5.2)"
    slines = [fmt]
    for icc, alias in enumerate(ordered, start=1):
        r = used[used.alias == alias].iloc[0]
        elev = int(round(-r.z_bsl_km * 1000.0))       # VELEST z = -elev/1000
        slines.append(f"{alias:<4s}{abs(r.latitude):7.4f}{'S' if r.latitude < 0 else 'N'} "
                      f"{abs(r.longitude):8.4f}{'W' if r.longitude < 0 else 'E'} "
                      f"{elev:5d} 1 {icc:3d}  0.00   0.00")
    slines.append("")
    (out / "orca.sta").write_text("\n".join(slines) + "\n")

    # ---------------- .mod ----------------
    vm = pd.read_csv(REPO / "configs" / "velocity_model.csv")
    vp = layer_velocities(vm, LAYER_TOPS, "vp_kms")
    vs = [v / args.vpvs for v in vp]
    # Layer 1 is above every OBS and is sampled only by the 379 land-station
    # picks; freeze it (999) and let the land station corrections absorb it.
    # Layers below 21 km are below the deepest strict-tier event; freeze them.
    dmp_p = [999.0 if (t < 0.7 or t >= 21.0) else 1.0 for t in LAYER_TOPS]
    dmp_s = list(dmp_p)
    write_model(out / "orca.mod", LAYER_TOPS, vp, vs, dmp_p, dmp_s,
                " ORCA start (rock-only, sea-level datum)")
    for tag, f in (("slow", 0.90), ("fast", 1.10)):
        write_model(out / f"orca_{tag}.mod", LAYER_TOPS,
                    [v * f for v in vp], [v * f / args.vpvs for v in vp],
                    dmp_p, dmp_s, f" ORCA start x{f:.2f} ({tag})")

    print("\n  start model (rock only, sea-level datum):")
    print(f"    {'top km BSL':>11}{'Vp':>8}{'Vs':>8}{'Vp/Vs':>8}{'vdamp':>8}")
    for t, a, b, d in zip(LAYER_TOPS, vp, vs, dmp_p):
        print(f"    {t:>11.2f}{a:>8.3f}{b:>8.3f}{a / b:>8.3f}{d:>8.0f}")

    # ---------------- velest.cmn template ----------------
    (out / "cmn_template.txt").write_text(CMN)
    meta = dict(olat=float(cat.lat.median()), olon=float(-cat.lon.median()),
                neqs=len(cat), neqs_sub=len(sub_order), dmax=args.dmax, vpvs=args.vpvs)
    pd.Series(meta).to_json(out / "meta.json")
    print(f"\n  wrote {out}/orca.cnv (.sta, .mod, orca_slow.mod, orca_fast.mod, meta.json)")
    print(f"  SDC origin olat {meta['olat']:.4f} (S negative)  olon {meta['olon']:.4f} (west positive)")


if __name__ == "__main__":
    main()
