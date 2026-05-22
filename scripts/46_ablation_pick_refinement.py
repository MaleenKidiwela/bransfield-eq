"""Pick-refinement ablation (50-event sanity test).

For N random events from the reliable v2 catalog:
  1. Use the NLLoc v2 hypocenter to predict tP/tS at every grid station.
  2. Build a refined pick set with gold-tier-relaxed logic from script 44
     (consensus PhaseNet/OBST within 0.5 s when both present, single-picker
     fallback when only one fires; S-P > 0.3 s and Vp/Vs in [1.4, 2.4]
     when both phases accepted). The refined set REPLACES the pyocto pick
     time where a gold pick exists at the same station-phase, and ADDS
     gold picks at stations the pyocto associator missed. Stations that
     pyocto picked but the picker CSVs do not corroborate keep the pyocto
     time.
  3. Write nlloc/obs/<label>.obs + event_order.csv.
  4. Re-run NLLoc on the same ORCA_v2 grids with the same control template.
  5. Join the new locations back to the baseline v2 reliable catalog and
     emit a per-event comparison CSV + a 4-panel diagnostic figure.

Usage:
    python scripts/46_ablation_pick_refinement.py --n-events 50 --seed 0
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

_s42_spec = importlib.util.spec_from_file_location(
    "_s42", REPO / "scripts" / "42_plot_event_picker_probs.py"
)
_s42 = importlib.util.module_from_spec(_s42_spec)
_s42_spec.loader.exec_module(_s42)

_s44_spec = importlib.util.spec_from_file_location(
    "_s44", REPO / "scripts" / "44_refine_picks_event.py"
)
_s44 = importlib.util.module_from_spec(_s44_spec)
_s44_spec.loader.exec_module(_s44)

REL_CSV    = REPO / "catalogs" / "nlloc_picker_only_no_shots_v2_reliable.csv"
PYO_EV_CSV = REPO / "catalogs" / "pyocto_events_picker_only_no_shots.csv"
PYO_PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv"
ST_CSV     = REPO / "catalogs" / "station_geometry.csv"
BASE_CTRL  = REPO / "nlloc" / "run" / "picker_only_no_shots_v2.in"

# Gold-tier-relaxed config (same as session-22 events 14341/14343).
P_WINDOW       = 1.5
S_WINDOW       = 2.5
CONSENSUS_TOL  = 0.5
MIN_PROB       = 0.15
REQUIRE_SP_GAP = 0.3
VPVS_LO, VPVS_HI = 1.4, 2.4
P_ERR_S = 0.1
S_ERR_S = 0.2


def gold_pick(per_phase: dict, phase: str) -> float | None:
    d = per_phase.get(phase, {})
    tn = d.get("PhaseNet", {}).get("dt")
    pn = d.get("PhaseNet", {}).get("prob")
    to_ = d.get("OBSTransformer", {}).get("dt")
    po_ = d.get("OBSTransformer", {}).get("prob")
    have_n = tn is not None and pn is not None and pn >= MIN_PROB
    have_o = to_ is not None and po_ is not None and po_ >= MIN_PROB
    if have_n and have_o and abs(tn - to_) <= CONSENSUS_TOL:
        return (pn * tn + po_ * to_) / (pn + po_)
    if have_n and have_o:
        return float(tn) if pn >= po_ else float(to_)
    if have_n:
        return float(tn)
    if have_o:
        return float(to_)
    return None


def refine_event(eidx, hypo_lat, hypo_lon, hypo_dep, ot, st_df, pyocto_picks_df,
                 corrections):
    pred = _s42.predict_arrivals(hypo_lat, hypo_lon, hypo_dep)
    if corrections:
        new_pred = {}
        for sta_bare, (tp, ts) in pred.items():
            full = None
            for (full_lbl, _ph) in corrections:
                if full_lbl.endswith("." + sta_bare):
                    full = full_lbl
                    break
            if full is None:
                new_pred[sta_bare] = (tp, ts)
                continue
            cp = corrections.get((full, "P"), 0.0)
            cs = corrections.get((full, "S"), 0.0)
            new_pred[sta_bare] = (tp + cp, ts + cs)
        pred = new_pred

    # pyocto times relative to OT, keyed by (full_station, phase)
    pyocto_lookup: dict[tuple[str, str], float] = {}
    for _, p in pyocto_picks_df.iterrows():
        pyocto_lookup[(p.station, p.phase)] = float(p.time) - ot.timestamp

    t0 = ot - 30
    t1 = ot + 30

    # Walk stations once; compute gold per station-phase.
    gold: dict[tuple[str, str], float] = {}
    for _, sr in st_df.iterrows():
        if sr.station not in pred:
            continue
        tp_pred, ts_pred = pred[sr.station]
        full = f"{sr.network}.{sr.station}"
        per_phase: dict[str, dict] = {}
        for picker_name, pdir in _s44.PICKER_DIRS.items():
            df_p = _s42.load_picker_picks(pdir, sr.network, sr.station, t0, t1)
            best_p = _s44.best_pick_in_window(df_p, "P", ot, tp_pred, P_WINDOW)
            best_s = _s44.best_pick_in_window(df_p, "S", ot, ts_pred, S_WINDOW)
            if best_p:
                per_phase.setdefault("P", {})[picker_name] = best_p
            if best_s:
                per_phase.setdefault("S", {})[picker_name] = best_s
        gp = gold_pick(per_phase, "P")
        gs = gold_pick(per_phase, "S")
        # Sanity gates when both present
        if gp is not None and gs is not None:
            if (gs - gp) < REQUIRE_SP_GAP:
                gs = None
            elif gp > 0:
                vpvs = gs / gp
                if not (VPVS_LO <= vpvs <= VPVS_HI):
                    gs = None
        if gp is not None:
            gold[(full, "P")] = gp
        if gs is not None:
            gold[(full, "S")] = gs

    # Merge: prefer gold, else pyocto.
    keys = set(pyocto_lookup) | set(gold)
    out_rows = []
    n_replaced = n_added = n_kept = 0
    for k in keys:
        if k in gold:
            dt_rel = gold[k]
            if k in pyocto_lookup:
                n_replaced += 1
            else:
                n_added += 1
        else:
            dt_rel = pyocto_lookup[k]
            n_kept += 1
        full, phase = k
        out_rows.append((full, phase, ot.timestamp + dt_rel))
    return out_rows, n_replaced, n_added, n_kept


def fmt_obs_line(bare_sta: str, dt: pd.Timestamp, phase: str, err_s: float) -> str:
    ymd = dt.strftime("%Y%m%d")
    hm = dt.strftime("%H%M")
    sec = dt.second + dt.microsecond * 1e-6
    return (
        f" {bare_sta:<7s} SP    Z E      {phase} ? {ymd} {hm} "
        f"{sec:7.4f} GAU  {err_s:.2e} -1.00e+00 -1.00e+00 -1.00e+00  1.00e+00"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="ablation_refined")
    ap.add_argument("--min-phases", type=int, default=4,
                    help="drop events whose final pick set falls below this")
    args = ap.parse_args()

    rel = pd.read_csv(REL_CSV)
    # Already strict-tier: on_boundary=False, in_hull=True. Extra safety filter:
    rel = rel[(~rel.on_boundary) & rel.in_hull].reset_index(drop=True)
    print(f"reliable+in_hull pool: {len(rel)}")
    sample = rel.sample(n=args.n_events, random_state=args.seed).sort_values(
        "origin_time").reset_index(drop=True)
    sample_idxs = sample.event_idx.astype(int).tolist()
    print(f"sampled {len(sample_idxs)} events; idx range "
          f"{min(sample_idxs)}..{max(sample_idxs)}")

    pyo_pk_all = pd.read_csv(PYO_PK_CSV)
    st_df = pd.read_csv(ST_CSV)

    corrections = _s44.load_station_corrections()

    obs_dir = REPO / "nlloc" / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    obs_path = obs_dir / f"{args.label}.obs"
    order_path = obs_dir / f"{args.label}.event_order.csv"

    summary = []
    with obs_path.open("w") as fh, order_path.open("w") as ofh:
        ofh.write("obs_order,event_idx\n")
        order_idx = 0
        for _, ev in sample.iterrows():
            eidx = int(ev.event_idx)
            ot = UTCDateTime(pd.Timestamp(ev.origin_time).to_pydatetime())
            pk_ev = pyo_pk_all[pyo_pk_all.event_idx == eidx]
            rows, n_repl, n_add, n_keep = refine_event(
                eidx, float(ev.lat), float(ev.lon), float(ev.depth_km),
                ot, st_df, pk_ev, corrections)
            if len(rows) < args.min_phases:
                print(f"  skip {eidx}: only {len(rows)} picks after refine")
                continue
            # Write obs block
            rows_sorted = sorted(rows, key=lambda r: r[2])
            for full, phase, abs_time in rows_sorted:
                bare = full.split(".", 1)[1]
                dt = pd.Timestamp(abs_time, unit="s", tz="UTC")
                err = P_ERR_S if phase == "P" else S_ERR_S
                fh.write(fmt_obs_line(bare, dt, phase, err) + "\n")
            fh.write("\n")
            ofh.write(f"{order_idx},{eidx}\n")
            order_idx += 1
            summary.append({"event_idx": eidx,
                            "n_pyocto": len(pk_ev),
                            "n_refined": len(rows),
                            "n_replaced": n_repl,
                            "n_added": n_add,
                            "n_kept": n_keep})
            if order_idx % 10 == 0:
                print(f"  refined {order_idx}/{len(sample_idxs)} events")

    sdf = pd.DataFrame(summary)
    print(f"\nwrote {obs_path}  ({len(sdf)} events)")
    print(f"      {order_path}")
    print("\nrefinement summary:")
    print(sdf.describe().to_string())

    sdf_path = REPO / "catalogs" / f"{args.label}_refinement_summary.csv"
    sdf.to_csv(sdf_path, index=False)
    print(f"      {sdf_path}")

    # Build a control file by copying the v2 template and rewriting obs path
    # + output root.
    base_text = BASE_CTRL.read_text()
    out_dir = REPO / "nlloc" / "output" / args.label
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = base_text.replace(
        "nlloc/obs/picker_only_no_shots_v2.obs",
        f"nlloc/obs/{args.label}.obs",
    ).replace(
        "nlloc/output/picker_only_no_shots_v2/loc",
        f"nlloc/output/{args.label}/loc",
    ).replace(
        "LOCCOM picker_only_no_shots_v2",
        f"LOCCOM {args.label}",
    )
    ctrl_path = REPO / "nlloc" / "run" / f"{args.label}.in"
    ctrl_path.write_text(text)
    print(f"      {ctrl_path}")
    print(f"\nNow run:  NLLoc nlloc/run/{args.label}.in")


if __name__ == "__main__":
    main()
