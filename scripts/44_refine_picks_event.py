"""One-event pick-refinement using NLLoc v2 predicted arrivals.

For a single event, predict tP/tS at every grid station, then search the
per-station-day pick CSVs (PhaseNet + OBSTransformer at the existing
threshold-0.1 catalog) for the highest-prob candidate within a window
around each prediction. Emits:

  - A table of accepted refined picks per station (P + S, per picker).
  - A figure overlaying:
      * waveform (gray)
      * existing pyocto picks (black solid/dashed)
      * predicted P/S (green/magenta dashed)
      * ACCEPTED refined picks: bold filled triangles at the accepted peak
        time, colored by picker.

This is the candidate-augmentation step that would feed a re-located NLLoc
run. Sub-threshold candidates are NOT visible (would require re-picking
the day at a lower threshold).

Usage:
    python scripts/44_refine_picks_event.py --event-idx 14341
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read, Stream

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
# Reuse the grid-prediction helpers from script 42
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_s42", REPO / "scripts" / "42_plot_event_picker_probs.py"
)
_s42 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s42)
predict_arrivals = _s42.predict_arrivals
haversine_km = _s42.haversine_km
load_z_trace = _s42.load_z_trace
load_picker_picks = _s42.load_picker_picks

WF_DIR = REPO / "data" / "waveforms"
ST_CSV = REPO / "catalogs" / "station_geometry.csv"
EV_CSV = REPO / "catalogs" / "pyocto_events_picker_only_no_shots.csv"
PK_CSV = REPO / "catalogs" / "pyocto_picks_picker_only_no_shots.csv"
NLLOC_CSV = REPO / "catalogs" / "nlloc_picker_only_no_shots_v2.csv"
OUT_DIR = REPO / "notes" / "figures" / "associations"

PICKER_DIRS = {
    "PhaseNet":       REPO / "catalogs" / "picks",
    "OBSTransformer": REPO / "catalogs" / "picks_obst_01",
}
STATION_CORR_CSV = REPO / "catalogs" / "station_corrections.csv"


def load_station_corrections(min_n: int = 20) -> dict[tuple[str, str], float]:
    """Return {(station, phase): median_residual_s}. Skip entries with
    fewer than `min_n` measurements (low confidence)."""
    if not STATION_CORR_CSV.exists():
        return {}
    df = pd.read_csv(STATION_CORR_CSV)
    df = df[df["n"] >= min_n]
    return {(r.station, r.phase): float(r.median_residual_s)
            for _, r in df.iterrows()}
PICKER_COLORS = {"PhaseNet": "#d62728", "OBSTransformer": "#ff8c00"}


def best_pick_in_window(df_picks: pd.DataFrame, phase: str,
                        ot: UTCDateTime, t_center: float, half_width: float
                        ) -> dict | None:
    """Return highest-prob pick of `phase` with peak time in
    [ot + t_center − hw, ot + t_center + hw]. None if no candidate."""
    if df_picks.empty:
        return None
    sub = df_picks[df_picks.phase == phase]
    if sub.empty:
        return None
    peak_dt = sub["time_obj"].map(lambda t: (UTCDateTime(t.to_pydatetime()) - ot))
    mask = (peak_dt >= t_center - half_width) & (peak_dt <= t_center + half_width)
    cand = sub[mask]
    if cand.empty:
        return None
    best = cand.loc[cand["prob"].idxmax()]
    return {
        "dt": float(peak_dt[best.name]),
        "prob": float(best["prob"]),
        "start": (UTCDateTime(best["start_obj"].to_pydatetime()) - ot),
        "end": (UTCDateTime(best["end_obj"].to_pydatetime()) - ot),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-idx", type=int, default=14341)
    ap.add_argument("--window-half-s", type=float, default=30.0)
    ap.add_argument("--band-low", type=float, default=2.0)
    ap.add_argument("--band-high", type=float, default=20.0)
    ap.add_argument("--p-window", type=float, default=None,
                    help="± seconds around predicted tP (default by tier)")
    ap.add_argument("--s-window", type=float, default=None,
                    help="± seconds around predicted tS (default by tier)")
    ap.add_argument("--min-prob", type=float, default=0.1,
                    help="floor on saved pick prob (catalog already ≥0.1)")
    ap.add_argument("--tier", choices=["gold", "standard"], default="gold")
    ap.add_argument("--consensus-tol", type=float, default=None,
                    help="max seconds between PN and OBST peaks (tier default)")
    ap.add_argument("--vpvs-lo", type=float, default=None)
    ap.add_argument("--vpvs-hi", type=float, default=None)
    ap.add_argument("--no-consensus", action="store_true",
                    help="accept single-picker picks (overrides tier default)")
    ap.add_argument("--no-corrections", action="store_true",
                    help="skip applying per-station timing corrections")
    args = ap.parse_args()

    # Tier defaults (relaxed gold; tune via flags above)
    if args.tier == "gold":
        p_window = args.p_window if args.p_window is not None else 1.5
        s_window = args.s_window if args.s_window is not None else 2.5
        require_consensus = not args.no_consensus
        consensus_tol = args.consensus_tol if args.consensus_tol is not None else 0.5
        min_prob = max(args.min_prob, 0.15)
        require_sp_gap = 0.3
        vpvs_lo = args.vpvs_lo if args.vpvs_lo is not None else 1.4
        vpvs_hi = args.vpvs_hi if args.vpvs_hi is not None else 2.4
    else:
        p_window = args.p_window if args.p_window is not None else 2.0
        s_window = args.s_window if args.s_window is not None else 3.0
        require_consensus = (not args.no_consensus) and False
        consensus_tol = args.consensus_tol if args.consensus_tol is not None else 0.7
        min_prob = args.min_prob
        require_sp_gap = 0.3
        vpvs_lo = args.vpvs_lo if args.vpvs_lo is not None else 1.3
        vpvs_hi = args.vpvs_hi if args.vpvs_hi is not None else 2.5
    print(f"  tier={args.tier}  P-window=±{p_window}s  S-window=±{s_window}s  "
          f"consensus={require_consensus}  min_prob={min_prob}")

    ev = pd.read_csv(EV_CSV)
    row = ev[ev.event_idx == args.event_idx]
    if row.empty:
        raise SystemExit(f"event_idx {args.event_idx} not found")
    ev_row = row.iloc[0]
    ot = UTCDateTime(float(ev_row.time))

    # Prefer NLLoc v2 hypocenter
    nl = pd.read_csv(NLLOC_CSV)
    m = nl[nl.event_idx == args.event_idx]
    if m.empty:
        raise SystemExit("NLLoc v2 location not found for this event")
    nr = m.iloc[0]
    hypo_lat, hypo_lon, hypo_dep = float(nr.lat), float(nr.lon), float(nr.depth_km)
    print(f"event {args.event_idx} @ {ot.isoformat()[:19]}")
    print(f"  NLLoc v2 hypo: lat={hypo_lat:.4f} lon={hypo_lon:.4f} "
          f"depth={hypo_dep:.3f} km  gap={float(nr.gap_deg):.0f}° "
          f"RMS={float(nr.rms_s):.2f}s")

    pred = predict_arrivals(hypo_lat, hypo_lon, hypo_dep)

    # Apply per-station timing corrections (shift the predicted arrival
    # so the search window is centered on the empirical mean residual).
    corrections = {} if args.no_corrections else load_station_corrections()
    if corrections:
        n_applied = 0
        new_pred = {}
        for sta_bare, (tp, ts) in pred.items():
            # Corrections keyed by full label (NET.STA) — find matching
            full = None
            for (full_lbl, ph) in corrections:
                if full_lbl.endswith("." + sta_bare):
                    full = full_lbl
                    break
            if full is None:
                new_pred[sta_bare] = (tp, ts)
                continue
            corr_p = corrections.get((full, "P"), 0.0)
            corr_s = corrections.get((full, "S"), 0.0)
            new_pred[sta_bare] = (tp + corr_p, ts + corr_s)
            if corr_p != 0.0 or corr_s != 0.0:
                n_applied += 1
        pred = new_pred
        print(f"  applied station corrections at {n_applied} stations")

    # Existing pyocto picks
    pk_all = pd.read_csv(PK_CSV)
    ev_pk = pk_all[pk_all.event_idx == args.event_idx]
    pyocto_lookup: dict[str, dict[str, float]] = {}
    for _, p in ev_pk.iterrows():
        pyocto_lookup.setdefault(p.station, {})[p.phase] = float(p.time) - ot.timestamp

    st = pd.read_csv(ST_CSV)
    st["dist_km"] = haversine_km(hypo_lat, hypo_lon,
                                 st.latitude.values, st.longitude.values)
    st = st.sort_values("dist_km").reset_index(drop=True)

    t0 = ot - args.window_half_s
    t1 = ot + args.window_half_s

    rows_out = []
    refined: dict[str, dict[str, dict]] = {}   # full_label -> phase -> {picker: pick}
    for _, sr in st.iterrows():
        if sr.station not in pred:
            continue
        tp_pred, ts_pred = pred[sr.station]
        full = f"{sr.network}.{sr.station}"
        already = pyocto_lookup.get(full, {})

        # Load picks once per picker
        rec = {"station": full, "dist_km": sr.dist_km,
               "tP_pred": tp_pred, "tS_pred": ts_pred,
               "pyocto_P": already.get("P"), "pyocto_S": already.get("S")}
        ph_rec: dict[str, dict] = {}
        for picker_name, pdir in PICKER_DIRS.items():
            df_p = load_picker_picks(pdir, sr.network, sr.station, t0, t1)
            best_p = best_pick_in_window(df_p, "P", ot, tp_pred, p_window)
            best_s = best_pick_in_window(df_p, "S", ot, ts_pred, s_window)
            rec[f"{picker_name}_P_dt"]   = best_p["dt"]   if best_p else None
            rec[f"{picker_name}_P_prob"] = best_p["prob"] if best_p else None
            rec[f"{picker_name}_S_dt"]   = best_s["dt"]   if best_s else None
            rec[f"{picker_name}_S_prob"] = best_s["prob"] if best_s else None
            if best_p: ph_rec.setdefault("P", {})[picker_name] = best_p
            if best_s: ph_rec.setdefault("S", {})[picker_name] = best_s
        rows_out.append(rec)
        refined[full] = ph_rec

    df = pd.DataFrame(rows_out)

    # ---- Gold-tier consensus filter -----------------------------------
    # Accept a phase pick at a station only when BOTH PhaseNet and OBST
    # have a candidate in the window AND their peak times agree within
    # consensus_tol. Final time = prob-weighted mean. Apply sanity gates.
    if args.tier == "gold":
        gold_accepted: dict[str, dict[str, dict]] = {}
        for _, r in df.iterrows():
            sta = r.station
            picks_here = {}
            for phase, win in (("P", p_window), ("S", s_window)):
                tn = r.get(f"PhaseNet_{phase}_dt")
                to = r.get(f"OBSTransformer_{phase}_dt")
                pn = r.get(f"PhaseNet_{phase}_prob")
                po = r.get(f"OBSTransformer_{phase}_prob")
                have_n = (not pd.isna(tn)) and (not pd.isna(pn)) and pn >= min_prob
                have_o = (not pd.isna(to)) and (not pd.isna(po)) and po >= min_prob
                if require_consensus:
                    if not (have_n and have_o):
                        continue
                    if abs(tn - to) > consensus_tol:
                        continue
                    dt_final = (pn * tn + po * to) / (pn + po)
                    picks_here[phase] = {
                        "dt": dt_final,
                        "prob_mean": (pn + po) / 2,
                    }
                else:
                    if have_n and have_o and abs(tn - to) <= consensus_tol:
                        dt_final = (pn * tn + po * to) / (pn + po)
                        prob = (pn + po) / 2
                    elif have_n and have_o:
                        # Disagreement → take higher-prob picker
                        if pn >= po:
                            dt_final, prob = float(tn), float(pn)
                        else:
                            dt_final, prob = float(to), float(po)
                    elif have_n:
                        dt_final, prob = float(tn), float(pn)
                    elif have_o:
                        dt_final, prob = float(to), float(po)
                    else:
                        continue
                    picks_here[phase] = {"dt": dt_final, "prob_mean": prob}
            # Sanity gates (prediction window + consensus already enforced;
            # add S–P spacing and Vp/Vs when both phases were accepted).
            if "P" in picks_here and "S" in picks_here:
                tp = picks_here["P"]["dt"]
                ts = picks_here["S"]["dt"]
                if (ts - tp) < require_sp_gap:
                    picks_here.pop("S", None)
                elif tp > 0:
                    vpvs = ts / tp
                    if not (vpvs_lo <= vpvs <= vpvs_hi):
                        picks_here.pop("S", None)
            if picks_here:
                gold_accepted[sta] = picks_here
        # Print gold-only table
        print()
        print("GOLD-tier accepted picks (consensus + sanity gates):")
        gold_rows = []
        for sta, ph in sorted(gold_accepted.items(),
                              key=lambda kv: df[df.station == kv[0]].iloc[0].dist_km):
            d = df[df.station == sta].iloc[0]
            tp_pred = float(d.tP_pred); ts_pred = float(d.tS_pred)
            tp = ph.get("P", {}).get("dt"); ts = ph.get("S", {}).get("dt")
            tp_prob = ph.get("P", {}).get("prob_mean")
            ts_prob = ph.get("S", {}).get("prob_mean")
            gold_rows.append({
                "station": sta, "dist_km": float(d.dist_km),
                "tP_pred": tp_pred, "tP_gold": tp, "P_res": (tp - tp_pred) if tp else None,
                "P_prob": tp_prob,
                "tS_pred": ts_pred, "tS_gold": ts, "S_res": (ts - ts_pred) if ts else None,
                "S_prob": ts_prob,
            })
        gold_df = pd.DataFrame(gold_rows)
        print(gold_df.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
        n_P = gold_df.tP_gold.notna().sum() if not gold_df.empty else 0
        n_S = gold_df.tS_gold.notna().sum() if not gold_df.empty else 0
        print(f"\n  GOLD picks accepted: {n_P} P  +  {n_S} S  "
              f"= {n_P + n_S} total")
        # Replace `refined` with only the gold-accepted picks so the
        # plot below shows the gold set.
        new_refined: dict = {}
        for sta, ph in gold_accepted.items():
            net = sta.split(".")[0]; sname = sta.split(".")[1]
            for phase, info in ph.items():
                # Mock the per-picker structure expected by the plot loop
                new_refined.setdefault(sta, {}).setdefault(phase, {})["GOLD"] = {
                    "dt": info["dt"], "prob": info["prob_mean"],
                    "start": info["dt"], "end": info["dt"],
                }
        refined = new_refined
        PICKER_COLORS["GOLD"] = "#ffd700"

    # Print summary table
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

    # Counts
    print()
    n_sta = len(df)
    n_py_P = df.pyocto_P.notna().sum()
    n_py_S = df.pyocto_S.notna().sum()
    n_pn_P = df.PhaseNet_P_dt.notna().sum()
    n_pn_S = df.PhaseNet_S_dt.notna().sum()
    n_ob_P = df.OBSTransformer_P_dt.notna().sum()
    n_ob_S = df.OBSTransformer_S_dt.notna().sum()
    n_consensus_P = ((df.PhaseNet_P_dt.notna()) & (df.OBSTransformer_P_dt.notna())).sum()
    n_consensus_S = ((df.PhaseNet_S_dt.notna()) & (df.OBSTransformer_S_dt.notna())).sum()
    print(f"  stations with grid prediction: {n_sta}")
    print(f"  pyocto      P/S:  {n_py_P}/{n_py_S}")
    print(f"  PhaseNet refined P/S: {n_pn_P}/{n_pn_S}")
    print(f"  OBST     refined P/S: {n_ob_P}/{n_ob_S}")
    print(f"  consensus (both)  P/S: {n_consensus_P}/{n_consensus_S}")
    # Net new picks (refined but pyocto missed)
    new_P = ((df.pyocto_P.isna()) & (df.PhaseNet_P_dt.notna() | df.OBSTransformer_P_dt.notna())).sum()
    new_S = ((df.pyocto_S.isna()) & (df.PhaseNet_S_dt.notna() | df.OBSTransformer_S_dt.notna())).sum()
    print(f"  NEW picks (pyocto missed but a picker found in window) P/S: {new_P}/{new_S}")

    # Figure
    rows_w = []
    for _, sr in st.iterrows():
        if sr.station not in pred:
            continue
        tr = load_z_trace(sr.network, sr.station, t0, t1, args.band_low, args.band_high)
        if tr is None:
            continue
        maxabs = float(np.max(np.abs(tr.data))) or 1.0
        rows_w.append((sr, tr, maxabs))

    n = len(rows_w)
    fig_h = max(5.0, 0.42 * n + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    yticks, ylabels = [], []
    for i, (sr, tr, maxabs) in enumerate(rows_w):
        t_rel = tr.times() - args.window_half_s
        y = tr.data / (2.4 * maxabs)
        y_offset = -i
        full = f"{sr.network}.{sr.station}"
        has_pyocto = full in pyocto_lookup
        ax.plot(t_rel, y + y_offset, color=("0.15" if has_pyocto else "0.55"),
                linewidth=0.5, zorder=4)
        yticks.append(y_offset)
        ylabels.append(f"{full}  {sr.dist_km:6.1f} km")

        # Refined accepted picks: solid vertical lines (P red, S blue)
        ph_rec = refined.get(full, {})
        for phase, by_pk in ph_rec.items():
            # All entries in by_pk are the same gold pick — take any
            info = next(iter(by_pk.values()))
            col = "red" if phase == "P" else "royalblue"
            ax.plot([info["dt"], info["dt"]],
                    [y_offset - 0.5, y_offset + 0.5],
                    color=col, linewidth=1.8, zorder=10)

    ax.axvline(0, color="goldenrod", linewidth=1.0, linestyle="--", zorder=3)
    ax.set_xlim(-args.window_half_s, args.window_half_s)
    ax.set_ylim(-(n - 1) - 1.1, 1.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("seconds from event origin")
    ax.grid(alpha=0.3, axis="x")

    legend = [
        plt.Line2D([0], [0], color="red", lw=1.8, label="refined P"),
        plt.Line2D([0], [0], color="royalblue", lw=1.8, label="refined S"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7, ncol=4,
              framealpha=0.9)
    ax.set_title(
        f"event {args.event_idx} refinement — {ot.isoformat()[:19]}\n"
        f"NLLoc v2 hypo ({hypo_lat:.3f}, {hypo_lon:.3f}, {hypo_dep:.2f} km)  "
        f"P-window ±{p_window}s  S-window ±{s_window}s  tier={args.tier}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ot.strftime("%Y%m%dT%H%M%S")
    out = OUT_DIR / f"event_{args.event_idx:06d}_{stamp}_refined.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
