"""Train a semi-supervised binary shot/EQ classifier on per-event log-power
spectra.

Label construction:
    Known SHOT    : ±5 s temporal-flagged events in Jan 21 - Feb 4 (from
                    flag_shot column in catalogs/pyocto_events_picker_only_with_shot_flag.csv).
    Known EQ      : Events with origin_time < 2019-01-21 OR >= 2019-02-05.
    Unlabeled     : Unflagged events in Jan 21 - Feb 4 (gray zone).

Model: sklearn HistGradientBoostingClassifier on the 151-D event spectra.

Outputs:
    catalogs/pyocto_events_picker_only_with_shot_flag_v2.csv
        adds columns: prob_shot, flag_shot_v2
    notes/figures/shot_classifier_diagnostics.png   (held-out confusion + ROC)
    notes/figures/event_spectra_v2_means.png        (mean spectra by class)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             classification_report)
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent

SHOT_WINDOW_START = pd.Timestamp("2019-01-21", tz="UTC")
SHOT_WINDOW_END   = pd.Timestamp("2019-02-05", tz="UTC")
PROB_THRESHOLD = 0.5


def main():
    print("Loading event spectra ...")
    spectra = np.load(REPO / "catalogs" / "event_spectra.npy")
    meta = pd.read_parquet(REPO / "catalogs" / "event_spectra_meta.parquet")
    print(f"  spectra: {spectra.shape}, meta: {len(meta):,} rows")

    print("Loading event flags ...")
    ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag.csv")
    ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True,
                                       format="mixed", errors="coerce")
    # Join spectra to events via event_idx
    ev = ev.merge(meta, on="event_idx", how="left")
    # Build feature matrix in the same order as ev
    # meta is ordered by event_idx; build a lookup
    eid_to_row = {int(eid): i for i, eid in enumerate(meta["event_idx"].values)}
    feat_rows = np.array([eid_to_row.get(int(e), -1) for e in ev["event_idx"]])
    has_spec = (feat_rows >= 0) & (ev["n_picks_used"].fillna(0).astype(int) > 0)
    print(f"  events with valid spectra: {has_spec.sum():,} / {len(ev):,}")

    in_shot_window = (ev.origin_time >= SHOT_WINDOW_START) & (ev.origin_time < SHOT_WINDOW_END)
    is_temporal_shot = ev["flag_shot"].astype(bool) & in_shot_window

    # Label assignment
    label = np.full(len(ev), -1, dtype=int)   # -1 = unlabeled
    # KNOWN EQ: outside shot window
    label[(~in_shot_window) & has_spec] = 0
    # KNOWN SHOT: temporal flag inside shot window
    label[is_temporal_shot & has_spec] = 1

    print(f"  label counts: -1 (unlabeled)={(label == -1).sum():,}  "
          f"0 (known EQ)={(label == 0).sum():,}  "
          f"1 (known shot)={(label == 1).sum():,}")

    # Build feature matrix only for labeled rows
    train_mask = (label >= 0) & has_spec
    X_all = spectra[feat_rows[train_mask]]
    y_all = label[train_mask]

    # Held-out validation: 20% per class
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    print(f"  train: {len(y_tr):,}  held-out: {len(y_te):,}")

    print("\nTraining HistGradientBoostingClassifier ...")
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=8,
        class_weight="balanced", random_state=42, verbose=0)
    clf.fit(X_tr, y_tr)

    # Held-out diagnostics
    proba_te = clf.predict_proba(X_te)[:, 1]
    pred_te = (proba_te >= PROB_THRESHOLD).astype(int)
    auc = roc_auc_score(y_te, proba_te)
    cm = confusion_matrix(y_te, pred_te)
    print(f"\nHeld-out AUC: {auc:.4f}")
    print(f"Confusion matrix (rows=true, cols=pred):")
    print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
    print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")
    print(f"\n{classification_report(y_te, pred_te, target_names=['EQ', 'SHOT'])}")

    # Diagnostics plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_te, proba_te)
    ax1.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("Held-out ROC")
    ax1.legend(); ax1.grid(alpha=0.3)

    im = ax2.imshow(cm, cmap="Blues", aspect="equal")
    for (i, j), v in np.ndenumerate(cm):
        ax2.text(j, i, f"{v}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black", fontsize=14)
    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["pred EQ", "pred SHOT"])
    ax2.set_yticklabels(["true EQ", "true SHOT"])
    ax2.set_title(f"Confusion matrix (threshold = {PROB_THRESHOLD})")
    fig.suptitle(f"Shot classifier diagnostics — HGBT on 151-D log-power spectra "
                 f"(n_train={len(y_tr):,}, n_test={len(y_te):,})",
                 fontsize=12)
    plt.tight_layout()
    out_diag = REPO / "notes" / "figures" / "shot_classifier_diagnostics.png"
    plt.savefig(out_diag, dpi=140, bbox_inches="tight")
    print(f"wrote {out_diag}")

    # Predict on ALL events with spectra
    proba_all = np.full(len(ev), np.nan, dtype=float)
    has_spec_idx = np.where(has_spec)[0]
    proba_all[has_spec_idx] = clf.predict_proba(spectra[feat_rows[has_spec]])[:, 1]

    # flag_shot_v2: temporal flag OR (in-window AND prob >= threshold)
    # Lock classifier verdict to in-window events; out-of-window stays False.
    in_window = in_shot_window.values
    flag_v2 = ev["flag_shot"].astype(bool).values.copy()
    classifier_shot = in_window & (proba_all >= PROB_THRESHOLD)
    flag_v2 = flag_v2 | classifier_shot
    print(f"\nFlag counts:")
    print(f"  temporal-only (±5 s):       {ev['flag_shot'].astype(bool).sum():,}")
    print(f"  classifier-added (in-win):  {(classifier_shot & ~ev['flag_shot'].astype(bool).values).sum():,}")
    print(f"  combined flag_shot_v2:      {flag_v2.sum():,}")

    out_csv = REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag_v2.csv"
    ev["prob_shot"] = proba_all
    ev["flag_shot_v2"] = flag_v2
    ev.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # Mean-spectrum plot for sanity
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    freq = np.linspace(0, 50, spectra.shape[1])
    mean_shot = spectra[feat_rows[(flag_v2) & has_spec]].mean(axis=0)
    mean_eq   = spectra[feat_rows[(~flag_v2) & has_spec]].mean(axis=0)
    ax.plot(freq, mean_shot, color="red", linewidth=2,
            label=f"flag_shot_v2=True ({int(((flag_v2) & has_spec).sum()):,})")
    ax.plot(freq, mean_eq, color="steelblue", linewidth=2,
            label=f"flag_shot_v2=False ({int(((~flag_v2) & has_spec).sum()):,})")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean log-power (normalized)")
    ax.set_title("Mean event spectrum by v2 class")
    ax.legend(); ax.grid(alpha=0.3)
    out_spec = REPO / "notes" / "figures" / "event_spectra_v2_means.png"
    plt.savefig(out_spec, dpi=140, bbox_inches="tight")
    print(f"wrote {out_spec}")


if __name__ == "__main__":
    main()
