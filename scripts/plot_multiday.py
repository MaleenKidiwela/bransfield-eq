"""Multi-day P/S recall comparison plot."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
df = pd.read_csv("/tmp/multiday_metrics.csv")
days = sorted(df.day.unique())

pn = df[df.picker=="PN_inst"].set_index("day").reindex(days)
ob = df[df.picker=="OBST_05"].set_index("day").reindex(days)

manual_p = (pn.P_TP + pn.P_FN).values
manual_s = (pn.S_TP + pn.S_FN).values

x = np.arange(len(days))
w = 0.38

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# P recall
ax = axes[0]
ax.bar(x - w/2, pn.P_rec, w, color="#1f77b4", label="PhaseNet instance @ 0.1")
ax.bar(x + w/2, ob.P_rec, w, color="#9467bd", label="OBSTransformer obst2024 @ 0.5")
for i, n in enumerate(manual_p):
    ax.text(i, 1.04, f"n={n}", ha="center", fontsize=8, color="0.4")
ax.set_ylabel("P recall vs mag07")
ax.set_ylim(0, 1.15)
ax.set_title("P recall, per day  (n = manual P picks that day)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3, axis="y")

# S recall
ax = axes[1]
ax.bar(x - w/2, pn.S_rec, w, color="#1f77b4", label="PhaseNet instance @ 0.1")
ax.bar(x + w/2, ob.S_rec, w, color="#9467bd", label="OBSTransformer obst2024 @ 0.5")
for i, n in enumerate(manual_s):
    ax.text(i, 1.04, f"n={n}", ha="center", fontsize=8, color="0.4")
ax.set_ylabel("S recall vs mag07")
ax.set_ylim(0, 1.15)
ax.set_title("S recall, per day  (n = manual S picks that day)")
ax.set_xticks(x)
ax.set_xticklabels(days, rotation=30, ha="right")
ax.legend(loc="lower right")
ax.grid(alpha=0.3, axis="y")

# Aggregate annotation
agg_pn_p = pn.P_TP.sum() / (pn.P_TP.sum() + pn.P_FN.sum())
agg_ob_p = ob.P_TP.sum() / (ob.P_TP.sum() + ob.P_FN.sum())
agg_pn_s = pn.S_TP.sum() / (pn.S_TP.sum() + pn.S_FN.sum())
agg_ob_s = ob.S_TP.sum() / (ob.S_TP.sum() + ob.S_FN.sum())

fig.suptitle(
    f"Picker comparison, 2019-02-04 → 2019-02-13  "
    f"(10 days, {manual_p.sum()} P + {manual_s.sum()} S manual picks, mag07)\n"
    f"Aggregate P-recall: PN={agg_pn_p:.3f}  OBST={agg_ob_p:.3f}    "
    f"Aggregate S-recall: PN={agg_pn_s:.3f}  OBST={agg_ob_s:.3f}",
    fontsize=11, y=0.995
)
fig.tight_layout()
out = REPO / "figures" / "multiday_2019-02-04_to_2019-02-13.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
