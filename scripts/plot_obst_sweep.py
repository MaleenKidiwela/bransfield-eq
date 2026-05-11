"""OBSTransformer threshold sweep on a single day — recall + FP plot."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent

# Hard-coded from validation runs on 2019-12-26 (mag07).
sweep = [
    # thresh, total, P_TP, P_FP, P_FN, S_TP, S_FP, S_FN
    (0.1, 70480, 5, 4991, 6, 12, 12334, 0),
    (0.3, 36581, 4, 1912, 7, 12,  7241, 0),
    (0.5, 20919, 3, 1072, 8, 12,  4198, 0),
    (0.7, 12687, 3,  534, 8, 11,  2700, 1),
]

thr   = [r[0] for r in sweep]
tot   = [r[1] for r in sweep]
p_rec = [r[2]/(r[2]+r[4]) for r in sweep]
s_rec = [r[5]/(r[5]+r[7]) for r in sweep]
p_fp  = [r[3] for r in sweep]
s_fp  = [r[6] for r in sweep]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(thr, p_rec, "o-", color="red",  lw=2, label="P recall")
ax.plot(thr, s_rec, "o-", color="blue", lw=2, label="S recall")
for x, y in zip(thr, p_rec):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=8, color="red")
for x, y in zip(thr, s_rec):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=8, color="blue")
ax.set_xlabel("Probability threshold")
ax.set_ylabel("Recall vs mag07 manual picks")
ax.set_ylim(0, 1.1)
ax.set_xticks(thr)
ax.set_title("OBSTransformer recall vs threshold (2019-12-26, 11 P + 12 S manual)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(thr, tot,  "o-", color="black", lw=2, label="all picks (sum P+S)")
ax.plot(thr, p_fp, "s--", color="red",   label="P FP")
ax.plot(thr, s_fp, "^--", color="blue",  label="S FP")
for x, y in zip(thr, tot):
    ax.annotate(f"{y:,}", (x, y), textcoords="offset points",
                xytext=(6, 4), fontsize=8)
ax.set_xlabel("Probability threshold")
ax.set_ylabel("Pick count (log scale)")
ax.set_xticks(thr)
ax.set_yscale("log")
ax.set_title("OBSTransformer pick volume vs threshold")
ax.legend()
ax.grid(alpha=0.3, which="both")

fig.suptitle("OBSTransformer threshold sweep — Bransfield OBS, 2019-12-26",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
out = REPO / "figures" / "picker_comparison_2019-12-26" / "obst_threshold_sweep.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
