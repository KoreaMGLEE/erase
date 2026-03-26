"""Plan 007 Final Figure 4:
Base = Human Easy 전체 (464개)의 지식 유형 분포
Compare = HE-MH (40개)가 base 대비 얼마나 skew되는가
Diverging bar chart with clean label placement.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

FIG_DIR = "/workspace/erase/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Full manual coding results (465→464 with confidence)
BASE = {"S": 129, "F": 316, "P": 19}   # Human-Easy 전체, n=464
HE_MH = {"S": 13, "F": 20, "P": 7}    # Human-Easy Model-Hard, n=40

n_base = sum(BASE.values())  # 464
n_hemh = sum(HE_MH.values())  # 40

base_pct = {k: v / n_base * 100 for k, v in BASE.items()}
hemh_pct = {k: v / n_hemh * 100 for k, v in HE_MH.items()}
delta = {k: hemh_pct[k] - base_pct[k] for k in BASE}

LABELS = ["Procedural", "Factual", "Situational"]
KEYS = ["P", "F", "S"]  # P on top (biggest positive), F middle (biggest negative), S bottom

fig, ax = plt.subplots(figsize=(5, 2.5))

y = np.arange(len(LABELS))
deltas = [delta[k] for k in KEYS]

family_colors = {"over": "#FF9800", "under": "#78909C"}
colors = [family_colors["over"] if d > 0 else family_colors["under"] for d in deltas]

bars = ax.barh(y, deltas, 0.55, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

# Annotate: delta pp right at bar end, detail percentages right-aligned as separate column
right_edge = 22  # fixed x for right-aligned detail column
for i, k in enumerate(KEYS):
    d = deltas[i]
    actual = hemh_pct[k]
    base = base_pct[k]

    # Delta pp label: right at bar tip, small
    if d >= 0:
        ax.text(d + 0.4, y[i], f"{d:+.0f}pp", ha="left", va="center", fontsize=7,
                color="#555555")
    else:
        ax.text(d - 0.4, y[i], f"{d:+.0f}pp", ha="right", va="center", fontsize=7,
                color="#555555")

    pass  # detail column removed

ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(LABELS, fontsize=9, fontweight="medium")
ax.set_xlabel("Difference from Human-Easy base rate (pp)", fontsize=8, labelpad=8)
ax.set_xlim(-21.5, 18)
ax.tick_params(axis='x', labelsize=8)
ax.grid(True, alpha=0.2, axis="x")

ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Subtitle
# ax.text(0.5, -0.3,
#         f"Model-hard subset (n={n_hemh})  vs  Human-Easy base rate (n={n_base})",
#         transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#666666")

# Legend above figure
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=family_colors["over"], alpha=0.85, label="Over-represented"),
    Patch(facecolor=family_colors["under"], alpha=0.85, label="Under-represented"),
]
ax.legend(handles=legend_elements, fontsize=7, loc="lower center",
          bbox_to_anchor=(0.5, 1.07), ncol=2, frameon=True,
          facecolor="white", edgecolor="#cccccc", framealpha=1.0)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_knowledge_type.pdf"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "fig4_knowledge_type.png"), dpi=200, bbox_inches="tight")
print(f"Saved: {FIG_DIR}/fig4_knowledge_type.png")

# Print summary
print(f"\nBase (n={n_base}): S={base_pct['S']:.1f}%, F={base_pct['F']:.1f}%, P={base_pct['P']:.1f}%")
print(f"HE-MH (n={n_hemh}): S={hemh_pct['S']:.1f}%, F={hemh_pct['F']:.1f}%, P={hemh_pct['P']:.1f}%")
print(f"Delta: S={delta['S']:+.1f}pp, F={delta['F']:+.1f}pp, P={delta['P']:+.1f}pp")
