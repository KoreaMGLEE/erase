"""Fig 5 variant: Jaccard overlap vs threshold k% — anchor = BERT-large for all panels.
(a) BERT-large vs other BERT models
(b) BERT-large vs all T5 models
(c) BERT-large vs all Pythia models
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

FIG_DIR = "/workspace/erase/figures"
os.makedirs(FIG_DIR, exist_ok=True)

ANCHORS = [
    ("bert-large", "BERT-large"),
    ("bert-medium", "BERT-medium"),
    ("bert-mini", "BERT-mini"),
]

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]

BERT_MODELS = ["bert-mini", "bert-small", "bert-medium", "bert-base"]
T5_MODELS = ["t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl"]
PYTHIA_MODELS = [
    "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-2.8b", "pythia-12b",
]

FAMILY_MARKERS = {"BERT": "o", "T5": "s", "Pythia": "^"}

K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]

SHORT_NAMES = {
    "bert-mini": "B-mini", "bert-small": "B-sm", "bert-medium": "B-med",
    "bert-base": "B-base", "bert-large": "B-lg",
    "t5-v1_1-small": "T5-sm", "t5-v1_1-base": "T5-base",
    "t5-v1_1-large": "T5-lg", "t5-v1_1-xl": "T5-xl", "t5-v1_1-xxl": "T5-xxl",
    "pythia-14m": "P-14m", "pythia-31m": "P-31m", "pythia-70m": "P-70m",
    "pythia-160m": "P-160m", "pythia-410m": "P-410m", "pythia-1b": "P-1b",
    "pythia-1.4b": "P-1.4b", "pythia-2.8b": "P-2.8b", "pythia-6.9b": "P-6.9b",
    "pythia-12b": "P-12b",
}

# Color gradients: small→large = light→dark
BERT_COLORS = {
    "bert-mini":   "#b3d9f7",
    "bert-small":  "#7cbae8",
    "bert-medium": "#4196d4",
    "bert-base":   "#1a6fb5",
    "bert-large":  "#0e3f6e",
}

T5_COLORS = {
    "t5-v1_1-small": "#f7c5a0",
    "t5-v1_1-base":  "#f5a050",
    "t5-v1_1-large": "#e07030",
    "t5-v1_1-xl":    "#d63a2a",
    "t5-v1_1-xxl":   "#8b1a1a",
}

PYTHIA_COLORS = {
    "pythia-14m":  "#c8e8a0",
    "pythia-31m":  "#a8d84e",
    "pythia-70m":  "#80c840",
    "pythia-160m": "#60b030",
    "pythia-410m": "#48a028",
    "pythia-1b":   "#389020",
    "pythia-1.4b": "#2d8018",
    "pythia-2.8b": "#247010",
    "pythia-6.9b": "#1a5e0a",
    "pythia-12b":  "#104a04",
}

MODEL_COLORS = {**BERT_COLORS, **T5_COLORS, **PYTHIA_COLORS}

# ── MNLI config ──────────────────────────────────────────────────
MNLI_PATHS = {
    "bert-mini": "/workspace/erase/outputs/plan0/confidence/bert-mini_split{}/avg_conf.json",
    "bert-small": "/workspace/erase/outputs/plan0/confidence/bert-small_split{}/avg_conf.json",
    "bert-medium": "/workspace/erase/outputs/plan0/confidence/bert-medium_split{}/avg_conf.json",
    "bert-base": "/workspace/erase/outputs/plan0/confidence/bert-base_split{}/avg_conf.json",
    "bert-large": "/workspace/erase/outputs/plan0/confidence/bert-large_split{}/avg_conf.json",
    "t5-v1_1-small": "/workspace/erase/outputs/plan5/confidence/t5-v1_1-small_sentinel_split{}/avg_conf.json",
    "t5-v1_1-base": "/workspace/erase/outputs/plan5/confidence/t5-v1_1-base_sentinel_1e-4_split{}/avg_conf.json",
    "t5-v1_1-large": "/workspace/erase/outputs/plan5/confidence/t5-v1_1-large_sentinel_split{}/avg_conf.json",
    "t5-v1_1-xl": "/workspace/erase/outputs/plan5/confidence/t5-v1_1-xl_sentinel_split{}/avg_conf.json",
    "t5-v1_1-xxl": "/workspace/erase/outputs/plan5/confidence/t5-v1_1-xxl_sentinel_split{}/avg_conf.json",
    "pythia-14m": "/workspace/erase/outputs/plan5/confidence/pythia-14m_split{}/avg_conf.json",
    "pythia-31m": "/workspace/erase/outputs/plan5/confidence/pythia-31m_split{}/avg_conf.json",
    "pythia-70m": "/workspace/erase/outputs/plan0/confidence/pythia-70m_split{}/avg_conf.json",
    "pythia-160m": "/workspace/erase/outputs/plan0/confidence/pythia-160m_split{}/avg_conf.json",
    "pythia-410m": "/workspace/erase/outputs/plan0/confidence/pythia-410m_split{}/avg_conf.json",
    "pythia-1b": "/workspace/erase/outputs/plan0/confidence/pythia-1b_split{}/avg_conf.json",
    "pythia-1.4b": "/workspace/erase/outputs/plan0/confidence/pythia-1.4b_split{}/avg_conf.json",
    "pythia-2.8b": "/workspace/erase/outputs/plan0/confidence/pythia-2.8b_split{}/avg_conf.json",
    "pythia-6.9b": "/workspace/erase/outputs/plan0/confidence/pythia-6.9b_split{}/avg_conf.json",
    "pythia-12b": "/workspace/erase/outputs/plan5/confidence/pythia-12b_split{}/avg_conf.json",
}
MNLI_SPLITS = [1, 2, 3]

# ── Helpers ──────────────────────────────────────────────────────

def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = max(1, int(len(items) * k_pct / 100))
    return set(item[0] for item in items[:n])


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0


def load_mnli_confs():
    all_confs = {}
    for model in ALL_MODELS:
        splits = {}
        for s in MNLI_SPLITS:
            path = MNLI_PATHS[model].format(s)
            if os.path.exists(path):
                with open(path) as f:
                    splits[s] = json.load(f)
        if splits:
            all_confs[model] = splits
            print(f"  {model}: {len(splits)} splits")
    return all_confs


def compute_model_curve(all_confs, anchor, model, k_values):
    if anchor not in all_confs or model not in all_confs:
        return None
    anchor_runs = all_confs[anchor]
    model_runs = all_confs[model]
    shared = set(anchor_runs.keys()) & set(model_runs.keys())

    curve = []
    for k_pct in k_values:
        if shared:
            jacs = []
            for rk in sorted(shared):
                common = set(anchor_runs[rk].keys()) & set(model_runs[rk].keys())
                if common:
                    ca = {k: v for k, v in anchor_runs[rk].items() if k in common}
                    cm = {k: v for k, v in model_runs[rk].items() if k in common}
                    jacs.append(jaccard(get_easy_set(ca, k_pct), get_easy_set(cm, k_pct)))
        else:
            jacs = []
            for rk_a in anchor_runs:
                for rk_m in model_runs:
                    common = set(anchor_runs[rk_a].keys()) & set(model_runs[rk_m].keys())
                    if common:
                        ca = {k: v for k, v in anchor_runs[rk_a].items() if k in common}
                        cm = {k: v for k, v in model_runs[rk_m].items() if k in common}
                        jacs.append(jaccard(get_easy_set(ca, k_pct), get_easy_set(cm, k_pct)))
        curve.append((float(np.mean(jacs)), float(np.std(jacs))) if jacs else (0.0, 0.0))
    return curve


# ── Plotting ─────────────────────────────────────────────────────

def set_pub_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.minor.width": 0.3, "ytick.minor.width": 0.3,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.direction": "in", "ytick.direction": "in",
    })


def draw_panel(ax, all_confs, anchor, models, family, label_fs, tick_fs):
    """Draw one panel: anchor vs all models in the given list."""
    x_positions = np.array(K_VALUES)
    marker = FAMILY_MARKERS[family]
    ms = 5.5 if family == "Pythia" else 4.5

    # Plot large→small so legend order = large first
    for model in reversed(models):
        if model == anchor:
            continue
        curve = compute_model_curve(all_confs, anchor, model, K_VALUES)
        if curve is None:
            continue
        ys = np.array([c[0] for c in curve])
        color = MODEL_COLORS[model]
        ax.plot(x_positions, ys, color=color, solid_capstyle="round",
                marker=marker, markersize=ms,
                linewidth=1.2, label=SHORT_NAMES[model],
                markeredgecolor="white", markeredgewidth=0.3,
                markerfacecolor=color, zorder=4)

    ax.yaxis.grid(True, linewidth=0.2, color="#cccccc", alpha=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Top k%", fontsize=label_fs, labelpad=2)
    major_ticks = [1, 5, 10, 20, 50]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(k) for k in major_ticks], fontsize=tick_fs)
    minor_ticks = [k for k in K_VALUES if k not in major_ticks]
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=tick_fs, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.8, 60)


def draw_combined_figure(mnli_confs):
    set_pub_style()

    title_fs = 7.5
    label_fs = 6.5
    tick_fs = 5.5
    legend_fs = 4.5

    all_bert = ["bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large"]
    col_families = [
        ("BERT", "vs BERT family"),
        ("T5", "vs T5 v1.1 family"),
        ("Pythia", "vs Pythia family"),
    ]
    row_anchors = [
        ("bert-large", "BERT-large"),
        ("bert-medium", "BERT-medium"),
        ("bert-mini", "BERT-mini"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.0))

    for row, (anchor, anchor_short) in enumerate(row_anchors):
        bert_panel_models = [m for m in all_bert if m != anchor]
        panel_model_lists = [
            bert_panel_models,
            T5_MODELS,
            PYTHIA_MODELS,
        ]

        # Compute row-level y max
        row_max = 0
        for col in range(3):
            family = col_families[col][0]
            models = panel_model_lists[col]
            for model in models:
                if model == anchor:
                    continue
                curve = compute_model_curve(mnli_confs, anchor, model, K_VALUES)
                if curve:
                    row_max = max(row_max, max(c[0] for c in curve))
        y_top = np.ceil((row_max + 0.03) * 10) / 10

        for col in range(3):
            ax = axes[row, col]
            family = col_families[col][0]
            models = panel_model_lists[col]
            draw_panel(ax, mnli_confs, anchor, models, family, label_fs, tick_fs)
            ax.set_ylim(-0.02, y_top)

            # Column headers (top row only)
            if row == 0:
                ax.set_title(col_families[col][1], fontsize=title_fs, pad=6)

            # Row labels (left column only)
            if col == 0:
                ax.set_ylabel("Jaccard similarity", fontsize=label_fs, labelpad=2)
            else:
                ax.set_ylabel("")

            # x-axis label only on bottom row
            if row < 2:
                ax.set_xlabel("")

            # Legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=legend_fs, loc="upper left",
                          frameon=False, handlelength=1.2, handletextpad=0.3,
                          labelspacing=0.25, borderpad=0.2, ncol=1)

    fig.tight_layout(w_pad=1.0, h_pad=1.2, rect=[0.05, 0, 1, 1])

    # Add row labels on the far left
    for row, (_, anchor_short) in enumerate(row_anchors):
        # Get vertical center of this row
        y_center = (axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2
        fig.text(0.01, y_center, anchor_short, fontsize=title_fs, fontweight="bold",
                 ha="center", va="center", rotation=90)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig5_bert_anchors_combined.{ext}"),
                    dpi=300, bbox_inches="tight")
    print("  Saved: fig5_bert_anchors_combined")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading MNLI confidences...")
    mnli_confs = load_mnli_confs()
    print("\nDrawing combined figure...")
    draw_combined_figure(mnli_confs)
    print("Done!")
