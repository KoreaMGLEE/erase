"""Fig 5 variants: 3×3 Jaccard overlap vs threshold k%.
Rows = 3 anchors (large/mid/small within one family), Cols = 3 target families.
Generate 6 figures: {BERT, T5, Pythia} anchors × {MNLI, ARC} datasets.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

FIG_DIR = "/workspace/erase/figures"
os.makedirs(FIG_DIR, exist_ok=True)

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]

ALL_BERT = ["bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large"]
ALL_T5 = ["t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl"]
ALL_PYTHIA = [
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

MODEL_COLORS = {
    "bert-mini": "#b3d9f7", "bert-small": "#7cbae8", "bert-medium": "#4196d4",
    "bert-base": "#1a6fb5", "bert-large": "#0e3f6e",
    "t5-v1_1-small": "#f7c5a0", "t5-v1_1-base": "#f5a050",
    "t5-v1_1-large": "#e07030", "t5-v1_1-xl": "#d63a2a", "t5-v1_1-xxl": "#8b1a1a",
    "pythia-14m": "#c8e8a0", "pythia-31m": "#a8d84e", "pythia-70m": "#80c840",
    "pythia-160m": "#60b030", "pythia-410m": "#48a028", "pythia-1b": "#389020",
    "pythia-1.4b": "#2d8018", "pythia-2.8b": "#247010", "pythia-6.9b": "#1a5e0a",
    "pythia-12b": "#104a04",
}

# ── Anchor configs ───────────────────────────────────────────────

ANCHOR_CONFIGS = {
    "bert": {
        "row_anchors": [
            ("bert-large", "BERT-large"),
            ("bert-medium", "BERT-medium"),
            ("bert-mini", "BERT-mini"),
        ],
        "tag": "bert",
    },
    "t5": {
        "row_anchors": [
            ("t5-v1_1-xxl", "T5-v1.1-xxl"),
            ("t5-v1_1-large", "T5-v1.1-large"),
            ("t5-v1_1-small", "T5-v1.1-small"),
        ],
        "tag": "t5",
    },
    "pythia": {
        "row_anchors": [
            ("pythia-12b", "Pythia-12B"),
            ("pythia-1.4b", "Pythia-1.4B"),
            ("pythia-70m", "Pythia-70M"),
        ],
        "tag": "pythia",
    },
}

COL_FAMILIES = [
    ("BERT", "vs BERT family", ALL_BERT),
    ("T5", "vs T5 v1.1 family", ALL_T5),
    ("Pythia", "vs Pythia family", ALL_PYTHIA),
]

# ── Data loading ─────────────────────────────────────────────────

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

ARC_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
ARC_SEEDS = {m: [1, 2, 3] for m in ALL_MODELS}
ARC_SEEDS["t5-v1_1-large"] = [2, 3, 4]
ARC_SEEDS["pythia-410m"] = [1, 3, 4]
ARC_SEEDS["pythia-6.9b"] = [1, 4, 5]
ARC_SEEDS["pythia-12b"] = [1, 3, 4]


def load_mnli_confs():
    all_confs = {}
    for model in ALL_MODELS:
        splits = {}
        for s in [1, 2, 3]:
            path = MNLI_PATHS[model].format(s)
            if os.path.exists(path):
                with open(path) as f:
                    splits[s] = json.load(f)
        if splits:
            all_confs[model] = splits
    print(f"  MNLI: {len(all_confs)} models loaded")
    return all_confs


def load_arc_confs():
    all_confs = {}
    for model in ALL_MODELS:
        seeds = {}
        for seed in ARC_SEEDS[model]:
            path = os.path.join(ARC_DIR, f"{model}_seed{seed}", "avg_conf.json")
            if os.path.exists(path):
                with open(path) as f:
                    seeds[seed] = json.load(f)
        if seeds:
            all_confs[model] = seeds
    print(f"  ARC:  {len(all_confs)} models loaded")
    return all_confs


# ── Computation ──────────────────────────────────────────────────

CACHE_DIR = "/workspace/erase/outputs/plan9/jaccard_cache"

def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = max(1, int(len(items) * k_pct / 100))
    return set(item[0] for item in items[:n])


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0


def precompute_all_curves(all_confs, dataset_name):
    """Precompute all pairwise Jaccard curves and cache to disk."""
    cache_path = os.path.join(CACHE_DIR, f"{dataset_name}_curves.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            print(f"  Loaded cached curves: {cache_path}")
            return json.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    models = [m for m in ALL_MODELS if m in all_confs]
    curves = {}
    total = len(models) * (len(models) - 1)
    done = 0

    for anchor in models:
        anchor_runs = all_confs[anchor]
        for model in models:
            if model == anchor:
                continue
            model_runs = all_confs[model]
            shared = set(anchor_runs.keys()) & set(model_runs.keys())

            curve = []
            for k_pct in K_VALUES:
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
                curve.append([float(np.mean(jacs)), float(np.std(jacs))] if jacs else [0.0, 0.0])

            key = f"{anchor}||{model}"
            curves[key] = curve
            done += 1

    with open(cache_path, "w") as f:
        json.dump({"k_values": K_VALUES, "curves": curves}, f)
    print(f"  Computed and cached {done} curves → {cache_path}")
    return {"k_values": K_VALUES, "curves": curves}


def compute_model_curve(cache, anchor, model, k_values):
    key = f"{anchor}||{model}"
    curve_data = cache.get("curves", {}).get(key)
    if curve_data is None:
        return None
    return [tuple(c) for c in curve_data]


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


def draw_panel(ax, cache, anchor, models, family, label_fs, tick_fs):
    x_positions = np.array(K_VALUES)
    marker = FAMILY_MARKERS[family]
    ms = 5.5 if family == "Pythia" else 4.5

    for model in reversed(models):
        if model == anchor:
            continue
        curve = compute_model_curve(cache, anchor, model, K_VALUES)
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


def draw_mega_figure(mnli_cache, arc_cache):
    """9 rows × 6 cols: all 3 anchor families stacked, MNLI left, ARC right."""
    set_pub_style()

    title_fs = 7.5
    label_fs = 6.5
    tick_fs = 5.5
    legend_fs = 5.5

    # Build all rows: BERT(3) + T5(3) + Pythia(3) = 9 rows
    all_rows = []
    for family_key in ["bert", "t5", "pythia"]:
        for anchor, anchor_short in ANCHOR_CONFIGS[family_key]["row_anchors"]:
            all_rows.append((anchor, anchor_short))
    n_rows = len(all_rows)  # 9

    fig, axes = plt.subplots(n_rows, 6, figsize=(15, 2.0 * n_rows))

    for row, (anchor, anchor_short) in enumerate(all_rows):
        panel_model_lists = [col_info[2] for col_info in COL_FAMILIES]

        for half, cache in enumerate([mnli_cache, arc_cache]):
            # Compute row-level y max for this half
            row_max = 0
            for col in range(3):
                models = panel_model_lists[col]
                for model in models:
                    if model == anchor:
                        continue
                    curve = compute_model_curve(cache, anchor, model, K_VALUES)
                    if curve:
                        row_max = max(row_max, max(c[0] for c in curve))
            y_top = np.ceil((row_max + 0.03) * 10) / 10

            for col in range(3):
                ax_col = half * 3 + col
                ax = axes[row, ax_col]
                family = COL_FAMILIES[col][0]
                models = panel_model_lists[col]
                draw_panel(ax, cache, anchor, models, family, label_fs, tick_fs)
                ax.set_ylim(-0.02, y_top)

                # Column sub-headers (top row only)
                if row == 0:
                    ax.set_title(COL_FAMILIES[col][1], fontsize=title_fs - 0.5, pad=4)

                # y-axis label: only on col 0 of the very middle row
                if ax_col == 0 and row == n_rows // 2:
                    ax.set_ylabel("Jaccard similarity", fontsize=label_fs, labelpad=2)
                else:
                    ax.set_ylabel("")

                # x-axis label only on bottom row
                if row < n_rows - 1:
                    ax.set_xlabel("")

                # Legend
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=legend_fs, loc="upper left",
                              frameon=False, handlelength=1.2, handletextpad=0.3,
                              labelspacing=0.25, borderpad=0.2, ncol=1)

    fig.tight_layout(w_pad=0.8, h_pad=0.8, rect=[0.025, 0, 1, 0.97])

    # Row labels: anchor model name, close to the plot
    for row, (_, anchor_short) in enumerate(all_rows):
        ax_left = axes[row, 0]
        pos = ax_left.get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(pos.x0 - 0.02, y_center, anchor_short, fontsize=title_fs - 0.5,
                 fontweight="bold", ha="right", va="center", rotation=90)

    # Dataset super-titles at top
    mnli_center = (axes[0, 0].get_position().x0 + axes[0, 2].get_position().x1) / 2
    arc_center = (axes[0, 3].get_position().x0 + axes[0, 5].get_position().x1) / 2
    fig.text(mnli_center, 0.97, "MNLI", fontsize=title_fs + 1, fontweight="bold",
             ha="center", va="center")
    fig.text(arc_center, 0.97, "ARC", fontsize=title_fs + 1, fontweight="bold",
             ha="center", va="center")

    fname = "fig5_all_anchors_combined"
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"{fname}.{ext}"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading confidences...")
    mnli_confs = load_mnli_confs()
    arc_confs = load_arc_confs()

    print("\nPrecomputing Jaccard curves...")
    mnli_cache = precompute_all_curves(mnli_confs, "mnli")
    arc_cache = precompute_all_curves(arc_confs, "arc")

    print("\nDrawing mega figure...")
    draw_mega_figure(mnli_cache, arc_cache)

    print("\nDone!")
