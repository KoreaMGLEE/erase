"""Fig 5: Jaccard overlap vs threshold k% — anchor = largest model per family."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = "/workspace/erase/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Model config ─────────────────────────────────────────────────

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]

PARAMS = {
    "bert-mini": 11e6, "bert-small": 29e6, "bert-medium": 42e6,
    "bert-base": 110e6, "bert-large": 335e6,
    "t5-v1_1-small": 77e6, "t5-v1_1-base": 248e6,
    "t5-v1_1-large": 783e6, "t5-v1_1-xl": 3e9, "t5-v1_1-xxl": 11e9,
    "pythia-14m": 14e6, "pythia-31m": 31e6, "pythia-70m": 70e6,
    "pythia-160m": 160e6, "pythia-410m": 410e6, "pythia-1b": 1e9,
    "pythia-1.4b": 1.4e9, "pythia-2.8b": 2.8e9, "pythia-6.9b": 6.9e9,
    "pythia-12b": 12e9,
}

FAMILIES = {}
for m in ALL_MODELS:
    if m.startswith("bert"):
        FAMILIES[m] = "BERT"
    elif m.startswith("t5"):
        FAMILIES[m] = "T5 v1.1"
    else:
        FAMILIES[m] = "Pythia"

FAMILY_MARKERS = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

ANCHORS = {
    "BERT": "bert-large",
    "T5 v1.1": "t5-v1_1-xxl",
    "Pythia": "pythia-12b",
}

K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50]

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

# ── ARC config ───────────────────────────────────────────────────

ARC_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
ARC_SEEDS = {m: [1, 2, 3] for m in ALL_MODELS}
ARC_SEEDS["t5-v1_1-large"] = [2, 3, 4]
ARC_SEEDS["pythia-410m"] = [1, 3, 4]
ARC_SEEDS["pythia-6.9b"] = [1, 4, 5]
ARC_SEEDS["pythia-12b"] = [1, 3, 4]

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
    """Load MNLI confidence dicts: {model: {split_idx: conf_dict}}"""
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
            print(f"  MNLI {model}: {len(splits)} splits")
        else:
            print(f"  MNLI {model}: MISSING")
    return all_confs


def load_arc_confs():
    """Load ARC confidence dicts: {model: {seed: conf_dict}}"""
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
            print(f"  ARC  {model}: {len(seeds)} seeds")
        else:
            print(f"  ARC  {model}: MISSING")
    return all_confs


def compute_anchor_jaccards(all_confs, anchor, k_pct):
    """For one anchor and one k%, compute Jaccard with every other model.
    Returns {model: (mean, std)} averaged over shared splits/seeds."""
    if anchor not in all_confs:
        return {}
    anchor_runs = all_confs[anchor]
    results = {}
    for model in ALL_MODELS:
        if model == anchor or model not in all_confs:
            continue
        model_runs = all_confs[model]
        # Find shared run keys
        shared = set(anchor_runs.keys()) & set(model_runs.keys())
        if shared:
            jacs = []
            for run_key in sorted(shared):
                conf_a = anchor_runs[run_key]
                conf_m = model_runs[run_key]
                common = set(conf_a.keys()) & set(conf_m.keys())
                if common:
                    ca = {k: v for k, v in conf_a.items() if k in common}
                    cm = {k: v for k, v in conf_m.items() if k in common}
                    jacs.append(jaccard(get_easy_set(ca, k_pct), get_easy_set(cm, k_pct)))
        else:
            # Cross-product
            jacs = []
            for rk_a in anchor_runs:
                for rk_m in model_runs:
                    conf_a = anchor_runs[rk_a]
                    conf_m = model_runs[rk_m]
                    common = set(conf_a.keys()) & set(conf_m.keys())
                    if common:
                        ca = {k: v for k, v in conf_a.items() if k in common}
                        cm = {k: v for k, v in conf_m.items() if k in common}
                        jacs.append(jaccard(get_easy_set(ca, k_pct), get_easy_set(cm, k_pct)))
        if jacs:
            results[model] = (float(np.mean(jacs)), float(np.std(jacs)))
    return results


# ── Plotting ─────────────────────────────────────────────────────

def set_pub_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def compute_model_curve(all_confs, anchor, model, k_values):
    """For one anchor-model pair, compute Jaccard at each k%.
    Returns list of (mean, std) for each k."""
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


# ── Model selection: same family all, cross-family 3 reps ────────

# 3 reps per family: mid, second-largest, largest
FAMILY_REPS = {
    "BERT": ["bert-medium", "bert-base", "bert-large"],
    "T5 v1.1": ["t5-v1_1-base", "t5-v1_1-xl", "t5-v1_1-xxl"],
    "Pythia": ["pythia-410m", "pythia-6.9b", "pythia-12b"],
}

FAMILY_BASE_COLORS = {"BERT": "#1f77b4", "T5 v1.1": "#d62728", "Pythia": "#2ca02c"}

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


# Explicit per-model colors — distinct hues within each family
MODEL_COLORS = {
    # BERT: light blue → medium blue → dark navy
    "bert-medium":    "#7cbae8",
    "bert-base":      "#2878b5",
    "bert-large":     "#0e3f6e",
    # T5: orange → red-orange → dark red
    "t5-v1_1-base":   "#f5a050",
    "t5-v1_1-xl":     "#d63a2a",
    "t5-v1_1-xxl":    "#8b1a1a",
    # Pythia: yellow-green → green → dark green
    "pythia-410m":    "#a8d84e",
    "pythia-6.9b":    "#3a9e3a",
    "pythia-12b":     "#1a5e1a",
}


def get_model_color(model):
    return MODEL_COLORS.get(model, "#888888")


def draw_panel(ax, all_confs, anchor, anchor_family, k_values,
               label_fs, tick_fs):
    """x = k%, y = Jaccard, lines = models.
    Same family: solid, size-gradient color, line-first.
    Cross family: dashed, muted, de-emphasised."""
    x_positions = np.array(k_values)

    # Draw in family order so legend is grouped: BERT, T5, Pythia
    for family in ["BERT", "T5 v1.1", "Pythia"]:
        reps = [m for m in FAMILY_REPS[family] if m != anchor]
        same = (family == anchor_family)
        # Pythia triangle needs bigger markersize to look same visual size
        ms_same = 5.5 if family == "Pythia" else 4.5
        ms_cross = 4.5 if family == "Pythia" else 3.5

        for model in reps:
            curve = compute_model_curve(all_confs, anchor, model, k_values)
            if curve is None:
                continue
            ys = np.array([c[0] for c in curve])
            color = get_model_color(model)
            if same:
                # Same family: solid, full alpha, prominent
                ax.plot(x_positions, ys, color=color, solid_capstyle="round",
                        marker=FAMILY_MARKERS[family], markersize=ms_same,
                        linewidth=1.2, label=SHORT_NAMES[model],
                        markeredgecolor="white", markeredgewidth=0.3,
                        markerfacecolor=color, zorder=4, alpha=1.0)
            else:
                # Cross-family: dashed, alpha on lines only
                ax.plot(x_positions, ys, color=color,
                        marker=FAMILY_MARKERS[family], markersize=ms_cross,
                        linewidth=0.6, linestyle="--", label=SHORT_NAMES[model],
                        markeredgecolor="white", markeredgewidth=0.2,
                        markerfacecolor=color, zorder=2, alpha=0.35)

    # Very light major grid
    ax.yaxis.grid(True, linewidth=0.2, color="#cccccc", alpha=0.5, zorder=0)

    ax.set_xscale("log")
    ax.set_xlabel("Top k%", fontsize=label_fs, labelpad=2)
    ax.set_ylabel("Jaccard similarity", fontsize=label_fs, labelpad=2)
    # Major ticks with labels
    major_ticks = [1, 5, 10, 20, 50]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(k) for k in major_ticks], fontsize=tick_fs)
    # Minor ticks without labels (2,3,7,15,30,40)
    minor_ticks = [k for k in k_values if k not in major_ticks]
    ax.set_xticks(minor_ticks, minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=tick_fs, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0.8, 60)
    ax.set_ylim(-0.02, 0.82)


def draw_row(confs, dataset, anchor_list, filename):
    """One row of 3 panels — appendix robustness figure."""
    set_pub_style()

    title_fs = 7.5
    label_fs = 6.5
    tick_fs = 5.5
    legend_fs = 4.5

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    for col, (family, anchor, anchor_short) in enumerate(anchor_list):
        ax = axes[col]
        draw_panel(ax, confs, anchor, family, K_VALUES, label_fs, tick_fs)
        panel_letter = chr(ord('a') + col)
        # Title: "Overlap with X" — immediately clear what anchor is
        ax.set_title(rf"$\bf{{({panel_letter})}}$ Overlap with {anchor_short}",
                     fontsize=title_fs, pad=4)
        if col > 0:
            ax.set_ylabel("")

    # Shared legend below panels
    # Collect all handles: same-family from each panel + cross-family markers
    # For simplicity, use rightmost panel (Pythia, most models) legend items
    # and add family marker indicators
    all_handles, all_labels = [], []
    for col in range(3):
        h, l = axes[col].get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    # Build legend handles manually at full alpha
    from matplotlib.lines import Line2D
    desired_order = [
        ("B-med", "BERT", "bert-medium"),
        ("B-base", "BERT", "bert-base"),
        ("B-lg", "BERT", "bert-large"),
        ("T5-base", "T5 v1.1", "t5-v1_1-base"),
        ("T5-xl", "T5 v1.1", "t5-v1_1-xl"),
        ("T5-xxl", "T5 v1.1", "t5-v1_1-xxl"),
        ("P-410m", "Pythia", "pythia-410m"),
        ("P-6.9b", "Pythia", "pythia-6.9b"),
        ("P-12b", "Pythia", "pythia-12b"),
    ]
    unique_h, unique_l = [], []
    for short, family, model in desired_order:
        color = get_model_color(model)
        marker = FAMILY_MARKERS[family]
        ms = 4.0 if family != "Pythia" else 5.0
        h = Line2D([0], [0], color=color, marker=marker, markersize=ms,
                   markerfacecolor=color, markeredgecolor="white",
                   markeredgewidth=0.3, linewidth=1.0, linestyle="-")
        unique_h.append(h)
        unique_l.append(short)

    if unique_h:
        fig.legend(unique_h, unique_l, fontsize=legend_fs + 1,
                   loc="lower center", ncol=min(len(unique_h), 10),
                   frameon=False, handlelength=1.2, handletextpad=0.3,
                   columnspacing=0.8, borderpad=0.2,
                   bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(w_pad=1.2, rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


def draw_figure(mnli_confs, arc_confs):
    # Largest anchors
    anchors_large = [
        ("BERT", "bert-large", "BERT-large"),
        ("T5 v1.1", "t5-v1_1-xxl", "T5-v1.1-xxl"),
        ("Pythia", "pythia-12b", "Pythia-12B"),
    ]
    # Mid-size anchors
    anchors_mid = [
        ("BERT", "bert-base", "BERT-base"),
        ("T5 v1.1", "t5-v1_1-xl", "T5-v1.1-xl"),
        ("Pythia", "pythia-6.9b", "Pythia-6.9B"),
    ]
    # Small anchors
    anchors_small = [
        ("BERT", "bert-medium", "BERT-medium"),
        ("T5 v1.1", "t5-v1_1-base", "T5-v1.1-base"),
        ("Pythia", "pythia-410m", "Pythia-410M"),
    ]

    for confs, dataset in [(mnli_confs, "mnli"), (arc_confs, "arc")]:
        draw_row(confs, dataset, anchors_large, f"fig_appendix_robustness_{dataset}_large")
        draw_row(confs, dataset, anchors_mid, f"fig_appendix_robustness_{dataset}_mid")
        draw_row(confs, dataset, anchors_small, f"fig_appendix_robustness_{dataset}_small")


if __name__ == "__main__":
    print("Loading MNLI confidences...")
    mnli_confs = load_mnli_confs()

    print("\nLoading ARC confidences...")
    arc_confs = load_arc_confs()

    print("\nDrawing figure...")
    draw_figure(mnli_confs, arc_confs)
    print("Done!")
