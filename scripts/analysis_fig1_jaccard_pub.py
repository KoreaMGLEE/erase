"""Fig 1: Pairwise Jaccard heatmap — publication style matching fig2 line plots."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUTPUT_DIR = "/workspace/erase/outputs/plan5/analysis"
FIG_DIR = "/workspace/erase/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = [
    "bert-small", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large",
    "pythia-70m", "pythia-410m", "pythia-1b",
]

FAMILY_LABELS = ["Encoder\n(BERT)", "Enc-Dec\n(T5 v1.1)", "Decoder\n(Pythia)"]
K = 30


def load_split_confidences(model_name):
    split_confs = {}
    for split in ["split1", "split2", "split3"]:
        candidates = [
            f"/workspace/erase/outputs/plan0/confidence/{model_name}_{split}/avg_conf.json",
            f"/workspace/erase/outputs/plan5/confidence/{model_name}_{split}/avg_conf.json",
            f"/workspace/erase/outputs/plan5/confidence/{model_name}_sentinel_{split}/avg_conf.json",
            f"/workspace/erase/outputs/plan5/confidence/{model_name}_sentinel_1e-4_{split}/avg_conf.json",
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path) as f:
                    split_confs[split] = json.load(f)
                break
    return split_confs


def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(item[0] for item in items[:n])


def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


def compute_matrices(model_split_confs):
    n = len(MODELS)
    split_matrices = []

    for split in ["split1", "split2", "split3"]:
        easy_sets = {}
        skip_split = False
        for model in MODELS:
            if split in model_split_confs[model]:
                easy_sets[model] = get_easy_set(model_split_confs[model][split], K)
            else:
                skip_split = True
                break
        if skip_split:
            continue

        matrix = np.zeros((n, n))
        for i, m1 in enumerate(MODELS):
            for j, m2 in enumerate(MODELS):
                common_indices = set(model_split_confs[m1][split].keys()) & set(model_split_confs[m2][split].keys())
                if len(common_indices) == 0:
                    matrix[i][j] = 0
                else:
                    conf1 = {k: v for k, v in model_split_confs[m1][split].items() if k in common_indices}
                    conf2 = {k: v for k, v in model_split_confs[m2][split].items() if k in common_indices}
                    easy1 = get_easy_set(conf1, K)
                    easy2 = get_easy_set(conf2, K)
                    matrix[i][j] = jaccard(easy1, easy2)
        split_matrices.append(matrix)

    mean_matrix = np.mean(split_matrices, axis=0)
    std_matrix = np.std(split_matrices, axis=0)
    return mean_matrix, std_matrix


def draw(mean_matrix, std_matrix, fig_w, fig_h, annot_fs, tick_fs, title_fs,
         label_fs, cbar_fs, spine_w, sep_w, sep_color, cmap_name, tag):

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": spine_w,
        "xtick.major.width": spine_w,
        "ytick.major.width": spine_w,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    n = len(MODELS)

    custom_cmaps = {
        "muted_warm": [
            "#faf6f0", "#f5e6ce", "#e8c9a0", "#d4a574", "#c0825a",
            "#a65d3f", "#8b3a2a", "#6b2020"],
        "muted_amber": [
            "#faf8f2", "#f2e4c8", "#e5c896", "#d1a660", "#b8843a",
            "#956428", "#704a1e", "#4a3018"],
        "muted_copper": [
            "#f9f5f0", "#f0e0cc", "#e0c4a0", "#cca678", "#b38858",
            "#966a40", "#7a4e2c", "#5c341c"],
        # --- NEW: red-tinted warm variants inspired by reference ---
        # f1: light beige -> blush pink -> muted rose -> terracotta -> deep brick
        # More red in mid-to-high range, cleaner low values
        "warm_rose": [
            "#faf5f2", "#f5e2d8", "#ecc8b4", "#dea98e", "#cc8870",
            "#b56652", "#994438", "#7a2a24"],
        # f2: similar but slightly more saturated red, reference-matching
        # Very light cream base, pink-salmon mid, rich terracotta high
        "warm_terracotta": [
            "#fbf6f3", "#f4e0d4", "#eac4ae", "#dca488", "#cb8268",
            "#b5604a", "#953e30", "#72261e"],
    }
    if cmap_name in custom_cmaps:
        cmap = LinearSegmentedColormap.from_list(cmap_name, custom_cmaps[cmap_name], N=256)
    else:
        cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mean_matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    short_labels = [
        m.replace("t5-v1_1-", "T5v1.1-").replace("bert-", "BERT-").replace("pythia-", "Pythia-")
        for m in MODELS
    ]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=tick_fs)
    ax.set_yticklabels(short_labels, fontsize=tick_fs)

    # Annotations: compact format
    for i in range(n):
        for j in range(n):
            val = mean_matrix[i][j]
            color = "white" if val > 0.55 else "#222222"
            if i == j:
                txt = "1.00"
            else:
                txt = f"{val:.2f}\n\u00b1{std_matrix[i][j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=annot_fs, color=color, fontfamily="serif")

    # Family boundary lines — thin, dark gray instead of thick black
    for pos in [2.5, 5.5]:
        ax.axhline(y=pos, color=sep_color, linewidth=sep_w, zorder=5)
        ax.axvline(x=pos, color=sep_color, linewidth=sep_w, zorder=5)

    # No extra family labels — tick labels + separator lines are sufficient

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Jaccard similarity", fontsize=cbar_fs, fontfamily="serif")
    cbar.ax.tick_params(labelsize=tick_fs - 0.5, width=spine_w, direction="in")
    cbar.outline.set_linewidth(spine_w)

    # Title — compact, matching fig2 style
    ax.set_title(f"Pairwise overlap of easy examples (top {K}%)",
                 fontsize=title_fs, fontfamily="serif", pad=8)

    # Remove default spines on image axes (they overlap with imshow border)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_w)
        spine.set_color("#666666")

    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.14, top=0.92)
    fig.savefig(os.path.join(FIG_DIR, f"fig1_pub_{tag}.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"fig1_pub_{tag}.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved: fig1_pub_{tag}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading per-split confidence for 9 models...")
    model_split_confs = {}
    for model in MODELS:
        split_confs = load_split_confidences(model)
        print(f"  {model}: {len(split_confs)} splits")
        model_split_confs[model] = split_confs

    mean_matrix, std_matrix = compute_matrices(model_split_confs)

    # Variants: (w, h, annot_fs, tick_fs, title_fs, label_fs, cbar_fs,
    #            spine_w, sep_w, sep_color, cmap, tag)
    variants = [
        # --- Red-tinted warm: e layout + new colormaps ---
        # f1: warm_rose — light base, blush-to-brick gradient
        (6.0, 5.2, 5.5, 7.0, 8.5, 6.5, 7.5, 0.4, 0.8, "#666666", "warm_rose", "f1"),
        # f2: warm_terracotta — slightly more saturated, reference-like
        (6.0, 5.2, 5.5, 7.0, 8.5, 6.5, 7.5, 0.4, 0.8, "#666666", "warm_terracotta", "f2"),
    ]

    for args in variants:
        draw(mean_matrix, std_matrix, *args)

    print("Done!")
