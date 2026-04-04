"""Fig 1 final: Pairwise Jaccard heatmaps — MNLI and ARC, warm_rose colormap."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = "/workspace/erase/figures"
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = [
    "bert-small", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large",
    "pythia-70m", "pythia-410m", "pythia-1b",
]
K = 30

# --- MNLI config ---
MNLI_CONF_DIRS = [
    "/workspace/erase/outputs/plan0/confidence",
    "/workspace/erase/outputs/plan5/confidence",
]
MNLI_SPLITS = ["split1", "split2", "split3"]

# --- ARC config ---
ARC_CONF_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
ARC_MODEL_SEEDS = {
    "t5-v1_1-large": [2, 3, 4],
    "pythia-410m": [1, 3, 4],
}
ARC_DEFAULT_SEEDS = [1, 2, 3]

# --- Colormap ---
# f1 original
WARM_ROSE = ["#faf5f2", "#f5e2d8", "#ecc8b4", "#dea98e", "#cc8870",
             "#b56652", "#994438", "#7a2a24"]
# h1: 무난한 논문용 — low #F5F1EC, mid #D8A06F, high #B4552D
WARM_ROSE_H1 = ["#F5F1EC", "#EDDCC8", "#E4C49E", "#D8A06F", "#CC7E4E",
                "#C06A3C", "#B4552D", "#963E20"]
# h2: 오렌지 강조 — low #F7F2EC, mid #D9965B, high #C24E24
WARM_ROSE_H2 = ["#F7F2EC", "#EDDAC0", "#E2BE8E", "#D9965B", "#D07840",
                "#C86332", "#C24E24", "#A03C1A"]


def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(item[0] for item in items[:n])


def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


# ── MNLI ──────────────────────────────────────────────────────────

def load_mnli_split_confs():
    model_split_confs = {}
    for model in MODELS:
        split_confs = {}
        for split in MNLI_SPLITS:
            for conf_dir in MNLI_CONF_DIRS:
                candidates = [
                    os.path.join(conf_dir, f"{model}_{split}", "avg_conf.json"),
                    os.path.join(conf_dir, f"{model}_sentinel_{split}", "avg_conf.json"),
                    os.path.join(conf_dir, f"{model}_sentinel_1e-4_{split}", "avg_conf.json"),
                ]
                found = False
                for path in candidates:
                    if os.path.exists(path):
                        with open(path) as f:
                            split_confs[split] = json.load(f)
                        found = True
                        break
                if found:
                    break
        print(f"  MNLI {model}: {len(split_confs)} splits")
        model_split_confs[model] = split_confs
    return model_split_confs


def compute_mnli_matrices(model_split_confs):
    n = len(MODELS)
    split_matrices = []
    for split in MNLI_SPLITS:
        skip = False
        for model in MODELS:
            if split not in model_split_confs[model]:
                skip = True
                break
        if skip:
            continue
        matrix = np.zeros((n, n))
        for i, m1 in enumerate(MODELS):
            for j, m2 in enumerate(MODELS):
                common = set(model_split_confs[m1][split].keys()) & set(model_split_confs[m2][split].keys())
                if len(common) == 0:
                    matrix[i][j] = 0
                else:
                    c1 = {k: v for k, v in model_split_confs[m1][split].items() if k in common}
                    c2 = {k: v for k, v in model_split_confs[m2][split].items() if k in common}
                    matrix[i][j] = jaccard(get_easy_set(c1, K), get_easy_set(c2, K))
        split_matrices.append(matrix)
    return np.mean(split_matrices, axis=0), np.std(split_matrices, axis=0)


# ── ARC ───────────────────────────────────────────────────────────

def load_arc_seed_confs():
    model_seed_confs = {}
    for model in MODELS:
        seeds = ARC_MODEL_SEEDS.get(model, ARC_DEFAULT_SEEDS)
        seed_confs = {}
        for seed in seeds:
            path = os.path.join(ARC_CONF_DIR, f"{model}_seed{seed}", "avg_conf.json")
            if os.path.exists(path):
                with open(path) as f:
                    seed_confs[f"seed{seed}"] = json.load(f)
        print(f"  ARC  {model}: {len(seed_confs)} seeds")
        model_seed_confs[model] = seed_confs
    return model_seed_confs


def compute_arc_matrices(model_seed_confs):
    n = len(MODELS)
    mean_matrix = np.zeros((n, n))
    std_matrix = np.zeros((n, n))
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i == j:
                mean_matrix[i][j] = 1.0
                continue
            shared = set(model_seed_confs[m1].keys()) & set(model_seed_confs[m2].keys())
            if len(shared) == 0:
                seeds1 = list(model_seed_confs[m1].keys())
                seeds2 = list(model_seed_confs[m2].keys())
                jacs = []
                for s1 in seeds1:
                    for s2 in seeds2:
                        common = set(model_seed_confs[m1][s1].keys()) & set(model_seed_confs[m2][s2].keys())
                        if not common:
                            continue
                        c1 = {k: v for k, v in model_seed_confs[m1][s1].items() if k in common}
                        c2 = {k: v for k, v in model_seed_confs[m2][s2].items() if k in common}
                        jacs.append(jaccard(get_easy_set(c1, K), get_easy_set(c2, K)))
            else:
                jacs = []
                for sk in sorted(shared):
                    common = set(model_seed_confs[m1][sk].keys()) & set(model_seed_confs[m2][sk].keys())
                    if not common:
                        continue
                    c1 = {k: v for k, v in model_seed_confs[m1][sk].items() if k in common}
                    c2 = {k: v for k, v in model_seed_confs[m2][sk].items() if k in common}
                    jacs.append(jaccard(get_easy_set(c1, K), get_easy_set(c2, K)))
            mean_matrix[i][j] = float(np.mean(jacs)) if jacs else 0
            std_matrix[i][j] = float(np.std(jacs)) if jacs else 0
    return mean_matrix, std_matrix


# ── Drawing ───────────────────────────────────────────────────────

def draw_heatmap(mean_matrix, std_matrix, title, filename, colormap_colors=None):
    if colormap_colors is None:
        colormap_colors = WARM_ROSE
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.4,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })

    cmap = LinearSegmentedColormap.from_list("warm_rose", colormap_colors, N=256)
    n = len(MODELS)

    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(mean_matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    short_labels = [
        m.replace("t5-v1_1-", "T5v1.1-").replace("bert-", "BERT-").replace("pythia-", "Pythia-")
        for m in MODELS
    ]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7.0)
    ax.set_yticklabels(short_labels, fontsize=7.0)

    for i in range(n):
        for j in range(n):
            val = mean_matrix[i][j]
            color = "white" if val > 0.55 else "#222222"
            if i == j:
                txt = "1.00"
            else:
                txt = f"{val:.2f}\n\u00b1{std_matrix[i][j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=5.5, color=color, fontfamily="serif")

    for pos in [2.5, 5.5]:
        ax.axhline(y=pos, color="#666666", linewidth=0.8, zorder=5)
        ax.axvline(x=pos, color="#666666", linewidth=0.8, zorder=5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Jaccard similarity", fontsize=7.5, fontfamily="serif")
    cbar.ax.tick_params(labelsize=6.5, width=0.4, direction="in")
    cbar.outline.set_linewidth(0.4)

    ax.set_title(title, fontsize=8.5, fontfamily="serif", pad=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#666666")

    fig.subplots_adjust(left=0.16, right=0.88, bottom=0.14, top=0.92)
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


if __name__ == "__main__":
    # --- MNLI ---
    print("Loading MNLI data...")
    mnli_confs = load_mnli_split_confs()
    mnli_mean, mnli_std = compute_mnli_matrices(mnli_confs)

    # --- ARC ---
    print("\nLoading ARC data...")
    arc_confs = load_arc_seed_confs()
    arc_mean, arc_std = compute_arc_matrices(arc_confs)

    # --- Generate variants ---
    for suffix, colors in [("h1", WARM_ROSE_H1), ("h2", WARM_ROSE_H2)]:
        draw_heatmap(mnli_mean, mnli_std,
                     f"Pairwise overlap of easy examples (top {K}%, MNLI)",
                     f"fig1_mnli_final_{suffix}", colors)
        draw_heatmap(arc_mean, arc_std,
                     f"Pairwise overlap of easy examples (top {K}%, ARC)",
                     f"fig1_arc_final_{suffix}", colors)

    print("\nDone!")
