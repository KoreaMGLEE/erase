"""Appendix: Full 20x20 pairwise Jaccard heatmaps — MNLI and ARC, h2 colormap."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = "/workspace/erase/figures"
os.makedirs(FIG_DIR, exist_ok=True)

K = 30

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]

SHORT_LABELS = [
    "B-mini", "B-small", "B-med", "B-base", "B-large",
    "T5-sm", "T5-base", "T5-lg", "T5-xl", "T5-xxl",
    "P-14m", "P-31m", "P-70m", "P-160m", "P-410m",
    "P-1b", "P-1.4b", "P-2.8b", "P-6.9b", "P-12b",
]

FAMILY_BOUNDARIES = [4.5, 9.5]

# h2 colormap (fig1 final)
WARM_ROSE_H2 = ["#F7F2EC", "#EDDAC0", "#E2BE8E", "#D9965B", "#D07840",
                "#C86332", "#C24E24", "#A03C1A"]

# === MNLI config ===
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
MNLI_SPLITS = ["split1", "split2", "split3"]

# === ARC config ===
ARC_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
ARC_SEEDS = {m: [1, 2, 3] for m in ALL_MODELS}
ARC_SEEDS["t5-v1_1-large"] = [2, 3, 4]
ARC_SEEDS["pythia-410m"] = [1, 3, 4]
ARC_SEEDS["pythia-6.9b"] = [1, 4, 5]
ARC_SEEDS["pythia-12b"] = [1, 3, 4]


def load_mnli_split_confs(model):
    confs = {}
    for split in MNLI_SPLITS:
        path = MNLI_PATHS[model].format(split.replace("split", ""))
        if not os.path.exists(path):
            path = MNLI_PATHS[model].format(split[-1])
        if os.path.exists(path):
            with open(path) as f:
                confs[split] = json.load(f)
    return confs


def load_arc_seed_confs(model):
    confs = {}
    for seed in ARC_SEEDS[model]:
        path = os.path.join(ARC_DIR, f"{model}_seed{seed}", "avg_conf.json")
        if os.path.exists(path):
            with open(path) as f:
                confs[f"seed{seed}"] = json.load(f)
    return confs


def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(item[0] for item in items[:n])


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0


def plot_heatmap(mean_matrix, std_matrix, title, filename):
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

    cmap = LinearSegmentedColormap.from_list("warm_rose_h2", WARM_ROSE_H2, N=256)
    n = len(ALL_MODELS)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mean_matrix, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SHORT_LABELS, rotation=45, ha="right", fontsize=6.0)
    ax.set_yticklabels(SHORT_LABELS, fontsize=6.0)

    # Annotations
    for i in range(n):
        for j in range(n):
            val = mean_matrix[i][j]
            color = "white" if val > 0.55 else "#222222"
            if i == j:
                txt = "1.00"
            else:
                txt = f"{val:.2f}\n\u00b1{std_matrix[i][j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=4.0, color=color, fontfamily="serif")

    # Family boundaries — thin dark gray
    for pos in FAMILY_BOUNDARIES:
        ax.axhline(y=pos, color="#666666", linewidth=0.8, zorder=5)
        ax.axvline(x=pos, color="#666666", linewidth=0.8, zorder=5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Jaccard similarity", fontsize=7.5, fontfamily="serif")
    cbar.ax.tick_params(labelsize=6.0, width=0.4, direction="in")
    cbar.outline.set_linewidth(0.4)

    ax.set_title(title, fontsize=9, fontfamily="serif", pad=8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#666666")

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.12, top=0.94)
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{filename}.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close(fig)


def main():
    n = len(ALL_MODELS)

    # === MNLI ===
    print("=== MNLI Full Jaccard ===")
    mnli_model_confs = {}
    for model in ALL_MODELS:
        confs = load_mnli_split_confs(model)
        if confs:
            mnli_model_confs[model] = confs
            print(f"  {model}: {len(confs)} splits")
        else:
            print(f"  {model}: MISSING")

    split_matrices = []
    for split in MNLI_SPLITS:
        skip = False
        for model in ALL_MODELS:
            if model not in mnli_model_confs or split not in mnli_model_confs[model]:
                skip = True
                break
        if skip:
            continue
        matrix = np.zeros((n, n))
        for i, m1 in enumerate(ALL_MODELS):
            for j, m2 in enumerate(ALL_MODELS):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    conf1 = mnli_model_confs[m1][split]
                    conf2 = mnli_model_confs[m2][split]
                    common = set(conf1.keys()) & set(conf2.keys())
                    if common:
                        c1 = {k: v for k, v in conf1.items() if k in common}
                        c2 = {k: v for k, v in conf2.items() if k in common}
                        matrix[i][j] = jaccard(get_easy_set(c1, K), get_easy_set(c2, K))
        split_matrices.append(matrix)

    if split_matrices:
        mnli_mean = np.mean(split_matrices, axis=0)
        mnli_std = np.std(split_matrices, axis=0)
        plot_heatmap(mnli_mean, mnli_std,
                     f"Pairwise overlap of easy examples (top {K}%, MNLI, 20 models)",
                     "appendix_jaccard_mnli_final")

    # === ARC ===
    print("\n=== ARC Full Jaccard ===")
    arc_model_confs = {}
    for model in ALL_MODELS:
        confs = load_arc_seed_confs(model)
        if confs:
            arc_model_confs[model] = confs
            print(f"  {model}: {len(confs)} seeds")
        else:
            print(f"  {model}: MISSING")

    arc_mean = np.zeros((n, n))
    arc_std = np.zeros((n, n))
    for i, m1 in enumerate(ALL_MODELS):
        for j, m2 in enumerate(ALL_MODELS):
            if i == j:
                arc_mean[i][j] = 1.0
                continue
            if m1 not in arc_model_confs or m2 not in arc_model_confs:
                continue
            shared = set(arc_model_confs[m1].keys()) & set(arc_model_confs[m2].keys())
            if not shared:
                jaccards = []
                for s1 in arc_model_confs[m1]:
                    for s2 in arc_model_confs[m2]:
                        conf1 = arc_model_confs[m1][s1]
                        conf2 = arc_model_confs[m2][s2]
                        common = set(conf1.keys()) & set(conf2.keys())
                        if common:
                            c1 = {k: v for k, v in conf1.items() if k in common}
                            c2 = {k: v for k, v in conf2.items() if k in common}
                            jaccards.append(jaccard(get_easy_set(c1, K), get_easy_set(c2, K)))
                if jaccards:
                    arc_mean[i][j] = np.mean(jaccards)
                    arc_std[i][j] = np.std(jaccards)
            else:
                jaccards = []
                for sk in sorted(shared):
                    conf1 = arc_model_confs[m1][sk]
                    conf2 = arc_model_confs[m2][sk]
                    common = set(conf1.keys()) & set(conf2.keys())
                    if common:
                        c1 = {k: v for k, v in conf1.items() if k in common}
                        c2 = {k: v for k, v in conf2.items() if k in common}
                        jaccards.append(jaccard(get_easy_set(c1, K), get_easy_set(c2, K)))
                if jaccards:
                    arc_mean[i][j] = np.mean(jaccards)
                    arc_std[i][j] = np.std(jaccards)

    plot_heatmap(arc_mean, arc_std,
                 f"Pairwise overlap of easy examples (top {K}%, ARC, 20 models)",
                 "appendix_jaccard_arc_final")


if __name__ == "__main__":
    main()
