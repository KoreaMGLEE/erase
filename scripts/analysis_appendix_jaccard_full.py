"""Appendix: Full pairwise Jaccard heatmap for all models — MNLI and ARC."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = "/workspace/erase/figures"
os.makedirs(FIG_DIR, exist_ok=True)

K = 30  # top 30% easy

# === Model lists (ordered by family then size) ===
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

FAMILY_BOUNDARIES = [4.5, 9.5]  # after bert-large, after t5-xxl

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
            # Try alternate numbering
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


def compute_jaccard_matrix(models, load_fn, unit_keys):
    """Compute mean ± std Jaccard matrix across splits/seeds."""
    n = len(models)
    model_confs = {}
    for model in models:
        confs = load_fn(model)
        if len(confs) > 0:
            model_confs[model] = confs
            print(f"  {model}: {len(confs)} units")
        else:
            print(f"  {model}: MISSING")

    # Per-unit Jaccard matrices
    unit_matrices = []
    for unit_key in unit_keys:
        skip = False
        for model in models:
            if model not in model_confs or unit_key not in model_confs[model]:
                skip = True
                break
        if skip:
            continue

        matrix = np.zeros((n, n))
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    conf1 = model_confs[m1][unit_key]
                    conf2 = model_confs[m2][unit_key]
                    common = set(conf1.keys()) & set(conf2.keys())
                    if len(common) == 0:
                        matrix[i][j] = 0
                    else:
                        c1 = {k: v for k, v in conf1.items() if k in common}
                        c2 = {k: v for k, v in conf2.items() if k in common}
                        matrix[i][j] = jaccard(get_easy_set(c1, K), get_easy_set(c2, K))
        unit_matrices.append(matrix)

    if not unit_matrices:
        return None, None

    # For models with non-matching unit keys (different seeds), do cross-comparison
    # First check if we got enough matrices
    if len(unit_matrices) >= 2:
        mean_matrix = np.mean(unit_matrices, axis=0)
        std_matrix = np.std(unit_matrices, axis=0)
    else:
        mean_matrix = unit_matrices[0]
        std_matrix = np.zeros_like(mean_matrix)

    return mean_matrix, std_matrix


def plot_heatmap(mean_matrix, std_matrix, models, labels, title, filename):
    n = len(models)
    fig, ax = plt.subplots(figsize=(12, 10.5))
    im = ax.imshow(mean_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    for i in range(n):
        for j in range(n):
            color = "white" if mean_matrix[i][j] > 0.6 else "black"
            if i == j:
                ax.text(j, i, "1.00", ha="center", va="center", fontsize=5, color=color)
            else:
                txt = f"{mean_matrix[i][j]:.2f}"
                if std_matrix is not None and std_matrix[i][j] > 0:
                    txt += f"\n±{std_matrix[i][j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=4.5, color=color)

    for pos in FAMILY_BOUNDARIES:
        ax.axhline(y=pos, color="black", linewidth=2)
        ax.axvline(x=pos, color="black", linewidth=2)

    plt.colorbar(im, ax=ax, label="Jaccard Similarity", shrink=0.8)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{filename}.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, f"{filename}.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {FIG_DIR}/{filename}.png")


def main():
    # === MNLI ===
    print("=== MNLI Full Jaccard ===")
    # For MNLI, all models share the same split keys
    mnli_model_confs = {}
    for model in ALL_MODELS:
        confs = load_mnli_split_confs(model)
        if confs:
            mnli_model_confs[model] = confs
            print(f"  {model}: {len(confs)} splits")
        else:
            print(f"  {model}: MISSING")

    # Compute per-split matrices
    n = len(ALL_MODELS)
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
        plot_heatmap(mnli_mean, mnli_std, ALL_MODELS, SHORT_LABELS,
                     f"Pairwise Jaccard of Easy Examples (top {K}%)\nMNLI — 20 models, mean ± std over 3 splits",
                     "appendix_jaccard_mnli_full")

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

    # For ARC, models may have different seed keys. Use pairwise shared seeds.
    arc_mean = np.zeros((n, n))
    arc_std = np.zeros((n, n))
    for i, m1 in enumerate(ALL_MODELS):
        for j, m2 in enumerate(ALL_MODELS):
            if i == j:
                arc_mean[i][j] = 1.0
                arc_std[i][j] = 0.0
                continue
            if m1 not in arc_model_confs or m2 not in arc_model_confs:
                continue
            shared = set(arc_model_confs[m1].keys()) & set(arc_model_confs[m2].keys())
            if not shared:
                # Cross-compare all seed pairs
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

    plot_heatmap(arc_mean, arc_std, ALL_MODELS, SHORT_LABELS,
                 f"Pairwise Jaccard of Easy Examples (top {K}%)\nARC — 20 models, mean ± std over 3 seeds",
                 "appendix_jaccard_arc_full")


if __name__ == "__main__":
    main()
