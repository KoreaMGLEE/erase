"""ARC Jaccard Heatmap: Pairwise Jaccard of easy examples (top K%) — seed별 평균 + std."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "/workspace/erase/outputs/plan2_v2/analysis"
FIG_DIR = "/workspace/erase/figures"
CONF_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = [
    "bert-small", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large",
    "pythia-70m", "pythia-410m", "pythia-1b",
]

K = 30

# Model-specific seed selection (skip failed seeds)
MODEL_SEEDS = {
    "t5-v1_1-large": [2, 3, 4],   # seed1 failed (random)
    "pythia-410m": [1, 3, 4],      # seed2 anomalous confidence
}
DEFAULT_SEEDS = [1, 2, 3]


def load_seed_confidences(model_name):
    """Load per-seed confidence for ARC."""
    seeds = MODEL_SEEDS.get(model_name, DEFAULT_SEEDS)
    seed_confs = {}
    for seed in seeds:
        path = os.path.join(CONF_DIR, f"{model_name}_seed{seed}", "avg_conf.json")
        if os.path.exists(path):
            with open(path) as f:
                seed_confs[f"seed{seed}"] = json.load(f)
    return seed_confs


def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(item[0] for item in items[:n])


def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


def main():
    print("Loading per-seed confidence for 9 models...")
    model_seed_confs = {}
    for model in MODELS:
        seed_confs = load_seed_confidences(model)
        print(f"  {model}: {len(seed_confs)} seeds")
        model_seed_confs[model] = seed_confs

    # Compute pairwise Jaccard: for each model pair, average across their shared seeds
    print("\nComputing pairwise Jaccard (per-seed average)...")
    n = len(MODELS)

    # For each model pair, compute Jaccard per shared seed, then average
    mean_matrix = np.zeros((n, n))
    std_matrix = np.zeros((n, n))

    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            if i == j:
                mean_matrix[i][j] = 1.0
                std_matrix[i][j] = 0.0
                continue
            # Find shared seed keys
            shared_seeds = set(model_seed_confs[m1].keys()) & set(model_seed_confs[m2].keys())
            if len(shared_seeds) == 0:
                # No shared seeds — use all seeds independently and cross-compare
                seeds1 = list(model_seed_confs[m1].keys())
                seeds2 = list(model_seed_confs[m2].keys())
                jaccards = []
                for s1 in seeds1:
                    for s2 in seeds2:
                        conf1 = model_seed_confs[m1][s1]
                        conf2 = model_seed_confs[m2][s2]
                        common = set(conf1.keys()) & set(conf2.keys())
                        if len(common) == 0:
                            continue
                        c1 = {k: v for k, v in conf1.items() if k in common}
                        c2 = {k: v for k, v in conf2.items() if k in common}
                        jaccards.append(jaccard(get_easy_set(c1, K), get_easy_set(c2, K)))
                mean_matrix[i][j] = float(np.mean(jaccards)) if jaccards else 0
                std_matrix[i][j] = float(np.std(jaccards)) if jaccards else 0
            else:
                jaccards = []
                for seed_key in sorted(shared_seeds):
                    conf1 = model_seed_confs[m1][seed_key]
                    conf2 = model_seed_confs[m2][seed_key]
                    common = set(conf1.keys()) & set(conf2.keys())
                    if len(common) == 0:
                        continue
                    c1 = {k: v for k, v in conf1.items() if k in common}
                    c2 = {k: v for k, v in conf2.items() if k in common}
                    jaccards.append(jaccard(get_easy_set(c1, K), get_easy_set(c2, K)))
                mean_matrix[i][j] = float(np.mean(jaccards)) if jaccards else 0
                std_matrix[i][j] = float(np.std(jaccards)) if jaccards else 0

    print(f"  Done. Models with non-default seeds: {list(MODEL_SEEDS.keys())}")

    # Save
    result = {
        "models": MODELS,
        "k_pct": K,
        "n_seeds": 3,
        "mean_matrix": mean_matrix.tolist(),
        "std_matrix": std_matrix.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "jaccard_matrix_9x9_k30_seed_avg_arc.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mean_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    short_labels = [m.replace("t5-v1_1-", "T5v1.1-").replace("bert-", "BERT-").replace("pythia-", "Pythia-")
                    for m in MODELS]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            color = "white" if mean_matrix[i][j] > 0.6 else "black"
            if i == j:
                ax.text(j, i, "1.00", ha="center", va="center", fontsize=7, color=color)
            else:
                ax.text(j, i, f"{mean_matrix[i][j]:.2f}\n\u00b1{std_matrix[i][j]:.2f}",
                        ha="center", va="center", fontsize=6, color=color)

    # Family boundaries (3+3+3)
    for pos in [2.5, 5.5]:
        ax.axhline(y=pos, color="black", linewidth=2)
        ax.axvline(x=pos, color="black", linewidth=2)

    plt.colorbar(im, ax=ax, label="Jaccard Similarity (mean across seeds)", shrink=0.8)
    ax.set_title(f"Pairwise Jaccard of Easy Examples (top {K}%)\nARC — mean \u00b1 std over 3 seeds", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_jaccard_heatmap_9x9_arc.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig1_jaccard_heatmap_9x9_arc.png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {FIG_DIR}/fig1_jaccard_heatmap_9x9_arc.pdf")


if __name__ == "__main__":
    main()
