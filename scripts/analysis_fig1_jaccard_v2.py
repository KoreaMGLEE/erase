"""본문 분석 1 v2: Pairwise Jaccard Heatmap — split별 평균 + std."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "/workspace/erase/outputs/plan5/analysis"
FIG_DIR = "/workspace/erase/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = [
    "bert-small", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl",
    "pythia-70m", "pythia-410m", "pythia-2.8b",
]

FAMILY_LABELS = ["Encoder\n(BERT)", "Enc-Dec\n(T5 v1.1)", "Decoder\n(Pythia)"]
K = 30


def load_split_confidences(model_name):
    """Load per-split confidence (not merged 90K)."""
    split_confs = {}
    for split in ["split1", "split2", "split3"]:
        # Try multiple naming patterns
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


def main():
    print("Loading per-split confidence for 9 models...")
    model_split_confs = {}
    for model in MODELS:
        split_confs = load_split_confidences(model)
        if len(split_confs) < 3:
            print(f"  {model}: only {len(split_confs)} splits found!")
        else:
            print(f"  {model}: {len(split_confs)} splits OK")
        model_split_confs[model] = split_confs

    # Compute per-split Jaccard matrices, then average
    print("\nComputing per-split Jaccard matrices...")
    n = len(MODELS)
    split_matrices = []

    for split in ["split1", "split2", "split3"]:
        # Get easy sets for this split
        easy_sets = {}
        skip_split = False
        for model in MODELS:
            if split in model_split_confs[model]:
                easy_sets[model] = get_easy_set(model_split_confs[model][split], K)
            else:
                print(f"  WARNING: {model} missing {split}, skipping this split")
                skip_split = True
                break

        if skip_split:
            continue

        # Compute Jaccard for this split
        matrix = np.zeros((n, n))
        for i, m1 in enumerate(MODELS):
            for j, m2 in enumerate(MODELS):
                # Only compare indices that exist in both models' split
                common_indices = set(model_split_confs[m1][split].keys()) & set(model_split_confs[m2][split].keys())
                if len(common_indices) == 0:
                    matrix[i][j] = 0
                else:
                    # Recompute easy sets restricted to common indices
                    conf1 = {k: v for k, v in model_split_confs[m1][split].items() if k in common_indices}
                    conf2 = {k: v for k, v in model_split_confs[m2][split].items() if k in common_indices}
                    easy1 = get_easy_set(conf1, K)
                    easy2 = get_easy_set(conf2, K)
                    matrix[i][j] = jaccard(easy1, easy2)

        split_matrices.append(matrix)
        print(f"  {split}: computed")

    if len(split_matrices) == 0:
        print("ERROR: No valid splits!")
        return

    # Mean and std across splits
    mean_matrix = np.mean(split_matrices, axis=0)
    std_matrix = np.std(split_matrices, axis=0)

    print(f"\nUsed {len(split_matrices)} splits for averaging")

    # Save
    result = {
        "models": MODELS,
        "k_pct": K,
        "n_splits": len(split_matrices),
        "mean_matrix": mean_matrix.tolist(),
        "std_matrix": std_matrix.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "jaccard_matrix_9x9_k30_split_avg.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Plot heatmap with mean ± std in cells
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mean_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    short_labels = [m.replace("t5-v1_1-", "T5v1.1-").replace("bert-", "BERT-").replace("pythia-", "Pythia-")
                    for m in MODELS]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # Annotate: mean ± std
    for i in range(n):
        for j in range(n):
            color = "white" if mean_matrix[i][j] > 0.6 else "black"
            if i == j:
                ax.text(j, i, "1.00", ha="center", va="center", fontsize=7, color=color)
            else:
                ax.text(j, i, f"{mean_matrix[i][j]:.2f}\n±{std_matrix[i][j]:.2f}",
                        ha="center", va="center", fontsize=6, color=color)

    # Family boundaries
    for pos in [2.5, 5.5]:
        ax.axhline(y=pos, color="black", linewidth=2)
        ax.axvline(x=pos, color="black", linewidth=2)

    plt.colorbar(im, ax=ax, label="Jaccard Similarity (mean across splits)", shrink=0.8)
    ax.set_title(f"Pairwise Jaccard of Easy Examples (top {K}%)\nMNLI — mean ± std over 3 splits", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_jaccard_heatmap_9x9_v2.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig1_jaccard_heatmap_9x9_v2.png"), dpi=150, bbox_inches="tight")
    print(f"Saved to {FIG_DIR}/fig1_jaccard_heatmap_9x9_v2.pdf")


if __name__ == "__main__":
    main()
