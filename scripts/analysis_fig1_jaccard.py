"""본문 분석 1: Pairwise Jaccard Heatmap (9 representative models, k=30%)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = "/workspace/erase/outputs/plan5/analysis"
FIG_DIR = "/workspace/erase/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# 9 representative models (3 families × 3 scales)
MODELS = [
    "bert-small", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large",
    "pythia-70m", "pythia-410m", "pythia-2.8b",
]

FAMILY_LABELS = ["Encoder\n(BERT)", "Enc-Dec\n(T5 v1.1)", "Decoder\n(Pythia)"]
K = 30  # top k%


def load_confidence(model_name):
    """Load 90K confidence from plan0 or plan5."""
    for plan_dir in ["plan0", "plan5"]:
        path = f"/workspace/erase/outputs/{plan_dir}/confidence/{model_name}_90k_avg_conf.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(f"Confidence not found for {model_name}")


def get_easy_set(conf_dict, k_pct):
    """Get top k% indices as easy set."""
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(item[0] for item in items[:n])


def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


def main():
    print("Loading confidence for 9 models...")
    confidences = {}
    easy_sets = {}
    for model in MODELS:
        try:
            conf = load_confidence(model)
            confidences[model] = conf
            easy_sets[model] = get_easy_set(conf, K)
            print(f"  {model}: {len(conf)} examples, easy set size: {len(easy_sets[model])}")
        except FileNotFoundError as e:
            print(f"  {model}: NOT FOUND - {e}")
            return

    # Compute 9x9 Jaccard matrix
    print("\nComputing Jaccard matrix...")
    matrix = np.zeros((len(MODELS), len(MODELS)))
    for i, m1 in enumerate(MODELS):
        for j, m2 in enumerate(MODELS):
            matrix[i][j] = jaccard(easy_sets[m1], easy_sets[m2])
            if i != j:
                print(f"  {m1} vs {m2}: {matrix[i][j]:.4f}")

    # Save matrix
    result = {
        "models": MODELS,
        "k_pct": K,
        "matrix": matrix.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "jaccard_matrix_9x9_k30.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)

    # Labels
    short_labels = [m.replace("t5-v1_1-", "T5v1.1-").replace("bert-", "BERT-").replace("pythia-", "Pythia-")
                    for m in MODELS]
    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # Annotate cells
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            color = "white" if matrix[i][j] > 0.6 else "black"
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    # Family boundaries
    for pos in [2.5, 5.5]:
        ax.axhline(y=pos, color="black", linewidth=2)
        ax.axvline(x=pos, color="black", linewidth=2)

    # Family labels on sides
    for idx, label in enumerate(FAMILY_LABELS):
        y_pos = idx * 3 + 1
        ax.text(-1.5, y_pos, label, ha="center", va="center", fontsize=8, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Jaccard Similarity", shrink=0.8)
    ax.set_title(f"Pairwise Jaccard Similarity of Easy Examples (top {K}%)\nMNLI 90K", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_jaccard_heatmap_9x9.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig1_jaccard_heatmap_9x9.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {FIG_DIR}/fig1_jaccard_heatmap_9x9.pdf")


if __name__ == "__main__":
    main()
