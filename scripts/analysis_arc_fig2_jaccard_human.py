"""ARC: Jaccard of Human Easy (Low) subset vs Model confidence top-K.
Per model, per seed → mean ± std.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

FIG_DIR = "/workspace/erase/figures"
CONF_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
ANNOTATION_PATH = "/workspace/erase/easy-to-hard-generalization/data/arc-challenge-easy-annotations.json"
os.makedirs(FIG_DIR, exist_ok=True)

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]

MODEL_SEEDS = {
    "t5-v1_1-large": [2, 3, 4],
    "pythia-410m": [1, 3, 4],
    "pythia-6.9b": [1, 4, 5],
    "pythia-12b": [1, 3, 4],
}
DEFAULT_SEEDS = [1, 2, 3]

FAMILIES = {
    "bert-mini": "BERT", "bert-small": "BERT", "bert-medium": "BERT",
    "bert-base": "BERT", "bert-large": "BERT",
    "t5-v1_1-small": "T5 v1.1", "t5-v1_1-base": "T5 v1.1",
    "t5-v1_1-large": "T5 v1.1", "t5-v1_1-xl": "T5 v1.1", "t5-v1_1-xxl": "T5 v1.1",
    "pythia-14m": "Pythia", "pythia-31m": "Pythia", "pythia-70m": "Pythia",
    "pythia-160m": "Pythia", "pythia-410m": "Pythia", "pythia-1b": "Pythia",
    "pythia-1.4b": "Pythia", "pythia-2.8b": "Pythia", "pythia-6.9b": "Pythia",
    "pythia-12b": "Pythia",
}

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


def jaccard(set_a, set_b):
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0


def load_annotations():
    with open(ANNOTATION_PATH) as f:
        annot = json.load(f)
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    train_ids = set(item["id"] for item in list(easy) + list(challenge))
    matched = {}
    for ann_id, info in annot.items():
        arc_id = ann_id
        for prefix in ["ARCCH_", "ARCEZ_"]:
            if ann_id.startswith(prefix):
                arc_id = ann_id[len(prefix):]
                break
        if arc_id in train_ids:
            matched[arc_id] = info
    return matched


def main():
    print("Loading annotations...")
    annotations = load_annotations()
    human_easy = set(k for k, v in annotations.items() if v["difficulty"] == "Low")
    annotated_ids = set(annotations.keys())
    n_easy = len(human_easy)
    N = len(annotated_ids)
    random_baseline = n_easy / (2 * N - n_easy)
    print(f"  Human Easy (Low): {n_easy}, Total: {N}, Random baseline: {random_baseline:.4f}")

    print("\nComputing Jaccard per model per seed...")
    results = {}
    for model in ALL_MODELS:
        seeds = MODEL_SEEDS.get(model, DEFAULT_SEEDS)
        jaccards = []
        for seed in seeds:
            path = os.path.join(CONF_DIR, f"{model}_seed{seed}", "avg_conf.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                conf = json.load(f)
            annotated_conf = {k: v for k, v in conf.items() if k in annotated_ids}
            sorted_by_conf = sorted(annotated_conf.items(), key=lambda x: -x[1])
            model_easy = set(k for k, _ in sorted_by_conf[:n_easy])
            jaccards.append(jaccard(model_easy, human_easy))

        if jaccards:
            results[model] = {"mean": float(np.mean(jaccards)), "std": float(np.std(jaccards))}
            print(f"  {model:20s}: J={np.mean(jaccards):.4f} ± {np.std(jaccards):.4f}")

    # Plot
    available = [m for m in ALL_MODELS if m in results]
    family_colors = {"BERT": "tab:blue", "T5 v1.1": "tab:orange", "Pythia": "tab:green"}
    family_markers = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

    fig, ax = plt.subplots(figsize=(7, 5))
    for family in ["BERT", "T5 v1.1", "Pythia"]:
        xs, ys, errs = [], [], []
        for model in available:
            if FAMILIES[model] == family:
                xs.append(PARAMS[model])
                ys.append(results[model]["mean"])
                errs.append(results[model]["std"])
        if xs:
            order = np.argsort(xs)
            xs = np.array([xs[i] for i in order])
            ys = np.array([ys[i] for i in order])
            errs = np.array([errs[i] for i in order])
            ax.plot(xs, ys, color=family_colors[family], marker=family_markers[family],
                    label=family, linewidth=1.5, markersize=6)
            ax.fill_between(xs, ys - errs, ys + errs, color=family_colors[family], alpha=0.15)

    ax.axhline(y=random_baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Random ({random_baseline:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters", fontsize=11)
    ax.set_ylabel("Jaccard Similarity", fontsize=11)
    ax.set_title(f"Human Easy (Low, K={n_easy}) vs Model Top-{n_easy}\n"
                 f"ARC train (N={N}) — mean ± std over 3 seeds", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_arc_jaccard_human_easy.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig2_arc_jaccard_human_easy.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {FIG_DIR}/fig2_arc_jaccard_human_easy.png")


if __name__ == "__main__":
    main()
