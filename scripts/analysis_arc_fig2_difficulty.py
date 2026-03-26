"""본문 분석 2 (ARC): Model Confidence vs Human Annotations.
Version A: Continuous confidence Spearman
Version B: Discretized confidence Spearman (bin sizes match label distribution)
Each version: (left) Difficulty alignment, (right) Grade alignment
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from datasets import load_dataset

OUTPUT_DIR = "/workspace/erase/outputs/plan2_v2/analysis"
FIG_DIR = "/workspace/erase/figures"
CONF_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
ANNOTATION_PATH = "/workspace/erase/easy-to-hard-generalization/data/arc-challenge-easy-annotations.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    "t5-v1_1-large": "T5 v1.1", "t5-v1_1-xl": "T5 v1.1",
    "t5-v1_1-xxl": "T5 v1.1",
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

DIFFICULTY_ORDER = {"Low": 0, "Medium": 1, "High": 2}
GRADE_ORDINAL = {"Grade 03": 3, "Grade 04": 4, "Grade 05": 5,
                 "Grade 06": 6, "Grade 07": 7, "Grade 08": 8}


def load_per_seed_confidence(model_name):
    seeds = MODEL_SEEDS.get(model_name, DEFAULT_SEEDS)
    seed_confs = {}
    for seed in seeds:
        path = os.path.join(CONF_DIR, f"{model_name}_seed{seed}", "avg_conf.json")
        if os.path.exists(path):
            with open(path) as f:
                seed_confs[seed] = json.load(f)
    return seed_confs


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


def discretize_by_label_dist(conf_vals, label_vals, label_to_ordinal):
    """Discretize confidence into bins matching label distribution.
    Highest confidence -> lowest ordinal (easiest).
    """
    from collections import Counter
    label_counts = Counter(label_vals)
    # Sort ordinals
    ordinals_sorted = sorted(label_to_ordinal.values())
    counts_per_level = [label_counts[o] for o in ordinals_sorted]

    # Sort indices by confidence descending
    sorted_idx = np.argsort(conf_vals)[::-1]
    disc = np.zeros(len(conf_vals), dtype=int)
    pos = 0
    for level, count in zip(ordinals_sorted, counts_per_level):
        disc[sorted_idx[pos:pos + count]] = level
        pos += count
    # Handle remainder (if any rounding)
    if pos < len(conf_vals):
        disc[sorted_idx[pos:]] = ordinals_sorted[-1]
    return disc


def compute_spearman_both(seed_confs, annotations, label_key, label_to_ordinal):
    """Compute continuous and discretized Spearman per seed."""
    cont_rhos, disc_rhos = [], []
    for seed, conf in seed_confs.items():
        conf_vals, label_vals = [], []
        for k, v in conf.items():
            if k in annotations and annotations[k][label_key] in label_to_ordinal:
                conf_vals.append(v)
                label_vals.append(label_to_ordinal[annotations[k][label_key]])
        if len(conf_vals) < 10:
            continue
        conf_arr = np.array(conf_vals)
        label_arr = np.array(label_vals)

        # Continuous
        rho_c, _ = stats.spearmanr(conf_arr, label_arr)
        cont_rhos.append(rho_c)

        # Discretized
        disc = discretize_by_label_dist(conf_arr, label_arr, label_to_ordinal)
        rho_d, _ = stats.spearmanr(disc, label_arr)
        disc_rhos.append(rho_d)

    return cont_rhos, disc_rhos


def plot_spearman_panel(ax, available_models, results, title, ylabel):
    """Plot model size vs -Spearman with shade."""
    family_colors = {"BERT": "tab:blue", "T5 v1.1": "tab:orange", "Pythia": "tab:green"}
    family_markers = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

    for family in ["BERT", "T5 v1.1", "Pythia"]:
        xs, ys, errs = [], [], []
        for model in available_models:
            if FAMILIES[model] == family and model in results:
                xs.append(PARAMS[model])
                ys.append(-results[model]["mean"])
                errs.append(results[model]["std"])
        if xs:
            order = np.argsort(xs)
            xs = np.array([xs[i] for i in order])
            ys = np.array([ys[i] for i in order])
            errs = np.array([errs[i] for i in order])
            ax.plot(xs, ys, color=family_colors[family], marker=family_markers[family],
                    label=family, linewidth=1.5, markersize=6)
            ax.fill_between(xs, ys - errs, ys + errs, color=family_colors[family], alpha=0.15)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    print("Loading annotations...")
    annotations = load_annotations()
    print(f"  Matched: {len(annotations)}")

    print("Loading model confidences...")
    model_seed_confs = {}
    for model in ALL_MODELS:
        sc = load_per_seed_confidence(model)
        if sc:
            model_seed_confs[model] = sc
    available_models = [m for m in ALL_MODELS if m in model_seed_confs]

    # Compute Spearman for difficulty and grade, both continuous and discretized
    diff_cont, diff_disc = {}, {}
    grade_cont, grade_disc = {}, {}

    print("\n=== Difficulty Spearman ===")
    print(f"{'Model':20s} {'Cont −ρ':>10s} {'Disc −ρ':>10s}")
    for model in available_models:
        c_rhos, d_rhos = compute_spearman_both(
            model_seed_confs[model], annotations, "difficulty", DIFFICULTY_ORDER)
        if c_rhos:
            diff_cont[model] = {"mean": float(np.mean(c_rhos)), "std": float(np.std(c_rhos))}
            diff_disc[model] = {"mean": float(np.mean(d_rhos)), "std": float(np.std(d_rhos))}
            print(f"  {model:20s} {-np.mean(c_rhos):+.4f}±{np.std(c_rhos):.4f}  "
                  f"{-np.mean(d_rhos):+.4f}±{np.std(d_rhos):.4f}")

    print("\n=== Grade Spearman (G3-G8 only) ===")
    print(f"{'Model':20s} {'Cont −ρ':>10s} {'Disc −ρ':>10s}")
    for model in available_models:
        c_rhos, d_rhos = compute_spearman_both(
            model_seed_confs[model], annotations, "grade", GRADE_ORDINAL)
        if c_rhos:
            grade_cont[model] = {"mean": float(np.mean(c_rhos)), "std": float(np.std(c_rhos))}
            grade_disc[model] = {"mean": float(np.mean(d_rhos)), "std": float(np.std(d_rhos))}
            print(f"  {model:20s} {-np.mean(c_rhos):+.4f}±{np.std(c_rhos):.4f}  "
                  f"{-np.mean(d_rhos):+.4f}±{np.std(d_rhos):.4f}")

    # ========== Version A: Continuous ==========
    fig_a, (ax_a1, ax_a2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_spearman_panel(ax_a1, available_models, diff_cont,
                        "(a) Difficulty Alignment", "−Spearman ρ (conf vs difficulty)\n↑ = better alignment")
    plot_spearman_panel(ax_a2, available_models, grade_cont,
                        "(b) Grade Level Alignment", "−Spearman ρ (conf vs grade)\n↑ = better alignment")
    fig_a.suptitle("ARC: Continuous Confidence vs Human Annotations\nmean ± std over 3 seeds",
                   fontsize=13, y=1.02)
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(FIG_DIR, "fig2_arc_continuous.pdf"), dpi=150, bbox_inches="tight")
    fig_a.savefig(os.path.join(FIG_DIR, "fig2_arc_continuous.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved Version A: {FIG_DIR}/fig2_arc_continuous.png")

    # ========== Version B: Discretized ==========
    fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_spearman_panel(ax_b1, available_models, diff_disc,
                        "(a) Difficulty Alignment", "−Spearman ρ (disc. conf vs difficulty)\n↑ = better alignment")
    plot_spearman_panel(ax_b2, available_models, grade_disc,
                        "(b) Grade Level Alignment", "−Spearman ρ (disc. conf vs grade)\n↑ = better alignment")
    fig_b.suptitle("ARC: Discretized Confidence vs Human Annotations\nmean ± std over 3 seeds",
                   fontsize=13, y=1.02)
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(FIG_DIR, "fig2_arc_discretized.pdf"), dpi=150, bbox_inches="tight")
    fig_b.savefig(os.path.join(FIG_DIR, "fig2_arc_discretized.png"), dpi=150, bbox_inches="tight")
    print(f"Saved Version B: {FIG_DIR}/fig2_arc_discretized.png")

    # Save data
    result = {
        "n_annotated": len(annotations),
        "difficulty_continuous": diff_cont,
        "difficulty_discretized": diff_disc,
        "grade_continuous": grade_cont,
        "grade_discretized": grade_disc,
    }
    with open(os.path.join(OUTPUT_DIR, "arc_difficulty_analysis.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
