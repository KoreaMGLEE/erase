"""Fig 2 Combined: 3-panel figure (Jaccard + Difficulty + Grade).
Academic style, narrow panels, light shading.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from scipy import stats
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

DIFFICULTY_ORDER = {"Low": 0, "Medium": 1, "High": 2}
GRADE_ORDINAL = {"Grade 03": 3, "Grade 04": 4, "Grade 05": 5,
                 "Grade 06": 6, "Grade 07": 7, "Grade 08": 8}


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


def load_per_seed_confidence(model_name):
    seeds = MODEL_SEEDS.get(model_name, DEFAULT_SEEDS)
    seed_confs = {}
    for seed in seeds:
        path = os.path.join(CONF_DIR, f"{model_name}_seed{seed}", "avg_conf.json")
        if os.path.exists(path):
            with open(path) as f:
                seed_confs[seed] = json.load(f)
    return seed_confs


def compute_jaccard_results(annotations, model_seed_confs):
    human_easy = set(k for k, v in annotations.items() if v["difficulty"] == "Low")
    annotated_ids = set(annotations.keys())
    n_easy = len(human_easy)
    N = len(annotated_ids)
    random_baseline = n_easy / (2 * N - n_easy)

    results = {}
    for model in ALL_MODELS:
        if model not in model_seed_confs:
            continue
        jaccards = []
        for seed, conf in model_seed_confs[model].items():
            annotated_conf = {k: v for k, v in conf.items() if k in annotated_ids}
            sorted_by_conf = sorted(annotated_conf.items(), key=lambda x: -x[1])
            model_easy = set(k for k, _ in sorted_by_conf[:n_easy])
            jaccards.append(jaccard(model_easy, human_easy))
        if jaccards:
            results[model] = {"mean": float(np.mean(jaccards)), "std": float(np.std(jaccards))}

    return results, random_baseline, n_easy, N


def compute_spearman_results(annotations, model_seed_confs, label_key, label_to_ordinal):
    results = {}
    for model in ALL_MODELS:
        if model not in model_seed_confs:
            continue
        rhos = []
        for seed, conf in model_seed_confs[model].items():
            conf_vals, label_vals = [], []
            for k, v in conf.items():
                if k in annotations and annotations[k][label_key] in label_to_ordinal:
                    conf_vals.append(v)
                    label_vals.append(label_to_ordinal[annotations[k][label_key]])
            if len(conf_vals) < 10:
                continue
            rho, _ = stats.spearmanr(np.array(conf_vals), np.array(label_vals))
            rhos.append(rho)
        if rhos:
            results[model] = {"mean": float(np.mean(rhos)), "std": float(np.std(rhos))}
    return results


def get_family_data(results, family):
    xs, ys, errs = [], [], []
    for model in ALL_MODELS:
        if FAMILIES[model] == family and model in results:
            xs.append(PARAMS[model])
            ys.append(results[model]["mean"])
            errs.append(results[model]["std"])
    if not xs:
        return None, None, None
    order = np.argsort(xs)
    return np.array([xs[i] for i in order]), np.array([ys[i] for i in order]), np.array([errs[i] for i in order])


def make_figure(version="A"):
    """
    Version A: serif font, thicker lines, very light shade
    Version B: sans-serif, thinner lines, slightly more shade
    Version C: Nature-style, compact
    """
    print(f"Loading data for version {version}...")
    annotations = load_annotations()
    model_seed_confs = {}
    for model in ALL_MODELS:
        sc = load_per_seed_confidence(model)
        if sc:
            model_seed_confs[model] = sc

    jaccard_res, random_bl, n_easy, N = compute_jaccard_results(annotations, model_seed_confs)
    diff_res = compute_spearman_results(annotations, model_seed_confs, "difficulty", DIFFICULTY_ORDER)
    grade_res = compute_spearman_results(annotations, model_seed_confs, "grade", GRADE_ORDINAL)

    # Style configs per version
    if version == "A":
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        })
        fig_w, fig_h = 14, 3.2
        lw, ms = 1.8, 5
        shade_alpha = 0.08
        title_fs, label_fs, tick_fs, legend_fs = 9, 8.5, 7.5, 7
        family_colors = {"BERT": "#1f77b4", "T5 v1.1": "#d62728", "Pythia": "#2ca02c"}
    elif version == "B":
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
        })
        fig_w, fig_h = 14, 3.2
        lw, ms = 1.5, 4.5
        shade_alpha = 0.10
        title_fs, label_fs, tick_fs, legend_fs = 9, 8.5, 7.5, 7
        family_colors = {"BERT": "#4472C4", "T5 v1.1": "#ED7D31", "Pythia": "#70AD47"}
    else:  # C
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
        })
        fig_w, fig_h = 13, 2.8
        lw, ms = 1.6, 4
        shade_alpha = 0.06
        title_fs, label_fs, tick_fs, legend_fs = 8.5, 8, 7, 6.5
        family_colors = {"BERT": "#0072B2", "T5 v1.1": "#D55E00", "Pythia": "#009E73"}

    family_markers = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    # === Panel (a): Jaccard ===
    ax = axes[0]
    for family in ["BERT", "T5 v1.1", "Pythia"]:
        xs, ys, errs = get_family_data(jaccard_res, family)
        if xs is None:
            continue
        ax.plot(xs, ys, color=family_colors[family], marker=family_markers[family],
                label=family, linewidth=lw, markersize=ms, markeredgecolor='white', markeredgewidth=0.3)
        ax.fill_between(xs, ys - errs, ys + errs, color=family_colors[family], alpha=shade_alpha)
    ax.axhline(y=random_bl, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(1.5e10, random_bl + 0.002, "random", fontsize=tick_fs, color="gray", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=label_fs)
    ax.set_ylabel("Jaccard similarity", fontsize=label_fs)
    ax.set_title("(a) Human-easy overlap", fontsize=title_fs, fontweight='bold')
    ax.tick_params(labelsize=tick_fs)

    # === Panel (b): Difficulty Spearman ===
    ax = axes[1]
    for family in ["BERT", "T5 v1.1", "Pythia"]:
        xs, ys, errs = get_family_data(diff_res, family)
        if xs is None:
            continue
        ax.plot(xs, -ys, color=family_colors[family], marker=family_markers[family],
                label=family, linewidth=lw, markersize=ms, markeredgecolor='white', markeredgewidth=0.3)
        ax.fill_between(xs, -ys - errs, -ys + errs, color=family_colors[family], alpha=shade_alpha)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=label_fs)
    ax.set_ylabel(r"$-\rho_s$ (conf. vs difficulty)", fontsize=label_fs)
    ax.set_title("(b) Difficulty alignment", fontsize=title_fs, fontweight='bold')
    ax.tick_params(labelsize=tick_fs)

    # === Panel (c): Grade Spearman ===
    ax = axes[2]
    for family in ["BERT", "T5 v1.1", "Pythia"]:
        xs, ys, errs = get_family_data(grade_res, family)
        if xs is None:
            continue
        ax.plot(xs, -ys, color=family_colors[family], marker=family_markers[family],
                label=family, linewidth=lw, markersize=ms, markeredgecolor='white', markeredgewidth=0.3)
        ax.fill_between(xs, -ys - errs, -ys + errs, color=family_colors[family], alpha=shade_alpha)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=label_fs)
    ax.set_ylabel(r"$-\rho_s$ (conf. vs grade level)", fontsize=label_fs)
    ax.set_title("(c) Grade-level alignment", fontsize=title_fs, fontweight='bold')
    ax.tick_params(labelsize=tick_fs)

    # Common formatting
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='both', direction='in')
        ax.tick_params(which='minor', length=2)
        ax.tick_params(which='major', length=4)

    # Single legend at right of last panel
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, labels, fontsize=legend_fs, loc='lower right',
                   frameon=True, framealpha=0.9, edgecolor='none')

    fig.tight_layout(w_pad=2.5)

    tag = version.lower()
    fig.savefig(os.path.join(FIG_DIR, f"fig2_combined_v{tag}.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"fig2_combined_v{tag}.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: fig2_combined_v{tag}.pdf/png")
    plt.close(fig)


if __name__ == "__main__":
    for v in ["A", "B", "C"]:
        make_figure(v)
    print("\nDone!")
