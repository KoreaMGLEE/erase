"""Fig 2: publication style — small text, large plot area, thin lines."""
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

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]
MODEL_SEEDS = {
    "t5-v1_1-large": [2, 3, 4], "pythia-410m": [1, 3, 4],
    "pythia-6.9b": [1, 4, 5], "pythia-12b": [1, 3, 4],
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

def jaccard(a, b):
    return len(a & b) / len(a | b) if len(a | b) > 0 else 0

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

def load_all_confs():
    confs = {}
    for model in ALL_MODELS:
        seeds = MODEL_SEEDS.get(model, DEFAULT_SEEDS)
        sc = {}
        for seed in seeds:
            p = os.path.join(CONF_DIR, f"{model}_seed{seed}", "avg_conf.json")
            if os.path.exists(p):
                with open(p) as f:
                    sc[seed] = json.load(f)
        if sc:
            confs[model] = sc
    return confs

def compute_all(annotations, confs):
    human_easy = set(k for k, v in annotations.items() if v["difficulty"] == "Low")
    ann_ids = set(annotations.keys())
    n_easy = len(human_easy)
    N = len(ann_ids)
    random_bl = n_easy / (2 * N - n_easy)
    jaccard_res, diff_res, grade_res = {}, {}, {}
    for model in ALL_MODELS:
        if model not in confs:
            continue
        jacs, d_rhos, g_rhos = [], [], []
        for seed, conf in confs[model].items():
            ac = {k: v for k, v in conf.items() if k in ann_ids}
            top = set(k for k, _ in sorted(ac.items(), key=lambda x: -x[1])[:n_easy])
            jacs.append(jaccard(top, human_easy))
            cv, lv = [], []
            for k, v in conf.items():
                if k in annotations and annotations[k]["difficulty"] in DIFFICULTY_ORDER:
                    cv.append(v); lv.append(DIFFICULTY_ORDER[annotations[k]["difficulty"]])
            if len(cv) >= 10:
                d_rhos.append(stats.spearmanr(cv, lv)[0])
            cv2, lv2 = [], []
            for k, v in conf.items():
                if k in annotations and annotations[k]["grade"] in GRADE_ORDINAL:
                    cv2.append(v); lv2.append(GRADE_ORDINAL[annotations[k]["grade"]])
            if len(cv2) >= 10:
                g_rhos.append(stats.spearmanr(cv2, lv2)[0])
        if jacs:
            jaccard_res[model] = {"mean": np.mean(jacs), "std": np.std(jacs)}
        if d_rhos:
            diff_res[model] = {"mean": np.mean(d_rhos), "std": np.std(d_rhos)}
        if g_rhos:
            grade_res[model] = {"mean": np.mean(g_rhos), "std": np.std(g_rhos)}
    return jaccard_res, diff_res, grade_res, random_bl

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


def draw(fig_w, fig_h, title_fs, label_fs, tick_fs, legend_fs,
         lw, ms, shade_alpha, spine_w, tag):

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": spine_w,
        "xtick.major.width": spine_w,
        "ytick.major.width": spine_w,
        "xtick.minor.width": spine_w * 0.7,
        "ytick.minor.width": spine_w * 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    family_colors = {"BERT": "#1f77b4", "T5 v1.1": "#d62728", "Pythia": "#2ca02c"}
    family_markers = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))

    panels = [
        (axes[0], jaccard_res, False, "Jaccard similarity", r"$\bf{(a)}$ Human-easy overlap"),
        (axes[1], diff_res, True, r"$-\rho_s$ (conf. vs difficulty)", r"$\bf{(b)}$ Difficulty alignment"),
        (axes[2], grade_res, True, r"$-\rho_s$ (conf. vs grade level)", r"$\bf{(c)}$ Grade-level alignment"),
    ]

    for ax, res, negate, ylabel, title in panels:
        for family in ["BERT", "T5 v1.1", "Pythia"]:
            xs, ys, errs = get_family_data(res, family)
            if xs is None:
                continue
            plot_ys = -ys if negate else ys
            ax.plot(xs, plot_ys, color=family_colors[family], marker=family_markers[family],
                    label=family, linewidth=lw, markersize=ms,
                    markeredgecolor=family_colors[family], markeredgewidth=0.3,
                    markerfacecolor=family_colors[family], zorder=3)
            ax.fill_between(xs, plot_ys - errs, plot_ys + errs,
                            color=family_colors[family], alpha=shade_alpha, zorder=1)

        if not negate:
            ax.axhline(y=random_bl, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
            ax.text(0.96, 0.06, "random", fontsize=tick_fs - 1, color="gray",
                    ha="right", transform=ax.transAxes, style='italic')
        else:
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

        ax.set_xscale("log")
        ax.set_xlabel("Parameters", fontsize=label_fs, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=label_fs, labelpad=2)
        ax.set_title(title, fontsize=title_fs, pad=4)
        ax.tick_params(labelsize=tick_fs, pad=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())

    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, labels, fontsize=legend_fs, loc='lower right',
                   frameon=True, framealpha=0.9, edgecolor='none',
                   handlelength=1.2, handletextpad=0.3,
                   borderpad=0.3, labelspacing=0.3)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.90, wspace=0.32)
    fig.savefig(os.path.join(FIG_DIR, f"fig2_pub_{tag}.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"fig2_pub_{tag}.png"), dpi=300, bbox_inches="tight")
    print(f"  Saved: fig2_pub_{tag}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading data...")
    annotations = load_annotations()
    confs = load_all_confs()
    jaccard_res, diff_res, grade_res, random_bl = compute_all(annotations, confs)

    # (w, h, title, label, tick, legend, lw, ms, shade, spine_w, tag)
    variants = [
        # Target: text ~5-7pt range, thin lines, small markers, max plot area
        (7.0, 2.0, 7.5, 6.5, 5.5, 5.5, 0.8, 2.5, 0.07, 0.4, "a"),
        (7.0, 2.2, 7.5, 6.5, 5.5, 5.5, 0.8, 2.5, 0.07, 0.4, "b"),
        (7.0, 2.0, 8,   7,   6,   6,   0.9, 3.0, 0.07, 0.4, "c"),
        (7.0, 2.2, 8,   7,   6,   6,   0.9, 3.0, 0.07, 0.4, "d"),
    ]

    for args in variants:
        draw(*args)

    print("Done!")
