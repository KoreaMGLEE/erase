"""본문 분석 3 v2: Model Size vs Shortcut 비율 — split별 평균 + std shade."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

OUTPUT_DIR = "/workspace/erase/outputs/plan5/analysis"
FIG_DIR = "/workspace/erase/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "bert-mini": {"params": 11e6, "family": "BERT"},
    "bert-small": {"params": 29e6, "family": "BERT"},
    "bert-medium": {"params": 42e6, "family": "BERT"},
    "bert-base": {"params": 110e6, "family": "BERT"},
    "bert-large": {"params": 335e6, "family": "BERT"},
    "t5-v1_1-small": {"params": 77e6, "family": "T5 v1.1"},
    "t5-v1_1-base": {"params": 248e6, "family": "T5 v1.1"},
    "t5-v1_1-large": {"params": 783e6, "family": "T5 v1.1"},
    "t5-v1_1-xl": {"params": 3e9, "family": "T5 v1.1"},
    "pythia-14m": {"params": 14e6, "family": "Pythia"},
    "pythia-31m": {"params": 31e6, "family": "Pythia"},
    "pythia-70m": {"params": 70e6, "family": "Pythia"},
    "pythia-160m": {"params": 160e6, "family": "Pythia"},
    "pythia-410m": {"params": 410e6, "family": "Pythia"},
    "pythia-1b": {"params": 1e9, "family": "Pythia"},
    "pythia-1.4b": {"params": 1.4e9, "family": "Pythia"},
    "pythia-2.8b": {"params": 2.8e9, "family": "Pythia"},
    "pythia-6.9b": {"params": 6.9e9, "family": "Pythia"},
}

NEGATION_WORDS = {"no", "not", "never", "nothing", "nobody", "neither",
                  "nor", "none", "cannot", "can't", "don't", "doesn't",
                  "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
                  "isn't", "aren't", "wasn't", "weren't"}

K = 30


def load_split_confidences(model_name):
    split_confs = {}
    for split in ["split1", "split2", "split3"]:
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


def has_negation(hypothesis):
    tokens = set(hypothesis.lower().split())
    return len(tokens & NEGATION_WORDS) > 0


def lexical_overlap(premise, hypothesis):
    p_tokens = set(premise.lower().split())
    h_tokens = set(hypothesis.lower().split())
    if len(h_tokens) == 0:
        return 0.0
    return len(p_tokens & h_tokens) / len(h_tokens)


def main():
    print("Loading MNLI dataset...")
    ds = load_dataset("glue", "mnli", split="train")

    with open("/workspace/erase/outputs/plan0/data/all_90k_indices.json") as f:
        all_90k = json.load(f)

    idx_to_data = {}
    for idx in all_90k:
        item = ds[idx]
        idx_to_data[str(idx)] = {
            "premise": item["premise"],
            "hypothesis": item["hypothesis"],
            "label": item["label"],
        }

    # Global baselines (class-specific)
    all_contradiction = [idx for idx, d in idx_to_data.items() if d["label"] == 2]
    all_entailment = [idx for idx, d in idx_to_data.items() if d["label"] == 0]
    global_negation = sum(has_negation(idx_to_data[idx]["hypothesis"]) for idx in all_contradiction) / len(all_contradiction)
    global_overlap = sum(lexical_overlap(idx_to_data[idx]["premise"], idx_to_data[idx]["hypothesis"]) >= 0.6
                         for idx in all_entailment) / len(all_entailment)
    print(f"Global: negation={global_negation:.4f}, overlap={global_overlap:.4f}")

    # Per-model, per-split shortcut ratios
    results = {"global_negation": global_negation, "global_overlap": global_overlap, "models": {}}

    for model_name, info in MODELS.items():
        split_confs = load_split_confidences(model_name)
        if len(split_confs) == 0:
            print(f"  {model_name}: no data, skipping")
            continue

        neg_ratios = []
        ovlp_ratios = []

        for split, conf in split_confs.items():
            easy_set = get_easy_set(conf, K)

            easy_contr = [idx for idx in easy_set if idx_to_data.get(idx, {}).get("label") == 2]
            easy_entail = [idx for idx in easy_set if idx_to_data.get(idx, {}).get("label") == 0]

            if len(easy_contr) > 0:
                neg_ratios.append(sum(has_negation(idx_to_data[idx]["hypothesis"]) for idx in easy_contr) / len(easy_contr))
            if len(easy_entail) > 0:
                ovlp_ratios.append(sum(lexical_overlap(idx_to_data[idx]["premise"], idx_to_data[idx]["hypothesis"]) >= 0.6
                                       for idx in easy_entail) / len(easy_entail))

        results["models"][model_name] = {
            "params": info["params"],
            "family": info["family"],
            "neg_mean": float(np.mean(neg_ratios)) if neg_ratios else 0,
            "neg_std": float(np.std(neg_ratios)) if neg_ratios else 0,
            "ovlp_mean": float(np.mean(ovlp_ratios)) if ovlp_ratios else 0,
            "ovlp_std": float(np.std(ovlp_ratios)) if ovlp_ratios else 0,
            "n_splits": len(split_confs),
        }
        print(f"  {model_name}: neg={results['models'][model_name]['neg_mean']:.4f}±{results['models'][model_name]['neg_std']:.4f}, "
              f"ovlp={results['models'][model_name]['ovlp_mean']:.4f}±{results['models'][model_name]['ovlp_std']:.4f} ({len(split_confs)} splits)")

    with open(os.path.join(OUTPUT_DIR, "shortcut_by_model_split_avg.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot with shade
    family_colors = {"BERT": "tab:blue", "T5 v1.1": "tab:orange", "Pythia": "tab:green"}
    family_markers = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for panel, (ax, metric, ylabel, title, global_val) in enumerate([
        (ax1, "neg", "Negation Ratio", "(a) Negation in Contradiction Easy Examples", global_negation),
        (ax2, "ovlp", "Lexical Overlap ≥ 0.6 Ratio", "(b) Lexical Overlap in Entailment Easy Examples", global_overlap),
    ]):
        for family in ["BERT", "T5 v1.1", "Pythia"]:
            xs, ys, errs = [], [], []
            for model_name, data in results["models"].items():
                if data["family"] == family:
                    xs.append(data["params"])
                    ys.append(data[f"{metric}_mean"])
                    errs.append(data[f"{metric}_std"])
            if xs:
                order = np.argsort(xs)
                xs = np.array([xs[i] for i in order])
                ys = np.array([ys[i] for i in order])
                errs = np.array([errs[i] for i in order])

                ax.plot(xs, ys, color=family_colors[family], marker=family_markers[family],
                        label=family, linewidth=1.5, markersize=6)
                ax.fill_between(xs, ys - errs, ys + errs, color=family_colors[family], alpha=0.15)

        ax.axhline(y=global_val, color="gray", linestyle="--", linewidth=1, label="Global baseline")
        ax.set_xscale("log")
        ax.set_xlabel("Model Parameters", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Model Size vs Shortcut Ratio (top {K}% easy, class-specific)\nmean ± std over 3 splits", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_scale_vs_shortcut_v2.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig3_scale_vs_shortcut_v2.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {FIG_DIR}/fig3_scale_vs_shortcut_v2.pdf")


if __name__ == "__main__":
    main()
