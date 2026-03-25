"""본문 분석 3: Model Size vs Shortcut 비율 (class-specific)."""
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

# All models with params (log scale x-axis)
MODELS = {
    # BERT family
    "bert-mini": {"params": 11e6, "family": "BERT"},
    "bert-small": {"params": 29e6, "family": "BERT"},
    "bert-medium": {"params": 42e6, "family": "BERT"},
    "bert-base": {"params": 110e6, "family": "BERT"},
    "bert-large": {"params": 335e6, "family": "BERT"},
    # T5 v1.1 family
    "t5-v1_1-small": {"params": 77e6, "family": "T5 v1.1"},
    "t5-v1_1-base": {"params": 248e6, "family": "T5 v1.1"},
    "t5-v1_1-large": {"params": 783e6, "family": "T5 v1.1"},
    # Pythia family
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

K = 30  # top k%


def load_confidence(model_name):
    for plan_dir in ["plan0", "plan5"]:
        path = f"/workspace/erase/outputs/{plan_dir}/confidence/{model_name}_90k_avg_conf.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


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
    # Load MNLI data
    print("Loading MNLI dataset...")
    ds = load_dataset("glue", "mnli", split="train")

    # Load 90K indices
    with open("/workspace/erase/outputs/plan0/data/all_90k_indices.json") as f:
        all_90k = json.load(f)
    all_90k_set = set(str(i) for i in all_90k)

    # Build index → data mapping for 90K
    idx_to_data = {}
    for idx in all_90k:
        item = ds[idx]
        idx_to_data[str(idx)] = {
            "premise": item["premise"],
            "hypothesis": item["hypothesis"],
            "label": item["label"],  # 0=entailment, 1=neutral, 2=contradiction
        }

    # Global baselines (class-specific)
    all_contradiction = [idx for idx, d in idx_to_data.items() if d["label"] == 2]
    all_entailment = [idx for idx, d in idx_to_data.items() if d["label"] == 0]

    global_negation = sum(has_negation(idx_to_data[idx]["hypothesis"]) for idx in all_contradiction) / len(all_contradiction)
    global_overlap = sum(lexical_overlap(idx_to_data[idx]["premise"], idx_to_data[idx]["hypothesis"]) >= 0.6
                         for idx in all_entailment) / len(all_entailment)

    print(f"Global baseline - Negation in contradiction: {global_negation:.4f}")
    print(f"Global baseline - Overlap>=0.6 in entailment: {global_overlap:.4f}")

    # Per-model shortcut ratios
    results = {
        "global_negation_ratio": global_negation,
        "global_overlap_ratio": global_overlap,
        "models": {},
    }

    for model_name, info in MODELS.items():
        conf = load_confidence(model_name)
        if conf is None:
            print(f"  {model_name}: confidence not found, skipping")
            continue

        # Filter to 90K indices only
        conf_90k = {k: v for k, v in conf.items() if k in all_90k_set}
        easy_set = get_easy_set(conf_90k, K)

        # Contradiction easy → negation ratio
        easy_contradiction = [idx for idx in easy_set if idx_to_data.get(idx, {}).get("label") == 2]
        if len(easy_contradiction) > 0:
            neg_ratio = sum(has_negation(idx_to_data[idx]["hypothesis"]) for idx in easy_contradiction) / len(easy_contradiction)
        else:
            neg_ratio = 0

        # Entailment easy → overlap ratio
        easy_entailment = [idx for idx in easy_set if idx_to_data.get(idx, {}).get("label") == 0]
        if len(easy_entailment) > 0:
            ovlp_ratio = sum(lexical_overlap(idx_to_data[idx]["premise"], idx_to_data[idx]["hypothesis"]) >= 0.6
                             for idx in easy_entailment) / len(easy_entailment)
        else:
            ovlp_ratio = 0

        results["models"][model_name] = {
            "params": info["params"],
            "family": info["family"],
            "negation_ratio": neg_ratio,
            "overlap_ratio": ovlp_ratio,
            "n_easy": len(easy_set),
            "n_easy_contradiction": len(easy_contradiction),
            "n_easy_entailment": len(easy_entailment),
        }
        print(f"  {model_name}: neg={neg_ratio:.4f} (n={len(easy_contradiction)}), overlap={ovlp_ratio:.4f} (n={len(easy_entailment)})")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "shortcut_by_model.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    family_colors = {"BERT": "tab:blue", "T5 v1.1": "tab:orange", "Pythia": "tab:green"}
    family_markers = {"BERT": "o", "T5 v1.1": "s", "Pythia": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Negation ratio in contradiction easy examples
    for family in ["BERT", "T5 v1.1", "Pythia"]:
        xs, ys = [], []
        for model_name, data in results["models"].items():
            if data["family"] == family:
                xs.append(data["params"])
                ys.append(data["negation_ratio"])
        if xs:
            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax1.plot(xs, ys, color=family_colors[family], marker=family_markers[family],
                     label=family, linewidth=1.5, markersize=6)

    ax1.axhline(y=global_negation, color="gray", linestyle="--", linewidth=1, label="Global baseline")
    ax1.set_xscale("log")
    ax1.set_xlabel("Model Parameters", fontsize=11)
    ax1.set_ylabel("Negation Ratio", fontsize=11)
    ax1.set_title("(a) Negation in Contradiction Easy Examples", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Overlap ratio in entailment easy examples
    for family in ["BERT", "T5 v1.1", "Pythia"]:
        xs, ys = [], []
        for model_name, data in results["models"].items():
            if data["family"] == family:
                xs.append(data["params"])
                ys.append(data["overlap_ratio"])
        if xs:
            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax2.plot(xs, ys, color=family_colors[family], marker=family_markers[family],
                     label=family, linewidth=1.5, markersize=6)

    ax2.axhline(y=global_overlap, color="gray", linestyle="--", linewidth=1, label="Global baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("Model Parameters", fontsize=11)
    ax2.set_ylabel("Lexical Overlap ≥ 0.6 Ratio", fontsize=11)
    ax2.set_title("(b) Lexical Overlap in Entailment Easy Examples", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Model Size vs Shortcut Ratio (top {K}% easy examples, class-specific)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_scale_vs_shortcut.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, "fig3_scale_vs_shortcut.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {FIG_DIR}/fig3_scale_vs_shortcut.pdf")


if __name__ == "__main__":
    main()
