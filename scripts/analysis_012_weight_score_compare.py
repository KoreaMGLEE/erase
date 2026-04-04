"""Compare weight score formulas on real 90K confidence data.
Shows distribution, SC ratio by weight bin, and hard-for-both behavior.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = "/workspace/erase/outputs"
OUT_DIR = "/workspace/erase/outputs/plan12_analysis"

# Load confidences
CONF_90K = {
    "t5-v1_1-large": "plan5/confidence/t5-v1_1-large_90k_avg_conf.json",
    "bert-mini":     "plan0/confidence/bert-mini_90k_avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_90k_avg_conf.json",
    "pythia-70m":    "plan0/confidence/pythia-70m_90k_avg_conf.json",
}


def load_heuristics(all_ids):
    """Compute HANS-style shortcut labels."""
    from datasets import load_dataset
    ds = load_dataset("glue", "mnli", split="train")
    negation_words = {"no", "not", "never", "nothing", "nobody", "none", "nor", "neither",
                      "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
                      "won't", "wouldn't", "couldn't", "shouldn't", "can't", "cannot"}
    id_set = set(all_ids)
    heuristics = {}
    for i, ex in enumerate(ds):
        idx = str(i)
        if idx not in id_set:
            continue
        premise_tokens = set(ex["premise"].lower().split())
        hyp_tokens = set(ex["hypothesis"].lower().split())
        overlap = len(premise_tokens & hyp_tokens) / max(len(hyp_tokens), 1)
        has_negation = bool(hyp_tokens & negation_words)
        label = ex["label"]
        is_shortcut = (overlap > 0.5 and label == 0) or (has_negation and label == 2)
        heuristics[idx] = is_shortcut
    return heuristics


def main():
    # Load all confidences
    confs = {}
    for model, path in CONF_90K.items():
        with open(os.path.join(BASE, path)) as f:
            confs[model] = json.load(f)
        print(f"  Loaded {model}: {len(confs[model])}")

    # Common IDs
    all_ids = sorted(set.intersection(*[set(c.keys()) for c in confs.values()]))
    print(f"  Common IDs: {len(all_ids)}")

    # Compute aggregate small model confidence (mean of 3)
    L = np.array([confs["t5-v1_1-large"].get(idx, 0) for idx in all_ids])
    S_models = ["bert-mini", "t5-v1_1-small", "pythia-70m"]
    S_each = np.array([[confs[m].get(idx, 0) for idx in all_ids] for m in S_models])
    S = S_each.mean(axis=0)  # mean aggregation

    print(f"\n  L stats: mean={L.mean():.3f}, std={L.std():.3f}, "
          f"<0.5: {(L<0.5).sum()}, >0.9: {(L>0.9).sum()}")
    print(f"  S stats: mean={S.mean():.3f}, std={S.std():.3f}, "
          f"<0.5: {(S<0.5).sum()}, >0.9: {(S>0.9).sum()}")

    # Load heuristics
    print("\n  Loading HANS heuristics...")
    heuristics = load_heuristics(all_ids)
    is_sc = np.array([heuristics.get(idx, False) for idx in all_ids])
    print(f"  Global SC ratio: {is_sc.mean():.3f}")

    # Also load hyp-only confidence as alternative SC measure
    hyp_path = os.path.join(OUT_DIR, "hyp_only_confidence.json")
    if os.path.exists(hyp_path):
        with open(hyp_path) as f:
            hyp_conf = json.load(f)
        hyp_sc = np.array([hyp_conf.get(idx, 0.33) for idx in all_ids])
    else:
        hyp_sc = None

    # === Weight formulas ===
    formulas = {}

    # F1: Linear (current 011B)
    formulas["F1: L×(1-S)"] = L * (1 - S)

    # F2: Quadratic penalty
    formulas["F2: L×(1-S²)"] = L * (1 - S**2)

    # F3: Cubic penalty
    formulas["F3: L×(1-S³)"] = L * (1 - S**3)

    # F4: Quality-only (no L upweight)
    formulas["F4: 1-S²"] = 1 - S**2

    # F5: Hybrid — quality base + L bonus (multiplicative)
    # (1-S^γ) ensures shortcut suppression, (ε+L) gives gentle L bonus
    # without crushing hard-for-both
    eps = 0.5
    formulas[f"F5: (1-S²)×({eps}+L)"] = (1 - S**2) * (eps + L)

    # F6: Hybrid — floor on L
    floor = 0.3
    formulas[f"F6: max(L,{floor})×(1-S²)"] = np.maximum(L, floor) * (1 - S**2)

    # Normalize all to [0, 1]
    for name in formulas:
        w = formulas[name]
        w_min, w_max = w.min(), w.max()
        if w_max > w_min:
            formulas[name] = (w - w_min) / (w_max - w_min)

    # === Analysis ===
    # Categorize examples
    # Genuine easy: L > 0.8, S < 0.5
    # Shortcut shared: L > 0.8, S > 0.8
    # Shortcut S-only: L < 0.5, S > 0.8
    # Hard for both: L < 0.5, S < 0.5
    categories = {}
    categories["Genuine easy (L>0.8, S<0.5)"] = (L > 0.8) & (S < 0.5)
    categories["Shortcut shared (L>0.8, S>0.8)"] = (L > 0.8) & (S > 0.8)
    categories["Shortcut S-only (L<0.5, S>0.8)"] = (L < 0.5) & (S > 0.8)
    categories["Hard for both (L<0.5, S<0.5)"] = (L < 0.5) & (S < 0.5)
    categories["Medium (rest)"] = ~((L > 0.8) & (S < 0.5)) & ~((L > 0.8) & (S > 0.8)) & \
                                   ~((L < 0.5) & (S > 0.8)) & ~((L < 0.5) & (S < 0.5))

    print("\n=== Category sizes ===")
    for cat, mask in categories.items():
        sc_rate = is_sc[mask].mean() if mask.sum() > 0 else 0
        print(f"  {cat}: n={mask.sum()}, SC={sc_rate:.3f}")

    print("\n=== Mean weight by category ===")
    header = f"  {'Category':<40}" + "".join(f"{name:>20}" for name in formulas)
    print(header)
    print("  " + "-" * (40 + 20 * len(formulas)))
    for cat, mask in categories.items():
        row = f"  {cat:<40}"
        for name, w in formulas.items():
            row += f"{w[mask].mean():>20.3f}"
        print(row)

    # === SC ratio by weight decile ===
    print("\n=== SC ratio by weight decile (HANS) ===")
    for name, w in formulas.items():
        print(f"\n  {name}:")
        percentiles = np.percentile(w, np.arange(0, 101, 10))
        for i in range(10):
            lo, hi = percentiles[i], percentiles[i+1]
            if i == 9:
                mask = (w >= lo) & (w <= hi)
            else:
                mask = (w >= lo) & (w < hi)
            if mask.sum() == 0:
                continue
            sc_rate = is_sc[mask].mean()
            mean_L = L[mask].mean()
            mean_S = S[mask].mean()
            print(f"    Decile {i}: w=[{lo:.3f},{hi:.3f}] n={mask.sum():>5} "
                  f"SC={sc_rate:.3f} L={mean_L:.3f} S={mean_S:.3f}")

    # === Figure ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Weight Score Comparison on Real 90K Data (L=T5-large, S=mean(mini,small,70m))",
                 fontsize=13, fontweight="bold")

    formula_names = list(formulas.keys())
    cat_colors = {
        "Genuine easy (L>0.8, S<0.5)": "#2ecc71",
        "Shortcut shared (L>0.8, S>0.8)": "#e74c3c",
        "Shortcut S-only (L<0.5, S>0.8)": "#e67e22",
        "Hard for both (L<0.5, S<0.5)": "#3498db",
        "Medium (rest)": "#95a5a6",
    }

    for fi, fname in enumerate(formula_names):
        ax = axes[fi // 3][fi % 3]
        w = formulas[fname]

        # Histogram by category
        for cat, mask in categories.items():
            if mask.sum() == 0:
                continue
            ax.hist(w[mask], bins=50, alpha=0.5, label=f"{cat.split('(')[0].strip()} ({mask.sum()})",
                    color=cat_colors[cat], density=True)

        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.set_xlabel("Weight (normalized [0,1])")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, "figures", "fig11_weight_score_compare.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {outpath}")

    # === Summary table for notes ===
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Formula':<25} {'Genuine':>10} {'SC-shared':>10} {'SC-Sonly':>10} {'Hard-both':>10} {'Medium':>10} {'Separation':>12}")
    print("-" * 87)
    for name, w in formulas.items():
        ge = w[categories["Genuine easy (L>0.8, S<0.5)"]].mean()
        ss = w[categories["Shortcut shared (L>0.8, S>0.8)"]].mean()
        so = w[categories["Shortcut S-only (L<0.5, S>0.8)"]].mean()
        hb = w[categories["Hard for both (L<0.5, S<0.5)"]].mean()
        md = w[categories["Medium (rest)"]].mean()
        sep = ge - max(ss, so)  # genuine easy vs worst shortcut
        print(f"{name:<25} {ge:>10.3f} {ss:>10.3f} {so:>10.3f} {hb:>10.3f} {md:>10.3f} {sep:>12.3f}")


if __name__ == "__main__":
    main()
