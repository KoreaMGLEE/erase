"""Plan 012 v2: GPU-free analysis with proper HANS heuristics.
Implements 3 HANS heuristics + negation, improved figures.
"""
import json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

BASE = "/workspace/erase/outputs"
OUT_DIR = "/workspace/erase/outputs/plan12_analysis"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODELS = {
    "bert-mini": (11, "BERT"), "bert-small": (29, "BERT"), "bert-medium": (41, "BERT"),
    "bert-base": (110, "BERT"), "bert-large": (335, "BERT"),
    "t5-v1_1-small": (77, "T5"), "t5-v1_1-base": (250, "T5"),
    "t5-v1_1-large": (770, "T5"), "t5-v1_1-xl": (3000, "T5"),
    "pythia-14m": (14, "Pythia"), "pythia-70m": (70, "Pythia"), "pythia-160m": (160, "Pythia"),
    "pythia-410m": (410, "Pythia"), "pythia-1b": (1000, "Pythia"),
    "pythia-1.4b": (1400, "Pythia"), "pythia-2.8b": (2800, "Pythia"), "pythia-6.9b": (6900, "Pythia"),
}

CONF_90K = {
    "bert-mini": "plan0/confidence/bert-mini_90k_avg_conf.json",
    "bert-small": "plan0/confidence/bert-small_90k_avg_conf.json",
    "bert-medium": "plan0/confidence/bert-medium_90k_avg_conf.json",
    "bert-base": "plan0/confidence/bert-base_90k_avg_conf.json",
    "bert-large": "plan0/confidence/bert-large_90k_avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_90k_avg_conf.json",
    "t5-v1_1-base": "plan5/confidence/t5-v1_1-base_90k_avg_conf.json",
    "t5-v1_1-large": "plan5/confidence/t5-v1_1-large_90k_avg_conf.json",
    "t5-v1_1-xl": "plan5/confidence/t5-v1_1-xl_90k_avg_conf.json",
    "pythia-14m": "plan5/confidence/pythia-14m_90k_avg_conf.json",
    "pythia-70m": "plan0/confidence/pythia-70m_90k_avg_conf.json",
    "pythia-160m": "plan0/confidence/pythia-160m_90k_avg_conf.json",
    "pythia-410m": "plan0/confidence/pythia-410m_90k_avg_conf.json",
    "pythia-1b": "plan0/confidence/pythia-1b_90k_avg_conf.json",
    "pythia-1.4b": "plan0/confidence/pythia-1.4b_90k_avg_conf.json",
    "pythia-2.8b": "plan0/confidence/pythia-2.8b_90k_avg_conf.json",
    "pythia-6.9b": "plan0/confidence/pythia-6.9b_90k_avg_conf.json",
}

TARGETS = ["bert-base", "bert-large", "t5-v1_1-base", "t5-v1_1-xl", "pythia-410m", "pythia-2.8b"]
THRESHOLDS = [10, 20, 30, 50]

NEG_WORDS = {"no", "not", "never", "nothing", "nobody", "none", "nor", "neither",
             "doesn't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't",
             "won't", "wouldn't", "couldn't", "shouldn't", "can't", "cannot", "hardly", "rarely", "seldom"}


# ── HANS Heuristic Implementation ──

def tokenize(text):
    """Simple whitespace + punctuation tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def has_lexical_overlap(premise_tokens, hyp_tokens):
    """HANS heuristic 1: All hypothesis content words appear in premise."""
    if not hyp_tokens:
        return False
    return set(hyp_tokens).issubset(set(premise_tokens))


def has_subsequence(premise_tokens, hyp_tokens):
    """HANS heuristic 2: Hypothesis is a contiguous subsequence of premise."""
    if not hyp_tokens or len(hyp_tokens) > len(premise_tokens):
        return False
    for i in range(len(premise_tokens) - len(hyp_tokens) + 1):
        if premise_tokens[i:i+len(hyp_tokens)] == hyp_tokens:
            return True
    return False


def has_high_overlap(premise_tokens, hyp_tokens, threshold=0.7):
    """Soft overlap: ≥70% of hypothesis tokens appear in premise."""
    if not hyp_tokens:
        return False
    return len(set(hyp_tokens) & set(premise_tokens)) / len(set(hyp_tokens)) >= threshold


def has_negation(tokens):
    return bool(set(tokens) & NEG_WORDS)


def compute_heuristic_labels(all_ids):
    """Compute multiple shortcut heuristic flags for MNLI examples."""
    from datasets import load_dataset
    ds = load_dataset("glue", "mnli", split="train")

    id_set = set(all_ids)
    heuristics = {}

    for i, ex in enumerate(ds):
        idx = str(i)
        if idx not in id_set:
            continue

        p_tokens = tokenize(ex["premise"])
        h_tokens = tokenize(ex["hypothesis"])
        label = ex["label"]  # 0=entailment, 1=neutral, 2=contradiction

        lex_overlap = has_lexical_overlap(p_tokens, h_tokens)
        subseq = has_subsequence(p_tokens, h_tokens)
        high_overlap = has_high_overlap(p_tokens, h_tokens)
        neg = has_negation(h_tokens)

        # Shortcut-consistent: heuristic fires AND correctly predicts label
        # HANS heuristics predict entailment when overlap/subsequence is present
        # Negation predicts contradiction
        sc_overlap = lex_overlap and label == 0       # full overlap → entailment (correct)
        sc_subseq = subseq and label == 0             # subsequence → entailment (correct)
        sc_high_overlap = high_overlap and label == 0  # high overlap → entailment (correct)
        sc_negation = neg and label == 2               # negation → contradiction (correct)

        # Combined: any shortcut-consistent pattern
        is_shortcut_strict = sc_overlap or sc_subseq or sc_negation
        is_shortcut_soft = sc_high_overlap or sc_negation

        heuristics[idx] = {
            "lex_overlap": lex_overlap,
            "subsequence": subseq,
            "high_overlap": high_overlap,
            "negation": neg,
            "label": label,
            "sc_strict": is_shortcut_strict,     # full overlap/subseq + entail, or neg + contra
            "sc_soft": is_shortcut_soft,          # 70% overlap + entail, or neg + contra
            "sc_overlap_ent": sc_overlap,
            "sc_subseq_ent": sc_subseq,
            "sc_negation": sc_negation,
        }

    # Stats
    n = len(heuristics)
    print(f"  Heuristic stats (n={n}):")
    print(f"    Lexical overlap (full subset):  {sum(1 for h in heuristics.values() if h['lex_overlap'])/ n:.3f}")
    print(f"    Subsequence:                    {sum(1 for h in heuristics.values() if h['subsequence'])/n:.3f}")
    print(f"    High overlap (≥70%):            {sum(1 for h in heuristics.values() if h['high_overlap'])/n:.3f}")
    print(f"    Negation:                       {sum(1 for h in heuristics.values() if h['negation'])/n:.3f}")
    print(f"    SC strict (overlap/subseq+ent OR neg+contra): {sum(1 for h in heuristics.values() if h['sc_strict'])/n:.3f}")
    print(f"    SC soft (70%overlap+ent OR neg+contra):       {sum(1 for h in heuristics.values() if h['sc_soft'])/n:.3f}")

    return heuristics


# ── Analysis functions ──

def load_all_confidences():
    confs = {}
    for model, path in CONF_90K.items():
        full_path = os.path.join(BASE, path)
        with open(full_path) as f:
            confs[model] = json.load(f)
    return confs


def get_easy_set(conf, ids, k_pct):
    items = sorted([(k, conf[k]) for k in ids], key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(k for k, _ in items[:n])


def sc_ratio(example_set, heuristics, mode="sc_soft"):
    if not example_set:
        return 0
    return sum(1 for idx in example_set if heuristics.get(idx, {}).get(mode, False)) / len(example_set)


def run_full_analysis(confs, all_ids, heuristics):
    """Run pairwise analysis for all targets × subjects × thresholds."""
    all_ids_set = set(all_ids)
    results = {}

    for target in TARGETS:
        results[target] = {}
        for t in THRESHOLDS:
            results[target][str(t)] = {}
            easy_t = get_easy_set(confs[target], all_ids, t)

            for subject in MODELS.keys():
                easy_s = get_easy_set(confs[subject], all_ids, t)

                t_only = easy_t - easy_s
                shared = easy_t & easy_s
                s_only = easy_s - easy_t
                neither = all_ids_set - easy_t - easy_s

                # Jaccard
                union = easy_t | easy_s
                jaccard = len(easy_t & easy_s) / len(union) if union else 0

                # Shortcut ratios (both strict and soft)
                entry = {
                    "jaccard": jaccard,
                    "zone_pcts": {
                        "t_only": len(t_only)/len(all_ids)*100,
                        "shared": len(shared)/len(all_ids)*100,
                        "s_only": len(s_only)/len(all_ids)*100,
                        "neither": len(neither)/len(all_ids)*100,
                    },
                    "sc_strict": {
                        "easy_s": sc_ratio(easy_s, heuristics, "sc_strict"),
                        "t_only": sc_ratio(t_only, heuristics, "sc_strict"),
                        "shared": sc_ratio(shared, heuristics, "sc_strict"),
                        "s_only": sc_ratio(s_only, heuristics, "sc_strict"),
                    },
                    "sc_soft": {
                        "easy_s": sc_ratio(easy_s, heuristics, "sc_soft"),
                        "t_only": sc_ratio(t_only, heuristics, "sc_soft"),
                        "shared": sc_ratio(shared, heuristics, "sc_soft"),
                        "s_only": sc_ratio(s_only, heuristics, "sc_soft"),
                    },
                    "conf_gap": (np.mean([confs[target].get(idx, 0) for idx in easy_s])
                                 - np.mean([confs[target].get(idx, 0) for idx in all_ids])) if easy_s else 0,
                    "size_ratio": MODELS[subject][0] / MODELS[target][0],
                    "same_family": MODELS[subject][1] == MODELS[target][1],
                    "is_self": subject == target,
                }
                results[target][str(t)][subject] = entry

    return results


# ── Figures ──

def make_figures(results):
    FAMILY_COLORS = {"BERT": "#e74c3c", "T5": "#3498db", "Pythia": "#2ecc71"}

    # ── Fig 1: Jaccard Heatmap ──
    SUBJECTS = list(MODELS.keys())
    fig, ax = plt.subplots(figsize=(10, 10))
    mat = np.zeros((len(SUBJECTS), len(TARGETS)))
    for i, s in enumerate(SUBJECTS):
        for j, t in enumerate(TARGETS):
            mat[i, j] = results.get(t, {}).get("30", {}).get(s, {}).get("jaccard", 0)

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0.15, vmax=0.6)
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([f"{t}\n({MODELS[t][0]}M)" for t in TARGETS], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels([f"{s} ({MODELS[s][0]}M)" for s in SUBJECTS], fontsize=9)
    ax.set_xlabel("Target Model", fontsize=11)
    ax.set_ylabel("Subject Model", fontsize=11)
    ax.set_title("Jaccard(Easy_Target, Easy_Subject) — k=30%\nHigher = more agreement on what's 'easy'", fontsize=12)

    for i in range(len(SUBJECTS)):
        for j in range(len(TARGETS)):
            v = mat[i, j]
            color = "white" if v > 0.45 else "black"
            text = "Self" if SUBJECTS[i] == TARGETS[j] else f"{v:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color,
                    fontweight="bold" if SUBJECTS[i] == TARGETS[j] else "normal")

    plt.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_jaccard_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved fig1")

    # ── Fig 2: S/T ratio vs SC(S-only) — Key question: "작은 모델의 easy는 shortcut인가?" ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        t_params, t_family = MODELS[target]

        for s, e in results[target]["30"].items():
            if s == target:
                continue
            s_params, s_family = MODELS[s]
            ratio = np.log10(s_params / t_params)
            sc = e["sc_soft"]["s_only"]
            family = s_family
            same = family == t_family

            ax.scatter(ratio, sc,
                       c=FAMILY_COLORS[family],
                       marker="o" if same else "x",
                       s=70 if same else 50,
                       alpha=0.8,
                       edgecolors="black" if same else "none",
                       linewidths=1)

        ax.set_xlabel("log₁₀(Subject / Target size)", fontsize=10)
        ax.set_ylabel("Shortcut ratio in S-only zone", fontsize=10)
        ax.set_title(f"Target: {target} ({t_params}M, {t_family})", fontsize=11)
        ax.axhline(y=sc_ratio(set(results[target]["30"][target]["zone_pcts"].keys()), {}, "sc_soft"),
                    color="gray", linestyle="--", alpha=0.3)
        ax.set_ylim(0.05, 0.65)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS["BERT"],
               markeredgecolor='black', markersize=9, label='BERT (same arch: ●)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS["T5"],
               markeredgecolor='black', markersize=9, label='T5 (same arch: ●)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=FAMILY_COLORS["Pythia"],
               markeredgecolor='black', markersize=9, label='Pythia (same arch: ●)'),
        Line2D([0], [0], marker='x', color='gray', markersize=9, label='Cross-arch: ✕'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.01), frameon=True)

    fig.suptitle("Does a smaller subject's easy set contain more shortcuts?\n"
                 "S-only zone = examples easy for Subject but hard for Target\n"
                 "← Subject smaller | Subject larger →", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    plt.savefig(os.path.join(FIG_DIR, "fig2_sizeratio_vs_sc_sonly.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2")

    # ── Fig 3: S/T ratio vs Jaccard ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()

    for idx, target in enumerate(TARGETS):
        ax = axes[idx]
        t_params, t_family = MODELS[target]

        for s, e in results[target]["30"].items():
            if s == target:
                continue
            s_params, s_family = MODELS[s]
            ratio = np.log10(s_params / t_params)
            family = s_family
            same = family == t_family

            ax.scatter(ratio, e["jaccard"],
                       c=FAMILY_COLORS[family],
                       marker="o" if same else "x",
                       s=70 if same else 50,
                       alpha=0.8,
                       edgecolors="black" if same else "none",
                       linewidths=1)

        ax.set_xlabel("log₁₀(Subject / Target size)", fontsize=10)
        ax.set_ylabel("Jaccard overlap", fontsize=10)
        ax.set_title(f"Target: {target} ({t_params}M, {t_family})", fontsize=11)
        ax.set_ylim(0.15, 0.65)

    fig.suptitle("How similar are easy sets? (Jaccard overlap, k=30%)\n"
                 "Higher Jaccard = Subject and Target agree more on what's 'easy'\n"
                 "← Subject smaller | Subject larger →", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(FIG_DIR, "fig3_sizeratio_vs_jaccard.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3")

    # ── Fig 4: Threshold sensitivity ──
    target = "t5-v1_1-xl"
    subjects_style = [
        ("bert-mini",      11,   "#e74c3c", "--", "v",  "BERT-mini (11M)"),
        ("t5-v1_1-small",  77,   "#2980b9", "--", "D",  "T5-small (77M)"),
        ("t5-v1_1-large",  770,  "#1a5276", "-.", "s",  "T5-large (770M)"),
        ("pythia-2.8b",    2800, "#27ae60", ":",  "o",  "Pythia-2.8B"),
        ("pythia-6.9b",    6900, "#145a32", "-",  "^",  "Pythia-6.9B"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = [
        ("jaccard", "Jaccard", "Higher = more overlap with Target's easy set"),
        ("sc_soft.s_only", "SC(S-only)", "Higher = more shortcuts among examples\nonly Subject finds easy"),
        ("sc_soft.easy_s", "SC(Easy_S)", "Higher = Subject's full easy set\ncontains more shortcuts"),
    ]

    for ax_idx, (metric_path, label, desc) in enumerate(metrics):
        ax = axes[ax_idx]
        for s_name, s_params, color, ls, marker, display_name in subjects_style:
            vals = []
            for t in THRESHOLDS:
                e = results[target][str(t)].get(s_name, {})
                if "." in metric_path:
                    parts = metric_path.split(".")
                    val = e.get(parts[0], {}).get(parts[1], 0)
                else:
                    val = e.get(metric_path, 0)
                vals.append(val)
            ax.plot(THRESHOLDS, vals, marker=marker, linestyle=ls, linewidth=2.2,
                    label=display_name, color=color, markersize=9)

        ax.set_xlabel("Threshold k%", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label}\n{desc}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Threshold Sensitivity (Target: T5-XL, 3B)\n"
                 "All models shown are smaller than Target — do patterns hold across thresholds?",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(FIG_DIR, "fig4_threshold_sensitivity.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4")

    # ── Fig 5: Key insight summary — SC(S-only) grouped by relationship type ──
    fig, ax = plt.subplots(figsize=(12, 6))

    categories = {
        "Same-arch\nsmaller": [],
        "Same-arch\nsimilar": [],
        "Same-arch\nlarger": [],
        "Cross-arch\nsmaller": [],
        "Cross-arch\nsimilar": [],
        "Cross-arch\nlarger": [],
    }

    for target in TARGETS:
        t_params, t_family = MODELS[target]
        for s, e in results[target]["30"].items():
            if s == target:
                continue
            s_params, s_family = MODELS[s]
            same = s_family == t_family
            ratio = s_params / t_params

            if ratio < 0.3:
                size_cat = "smaller"
            elif ratio > 3:
                size_cat = "larger"
            else:
                size_cat = "similar"

            arch_cat = "Same-arch" if same else "Cross-arch"
            key = f"{arch_cat}\n{size_cat}"
            if key in categories:
                categories[key].append(e["sc_soft"]["s_only"])

    positions = range(len(categories))
    labels = list(categories.keys())
    data = [categories[k] for k in labels]

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    colors = ["#3498db", "#2980b9", "#1a5276", "#e74c3c", "#c0392b", "#922b21"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Shortcut ratio in S-only zone", fontsize=12)
    ax.set_title("Shortcut Contamination by Subject-Target Relationship\n"
                 "S-only = examples easy for Subject but hard for Target\n"
                 "Higher = Subject's unique easy examples are more shortcut-laden", fontsize=12)
    ax.axhline(y=np.mean([sc_ratio(set(), {}) for _ in range(1)]), color="gray", linestyle="--", alpha=0.3)

    for i, d in enumerate(data):
        ax.text(i, max(d) + 0.01 if d else 0.5, f"n={len(d)}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_sc_by_relationship.png"), dpi=150)
    plt.close()
    print("  Saved fig5")


# ── Main ──
def main():
    print("Phase A: Loading data")
    confs = load_all_confidences()
    all_ids = sorted(set.intersection(*(set(c.keys()) for c in confs.values())))
    print(f"  Common examples: {len(all_ids)}")

    print("\nComputing HANS heuristics (improved)...")
    heuristics = compute_heuristic_labels(all_ids)

    print("\nPhase B: Pairwise analysis")
    results = run_full_analysis(confs, all_ids, heuristics)

    with open(os.path.join(OUT_DIR, "pairwise_results_v2.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved pairwise_results_v2.json")

    print("\nGenerating figures...")
    make_figures(results)

    # ── Summary table ──
    print(f"\n{'='*110}")
    print("  Summary: SC(S-only) by Subject-Target relationship (k=30%, sc_soft)")
    print(f"{'='*110}")

    for target in TARGETS:
        t_params, t_family = MODELS[target]
        print(f"\n  Target: {target} ({t_params}M, {t_family})")
        print(f"  {'Subject':<18} {'S/T':>6} {'Arch':>6} {'Jaccard':>8} {'SC:S-only':>10} {'SC:T-only':>10} {'SC:Shared':>10} {'ConfGap':>8}")
        print(f"  {'-'*78}")

        entries = [(s, e) for s, e in results[target]["30"].items() if s != target]
        entries.sort(key=lambda x: MODELS[x[0]][0])

        for s, e in entries:
            s_params, s_family = MODELS[s]
            same = "same" if s_family == t_family else "cross"
            print(f"  {s:<18} {s_params/t_params:>6.2f} {same:>6} {e['jaccard']:>8.3f} "
                  f"{e['sc_soft']['s_only']:>10.3f} {e['sc_soft']['t_only']:>10.3f} "
                  f"{e['sc_soft']['shared']:>10.3f} {e['conf_gap']:>8.3f}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
