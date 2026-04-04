"""Plan 012: GPU-free analysis — which subject's easy examples help/hurt?
Computes Jaccard overlap, zone distributions, shortcut ratios, confidence gaps.
"""
import json, os, glob
import numpy as np
from collections import defaultdict

BASE = "/workspace/erase/outputs"
OUT_DIR = "/workspace/erase/outputs/plan12_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model metadata ──
MODELS = {
    "bert-mini":     {"params": 11e6,  "family": "BERT"},
    "bert-small":    {"params": 29e6,  "family": "BERT"},
    "bert-medium":   {"params": 41e6,  "family": "BERT"},
    "bert-base":     {"params": 110e6, "family": "BERT"},
    "bert-large":    {"params": 335e6, "family": "BERT"},
    "t5-v1_1-small": {"params": 77e6,  "family": "T5"},
    "t5-v1_1-base":  {"params": 250e6, "family": "T5"},
    "t5-v1_1-large": {"params": 770e6, "family": "T5"},
    "t5-v1_1-xl":    {"params": 3e9,   "family": "T5"},
    "pythia-14m":    {"params": 14e6,  "family": "Pythia"},
    "pythia-70m":    {"params": 70e6,  "family": "Pythia"},
    "pythia-160m":   {"params": 160e6, "family": "Pythia"},
    "pythia-410m":   {"params": 410e6, "family": "Pythia"},
    "pythia-1b":     {"params": 1e9,   "family": "Pythia"},
    "pythia-1.4b":   {"params": 1.4e9, "family": "Pythia"},
    "pythia-2.8b":   {"params": 2.8e9, "family": "Pythia"},
    "pythia-6.9b":   {"params": 6.9e9, "family": "Pythia"},
}

# 90K avg confidence paths
CONF_90K = {
    "bert-mini":     "plan0/confidence/bert-mini_90k_avg_conf.json",
    "bert-small":    "plan0/confidence/bert-small_90k_avg_conf.json",
    "bert-medium":   "plan0/confidence/bert-medium_90k_avg_conf.json",
    "bert-base":     "plan0/confidence/bert-base_90k_avg_conf.json",
    "bert-large":    "plan0/confidence/bert-large_90k_avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_90k_avg_conf.json",
    "t5-v1_1-base":  "plan5/confidence/t5-v1_1-base_90k_avg_conf.json",
    "t5-v1_1-large": "plan5/confidence/t5-v1_1-large_90k_avg_conf.json",
    "t5-v1_1-xl":    "plan5/confidence/t5-v1_1-xl_90k_avg_conf.json",
    "pythia-14m":    "plan5/confidence/pythia-14m_90k_avg_conf.json",
    "pythia-70m":    "plan0/confidence/pythia-70m_90k_avg_conf.json",
    "pythia-160m":   "plan0/confidence/pythia-160m_90k_avg_conf.json",
    "pythia-410m":   "plan0/confidence/pythia-410m_90k_avg_conf.json",
    "pythia-1b":     "plan0/confidence/pythia-1b_90k_avg_conf.json",
    "pythia-1.4b":   "plan0/confidence/pythia-1.4b_90k_avg_conf.json",
    "pythia-2.8b":   "plan0/confidence/pythia-2.8b_90k_avg_conf.json",
    "pythia-6.9b":   "plan0/confidence/pythia-6.9b_90k_avg_conf.json",
}

TARGETS = ["bert-base", "bert-large", "t5-v1_1-base", "t5-v1_1-xl", "pythia-410m", "pythia-2.8b"]
THRESHOLDS = [10, 20, 30, 50]


# ── Phase A: Load data ──

def load_all_confidences():
    """Load 90K avg confidence for all models."""
    confs = {}
    for model, path in CONF_90K.items():
        full_path = os.path.join(BASE, path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                confs[model] = json.load(f)
            print(f"  Loaded {model}: {len(confs[model])} examples")
        else:
            print(f"  MISSING {model}: {full_path}")
    return confs


def compute_easy_sets(confs, thresholds=THRESHOLDS):
    """Compute easy sets for all models at all thresholds."""
    # Use common example IDs across all models
    all_ids = None
    for model, conf in confs.items():
        ids = set(conf.keys())
        all_ids = ids if all_ids is None else all_ids & ids
    all_ids = sorted(all_ids)
    print(f"\n  Common examples across all models: {len(all_ids)}")

    easy_sets = {}
    for model, conf in confs.items():
        items = sorted([(k, conf[k]) for k in all_ids], key=lambda x: -x[1])
        for t in thresholds:
            n = int(len(items) * t / 100)
            easy_sets[(model, t)] = set(k for k, _ in items[:n])

    return all_ids, easy_sets


def compute_heuristic_labels(all_ids):
    """Compute lexical overlap and negation heuristics for MNLI examples."""
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

        # Lexical overlap: fraction of hypothesis tokens in premise
        overlap = len(premise_tokens & hyp_tokens) / max(len(hyp_tokens), 1)

        # Negation: hypothesis contains negation word
        has_negation = bool(hyp_tokens & negation_words)

        # Subsequence: all hypothesis words appear in premise
        is_subsequence = hyp_tokens.issubset(premise_tokens)

        # Label
        label = ex["label"]  # 0=entailment, 1=neutral, 2=contradiction

        # HANS-style heuristic: high overlap + entailment = shortcut
        # Shortcut = (overlap > 0.5 and label == 0) or (has_negation and label == 2)
        is_shortcut = (overlap > 0.5 and label == 0) or (has_negation and label == 2)

        heuristics[idx] = {
            "overlap": overlap,
            "has_negation": has_negation,
            "is_subsequence": is_subsequence,
            "label": label,
            "is_shortcut": is_shortcut,
        }

    print(f"  Computed heuristics for {len(heuristics)} examples")
    shortcut_count = sum(1 for h in heuristics.values() if h["is_shortcut"])
    print(f"  Global shortcut ratio: {shortcut_count/len(heuristics):.3f}")
    return heuristics


# ── Phase B: Pairwise analysis ──

def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0
    return len(set_a & set_b) / len(set_a | set_b)


def asymmetric_overlap(set_a, set_b):
    """Fraction of set_a that is also in set_b."""
    if not set_a:
        return 0
    return len(set_a & set_b) / len(set_a)


def compute_zones(easy_t, easy_s, all_ids_set):
    t_only = easy_t - easy_s
    shared = easy_t & easy_s
    s_only = easy_s - easy_t
    neither = all_ids_set - easy_t - easy_s
    return {"t_only": t_only, "shared": shared, "s_only": s_only, "neither": neither}


def shortcut_ratio(example_set, heuristics):
    if not example_set:
        return 0
    count = sum(1 for idx in example_set if heuristics.get(idx, {}).get("is_shortcut", False))
    return count / len(example_set)


def confidence_gap(confs_t, easy_s, all_ids):
    """Mean T-confidence of S-easy examples vs overall mean."""
    overall_mean = np.mean([confs_t.get(idx, 0) for idx in all_ids])
    s_easy_mean = np.mean([confs_t.get(idx, 0) for idx in easy_s]) if easy_s else 0
    return s_easy_mean - overall_mean


def run_pairwise_analysis(confs, all_ids, easy_sets, heuristics):
    """Run all pairwise analyses for targets × subjects × thresholds."""
    all_ids_set = set(all_ids)
    results = {}

    for target in TARGETS:
        if target not in confs:
            continue
        results[target] = {}

        for threshold in THRESHOLDS:
            results[target][threshold] = {}
            easy_t = easy_sets.get((target, threshold), set())

            for subject in MODELS.keys():
                if subject not in confs:
                    continue
                easy_s = easy_sets.get((subject, threshold), set())

                zones = compute_zones(easy_t, easy_s, all_ids_set)

                entry = {
                    "jaccard": jaccard(easy_t, easy_s),
                    "asym_overlap_t_in_s": asymmetric_overlap(easy_t, easy_s),
                    "asym_overlap_s_in_t": asymmetric_overlap(easy_s, easy_t),
                    "zone_sizes": {z: len(v) for z, v in zones.items()},
                    "zone_pcts": {z: len(v)/len(all_ids)*100 for z, v in zones.items()},
                    "shortcut_ratio_easy_s": shortcut_ratio(easy_s, heuristics),
                    "shortcut_ratio_t_only": shortcut_ratio(zones["t_only"], heuristics),
                    "shortcut_ratio_shared": shortcut_ratio(zones["shared"], heuristics),
                    "shortcut_ratio_s_only": shortcut_ratio(zones["s_only"], heuristics),
                    "conf_gap": confidence_gap(confs[target], easy_s, all_ids),
                    "size_ratio": MODELS[subject]["params"] / MODELS[target]["params"],
                    "same_family": MODELS[subject]["family"] == MODELS[target]["family"],
                    "is_self": subject == target,
                }
                results[target][threshold][subject] = entry

    return results


# ── Phase C: Summary tables ──

def print_summary(results, threshold=30):
    """Print summary tables for a given threshold."""
    print(f"\n{'='*120}")
    print(f"  SUMMARY (threshold = top {threshold}%)")
    print(f"{'='*120}")

    for target in TARGETS:
        if target not in results:
            continue
        t_params = MODELS[target]["params"]
        t_family = MODELS[target]["family"]

        print(f"\n  Target: {target} ({t_params/1e6:.0f}M, {t_family})")
        print(f"  {'Subject':<20} {'Params':>8} {'Family':>8} {'Jaccard':>8} {'T-only%':>8} {'Shared%':>8} {'S-only%':>8} "
              f"{'SC:Easy_S':>9} {'SC:T-only':>9} {'SC:S-only':>9} {'ConfGap':>8} {'S/T':>6}")
        print(f"  {'-'*118}")

        # Sort by size ratio
        entries = []
        for subject, entry in results[target].get(threshold, {}).items():
            entries.append((subject, entry))
        entries.sort(key=lambda x: x[1]["size_ratio"])

        for subject, e in entries:
            s_params = MODELS[subject]["params"]
            s_family = MODELS[subject]["family"]
            marker = " ←SELF" if e["is_self"] else ""
            same = "same" if e["same_family"] else "cross"

            print(f"  {subject:<20} {s_params/1e6:>7.0f}M {same:>8} "
                  f"{e['jaccard']:>8.3f} "
                  f"{e['zone_pcts']['t_only']:>7.1f}% "
                  f"{e['zone_pcts']['shared']:>7.1f}% "
                  f"{e['zone_pcts']['s_only']:>7.1f}% "
                  f"{e['shortcut_ratio_easy_s']:>9.3f} "
                  f"{e['shortcut_ratio_t_only']:>9.3f} "
                  f"{e['shortcut_ratio_s_only']:>9.3f} "
                  f"{e['conf_gap']:>8.3f} "
                  f"{e['size_ratio']:>6.2f}{marker}")


def print_threshold_comparison(results):
    """Show how key metrics change across thresholds."""
    print(f"\n{'='*100}")
    print(f"  THRESHOLD SENSITIVITY (Target: t5-v1_1-xl)")
    print(f"{'='*100}")

    target = "t5-v1_1-xl"
    if target not in results:
        return

    subjects_of_interest = ["bert-mini", "t5-v1_1-small", "t5-v1_1-large",
                            "t5-v1_1-xl", "pythia-2.8b", "pythia-6.9b"]

    for subject in subjects_of_interest:
        print(f"\n  Subject: {subject}")
        print(f"  {'Threshold':>10} {'Jaccard':>8} {'T-only%':>8} {'S-only%':>8} {'SC:Easy_S':>9} {'SC:T-only':>9} {'SC:S-only':>9}")
        for t in THRESHOLDS:
            e = results[target].get(t, {}).get(subject)
            if not e:
                continue
            print(f"  {t:>9}% {e['jaccard']:>8.3f} {e['zone_pcts']['t_only']:>7.1f}% "
                  f"{e['zone_pcts']['s_only']:>7.1f}% {e['shortcut_ratio_easy_s']:>9.3f} "
                  f"{e['shortcut_ratio_t_only']:>9.3f} {e['shortcut_ratio_s_only']:>9.3f}")


# ── Main ──

def main():
    print("=" * 60)
    print("Phase A: Loading data")
    print("=" * 60)

    confs = load_all_confidences()
    all_ids, easy_sets = compute_easy_sets(confs)
    print("\nComputing MNLI heuristic labels...")
    heuristics = compute_heuristic_labels(all_ids)

    print("\n" + "=" * 60)
    print("Phase B: Pairwise analysis")
    print("=" * 60)
    results = run_pairwise_analysis(confs, all_ids, easy_sets, heuristics)

    # Save raw results
    # Convert sets to counts for JSON serialization
    with open(os.path.join(OUT_DIR, "pairwise_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUT_DIR}/pairwise_results.json")

    print("\n" + "=" * 60)
    print("Phase C: Summary")
    print("=" * 60)

    for t in [30, 10]:
        print_summary(results, threshold=t)

    print_threshold_comparison(results)

    print("\n\n=== Analysis complete ===")


if __name__ == "__main__":
    main()
