"""Plan 009 Phase 1: Data preparation for §5 retraining experiment.
1. Extract easy sets from 3 small models (BERT-small, T5-small, Pythia-14M)
2. Compute union and intersection
3. Build 20-dim binary easy vectors for union examples
4. k-means clustering + dedup (keep 10% per cluster)
5. Prepare 6 conditions of training data for MNLI (3 splits) and ARC (3 seeds)
"""
import json
import os
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUT_DIR = "/workspace/erase/outputs/plan9"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Model lists ===
ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]

SMALL_MODELS = ["bert-small", "t5-v1_1-small", "pythia-14m"]
K_PCT = 30  # top 30% = easy

# === MNLI confidence paths ===
MNLI_PATHS = {
    "bert-mini": "plan0/confidence/bert-mini_split{}/avg_conf.json",
    "bert-small": "plan0/confidence/bert-small_split{}/avg_conf.json",
    "bert-medium": "plan0/confidence/bert-medium_split{}/avg_conf.json",
    "bert-base": "plan0/confidence/bert-base_split{}/avg_conf.json",
    "bert-large": "plan0/confidence/bert-large_split{}/avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_sentinel_split{}/avg_conf.json",
    "t5-v1_1-base": "plan5/confidence/t5-v1_1-base_sentinel_1e-4_split{}/avg_conf.json",
    "t5-v1_1-large": "plan5/confidence/t5-v1_1-large_sentinel_split{}/avg_conf.json",
    "t5-v1_1-xl": "plan5/confidence/t5-v1_1-xl_sentinel_split{}/avg_conf.json",
    "t5-v1_1-xxl": "plan5/confidence/t5-v1_1-xxl_sentinel_split{}/avg_conf.json",
    "pythia-14m": "plan5/confidence/pythia-14m_split{}/avg_conf.json",
    "pythia-31m": "plan5/confidence/pythia-31m_split{}/avg_conf.json",
    "pythia-70m": "plan0/confidence/pythia-70m_split{}/avg_conf.json",
    "pythia-160m": "plan0/confidence/pythia-160m_split{}/avg_conf.json",
    "pythia-410m": "plan0/confidence/pythia-410m_split{}/avg_conf.json",
    "pythia-1b": "plan0/confidence/pythia-1b_split{}/avg_conf.json",
    "pythia-1.4b": "plan0/confidence/pythia-1.4b_split{}/avg_conf.json",
    "pythia-2.8b": "plan0/confidence/pythia-2.8b_split{}/avg_conf.json",
    "pythia-6.9b": "plan0/confidence/pythia-6.9b_split{}/avg_conf.json",
    "pythia-12b": "plan5/confidence/pythia-12b_split{}/avg_conf.json",
}

# ARC confidence
ARC_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"
ARC_SEEDS = {m: [1, 2, 3] for m in ALL_MODELS}
ARC_SEEDS["t5-v1_1-large"] = [2, 3, 4]
ARC_SEEDS["pythia-410m"] = [1, 3, 4]
ARC_SEEDS["pythia-6.9b"] = [1, 4, 5]
ARC_SEEDS["pythia-12b"] = [1, 3, 4]

BASE = "/workspace/erase/outputs"


def get_easy_set(conf_dict, k_pct):
    items = sorted(conf_dict.items(), key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(item[0] for item in items[:n])


def load_mnli_conf(model, split_num):
    path = os.path.join(BASE, MNLI_PATHS[model].format(split_num))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_arc_avg_conf(model):
    seeds = ARC_SEEDS[model]
    all_confs = []
    for seed in seeds:
        path = os.path.join(ARC_DIR, f"{model}_seed{seed}", "avg_conf.json")
        if os.path.exists(path):
            with open(path) as f:
                all_confs.append(json.load(f))
    if not all_confs:
        return None
    common = set(all_confs[0].keys())
    for c in all_confs[1:]:
        common &= set(c.keys())
    return {k: float(np.mean([c[k] for c in all_confs])) for k in common}


def cluster_and_dedup(binary_vectors, ids, n_clusters_range=(10, 100, 10), keep_ratio=0.1):
    """k-means on binary vectors, keep 10% per cluster."""
    X = np.array(binary_vectors)

    # Find best k by silhouette
    best_k, best_score = 20, -1
    for k in range(n_clusters_range[0], n_clusters_range[1] + 1, n_clusters_range[2]):
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric="hamming", sample_size=min(5000, len(X)))
        if score > best_score:
            best_score = score
            best_k = k

    print(f"  Best k={best_k} (silhouette={best_score:.4f})")

    # Cluster with best k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # Keep representative examples (10% per cluster, min 1)
    keep_ids = set()
    remove_ids = set()
    for c in range(best_k):
        cluster_idx = np.where(labels == c)[0]
        n_keep = max(1, int(len(cluster_idx) * keep_ratio))
        # Keep examples closest to centroid
        centroid = km.cluster_centers_[c]
        dists = np.sum(np.abs(X[cluster_idx] - centroid), axis=1)
        sorted_idx = cluster_idx[np.argsort(dists)]
        for idx in sorted_idx[:n_keep]:
            keep_ids.add(ids[idx])
        for idx in sorted_idx[n_keep:]:
            remove_ids.add(ids[idx])

    return keep_ids, remove_ids, best_k, best_score


def prepare_mnli():
    """Prepare 6 conditions for MNLI × 3 splits."""
    print("=" * 60)
    print("=== MNLI Data Preparation ===")
    print("=" * 60)

    for split_num in [1, 2, 3]:
        split_name = f"split{split_num}"
        print(f"\n--- {split_name} ---")
        split_dir = os.path.join(OUTPUT_DIR, "mnli", split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Load all model confidences for this split
        model_confs = {}
        for model in ALL_MODELS:
            conf = load_mnli_conf(model, split_num)
            if conf:
                model_confs[model] = conf

        # Get common IDs
        common_ids = set(list(model_confs.values())[0].keys())
        for conf in model_confs.values():
            common_ids &= set(conf.keys())
        all_ids = sorted(common_ids)
        n_total = len(all_ids)
        print(f"  Total examples: {n_total}")

        # Easy sets for each model
        easy_sets = {}
        for model in ALL_MODELS:
            if model in model_confs:
                easy_sets[model] = get_easy_set(
                    {k: model_confs[model][k] for k in all_ids}, K_PCT)

        # Small model easy sets
        small_easy = {m: easy_sets[m] for m in SMALL_MODELS}
        union_easy = set()
        for s in small_easy.values():
            union_easy |= s
        intersect_easy = small_easy[SMALL_MODELS[0]].copy()
        for m in SMALL_MODELS[1:]:
            intersect_easy &= small_easy[m]

        # Self-easy (BERT-large for MNLI)
        self_easy = easy_sets.get("bert-large", set())

        print(f"  Small model easy sets: " +
              ", ".join(f"{m}={len(small_easy[m])}" for m in SMALL_MODELS))
        print(f"  Union: {len(union_easy)} ({len(union_easy)/n_total*100:.1f}%)")
        print(f"  Intersection: {len(intersect_easy)} ({len(intersect_easy)/n_total*100:.1f}%)")
        print(f"  Self-easy (bert-large): {len(self_easy)} ({len(self_easy)/n_total*100:.1f}%)")

        # Binary easy vectors for union examples (for dedup)
        union_ids = sorted(union_easy)
        binary_vectors = []
        for idx in union_ids:
            vec = [1 if idx in easy_sets.get(m, set()) else 0 for m in ALL_MODELS]
            binary_vectors.append(vec)

        # Cluster and dedup
        print(f"  Clustering {len(union_ids)} union examples...")
        keep_ids, remove_ids, best_k, sil = cluster_and_dedup(binary_vectors, union_ids)
        print(f"  Dedup: keep={len(keep_ids)}, remove={len(remove_ids)}")

        # === Build 6 conditions ===
        conditions = {}

        # C1: Full
        conditions["C1_full"] = set(all_ids)

        # C2: Random-K% (same removal rate as C5 union-all)
        n_remove_union = len(union_easy)
        np.random.seed(42 + split_num)
        random_remove = set(np.random.choice(all_ids, size=n_remove_union, replace=False))
        conditions["C2_random"] = set(all_ids) - random_remove

        # C3: Remove-Self-Easy
        conditions["C3_self_easy"] = set(all_ids) - self_easy

        # C4: Remove-Intersection-All
        conditions["C4_intersect"] = set(all_ids) - intersect_easy

        # C5: Remove-Union-All
        conditions["C5_union"] = set(all_ids) - union_easy

        # C6: Remove-Union-Dedup (keep representatives)
        conditions["C6_dedup"] = set(all_ids) - remove_ids  # keep = full - remove_ids

        # Save conditions
        for cname, cids in conditions.items():
            cids_list = sorted(cids)
            with open(os.path.join(split_dir, f"{cname}_indices.json"), "w") as f:
                json.dump(cids_list, f)
            print(f"  {cname}: {len(cids_list)} examples ({len(cids_list)/n_total*100:.1f}%)")

        # Save metadata
        meta = {
            "split": split_name,
            "n_total": n_total,
            "n_union": len(union_easy),
            "n_intersect": len(intersect_easy),
            "n_self_easy": len(self_easy),
            "n_dedup_keep": len(keep_ids),
            "n_dedup_remove": len(remove_ids),
            "best_k": best_k,
            "silhouette": sil,
            "condition_sizes": {k: len(v) for k, v in conditions.items()},
        }
        with open(os.path.join(split_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


def prepare_arc():
    """Prepare 6 conditions for ARC."""
    print("\n" + "=" * 60)
    print("=== ARC Data Preparation ===")
    print("=" * 60)

    arc_dir = os.path.join(OUTPUT_DIR, "arc")
    os.makedirs(arc_dir, exist_ok=True)

    # Load avg confidence for all models
    model_confs = {}
    for model in ALL_MODELS:
        conf = load_arc_avg_conf(model)
        if conf:
            model_confs[model] = conf

    common_ids = set(list(model_confs.values())[0].keys())
    for conf in model_confs.values():
        common_ids &= set(conf.keys())
    all_ids = sorted(common_ids)
    n_total = len(all_ids)
    print(f"  Total examples: {n_total}")

    # Easy sets
    easy_sets = {}
    for model in ALL_MODELS:
        if model in model_confs:
            easy_sets[model] = get_easy_set(
                {k: model_confs[model][k] for k in all_ids}, K_PCT)

    # Small model easy
    small_easy = {m: easy_sets[m] for m in SMALL_MODELS}
    union_easy = set()
    for s in small_easy.values():
        union_easy |= s
    intersect_easy = small_easy[SMALL_MODELS[0]].copy()
    for m in SMALL_MODELS[1:]:
        intersect_easy &= small_easy[m]

    # Self-easy (Pythia-1b for ARC)
    self_easy = easy_sets.get("pythia-1b", set())

    print(f"  Small model easy: " +
          ", ".join(f"{m}={len(small_easy[m])}" for m in SMALL_MODELS))
    print(f"  Union: {len(union_easy)} ({len(union_easy)/n_total*100:.1f}%)")
    print(f"  Intersection: {len(intersect_easy)} ({len(intersect_easy)/n_total*100:.1f}%)")
    print(f"  Self-easy (pythia-1b): {len(self_easy)} ({len(self_easy)/n_total*100:.1f}%)")

    # Binary vectors + dedup
    union_ids = sorted(union_easy)
    binary_vectors = []
    for idx in union_ids:
        vec = [1 if idx in easy_sets.get(m, set()) else 0 for m in ALL_MODELS]
        binary_vectors.append(vec)

    print(f"  Clustering {len(union_ids)} union examples...")
    keep_ids, remove_ids, best_k, sil = cluster_and_dedup(binary_vectors, union_ids)
    print(f"  Dedup: keep={len(keep_ids)}, remove={len(remove_ids)}")

    # === Build 6 conditions ===
    conditions = {}
    conditions["C1_full"] = set(all_ids)

    n_remove_union = len(union_easy)
    np.random.seed(42)
    random_remove = set(np.random.choice(all_ids, size=min(n_remove_union, n_total), replace=False))
    conditions["C2_random"] = set(all_ids) - random_remove

    conditions["C3_self_easy"] = set(all_ids) - self_easy
    conditions["C4_intersect"] = set(all_ids) - intersect_easy
    conditions["C5_union"] = set(all_ids) - union_easy
    conditions["C6_dedup"] = set(all_ids) - remove_ids

    for cname, cids in conditions.items():
        cids_list = sorted(cids)
        with open(os.path.join(arc_dir, f"{cname}_indices.json"), "w") as f:
            json.dump(cids_list, f)
        print(f"  {cname}: {len(cids_list)} examples ({len(cids_list)/n_total*100:.1f}%)")

    meta = {
        "n_total": n_total,
        "n_union": len(union_easy),
        "n_intersect": len(intersect_easy),
        "n_self_easy": len(self_easy),
        "n_dedup_keep": len(keep_ids),
        "n_dedup_remove": len(remove_ids),
        "best_k": best_k,
        "silhouette": sil,
        "condition_sizes": {k: len(v) for k, v in conditions.items()},
    }
    with open(os.path.join(arc_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    prepare_mnli()
    prepare_arc()
    print("\n=== Phase 1 Complete ===")
