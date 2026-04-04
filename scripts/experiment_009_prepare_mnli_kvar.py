"""Plan 009: Prepare MNLI conditions for K=5% and K=10% easy thresholds.
Adapted from experiment_009_prepare.py with variable K_PCT.
Self-easy target: BERT-large.
"""
import json, os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUT_DIR = "/workspace/erase/outputs/plan9"
BASE = "/workspace/erase/outputs"

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]
SMALL_MODELS = ["bert-small", "t5-v1_1-small", "pythia-14m"]

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


def load_mnli_conf(model, split_num):
    path = os.path.join(BASE, MNLI_PATHS[model].format(split_num))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_easy_set(conf_dict, ids, k_pct):
    items = sorted([(k, conf_dict[k]) for k in ids], key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(k for k, _ in items[:n])


def cluster_and_dedup(binary_vectors, ids, keep_ratio=0.1):
    X = np.array(binary_vectors)
    if len(X) < 10:
        return set(ids), set(), 1, 0

    best_k, best_score = 10, -1
    for k in range(10, min(101, len(X)), 10):
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric="hamming", sample_size=min(5000, len(X)))
        if score > best_score:
            best_score = score
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    keep, remove = set(), set()
    for c in range(best_k):
        cluster_idx = np.where(labels == c)[0]
        n_keep = max(1, int(len(cluster_idx) * keep_ratio))
        centroid = km.cluster_centers_[c]
        dists = np.sum(np.abs(X[cluster_idx] - centroid), axis=1)
        sorted_idx = cluster_idx[np.argsort(dists)]
        for idx in sorted_idx[:n_keep]:
            keep.add(ids[idx])
        for idx in sorted_idx[n_keep:]:
            remove.add(ids[idx])
    return keep, remove, best_k, best_score


def prepare_mnli(k_pct):
    print(f"\n{'='*60}")
    print(f"=== MNLI K={k_pct}% ===")
    print(f"{'='*60}")

    for split_num in [1, 2, 3]:
        split_name = f"split{split_num}"
        print(f"\n--- {split_name} ---")
        split_dir = os.path.join(OUTPUT_DIR, f"mnli_bert_k{k_pct}", split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Load confidences
        model_confs = {}
        for model in ALL_MODELS:
            conf = load_mnli_conf(model, split_num)
            if conf:
                model_confs[model] = conf

        common_ids = set(list(model_confs.values())[0].keys())
        for conf in model_confs.values():
            common_ids &= set(conf.keys())
        all_ids = sorted(common_ids)
        n = len(all_ids)
        print(f"  Total: {n}")

        # Easy sets
        easy_sets = {}
        for model in ALL_MODELS:
            if model in model_confs:
                easy_sets[model] = get_easy_set(model_confs[model], all_ids, k_pct)

        small_easy = {m: easy_sets[m] for m in SMALL_MODELS}
        union = set()
        for s in small_easy.values():
            union |= s
        intersect = small_easy[SMALL_MODELS[0]].copy()
        for m in SMALL_MODELS[1:]:
            intersect &= small_easy[m]

        self_easy = easy_sets.get("bert-large", set())

        print(f"  Each small top-{k_pct}%={int(n*k_pct/100)}")
        print(f"  Union: {len(union)} ({len(union)/n*100:.1f}%)")
        print(f"  Intersect: {len(intersect)} ({len(intersect)/n*100:.1f}%)")
        print(f"  Self-easy (bert-large): {len(self_easy)}")

        # Binary vectors + dedup
        union_ids = sorted(union)
        if len(union_ids) > 0:
            binary_vectors = [
                [1 if idx in easy_sets.get(m, set()) else 0 for m in ALL_MODELS]
                for idx in union_ids
            ]
            keep, remove, best_k, sil = cluster_and_dedup(binary_vectors, union_ids)
            print(f"  Dedup: keep={len(keep)}, remove={len(remove)}, k={best_k}, sil={sil:.4f}")
        else:
            keep, remove = set(), set()

        # Build conditions
        conditions = {
            "C1_full": set(all_ids),
            "C3_self_easy": set(all_ids) - self_easy,
            "C4_intersect": set(all_ids) - intersect,
            "C5_union": set(all_ids) - union,
            "C6_dedup": set(all_ids) - remove,
        }

        # C2 random: match union removal
        np.random.seed(42 + split_num)
        n_remove = len(union)
        if n_remove < n:
            random_remove = set(np.random.choice(all_ids, size=n_remove, replace=False))
        else:
            random_remove = set(all_ids)
        conditions["C2_random"] = set(all_ids) - random_remove

        for cname, cids in sorted(conditions.items()):
            cids_list = sorted(cids)
            with open(os.path.join(split_dir, f"{cname}_indices.json"), "w") as f:
                json.dump(cids_list, f)
            print(f"  {cname}: {len(cids_list)} ({len(cids_list)/n*100:.1f}%)")

        meta = {
            "k_pct": k_pct, "split": split_name, "n_total": n,
            "n_union": len(union), "n_intersect": len(intersect),
            "n_self_easy": len(self_easy),
            "n_dedup_keep": len(keep), "n_dedup_remove": len(remove),
            "condition_sizes": {k: len(v) for k, v in conditions.items()},
        }
        with open(os.path.join(split_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    prepare_mnli(5)
    prepare_mnli(10)
    print("\n=== Done ===")
