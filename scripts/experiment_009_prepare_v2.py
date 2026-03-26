"""Plan 009 Phase 1 v2: Prepare ARC conditions for K=5% and K=10%."""
import json, os, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUT_DIR = "/workspace/erase/outputs/plan9"
ARC_DIR = "/workspace/erase/outputs/plan2_v2/confidence_v2"

ALL_MODELS = [
    "bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large", "t5-v1_1-xl", "t5-v1_1-xxl",
    "pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m",
    "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
]
SMALL_MODELS = ["bert-small", "t5-v1_1-small", "pythia-14m"]
ARC_SEEDS = {m: [1, 2, 3] for m in ALL_MODELS}
ARC_SEEDS["t5-v1_1-large"] = [2, 3, 4]
ARC_SEEDS["pythia-410m"] = [1, 3, 4]
ARC_SEEDS["pythia-6.9b"] = [1, 4, 5]
ARC_SEEDS["pythia-12b"] = [1, 3, 4]


def load_avg(model):
    seeds = ARC_SEEDS[model]
    confs = []
    for s in seeds:
        p = f"{ARC_DIR}/{model}_seed{s}/avg_conf.json"
        if os.path.exists(p):
            with open(p) as f:
                confs.append(json.load(f))
    if not confs:
        return None
    common = set(confs[0].keys())
    for c in confs[1:]:
        common &= set(c.keys())
    return {k: float(np.mean([c[k] for c in confs])) for k in common}


def get_easy_set(conf_dict, ids, k_pct):
    items = sorted([(k, conf_dict[k]) for k in ids], key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(k for k, _ in items[:n])


def cluster_dedup(binary_vectors, ids, keep_ratio=0.1):
    X = np.array(binary_vectors)
    if len(X) < 10:
        return set(ids), set(), 1, 0

    best_k, best_score = 10, -1
    for k in range(5, min(51, len(X)), 5):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric="hamming", sample_size=min(3000, len(X)))
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


def prepare_arc(k_pct):
    print(f"\n{'='*50}")
    print(f"=== ARC K={k_pct}% ===")
    print(f"{'='*50}")

    arc_dir = os.path.join(OUTPUT_DIR, f"arc_k{k_pct}")
    os.makedirs(arc_dir, exist_ok=True)

    model_confs = {m: load_avg(m) for m in ALL_MODELS}
    common_ids = None
    for c in model_confs.values():
        if c:
            ids = set(c.keys())
            common_ids = ids if common_ids is None else common_ids & ids
    all_ids = sorted(common_ids)
    n = len(all_ids)

    # Easy sets
    easy_sets = {}
    for m in ALL_MODELS:
        if model_confs[m]:
            easy_sets[m] = get_easy_set(model_confs[m], all_ids, k_pct)

    small_easy = {m: easy_sets[m] for m in SMALL_MODELS}
    union = set()
    for s in small_easy.values():
        union |= s
    intersect = small_easy[SMALL_MODELS[0]].copy()
    for m in SMALL_MODELS[1:]:
        intersect &= small_easy[m]

    # Self-easy for BERT-large
    self_easy_bert = easy_sets.get("bert-large", set())
    # Self-easy for Pythia-1b
    self_easy_pythia = easy_sets.get("pythia-1b", set())

    print(f"  N={n}, each small top-{k_pct}%={int(n*k_pct/100)}")
    print(f"  Union: {len(union)} ({len(union)/n*100:.1f}%)")
    print(f"  Intersect: {len(intersect)} ({len(intersect)/n*100:.1f}%)")
    print(f"  Self-easy BERT: {len(self_easy_bert)}, Pythia: {len(self_easy_pythia)}")

    # Binary vectors + dedup
    union_ids = sorted(union)
    if len(union_ids) > 0:
        binary_vectors = [[1 if idx in easy_sets.get(m, set()) else 0 for m in ALL_MODELS] for idx in union_ids]
        keep, remove, best_k, sil = cluster_dedup(binary_vectors, union_ids)
        print(f"  Dedup: keep={len(keep)}, remove={len(remove)}, k={best_k}")
    else:
        keep, remove = set(), set()

    # Build conditions
    conditions = {
        "C1_full": set(all_ids),
        "C3_self_easy_bert": set(all_ids) - self_easy_bert,
        "C3_self_easy_pythia": set(all_ids) - self_easy_pythia,
        "C4_intersect": set(all_ids) - intersect,
        "C5_union": set(all_ids) - union,
        "C6_dedup": set(all_ids) - remove,
    }

    # C2 random: match union removal rate
    np.random.seed(42)
    n_remove = len(union)
    if n_remove < n:
        random_remove = set(np.random.choice(all_ids, size=n_remove, replace=False))
    else:
        random_remove = set(all_ids)
    conditions["C2_random"] = set(all_ids) - random_remove

    for cname, cids in conditions.items():
        with open(os.path.join(arc_dir, f"{cname}_indices.json"), "w") as f:
            json.dump(sorted(cids), f)
        print(f"  {cname}: {len(cids)} ({len(cids)/n*100:.1f}%)")

    meta = {
        "k_pct": k_pct, "n_total": n,
        "n_union": len(union), "n_intersect": len(intersect),
        "n_self_easy_bert": len(self_easy_bert), "n_self_easy_pythia": len(self_easy_pythia),
        "n_dedup_keep": len(keep), "n_dedup_remove": len(remove),
        "condition_sizes": {k: len(v) for k, v in conditions.items()},
    }
    with open(os.path.join(arc_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    prepare_arc(5)
    prepare_arc(10)
    print("\n=== Done ===")
