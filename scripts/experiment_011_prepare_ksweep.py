"""Plan 011 Phase 4: Prepare zones for k=10% and k=20%."""
import json, os, random

OUTPUT_DIR = "/workspace/erase/outputs/plan11"
BASE = "/workspace/erase/outputs"

SMALL_MODELS = ["bert-small", "t5-v1_1-small", "pythia-14m"]
LARGE_MODEL = "bert-large"

MNLI_PATHS = {
    "bert-small": "plan0/confidence/bert-small_split{}/avg_conf.json",
    "bert-large": "plan0/confidence/bert-large_split{}/avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_sentinel_split{}/avg_conf.json",
    "pythia-14m": "plan5/confidence/pythia-14m_split{}/avg_conf.json",
}

# B3 graded weights
B3_WEIGHTS = {"S_all": 0.0, "S_partial": 0.5, "Shared": 1.0, "L_only": 2.0, "Neither": 1.0}


def load_conf(model, split_num):
    path = os.path.join(BASE, MNLI_PATHS[model].format(split_num))
    with open(path) as f:
        return json.load(f)


def get_easy_set(conf_dict, ids, k_pct):
    items = sorted([(k, conf_dict[k]) for k in ids], key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(k for k, _ in items[:n])


def assign_zones(all_ids, small_easy_sets, large_easy_set):
    zones = {}
    counts = {"S_all": 0, "S_partial": 0, "Shared": 0, "L_only": 0, "Neither": 0}
    for idx in all_ids:
        n_small = sum(1 for s in small_easy_sets if idx in s)
        is_large = idx in large_easy_set
        if n_small == 3 and not is_large:
            zone = "S_all"
        elif n_small >= 1 and not is_large:
            zone = "S_partial"
        elif n_small >= 1 and is_large:
            zone = "Shared"
        elif n_small == 0 and is_large:
            zone = "L_only"
        else:
            zone = "Neither"
        zones[idx] = zone
        counts[zone] += 1
    return zones, counts


def build_train_indices(all_ids, zones, weights, rng):
    indices = []
    for idx in all_ids:
        w = weights[zones[idx]]
        if w == 0.0:
            continue
        elif w == 1.0:
            indices.append(idx)
        elif w == 2.0:
            indices.append(idx)
            indices.append(idx)
        elif 0 < w < 1:
            if rng.random() < w:
                indices.append(idx)
        elif w > 1:
            indices.append(idx)
            if rng.random() < (w - 1):
                indices.append(idx)
    return indices


def prepare(k_pct, split_num):
    split_name = f"split{split_num}"
    tag = f"k{k_pct}"
    out_dir = os.path.join(OUTPUT_DIR, f"{tag}_{split_name}")
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{BASE}/plan0/data/{split_name}_indices.json") as f:
        split_indices = json.load(f)
    all_ids = [str(i) for i in split_indices]
    n = len(all_ids)

    small_confs = {m: load_conf(m, split_num) for m in SMALL_MODELS}
    large_conf = load_conf(LARGE_MODEL, split_num)

    small_easy = [get_easy_set(small_confs[m], all_ids, k_pct) for m in SMALL_MODELS]
    large_easy = get_easy_set(large_conf, all_ids, k_pct)

    zones, counts = assign_zones(all_ids, small_easy, large_easy)

    print(f"  {tag}/{split_name}: ", end="")
    for z in ["S_all", "S_partial", "Shared", "L_only", "Neither"]:
        print(f"{z}={counts[z]}({counts[z]/n*100:.1f}%) ", end="")
    print()

    with open(os.path.join(out_dir, "zones.json"), "w") as f:
        json.dump(zones, f)

    # B0 baseline
    rng = random.Random(42 + split_num)
    b0 = list(all_ids)
    with open(os.path.join(out_dir, "B0_baseline_indices.json"), "w") as f:
        json.dump(b0, f)

    # B3 graded
    b3 = build_train_indices(all_ids, zones, B3_WEIGHTS, rng)
    with open(os.path.join(out_dir, "B3_graded_indices.json"), "w") as f:
        json.dump(b3, f)

    print(f"    B0: {len(b0)}, B3: {len(b3)}")

    meta = {"k_pct": k_pct, "split": split_name, "n_total": n,
            "zone_counts": counts,
            "condition_sizes": {"B0_baseline": len(b0), "B3_graded": len(b3)}}
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    for k in [10, 20]:
        print(f"\n=== k={k}% ===")
        for s in [1, 2, 3]:
            prepare(k, s)
    print("\n=== Done ===")
