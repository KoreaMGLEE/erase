"""Plan 011: Bidirectional Intervention — Data Preparation.
5-zone assignment + condition train sets for B0-B9.
"""
import json, os, random
import numpy as np

OUTPUT_DIR = "/workspace/erase/outputs/plan11"
BASE = "/workspace/erase/outputs"
K_PCT = 30

SMALL_MODELS = ["bert-small", "t5-v1_1-small", "pythia-14m"]
LARGE_MODEL = "bert-large"

MNLI_PATHS = {
    "bert-small": "plan0/confidence/bert-small_split{}/avg_conf.json",
    "bert-large": "plan0/confidence/bert-large_split{}/avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_sentinel_split{}/avg_conf.json",
    "pythia-14m": "plan5/confidence/pythia-14m_split{}/avg_conf.json",
}

CONDITIONS = {
    # name: {zone: weight}
    "B0_baseline":    {"S_all": 1.0, "S_partial": 1.0, "Shared": 1.0, "L_only": 1.0, "Neither": 1.0},
    "B1_down_s_all":  {"S_all": 0.0, "S_partial": 1.0, "Shared": 1.0, "L_only": 1.0, "Neither": 1.0},
    "B2_up_l_only":   {"S_all": 1.0, "S_partial": 1.0, "Shared": 1.0, "L_only": 2.0, "Neither": 1.0},
    "B3_graded":      {"S_all": 0.0, "S_partial": 0.5, "Shared": 1.0, "L_only": 2.0, "Neither": 1.0},
    "B4_graded_soft": {"S_all": 0.5, "S_partial": 0.75, "Shared": 1.0, "L_only": 1.5, "Neither": 1.0},
    "B5_binary":      {"S_all": 0.0, "S_partial": 0.0, "Shared": 1.0, "L_only": 2.0, "Neither": 1.0},
    # B6 = random matched to B3 effective size
    "B7_down_all_s":  {"S_all": 0.0, "S_partial": 0.0, "Shared": 1.0, "L_only": 1.0, "Neither": 1.0},
    "B8_up_s_all":    {"S_all": 2.0, "S_partial": 1.0, "Shared": 1.0, "L_only": 1.0, "Neither": 1.0},
    "B9_down_l_only": {"S_all": 1.0, "S_partial": 1.0, "Shared": 1.0, "L_only": 0.0, "Neither": 1.0},
}


def load_conf(model, split_num):
    path = os.path.join(BASE, MNLI_PATHS[model].format(split_num))
    with open(path) as f:
        return json.load(f)


def get_easy_set(conf_dict, ids, k_pct):
    items = sorted([(k, conf_dict[k]) for k in ids], key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(k for k, _ in items[:n])


def assign_zones(all_ids, small_easy_sets, large_easy_set):
    """Assign each example to one of 5 zones."""
    zones = {}
    zone_counts = {"S_all": 0, "S_partial": 0, "Shared": 0, "L_only": 0, "Neither": 0}

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
        zone_counts[zone] += 1

    return zones, zone_counts


def build_train_indices(all_ids, zones, weights, rng):
    """Build weighted train indices. 0×=remove, 0.5×=sample half, 2×=duplicate."""
    indices = []
    for idx in all_ids:
        zone = zones[idx]
        w = weights[zone]
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


def prepare_split(split_num):
    split_name = f"split{split_num}"
    print(f"\n{'='*50}")
    print(f"  {split_name}")
    print(f"{'='*50}")

    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Load split indices
    split_idx_path = f"{BASE}/plan0/data/{split_name}_indices.json"
    with open(split_idx_path) as f:
        split_indices = json.load(f)
    all_ids = [str(i) for i in split_indices]
    n = len(all_ids)

    # Load confidences
    small_confs = {m: load_conf(m, split_num) for m in SMALL_MODELS}
    large_conf = load_conf(LARGE_MODEL, split_num)

    # Easy sets
    small_easy = [get_easy_set(small_confs[m], all_ids, K_PCT) for m in SMALL_MODELS]
    large_easy = get_easy_set(large_conf, all_ids, K_PCT)

    # Zone assignment
    zones, counts = assign_zones(all_ids, small_easy, large_easy)

    print(f"  Total: {n}")
    for z in ["S_all", "S_partial", "Shared", "L_only", "Neither"]:
        print(f"  {z}: {counts[z]} ({counts[z]/n*100:.1f}%)")

    # Save zone assignment
    with open(os.path.join(split_dir, "zones.json"), "w") as f:
        json.dump(zones, f)

    # Build condition train sets
    rng = random.Random(42 + split_num)
    condition_sizes = {}

    for cond_name, weights in CONDITIONS.items():
        indices = build_train_indices(all_ids, zones, weights, rng)
        condition_sizes[cond_name] = len(indices)
        with open(os.path.join(split_dir, f"{cond_name}_indices.json"), "w") as f:
            json.dump(indices, f)
        print(f"  {cond_name}: {len(indices)} ({len(indices)/n*100:.1f}%)")

    # B6: Random matched to B3 effective size
    b3_size = condition_sizes["B3_graded"]
    rng_b6 = random.Random(123 + split_num)
    b6_indices = rng_b6.sample(all_ids, min(b3_size, n))
    with open(os.path.join(split_dir, "B6_random_matched_indices.json"), "w") as f:
        json.dump(b6_indices, f)
    condition_sizes["B6_random_matched"] = len(b6_indices)
    print(f"  B6_random_matched: {len(b6_indices)} ({len(b6_indices)/n*100:.1f}%)")

    # Save metadata
    meta = {
        "split": split_name, "k_pct": K_PCT, "n_total": n,
        "zone_counts": counts,
        "condition_sizes": condition_sizes,
    }
    with open(os.path.join(split_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for s in [1, 2, 3]:
        prepare_split(s)
    print("\n=== Done ===")
