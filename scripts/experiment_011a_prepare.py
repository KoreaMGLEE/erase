"""Plan 011 Experiment A: Prepare condition indices for T5-XL target.
L variants: Self (T5-XL), Pythia-2.8B
S detectors: BERT-mini, T5-v1.1-small, Pythia-70M (union)
"""
import json, os, random

BASE = "/workspace/erase/outputs"
OUT_DIR = "/workspace/erase/outputs/plan11_expA"

MNLI_CONF = {
    "bert-mini":     "plan0/confidence/bert-mini_split{}/avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_sentinel_split{}/avg_conf.json",
    "pythia-70m":    "plan0/confidence/pythia-70m_split{}/avg_conf.json",
    "t5-v1_1-xl":    "plan5/confidence/t5-v1_1-xl_sentinel_split{}/avg_conf.json",
    "pythia-2.8b":   "plan0/confidence/pythia-2.8b_split{}/avg_conf.json",
}

SMALL_MODELS = ["bert-mini", "t5-v1_1-small", "pythia-70m"]
L_VARIANTS = {"self": "t5-v1_1-xl", "pythia2.8b": "pythia-2.8b"}
K_PCT = 30


def load_conf(model, split_num):
    path = os.path.join(BASE, MNLI_CONF[model].format(split_num))
    with open(path) as f:
        return json.load(f)


def get_easy_set(conf_dict, ids, k_pct):
    items = sorted([(k, conf_dict.get(k, 0)) for k in ids], key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return set(k for k, _ in items[:n])


def prepare_split(split_num):
    split_name = f"split{split_num}"

    with open(f"{BASE}/plan0/data/{split_name}_indices.json") as f:
        split_indices = json.load(f)
    all_ids = [str(i) for i in split_indices]
    n = len(all_ids)

    # Load confidences
    confs = {}
    for m in SMALL_MODELS + list(L_VARIANTS.values()):
        confs[m] = load_conf(m, split_num)

    # Small easy sets
    small_easy = {m: get_easy_set(confs[m], all_ids, K_PCT) for m in SMALL_MODELS}
    s_union = small_easy[SMALL_MODELS[0]] | small_easy[SMALL_MODELS[1]] | small_easy[SMALL_MODELS[2]]

    rng = random.Random(42 + split_num)

    for l_name, l_model in L_VARIANTS.items():
        out_dir = os.path.join(OUT_DIR, f"L_{l_name}", split_name)
        os.makedirs(out_dir, exist_ok=True)

        l_easy = get_easy_set(confs[l_model], all_ids, K_PCT)

        conditions = {
            "A2_L_easy":  sorted(l_easy),
            "A3_L_only":  sorted(l_easy - s_union),
            "A4_Shared":  sorted(l_easy & s_union),
            "A5_S_easy":  sorted(s_union),
            "A6_S_only":  sorted(s_union - l_easy),
        }

        # Save condition indices
        for cond_name, indices in conditions.items():
            with open(os.path.join(out_dir, f"{cond_name}_indices.json"), "w") as f:
                json.dump(indices, f)

        # Size-matched random baselines for each condition
        for cond_name, indices in conditions.items():
            size = len(indices)
            shuffled = list(all_ids)
            rng.shuffle(shuffled)
            random_indices = sorted(shuffled[:size])
            with open(os.path.join(out_dir, f"A0_random_n{size}_indices.json"), "w") as f:
                json.dump(random_indices, f)

        # Print summary
        print(f"  L={l_name} / {split_name}:")
        for cond_name, indices in conditions.items():
            print(f"    {cond_name}: {len(indices)}")

        # Save metadata
        meta = {
            "split": split_name, "L_variant": l_name, "L_model": l_model,
            "S_models": SMALL_MODELS, "k_pct": K_PCT, "n_total": n,
            "condition_sizes": {k: len(v) for k, v in conditions.items()},
        }
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    print("=== Preparing Experiment A indices ===\n")
    for s in [1, 2, 3]:
        prepare_split(s)
    print("\n=== Done ===")
