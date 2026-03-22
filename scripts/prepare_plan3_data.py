"""Prepare Plan 3 data: match ARC train with human difficulty annotations + model confidence."""
import json
import os
import numpy as np
from datasets import load_dataset
from collections import Counter

OUTPUT_DIR = "/workspace/erase/outputs/plan3/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load annotations
with open("/workspace/erase/easy-to-hard-generalization/data/arc-challenge-easy-annotations.json") as f:
    annotations = json.load(f)

# Map annotation keys to ARC IDs
ann_map = {}
for k, v in annotations.items():
    arc_id = k.replace("ARCCH_", "").replace("ARCEZ_", "")
    ann_map[arc_id] = v

# Load ARC train
easy = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

# Build matched dataset with difficulty
matched = []
for ds_name, ds in [("easy", easy), ("challenge", challenge)]:
    for item in ds:
        if item["id"] in ann_map:
            ann = ann_map[item["id"]]
            matched.append({
                "id": item["id"],
                "question": item["question"],
                "choices": item["choices"]["text"],
                "choice_labels": item["choices"]["label"],
                "answerKey": item["answerKey"],
                "correct_idx": item["choices"]["label"].index(item["answerKey"]),
                "num_choices": len(item["choices"]["text"]),
                "arc_split": ds_name,
                "difficulty": ann["difficulty"],
                "grade": ann["grade"],
                "bloom": ann["bloom"],
            })

print(f"Total matched: {len(matched)}")

# Split by difficulty
by_diff = {"Low": [], "Medium": [], "High": []}
for i, item in enumerate(matched):
    by_diff[item["difficulty"]].append(i)

for d, indices in by_diff.items():
    print(f"  {d}: {len(indices)}")

# Save matched dataset
with open(os.path.join(OUTPUT_DIR, "matched_train.json"), "w") as f:
    json.dump(matched, f, indent=2)

# Save difficulty indices
for d, indices in by_diff.items():
    with open(os.path.join(OUTPUT_DIR, f"indices_{d.lower()}.json"), "w") as f:
        json.dump(indices, f)

# Generate Exp 1 indices
rng = np.random.RandomState(42)
all_indices = list(range(len(matched)))
N_easy = len(by_diff["Low"])
N_med = len(by_diff["Medium"])
N_hard = len(by_diff["High"])

exp1 = {
    "easy_only": by_diff["Low"],
    "medium_only": by_diff["Medium"],
    "hard_only": by_diff["High"],
    "random_Neasy": rng.choice(all_indices, N_easy, replace=False).tolist(),
    "random_Nmed": rng.choice(all_indices, N_med, replace=False).tolist(),
    "full": all_indices,
}
with open(os.path.join(OUTPUT_DIR, "exp1_conditions.json"), "w") as f:
    json.dump(exp1, f)

# Generate Exp 2 indices (all N_hard=287)
exp2 = {
    "hard": by_diff["High"],
    "easy_sub": rng.choice(by_diff["Low"], N_hard, replace=False).tolist(),
    "medium_sub": rng.choice(by_diff["Medium"], N_hard, replace=False).tolist(),
    "random": rng.choice(all_indices, N_hard, replace=False).tolist(),
}
with open(os.path.join(OUTPUT_DIR, "exp2_conditions.json"), "w") as f:
    json.dump(exp2, f)

# Generate Exp 3 indices using model confidence from Plan 2
# Load confidence for each model
CONF_DIR = "/workspace/erase/outputs/plan2/confidence"
models = ["bert-mini", "bert-small", "bert-medium", "bert-base", "bert-large",
          "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b"]

# Map ARC IDs in matched to confidence
id_to_idx = {item["id"]: i for i, item in enumerate(matched)}

exp3 = {}
for model in models:
    # Try to load confidence from seed1
    conf_path = os.path.join(CONF_DIR, f"{model}_seed1", "train_confidence.json")
    if not os.path.exists(conf_path):
        print(f"  {model}: confidence not found, skipping")
        continue

    with open(conf_path) as f:
        conf_data = json.load(f)

    # Match confidence to our matched dataset
    confs = []
    valid_indices = []
    for i, item in enumerate(matched):
        if item["id"] in conf_data:
            confs.append(conf_data[item["id"]])
            valid_indices.append(i)

    if len(valid_indices) < N_easy:
        print(f"  {model}: only {len(valid_indices)} matched, need {N_easy}, skipping")
        continue

    print(f"  {model}: {len(valid_indices)} matched with confidence")

    # Sort by confidence (high = model-easy)
    sorted_pairs = sorted(zip(valid_indices, confs), key=lambda x: -x[1])
    sorted_indices = [p[0] for p in sorted_pairs]

    exp3[model] = {
        "model_easy_top_Neasy": sorted_indices[:N_easy],
        "model_easy_top_Nhard": sorted_indices[:N_hard],
        "model_hard_bottom_Neasy": sorted_indices[-N_easy:],
        "model_hard_bottom_Nhard": sorted_indices[-N_hard:],
    }

with open(os.path.join(OUTPUT_DIR, "exp3_conditions.json"), "w") as f:
    json.dump(exp3, f)

print(f"\nSaved all indices to {OUTPUT_DIR}")
print(f"N_easy={N_easy}, N_med={N_med}, N_hard={N_hard}")
print(f"Exp3 models with confidence: {list(exp3.keys())}")
