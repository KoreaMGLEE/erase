"""Step 1: MNLI 90K sampling - 3 non-overlapping splits of 30K each (stratified)."""
import json
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import os

OUTPUT_DIR = "/workspace/erase/outputs/plan0/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MNLI train...")
ds = load_dataset("glue", "mnli", split="train")
print(f"Total examples: {len(ds)}")

labels = np.array(ds["label"])
all_indices = np.arange(len(ds))

# Stratified sample 90K from full dataset
rng = np.random.RandomState(42)
_, sampled_idx = train_test_split(
    all_indices, test_size=90000, stratify=labels, random_state=42
)
sampled_idx = np.sort(sampled_idx)
sampled_labels = labels[sampled_idx]

# Split 90K into 3 x 30K (stratified)
idx_temp, split3_idx = train_test_split(
    np.arange(len(sampled_idx)), test_size=30000, stratify=sampled_labels, random_state=42
)
temp_labels = sampled_labels[idx_temp]
split1_idx, split2_idx = train_test_split(
    idx_temp, test_size=30000, stratify=temp_labels, random_state=42
)

# Map back to original dataset indices
split1_indices = sorted(sampled_idx[split1_idx].tolist())
split2_indices = sorted(sampled_idx[split2_idx].tolist())
split3_indices = sorted(sampled_idx[split3_idx].tolist())

# Verify non-overlapping
assert len(set(split1_indices) & set(split2_indices)) == 0
assert len(set(split1_indices) & set(split3_indices)) == 0
assert len(set(split2_indices) & set(split3_indices)) == 0
assert len(split1_indices) == 30000
assert len(split2_indices) == 30000
assert len(split3_indices) == 30000

# Print label distributions
for name, idx in [("split1", split1_indices), ("split2", split2_indices), ("split3", split3_indices)]:
    lbls = [ds[i]["label"] for i in idx]
    c = Counter(lbls)
    total = sum(c.values())
    print(f"{name}: {dict(c)} | ratios: {', '.join(f'{k}:{v/total:.3f}' for k,v in sorted(c.items()))}")

# Save
for name, idx in [("split1_indices", split1_indices), ("split2_indices", split2_indices),
                   ("split3_indices", split3_indices), ("all_90k_indices", sorted(sampled_idx.tolist()))]:
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(idx, f)
    print(f"Saved {path} ({len(idx)} indices)")

print("Data preparation done!")
