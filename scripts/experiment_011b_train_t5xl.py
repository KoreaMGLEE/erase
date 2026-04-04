"""Plan 011 Experiment B: Continuous weight-based curation for T5-XL.
Weighted sampling based on conf_large × (1 - agg_small).
"""
import argparse, json, os, time, math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Adafactor,
    get_constant_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

BASE = "/workspace/erase/outputs"
EXPA_DIR = "/workspace/erase/outputs/plan11_expA"
EXPB_DIR = "/workspace/erase/outputs/plan11_expB"
PLAN9_DIR = "/workspace/erase/outputs/plan9"
LABELS = ["yes", "maybe", "no"]

MNLI_CONF = {
    "bert-mini":     "plan0/confidence/bert-mini_split{}/avg_conf.json",
    "t5-v1_1-small": "plan5/confidence/t5-v1_1-small_sentinel_split{}/avg_conf.json",
    "pythia-70m":    "plan0/confidence/pythia-70m_split{}/avg_conf.json",
    "t5-v1_1-xl":    "plan5/confidence/t5-v1_1-xl_sentinel_split{}/avg_conf.json",
}

SMALL_MODELS = ["bert-mini", "t5-v1_1-small", "pythia-70m"]
LARGE_MODEL = "t5-v1_1-xl"


def load_conf(model, split_num):
    path = os.path.join(BASE, MNLI_CONF[model].format(split_num))
    with open(path) as f:
        return json.load(f)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def compute_weights(all_ids, split_num, method, agg_type, scale_type):
    """Compute per-example weights based on method/agg/scale."""
    # Load confidences
    small_confs = {m: load_conf(m, split_num) for m in SMALL_MODELS}
    large_conf = load_conf(LARGE_MODEL, split_num)

    # Optionally convert to rank (percentile)
    if scale_type == "rank":
        def to_rank(conf_dict, ids):
            vals = sorted([(k, conf_dict.get(k, 0)) for k in ids], key=lambda x: x[1])
            rank = {k: i / len(vals) for i, (k, _) in enumerate(vals)}
            return rank
        large_rank = to_rank(large_conf, all_ids)
        small_ranks = {m: to_rank(small_confs[m], all_ids) for m in SMALL_MODELS}
    else:
        large_rank = large_conf
        small_ranks = small_confs

    weights = {}
    for idx in all_ids:
        conf_l = large_rank.get(idx, 0)

        # Aggregate small model confidences
        s_vals = [small_ranks[m].get(idx, 0) for m in SMALL_MODELS]
        if agg_type == "mean":
            agg_s = np.mean(s_vals)
        else:  # max
            agg_s = max(s_vals)

        # Weight function
        if method == "prod":
            w = conf_l * (1.0 - agg_s)
        elif method == "diff":
            w = sigmoid(conf_l - agg_s)
        else:
            w = 1.0

        weights[idx] = max(w, 1e-6)  # floor to avoid zero

    return weights


def normalize_weights(weights):
    """Normalize weights to have mean=1."""
    vals = np.array(list(weights.values()))
    mean_w = vals.mean()
    if mean_w > 0:
        return {k: v / mean_w for k, v in weights.items()}
    return weights


def compute_random_weights(all_ids, seed=999):
    """Random weights for control."""
    rng = np.random.RandomState(seed)
    return {idx: rng.random() for idx in all_ids}


class NLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, max_length=256):
    inputs = [f'mnli hypothesis: {b["hypothesis"]} premise: {b["premise"]} answer: <extra_id_0>' for b in batch]
    targets = [f'<extra_id_0> {LABELS[b["label"]]}' for b in batch]
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors="pt").input_ids
    tgt[tgt == tokenizer.pad_token_id] = -100
    return {**enc, "decoder_labels": tgt, "labels_int": torch.tensor([b["label"] for b in batch])}


def load_hans():
    with open(os.path.join(PLAN9_DIR, "hans_eval.json")) as f:
        return json.load(f)


def load_anli(round_tag):
    ds = load_dataset("anli", split=f"dev_{round_tag}")
    return [{"premise": it["premise"], "hypothesis": it["hypothesis"],
             "label": it["label"]} for it in ds]


def evaluate(model, tokenizer, examples, device, sentinel_id, label_ids, batch_size=32, hans=False):
    model.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels_int"].to(device)
            bs = input_ids.size(0)
            dec = torch.full((bs, 2), tokenizer.pad_token_id, dtype=torch.long, device=device)
            dec[:, 1] = sentinel_id
            out = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec)
            logits = out.logits[:, 1, :]
            lid = [label_ids[l] for l in LABELS]
            label_logits = logits[:, lid]
            if hans:
                ent = label_logits[:, 0]
                non_ent = torch.logsumexp(label_logits[:, 1:], dim=1)
                preds = (non_ent > ent).long()
            else:
                preds = label_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


def train_and_eval(train_examples, sample_weights, split_name, condition, args, lr=None):
    device = torch.device(f"cuda:{args.gpu}")
    n_train = len(train_examples)
    seed = int(split_name[-1])
    lr = lr or args.lr

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    mnli_dev = load_dataset("glue", "mnli_matched", split="validation")
    dev_examples = [{"premise": it["premise"], "hypothesis": it["hypothesis"],
                     "label": it["label"]} for it in mnli_dev]
    hans_examples = load_hans()

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xl", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-xl", torch_dtype=torch.bfloat16).to(device)

    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
                              lora_dropout=0.05, target_modules=["q", "v"])
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}

    # Weighted random sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=n_train, replacement=True)
    loader = DataLoader(NLIDataset(train_examples), batch_size=8, sampler=sampler,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
                          lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
    epochs = 5
    total_steps = len(loader) * epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))

    best_dev, best_epoch, best_state, best_hans = 0, 0, None, 0
    results = {"condition": condition, "split": split_name, "model": "t5-v1_1-xl",
               "lr": lr, "n_train": n_train, "epochs_config": epochs, "epoch_log": []}

    for epoch in range(epochs):
        model.train()
        total_loss = n_steps = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["decoder_labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            n_steps += 1

        avg_loss = total_loss / n_steps
        dev_acc = evaluate(model, tokenizer, dev_examples, device, sentinel_id, label_ids)
        hans_acc = evaluate(model, tokenizer, hans_examples, device, sentinel_id, label_ids, hans=True)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")
        results["epoch_log"].append({"epoch": epoch+1, "loss": avg_loss, "dev": dev_acc, "hans": hans_acc})

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_epoch = epoch + 1
            best_hans = hans_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    del best_state

    anli = {}
    for r in ["r1", "r2", "r3"]:
        try:
            anli[f"anli_{r}"] = evaluate(model, tokenizer, load_anli(r), device, sentinel_id, label_ids)
            print(f"  ANLI-{r}: {anli[f'anli_{r}']:.4f}")
        except Exception as e:
            print(f"  ANLI-{r}: failed ({e})")

    results["final"] = {"mnli_dev": best_dev, "hans": best_hans, "best_epoch": best_epoch,
                         **anli, "n_train": n_train}

    # Weight statistics
    w_arr = np.array(sample_weights)
    results["weight_stats"] = {
        "mean": float(w_arr.mean()), "std": float(w_arr.std()),
        "min": float(w_arr.min()), "max": float(w_arr.max()),
        "median": float(np.median(w_arr)),
    }

    return results


# Condition definitions
CONDITIONS = {
    "B2_prod_mean_raw":  ("prod", "mean", "raw"),
    "B3_prod_max_raw":   ("prod", "max",  "raw"),
    "B4_prod_mean_rank": ("prod", "mean", "rank"),
    "B5_prod_max_rank":  ("prod", "max",  "rank"),
    "B6_diff_mean_raw":  ("diff", "mean", "raw"),
    "B7_diff_max_raw":   ("diff", "max",  "raw"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()) + ["B8_random"])
    parser.add_argument("--splits", nargs="+", default=["split1", "split2", "split3"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--normalize", action="store_true", help="Normalize weights to mean=1")
    parser.add_argument("--tag", type=str, default="", help="Output subdirectory tag")
    args = parser.parse_args()

    tag = args.tag if args.tag else ("norm" if args.normalize else "lr" + str(args.lr))
    outdir = os.path.join(EXPB_DIR, f"results_t5xl_{tag}")
    os.makedirs(outdir, exist_ok=True)

    for split in args.splits:
        split_num = int(split[-1])

        # Load split data
        with open(f"{BASE}/plan0/data/{split}_indices.json") as f:
            split_indices = json.load(f)
        all_ids = [str(i) for i in split_indices]

        ds = load_dataset("glue", "mnli", split="train")
        train_examples = [{"premise": ds[int(i)]["premise"], "hypothesis": ds[int(i)]["hypothesis"],
                            "label": ds[int(i)]["label"]} for i in all_ids]

        for cond in args.conditions:
            outfile = os.path.join(outdir, f"{cond}_{split}.json")
            if os.path.exists(outfile):
                print(f"\n=== {cond} / {split} === SKIP")
                continue

            print(f"\n=== {cond} / {split} ===")

            if cond == "B8_random":
                weights_dict = compute_random_weights(all_ids, seed=42 + split_num)
            elif cond in CONDITIONS:
                method, agg, scale = CONDITIONS[cond]
                weights_dict = compute_weights(all_ids, split_num, method, agg, scale)
            else:
                continue

            if args.normalize:
                weights_dict = normalize_weights(weights_dict)

            sample_weights = [weights_dict[idx] for idx in all_ids]

            start = time.time()
            res = train_and_eval(train_examples, sample_weights, split, cond, args, lr=args.lr)
            res["time_sec"] = time.time() - start
            res["normalized"] = args.normalize

            with open(outfile, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Done in {res['time_sec']:.0f}s")

    print("\n=== All runs complete ===")


if __name__ == "__main__":
    main()
