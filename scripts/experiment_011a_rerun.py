"""Re-run collapsed runs with different seeds."""
import json, glob, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Adafactor,
    get_constant_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

EXPA_DIR = "/workspace/erase/outputs/plan11_expA"
PLAN9_DIR = "/workspace/erase/outputs/plan9"
LABELS = ["yes", "maybe", "no"]

LR_MAP = {3000: 1e-4, 8000: 2e-4, 14000: 1e-4, 30000: 1e-4}

def get_lr(n): return LR_MAP[3000] if n<=4000 else LR_MAP[8000] if n<=10000 else LR_MAP[14000] if n<=20000 else LR_MAP[30000]
def get_epochs(n): return 10 if n <= 5000 else 5

class NLIDataset(Dataset):
    def __init__(self, e): self.examples = e
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

def collate_fn(batch, tokenizer, max_length=256):
    inputs = [f'mnli hypothesis: {b["hypothesis"]} premise: {b["premise"]} answer: <extra_id_0>' for b in batch]
    targets = [f'<extra_id_0> {LABELS[b["label"]]}' for b in batch]
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors="pt").input_ids
    tgt[tgt == tokenizer.pad_token_id] = -100
    return {**enc, "decoder_labels": tgt, "labels_int": torch.tensor([b["label"] for b in batch])}

def load_hans():
    with open(os.path.join(PLAN9_DIR, "hans_eval.json")) as f: return json.load(f)

def load_anli(r):
    ds = load_dataset("anli", split=f"dev_{r}")
    return [{"premise": it["premise"], "hypothesis": it["hypothesis"], "label": it["label"]} for it in ds]

def evaluate(model, tokenizer, examples, device, sentinel_id, label_ids, batch_size=32, hans=False):
    model.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))
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
                ent = label_logits[:, 0]; non_ent = torch.logsumexp(label_logits[:, 1:], dim=1)
                preds = (non_ent > ent).long()
            else:
                preds = label_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0

def train_and_eval(train_indices, split_name, condition, l_variant, device, seed_offset=10):
    ds = load_dataset("glue", "mnli", split="train")
    train_examples = [{"premise": ds[int(i)]["premise"], "hypothesis": ds[int(i)]["hypothesis"],
                        "label": ds[int(i)]["label"]} for i in train_indices]
    n_train = len(train_examples)
    lr = get_lr(n_train)
    epochs = get_epochs(n_train)
    seed = int(split_name[-1]) + seed_offset

    print(f"  N={n_train}, lr={lr}, epochs={epochs}, seed={seed}")

    mnli_dev = load_dataset("glue", "mnli_matched", split="validation")
    dev_examples = [{"premise": it["premise"], "hypothesis": it["hypothesis"], "label": it["label"]} for it in mnli_dev]
    hans_examples = load_hans()

    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xl", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-xl", torch_dtype=torch.bfloat16).to(device)
    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q", "v"])
    model = get_peft_model(model, lora_config); model.enable_input_require_grads()

    sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}

    loader = DataLoader(NLIDataset(train_examples), batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad], lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(loader) * epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))

    best_dev, best_epoch, best_state, best_hans = 0, 0, None, 0
    results = {"condition": condition, "split": split_name, "L_variant": l_variant,
               "model": "t5-v1_1-xl", "lr": lr, "n_train": n_train, "seed": seed, "rerun": True, "epoch_log": []}

    for epoch in range(epochs):
        model.train(); total_loss = n_steps = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["decoder_labels"].to(device))
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step()
            total_loss += out.loss.item(); n_steps += 1
        avg_loss = total_loss / n_steps
        dev_acc = evaluate(model, tokenizer, dev_examples, device, sentinel_id, label_ids)
        hans_acc = evaluate(model, tokenizer, hans_examples, device, sentinel_id, label_ids, hans=True)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")
        results["epoch_log"].append({"epoch": epoch+1, "loss": avg_loss, "dev": dev_acc, "hans": hans_acc})
        if dev_acc > best_dev:
            best_dev = dev_acc; best_epoch = epoch+1; best_hans = hans_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()}); del best_state
    anli = {}
    for r in ["r1", "r2", "r3"]:
        try:
            anli[f"anli_{r}"] = evaluate(model, tokenizer, load_anli(r), device, sentinel_id, label_ids)
            print(f"  ANLI-{r}: {anli[f'anli_{r}']:.4f}")
        except Exception as e: print(f"  ANLI-{r}: failed ({e})")

    results["final"] = {"mnli_dev": best_dev, "hans": best_hans, "best_epoch": best_epoch, **anli, "n_train": n_train}
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dev_threshold", type=float, default=0.70)
    parser.add_argument("--seed_offset", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    results_dir = os.path.join(EXPA_DIR, "results_t5xl")

    # Find collapsed runs
    collapsed = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(f) as fh: d = json.load(fh)
        if d.get("final", {}).get("mnli_dev", 0) < args.dev_threshold:
            collapsed.append((f, d))

    print(f"Found {len(collapsed)} collapsed runs (dev < {args.dev_threshold})")
    for f, d in collapsed:
        print(f"  {os.path.basename(f)}: dev={d['final']['mnli_dev']:.4f}")

    for filepath, old_data in collapsed:
        fname = os.path.basename(filepath)
        cond = old_data.get("condition", "")
        split = old_data.get("split", "")
        l_var = old_data.get("L_variant", "")

        # Determine index file
        if cond.startswith("A0_random"):
            n = old_data["final"]["n_train"]
            idx_path = os.path.join(EXPA_DIR, f"L_{l_var}", split, f"A0_random_n{n}_indices.json")
        else:
            idx_path = os.path.join(EXPA_DIR, f"L_{l_var}", split, f"{cond}_indices.json")

        if not os.path.exists(idx_path):
            print(f"\n=== SKIP {fname} (index not found: {idx_path}) ===")
            continue

        with open(idx_path) as f: indices = json.load(f)

        print(f"\n=== RERUN {fname} ===")
        start = time.time()
        res = train_and_eval(indices, split, cond, l_var, device, seed_offset=args.seed_offset)
        res["time_sec"] = time.time() - start

        if res["final"]["mnli_dev"] >= args.dev_threshold:
            # Overwrite with successful rerun
            with open(filepath, "w") as f: json.dump(res, f, indent=2)
            print(f"  SUCCESS: dev={res['final']['mnli_dev']:.4f} -> saved")
        else:
            print(f"  STILL COLLAPSED: dev={res['final']['mnli_dev']:.4f} -> keeping old")

    print("\n=== Rerun complete ===")

if __name__ == "__main__":
    main()
