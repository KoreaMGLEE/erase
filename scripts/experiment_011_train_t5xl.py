"""Plan 011 Phase 5: T5-XL cross-validation with bidirectional conditions.
Based on experiment_009_train_mnli_t5xl.py, adapted for plan11 indices.
"""
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Adafactor,
    get_constant_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

PLAN11_DIR = "/workspace/erase/outputs/plan11"
PLAN9_DIR = "/workspace/erase/outputs/plan9"
LABELS = ["yes", "maybe", "no"]
CONDITIONS = ["B0_baseline", "B1_down_s_all", "B2_up_l_only",
              "B3_graded", "B4_graded_soft", "B5_binary"]


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


def train_and_eval(condition, split_name, args):
    device = torch.device(f"cuda:{args.gpu}")

    with open(os.path.join(PLAN11_DIR, split_name, f"{condition}_indices.json")) as f:
        indices = json.load(f)

    ds = load_dataset("glue", "mnli", split="train")
    with open(f"/workspace/erase/outputs/plan0/data/{split_name}_indices.json") as f:
        split_indices = json.load(f)

    valid = set(str(i) for i in split_indices)
    train_idx = [int(idx) for idx in indices if idx in valid]
    train_examples = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"],
                        "label": ds[i]["label"]} for i in train_idx]
    print(f"  Train: {len(train_examples)}")

    mnli_dev = load_dataset("glue", "mnli_matched", split="validation")
    dev_examples = [{"premise": it["premise"], "hypothesis": it["hypothesis"],
                     "label": it["label"]} for it in mnli_dev]
    hans_examples = load_hans()

    torch.manual_seed(int(split_name[-1]))
    np.random.seed(int(split_name[-1]))
    torch.cuda.manual_seed_all(int(split_name[-1]))

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xl", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-xl", torch_dtype=torch.bfloat16).to(device)

    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
                              lora_dropout=0.05, target_modules=["q", "v"])
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}

    loader = DataLoader(NLIDataset(train_examples), batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))
    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(loader) * args.epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))

    best_dev, best_epoch, best_state, best_hans = 0, 0, None, 0
    results = {"condition": condition, "split": split_name, "model": "t5-v1_1-xl", "epochs": []}

    for epoch in range(args.epochs):
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
        results["epochs"].append({"epoch": epoch+1, "loss": avg_loss, "mnli_dev": dev_acc, "hans": hans_acc})

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
                         **anli, "n_train": len(train_examples)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--splits", nargs="+", default=["split1", "split2", "split3"])
    args = parser.parse_args()

    outdir = os.path.join(PLAN11_DIR, "results_t5xl")
    os.makedirs(outdir, exist_ok=True)

    for cond in args.conditions:
        for split in args.splits:
            outfile = os.path.join(outdir, f"{cond}_{split}.json")
            if os.path.exists(outfile):
                print(f"\n=== {cond} / {split} === SKIP")
                continue
            print(f"\n=== {cond} / {split} ===")
            start = time.time()
            res = train_and_eval(cond, split, args)
            res["time_sec"] = time.time() - start
            with open(outfile, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  Done in {res['time_sec']:.0f}s")

    print("\n=== All T5-XL runs complete ===")


if __name__ == "__main__":
    main()
