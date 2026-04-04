"""Plan 011 Experiment A: LR sweep for different data sizes.
Tests LR candidates on random subsets of size 3K and 8K.
"""
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# ── T5-XL imports ──
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Adafactor,
    get_constant_schedule_with_warmup,
    AutoTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

PLAN9_DIR = "/workspace/erase/outputs/plan9"
EXPA_DIR = "/workspace/erase/outputs/plan11_expA"
LABELS = ["yes", "maybe", "no"]


class NLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


# ── Collate functions ──

def collate_bert(batch, tokenizer, max_length=128):
    premises = [b["premise"] for b in batch]
    hypotheses = [b["hypothesis"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(premises, hypotheses, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def collate_t5(batch, tokenizer, max_length=256):
    inputs = [f'mnli hypothesis: {b["hypothesis"]} premise: {b["premise"]} answer: <extra_id_0>' for b in batch]
    targets = [f'<extra_id_0> {LABELS[b["label"]]}' for b in batch]
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors="pt").input_ids
    tgt[tgt == tokenizer.pad_token_id] = -100
    return {**enc, "decoder_labels": tgt, "labels_int": torch.tensor([b["label"] for b in batch])}


# ── Eval functions ──

def eval_bert(model, tokenizer, examples, device, batch_size=64, hans=False):
    model.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_bert(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            labels = batch["labels"].to(device)
            if hans:
                ent = out.logits[:, 0]
                non_ent = torch.logsumexp(out.logits[:, 1:], dim=1)
                preds = (non_ent > ent).long()
            else:
                preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


def eval_t5(model, tokenizer, examples, device, sentinel_id, label_ids, batch_size=32, hans=False):
    model.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_t5(b, tokenizer))
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


def load_hans():
    with open(os.path.join(PLAN9_DIR, "hans_eval.json")) as f:
        return json.load(f)


# ── Training functions ──

def train_bert(train_examples, dev_examples, hans_examples, lr, epochs, device, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=3).to(device)

    loader = DataLoader(NLIDataset(train_examples), batch_size=32, shuffle=True,
                        collate_fn=lambda b: collate_bert(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_dev, best_hans, best_epoch = 0, 0, 0
    epoch_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = n_steps = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            n_steps += 1

        avg_loss = total_loss / n_steps
        dev_acc = eval_bert(model, tokenizer, dev_examples, device)
        hans_acc = eval_bert(model, tokenizer, hans_examples, device, hans=True)
        epoch_log.append({"epoch": epoch+1, "loss": avg_loss, "dev": dev_acc, "hans": hans_acc})

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_hans = hans_acc
            best_epoch = epoch + 1

    return {"best_dev": best_dev, "best_hans": best_hans, "best_epoch": best_epoch, "epochs": epoch_log}


def train_t5xl(train_examples, dev_examples, hans_examples, lr, epochs, device, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xl", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-xl", torch_dtype=torch.bfloat16).to(device)

    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
                              lora_dropout=0.05, target_modules=["q", "v"])
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}

    loader = DataLoader(NLIDataset(train_examples), batch_size=8, shuffle=True,
                        collate_fn=lambda b: collate_t5(b, tokenizer))
    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
                          lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(loader) * epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))

    best_dev, best_hans, best_epoch = 0, 0, 0
    epoch_log = []

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
        dev_acc = eval_t5(model, tokenizer, dev_examples, device, sentinel_id, label_ids)
        hans_acc = eval_t5(model, tokenizer, hans_examples, device, sentinel_id, label_ids, hans=True)
        epoch_log.append({"epoch": epoch+1, "loss": avg_loss, "dev": dev_acc, "hans": hans_acc})

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_hans = hans_acc
            best_epoch = epoch + 1

    return {"best_dev": best_dev, "best_hans": best_hans, "best_epoch": best_epoch, "epochs": epoch_log}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["t5xl", "bert"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sizes", nargs="+", type=int, default=[3000, 8000])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    if args.model == "t5xl":
        lrs = [3e-5, 5e-5, 1e-4, 2e-4, 3e-4]
    else:
        lrs = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5]

    # Load MNLI data
    ds = load_dataset("glue", "mnli", split="train")
    mnli_dev = load_dataset("glue", "mnli_matched", split="validation")
    dev_examples = [{"premise": it["premise"], "hypothesis": it["hypothesis"],
                     "label": it["label"]} for it in mnli_dev]
    hans_examples = load_hans()

    outdir = os.path.join(EXPA_DIR, "lr_sweep", f"results_{args.model}")
    os.makedirs(outdir, exist_ok=True)

    for size in args.sizes:
        # Load random subset
        idx_path = os.path.join(EXPA_DIR, "lr_sweep", f"random_{size}_split1.json")
        with open(idx_path) as f:
            indices = json.load(f)

        train_examples = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"],
                            "label": ds[i]["label"]} for i in indices]
        print(f"\n{'='*60}")
        print(f"Size={size}, Model={args.model}, N_train={len(train_examples)}")
        print(f"{'='*60}")

        for lr in lrs:
            outfile = os.path.join(outdir, f"n{size}_lr{lr}.json")
            if os.path.exists(outfile):
                print(f"  lr={lr} SKIP")
                continue

            print(f"  lr={lr} ...", end=" ", flush=True)
            start = time.time()

            if args.model == "t5xl":
                result = train_t5xl(train_examples, dev_examples, hans_examples, lr, args.epochs, device)
            else:
                result = train_bert(train_examples, dev_examples, hans_examples, lr, args.epochs, device)

            elapsed = time.time() - start
            result.update({"model": args.model, "size": size, "lr": lr, "time_sec": elapsed})

            with open(outfile, "w") as f:
                json.dump(result, f, indent=2)

            print(f"dev={result['best_dev']:.4f} hans={result['best_hans']:.4f} ep={result['best_epoch']} ({elapsed:.0f}s)")

    print("\n=== LR sweep complete ===")


if __name__ == "__main__":
    main()
