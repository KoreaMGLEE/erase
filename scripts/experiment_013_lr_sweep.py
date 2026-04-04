"""Plan 013 Phase 1: LR sweep for all target models.
For each target × data_size(k%), find optimal LR using Random subset.
"""
import argparse, json, os, time, gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

BASE = "/workspace/erase/outputs"
OUT_DIR = "/workspace/erase/outputs/plan13"
PLAN9_DIR = "/workspace/erase/outputs/plan9"

# ── Target model configs ──
TARGETS = {
    "bert-small": {
        "hf_name": "google/bert_uncased_L-4_H-512_A-8",
        "type": "bert", "params": 29e6,
        "lrs": [5e-6, 1e-5, 2e-5, 3e-5, 5e-5],
    },
    "bert-base": {
        "hf_name": "bert-base-uncased",
        "type": "bert", "params": 110e6,
        "lrs": [5e-6, 1e-5, 2e-5, 3e-5, 5e-5],
    },
    "bert-large": {
        "hf_name": "bert-large-uncased",
        "type": "bert", "params": 335e6,
        "lrs": [5e-6, 1e-5, 2e-5, 3e-5, 5e-5],
    },
    "pythia-70m": {
        "hf_name": "EleutherAI/pythia-70m",
        "type": "pythia", "params": 70e6,
        "lrs": [5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    },
    "pythia-160m": {
        "hf_name": "EleutherAI/pythia-160m",
        "type": "pythia", "params": 160e6,
        "lrs": [5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    },
    "pythia-410m": {
        "hf_name": "EleutherAI/pythia-410m",
        "type": "pythia", "params": 410e6,
        "lrs": [5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    },
    "pythia-1b": {
        "hf_name": "EleutherAI/pythia-1b",
        "type": "pythia", "params": 1e9,
        "lrs": [5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    },
}

K_EPOCHS = {5: 20, 10: 15, 30: 8}
K_SIZES = [5, 10, 30]


class NLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


# ── BERT training ──

def bert_collate(batch, tokenizer, max_length=128):
    premises = [b["premise"] for b in batch]
    hypotheses = [b["hypothesis"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(premises, hypotheses, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def train_bert(model, tokenizer, train_examples, dev_examples, hans_examples,
               device, lr, epochs, batch_size=16):
    from transformers import get_linear_schedule_with_warmup
    loader = DataLoader(NLIDataset(train_examples), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: bert_collate(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_dev = 0
    best_hans = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = n = 0
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
            n += 1

        dev_acc = eval_bert(model, tokenizer, dev_examples, device)
        hans_acc = eval_bert(model, tokenizer, hans_examples, device)
        print(f"    Ep {epoch+1}: loss={total_loss/n:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_hans = hans_acc
            best_epoch = epoch + 1

    return {"dev": best_dev, "hans": best_hans, "best_epoch": best_epoch}


def eval_bert(model, tokenizer, examples, device, batch_size=64):
    model.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: bert_collate(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += len(batch["labels"])
    return correct / total if total > 0 else 0


# ── Pythia training ──

def pythia_collate(batch, tokenizer, max_length=256):
    texts = [f"Premise: {b['premise']} Hypothesis: {b['hypothesis']} Label:" for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def train_pythia(model, tokenizer, train_examples, dev_examples, hans_examples,
                 device, lr, epochs, batch_size=8, grad_accum=2):
    from transformers import get_linear_schedule_with_warmup

    # Add classification head
    from torch import nn
    hidden_size = model.config.hidden_size
    classifier = nn.Linear(hidden_size, 3).to(device)

    loader = DataLoader(NLIDataset(train_examples), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: pythia_collate(b, tokenizer))
    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_dev = 0
    best_hans = 0
    best_epoch = 0
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        classifier.train()
        total_loss = n = 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            # Use last token hidden state for classification
            hidden = outputs.hidden_states[-1]
            # Find last non-pad token
            seq_lens = attention_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(hidden.size(0)), seq_lens]
            logits = classifier(last_hidden)
            loss = loss_fn(logits, labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum
            n += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        dev_acc = eval_pythia(model, classifier, tokenizer, dev_examples, device)
        hans_acc = eval_pythia(model, classifier, tokenizer, hans_examples, device)
        print(f"    Ep {epoch+1}: loss={total_loss/n:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_hans = hans_acc
            best_epoch = epoch + 1

    del classifier
    return {"dev": best_dev, "hans": best_hans, "best_epoch": best_epoch}


def eval_pythia(model, classifier, tokenizer, examples, device, batch_size=32):
    model.eval()
    classifier.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: pythia_collate(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            seq_lens = attention_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(hidden.size(0)), seq_lens]
            logits = classifier(last_hidden)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS.keys()))
    parser.add_argument("--ks", nargs="+", type=int, default=K_SIZES)
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(OUT_DIR, exist_ok=True)
    outdir = os.path.join(OUT_DIR, "lr_sweep")
    os.makedirs(outdir, exist_ok=True)

    # Load data
    split_num = args.split
    with open(f"{BASE}/plan0/data/split{split_num}_indices.json") as f:
        split_indices = json.load(f)
    all_ids = [str(i) for i in split_indices]

    ds = load_dataset("glue", "mnli", split="train")
    all_examples = {str(i): {"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"],
                              "label": ds[i]["label"]} for i in range(len(ds))}

    mnli_dev = load_dataset("glue", "mnli_matched", split="validation")
    dev_examples = [{"premise": it["premise"], "hypothesis": it["hypothesis"],
                     "label": it["label"]} for it in mnli_dev]

    with open(os.path.join(PLAN9_DIR, "hans_eval.json")) as f:
        hans_examples = json.load(f)

    print(f"Split {split_num}: {len(all_ids)} train, {len(dev_examples)} dev, {len(hans_examples)} HANS")

    for target_name in args.targets:
        cfg = TARGETS[target_name]
        print(f"\n{'='*60}")
        print(f"  Target: {target_name} ({cfg['params']/1e6:.0f}M)")
        print(f"{'='*60}")

        for k in args.ks:
            n_examples = int(len(all_ids) * k / 100)
            epochs = K_EPOCHS[k]

            # Random subset
            rng = np.random.RandomState(42 + split_num + k)
            random_ids = rng.choice(all_ids, size=n_examples, replace=False)
            train_examples = [all_examples[idx] for idx in random_ids]

            for lr in cfg["lrs"]:
                tag = f"{target_name}_k{k}_lr{lr:.0e}_split{split_num}"
                outfile = os.path.join(outdir, f"{tag}.json")
                if os.path.exists(outfile):
                    print(f"\n  SKIP: {tag}")
                    continue

                print(f"\n  --- {target_name} k={k}% lr={lr} ({n_examples} examples, {epochs} epochs) ---")

                start = time.time()
                if cfg["type"] == "bert":
                    from transformers import AutoTokenizer, BertForSequenceClassification
                    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
                    model = BertForSequenceClassification.from_pretrained(
                        cfg["hf_name"], num_labels=3).to(device)
                    result = train_bert(model, tokenizer, train_examples, dev_examples,
                                       hans_examples, device, lr, epochs)
                elif cfg["type"] == "pythia":
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    model = AutoModelForCausalLM.from_pretrained(cfg["hf_name"], torch_dtype=torch.float32).to(device)
                    result = train_pythia(model, tokenizer, train_examples, dev_examples,
                                         hans_examples, device, lr, epochs)

                elapsed = time.time() - start
                result.update({"target": target_name, "k": k, "lr": lr,
                               "split": split_num, "n_train": n_examples,
                               "epochs": epochs, "time_sec": elapsed})

                with open(outfile, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    → dev={result['dev']:.4f} hans={result['hans']:.4f} ({elapsed:.0f}s)")

                # Cleanup
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

    print("\n=== LR sweep complete ===")


if __name__ == "__main__":
    main()
