"""Plan 009: BERT-large retraining on MNLI with K-variant conditions (k=5%, k=10%).
Adapted from experiment_009_train_mnli.py.
"""
import argparse, json, os, time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

PLAN9_DIR = "/workspace/erase/outputs/plan9"
CONDITIONS = ["C1_full", "C2_random", "C3_self_easy", "C4_intersect", "C5_union", "C6_dedup"]


class NLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, max_length=128):
    premises = [b["premise"] for b in batch]
    hypotheses = [b["hypothesis"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(premises, hypotheses, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def load_hans():
    hans_path = os.path.join(PLAN9_DIR, "hans_eval.json")
    with open(hans_path) as f:
        return json.load(f)


def load_anli(round_tag):
    ds = load_dataset("anli", split=f"dev_{round_tag}", trust_remote_code=True)
    return [{"premise": item["premise"], "hypothesis": item["hypothesis"],
             "label": item["label"]} for item in ds]


def evaluate(model, tokenizer, examples, device, batch_size=64, num_labels=3):
    model.eval()
    dataset = NLIDataset(examples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            if num_labels == 2:
                logits_3 = out.logits
                ent_logit = logits_3[:, 0]
                non_ent_logit = torch.logsumexp(logits_3[:, 1:], dim=1)
                preds = (non_ent_logit > ent_logit).long()
            else:
                preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


def train_and_eval(condition, split_name, args):
    device = torch.device(f"cuda:{args.gpu}")

    # Load indices from k-variant directory
    indices_dir = os.path.join(PLAN9_DIR, f"mnli_bert_k{args.k_pct}", split_name)
    indices_path = os.path.join(indices_dir, f"{condition}_indices.json")
    with open(indices_path) as f:
        indices = json.load(f)

    # Load MNLI train
    ds = load_dataset("glue", "mnli", split="train")

    # Load split indices (which 30K subset)
    split_indices_path = f"/workspace/erase/outputs/plan0/data/{split_name}_indices.json"
    with open(split_indices_path) as f:
        split_indices = json.load(f)

    valid_indices = set(str(i) for i in split_indices)
    train_indices = [int(idx) for idx in indices if idx in valid_indices]

    train_examples = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"],
                        "label": ds[i]["label"]} for i in train_indices]

    print(f"  Train: {len(train_examples)} examples")

    # Load eval sets
    mnli_dev = load_dataset("glue", "mnli_matched", split="validation")
    dev_examples = [{"premise": item["premise"], "hypothesis": item["hypothesis"],
                     "label": item["label"]} for item in mnli_dev]
    hans_examples = load_hans()

    # Model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=3).to(device)

    # Training
    train_dataset = NLIDataset(train_examples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=int(0.06 * total_steps),
                                                 num_training_steps=total_steps)

    best_dev_acc = 0
    best_epoch_results = None
    results = {"condition": condition, "split": split_name, "k_pct": args.k_pct, "epochs": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_steps = 0
        for batch in train_loader:
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
        dev_acc = evaluate(model, tokenizer, dev_examples, device)
        hans_acc = evaluate(model, tokenizer, hans_examples, device, num_labels=2)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")
        results["epochs"].append({
            "epoch": epoch + 1, "loss": avg_loss,
            "mnli_dev": dev_acc, "hans": hans_acc,
        })

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc

    # Final ANLI eval
    anli_results = {}
    for r in ["r1", "r2", "r3"]:
        try:
            anli_ex = load_anli(r)
            anli_acc = evaluate(model, tokenizer, anli_ex, device)
            anli_results[f"anli_{r}"] = anli_acc
            print(f"  ANLI-{r}: {anli_acc:.4f}")
        except Exception as e:
            print(f"  ANLI-{r}: failed ({e})")

    results["final"] = {
        "mnli_dev": dev_acc, "hans": hans_acc,
        **anli_results, "n_train": len(train_examples),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k_pct", type=int, required=True, help="Easy threshold: 5 or 10")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--splits", nargs="+", default=["split1", "split2", "split3"])
    args = parser.parse_args()

    outdir = os.path.join(PLAN9_DIR, f"mnli_bert_k{args.k_pct}", "results")
    os.makedirs(outdir, exist_ok=True)

    for condition in args.conditions:
        for split in args.splits:
            outfile = os.path.join(outdir, f"{condition}_{split}.json")
            if os.path.exists(outfile):
                print(f"\n=== {condition} / {split} === SKIP (already done)")
                continue

            print(f"\n=== {condition} / {split} (k={args.k_pct}%) ===")
            start = time.time()
            results = train_and_eval(condition, split, args)
            results["time_sec"] = time.time() - start
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Done in {results['time_sec']:.0f}s")

    print(f"\n=== All MNLI BERT-large k={args.k_pct}% runs complete ===")


if __name__ == "__main__":
    main()
