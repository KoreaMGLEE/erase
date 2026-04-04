"""Plan 011 Phase 4: k% sweep — B3 graded at k=10%, 20%."""
import argparse, json, os, time
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset

PLAN11_DIR = "/workspace/erase/outputs/plan11"
PLAN9_DIR = "/workspace/erase/outputs/plan9"


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
    with open(os.path.join(PLAN9_DIR, "hans_eval.json")) as f:
        return json.load(f)


def load_anli(round_tag):
    ds = load_dataset("anli", split=f"dev_{round_tag}")
    return [{"premise": it["premise"], "hypothesis": it["hypothesis"],
             "label": it["label"]} for it in ds]


def evaluate(model, tokenizer, examples, device, batch_size=64, num_labels=3):
    model.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            labels = batch["labels"].to(device)
            if num_labels == 2:
                ent = out.logits[:, 0]
                non_ent = torch.logsumexp(out.logits[:, 1:], dim=1)
                preds = (non_ent > ent).long()
            else:
                preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


def train_and_eval(condition, split_name, k_pct, args):
    device = torch.device(f"cuda:{args.gpu}")

    idx_dir = os.path.join(PLAN11_DIR, f"k{k_pct}_{split_name}")
    with open(os.path.join(idx_dir, f"{condition}_indices.json")) as f:
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

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=3).to(device)

    loader = DataLoader(NLIDataset(train_examples), batch_size=32, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(loader) * 5
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_dev, best_epoch, best_state = 0, 0, None
    results = {"condition": condition, "split": split_name, "k_pct": k_pct, "epochs": []}

    for epoch in range(5):
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
        dev_acc = evaluate(model, tokenizer, dev_examples, device)
        hans_acc = evaluate(model, tokenizer, hans_examples, device, num_labels=2)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")
        results["epochs"].append({"epoch": epoch+1, "loss": avg_loss, "mnli_dev": dev_acc, "hans": hans_acc})

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    final_dev = evaluate(model, tokenizer, dev_examples, device)
    final_hans = evaluate(model, tokenizer, hans_examples, device, num_labels=2)
    anli = {}
    for r in ["r1", "r2", "r3"]:
        try:
            anli[f"anli_{r}"] = evaluate(model, tokenizer, load_anli(r), device)
            print(f"  ANLI-{r}: {anli[f'anli_{r}']:.4f}")
        except Exception as e:
            print(f"  ANLI-{r}: failed ({e})")

    results["final"] = {"mnli_dev": final_dev, "hans": final_hans, **anli,
                         "n_train": len(train_examples), "best_epoch": best_epoch}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k_pcts", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--conditions", nargs="+", default=["B0_baseline", "B3_graded"])
    parser.add_argument("--splits", nargs="+", default=["split1", "split2", "split3"])
    args = parser.parse_args()

    outdir = os.path.join(PLAN11_DIR, "results_ksweep")
    os.makedirs(outdir, exist_ok=True)

    for k in args.k_pcts:
        for cond in args.conditions:
            for split in args.splits:
                outfile = os.path.join(outdir, f"{cond}_k{k}_{split}.json")
                if os.path.exists(outfile):
                    print(f"\n=== {cond}/k{k}/{split} === SKIP")
                    continue
                print(f"\n=== {cond} / k={k}% / {split} ===")
                start = time.time()
                res = train_and_eval(cond, split, k, args)
                res["time_sec"] = time.time() - start
                with open(outfile, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  Done in {res['time_sec']:.0f}s")

    print("\n=== All k-sweep runs complete ===")


if __name__ == "__main__":
    main()
