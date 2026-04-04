"""Hypothesis-only baseline for shortcut proxy.
Train BERT-mini on hypothesis only for 1 epoch, measure per-example confidence.
Light enough to share GPU with running experiments.
"""
import json, os, time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset

OUT_DIR = "/workspace/erase/outputs/plan12_analysis"


class HypOnlyDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, max_length=64):
    # Hypothesis only — no premise
    texts = [b["hypothesis"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # Load MNLI train
    ds = load_dataset("glue", "mnli", split="train")
    examples = [{"idx": str(i), "hypothesis": ds[i]["hypothesis"], "label": ds[i]["label"]}
                for i in range(len(ds))]
    print(f"MNLI train: {len(examples)} examples")

    # Train BERT-mini on hypothesis only
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")  # bert-mini
    model = BertForSequenceClassification.from_pretrained("google/bert_uncased_L-4_H-256_A-4", num_labels=3).to(device)

    loader = DataLoader(HypOnlyDataset(examples), batch_size=128, shuffle=True,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * len(loader)), len(loader))

    # 1 epoch training
    print("Training hypothesis-only BERT-mini (1 epoch)...")
    model.train()
    total_loss = n_steps = 0
    start = time.time()
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
    print(f"  Train loss: {total_loss/n_steps:.4f} ({time.time()-start:.0f}s)")

    # Measure confidence on all examples
    print("Measuring hypothesis-only confidence...")
    model.eval()
    eval_loader = DataLoader(HypOnlyDataset(examples), batch_size=256, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer))

    hyp_conf = {}
    with torch.no_grad():
        batch_start = 0
        for batch in eval_loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=-1)
            labels = batch["labels"]

            for i in range(len(labels)):
                idx = str(batch_start + i)
                # Confidence = probability assigned to the correct label
                conf = probs[i, labels[i]].item()
                hyp_conf[idx] = conf

            batch_start += len(labels)

    # Save
    outpath = os.path.join(OUT_DIR, "hyp_only_confidence.json")
    with open(outpath, "w") as f:
        json.dump(hyp_conf, f)

    # Stats
    vals = list(hyp_conf.values())
    print(f"\n  Saved {len(hyp_conf)} confidences to {outpath}")
    print(f"  Mean: {np.mean(vals):.4f}, Std: {np.std(vals):.4f}")
    print(f"  Top-30% threshold: {np.percentile(vals, 70):.4f}")

    # Verify: hypothesis-only accuracy
    correct = sum(1 for idx, conf in hyp_conf.items()
                  if conf == max(hyp_conf.get(idx, 0) for _ in range(1)))
    # Actually compute accuracy properly
    eval_loader2 = DataLoader(HypOnlyDataset(examples), batch_size=256, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in eval_loader2:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += len(batch["labels"])
    print(f"  Hypothesis-only accuracy: {correct/total:.4f} (chance=0.333)")


if __name__ == "__main__":
    main()
