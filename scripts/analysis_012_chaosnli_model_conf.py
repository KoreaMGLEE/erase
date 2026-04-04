"""ChaosNLI: Compare human agreement vs model confidence by model size.
Train BERT-mini/small/base on MNLI train (1 epoch), measure confidence on ChaosNLI examples.
"""
import json, os, time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset

OUT_DIR = "/workspace/erase/outputs/plan12_analysis"

MODELS = {
    "bert-mini":  ("google/bert_uncased_L-4_H-256_A-4", 11e6),
    "bert-small": ("google/bert_uncased_L-4_H-512_A-8", 29e6),
    "bert-base":  ("bert-base-uncased", 110e6),
}


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # Load ChaosNLI
    with open(os.path.join(OUT_DIR, "chaosnli_mnli.json")) as f:
        chaosnli = json.load(f)
    print(f"ChaosNLI: {len(chaosnli)} examples")

    # Prepare ChaosNLI eval examples
    chaos_examples = []
    for uid, info in chaosnli.items():
        chaos_examples.append({
            "uid": uid,
            "premise": info["premise"],
            "hypothesis": info["hypothesis"],
            "label": info["old_label"],  # original gold label
            "agreement": info["agreement"],
        })

    # Load MNLI train
    ds = load_dataset("glue", "mnli", split="train")
    train_examples = [{"premise": ds[i]["premise"], "hypothesis": ds[i]["hypothesis"],
                        "label": ds[i]["label"]} for i in range(len(ds))]
    print(f"MNLI train: {len(train_examples)} examples")

    results = {}

    for model_name, (hf_name, params) in MODELS.items():
        print(f"\n=== {model_name} ({params/1e6:.0f}M) ===")

        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = BertForSequenceClassification.from_pretrained(hf_name, num_labels=3).to(device)

        # Train 1 epoch on MNLI
        loader = DataLoader(NLIDataset(train_examples), batch_size=64, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer))
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * len(loader)), len(loader))

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

        # Evaluate on ChaosNLI
        model.eval()
        eval_loader = DataLoader(NLIDataset(chaos_examples), batch_size=128, shuffle=False,
                                  collate_fn=lambda b: collate_fn(b, tokenizer))

        per_example = {}
        correct = total = 0
        with torch.no_grad():
            batch_start = 0
            for batch in eval_loader:
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
                probs = torch.softmax(out.logits, dim=-1)
                preds = out.logits.argmax(dim=-1)
                labels = batch["labels"]

                for i in range(len(labels)):
                    ex = chaos_examples[batch_start + i]
                    uid = ex["uid"]
                    conf = probs[i, labels[i]].item()
                    pred = preds[i].item()
                    per_example[uid] = {
                        "model_conf": conf,
                        "model_pred": pred,
                        "gold_label": labels[i].item(),
                        "correct": pred == labels[i].item(),
                        "agreement": ex["agreement"],
                    }
                    correct += (pred == labels[i].item())
                    total += 1
                batch_start += len(labels)

        acc = correct / total
        print(f"  ChaosNLI acc: {acc:.4f}")

        results[model_name] = {
            "params": params,
            "acc": acc,
            "per_example": per_example,
        }

        # Free memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # Save results
    save_path = os.path.join(OUT_DIR, "chaosnli_model_confidence.json")
    # Convert for JSON serialization
    save_data = {}
    for model_name, res in results.items():
        save_data[model_name] = {
            "params": res["params"],
            "acc": res["acc"],
            "per_example": res["per_example"],
        }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {save_path}")

    # Analysis: human agreement vs model confidence correlation
    print("\n=== Analysis: Human Agreement vs Model Confidence ===")
    for model_name, res in results.items():
        agreements = []
        model_confs = []
        for uid, info in res["per_example"].items():
            agreements.append(info["agreement"])
            model_confs.append(info["model_conf"])
        corr = np.corrcoef(agreements, model_confs)[0, 1]
        print(f"  {model_name}: r={corr:.4f}, acc={res['acc']:.4f}")

    # Binned analysis: group by human agreement, look at model confidence
    print("\n=== Binned: Human Agreement → Model Confidence ===")
    bins = [(0, 0.5, "Low (<50%)"), (0.5, 0.7, "Med (50-70%)"), (0.7, 0.9, "High (70-90%)"), (0.9, 1.01, "VHigh (90%+)")]
    for model_name, res in results.items():
        print(f"  {model_name}:")
        for lo, hi, label in bins:
            subset = [v for v in res["per_example"].values() if lo <= v["agreement"] < hi]
            if subset:
                mean_conf = np.mean([v["model_conf"] for v in subset])
                mean_acc = np.mean([v["correct"] for v in subset])
                print(f"    {label}: n={len(subset)}, model_conf={mean_conf:.3f}, model_acc={mean_acc:.3f}")


if __name__ == "__main__":
    main()
