"""Plan 009: BERT-large retraining on ARC with 6 conditions.
Uses BertForMultipleChoice. Eval on ARC-Easy val, ARC-Challenge val/test.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForMultipleChoice
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

PLAN9_DIR = "/workspace/erase/outputs/plan9"
CONDITIONS = ["C1_full", "C2_random", "C3_self_easy", "C4_intersect", "C5_union", "C6_dedup"]
MAX_CHOICES = 5


class ARCDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_bert_mc(batch, tokenizer, max_length=128):
    batch_input_ids, batch_attention_mask, batch_token_type_ids, labels = [], [], [], []
    for item in batch:
        choices = list(item["choices"])
        while len(choices) < MAX_CHOICES:
            choices.append("")
        enc = tokenizer([item["question"]] * MAX_CHOICES, choices[:MAX_CHOICES],
                        padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        batch_input_ids.append(enc["input_ids"])
        batch_attention_mask.append(enc["attention_mask"])
        if "token_type_ids" in enc:
            batch_token_type_ids.append(enc["token_type_ids"])
        labels.append(item["correct_idx"])
    result = {"input_ids": torch.stack(batch_input_ids), "attention_mask": torch.stack(batch_attention_mask),
              "labels": torch.tensor(labels)}
    if batch_token_type_ids:
        result["token_type_ids"] = torch.stack(batch_token_type_ids)
    return result


def load_arc_data():
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def parse(split_data):
        examples = []
        for item in split_data:
            choices = item["choices"]["text"]
            labels_list = item["choices"]["label"]
            answer_key = item["answerKey"]
            correct_idx = labels_list.index(answer_key) if answer_key in labels_list else 0
            examples.append({
                "id": item["id"], "question": item["question"],
                "choices": choices, "correct_idx": correct_idx,
            })
        return examples

    train_all = parse(easy["train"]) + parse(challenge["train"])
    val_easy = parse(easy["validation"])
    val_challenge = parse(challenge["validation"])
    test_challenge = parse(challenge["test"])
    return train_all, val_easy, val_challenge, test_challenge


def evaluate(model, tokenizer, data, device):
    model.eval()
    loader = DataLoader(ARCDataset(data), batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate_bert_mc(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device) if "token_type_ids" in batch else None)
            preds = out.logits.argmax(dim=-1)
            labels = batch["labels"].to(device)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


def train_and_eval(condition, seed, args):
    device = torch.device(f"cuda:{args.gpu}")

    # Load condition indices
    indices_path = os.path.join(PLAN9_DIR, args.data_dir, f"{condition}_indices.json")
    with open(indices_path) as f:
        condition_ids = set(json.load(f))

    # Load ARC data
    train_all, val_easy, val_challenge, test_challenge = load_arc_data()
    train_filtered = [item for item in train_all if item["id"] in condition_ids]
    print(f"  Train: {len(train_filtered)} examples")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMultipleChoice.from_pretrained("bert-large-uncased").to(device)

    # Training
    train_loader = DataLoader(ARCDataset(train_filtered), batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_bert_mc(b, tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=int(0.06 * total_steps),
                                                 num_training_steps=total_steps)

    results = {"condition": condition, "seed": seed, "model": "bert-large", "epochs": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device) if "token_type_ids" in batch else None,
                        labels=batch["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            n_steps += 1

        avg_loss = total_loss / n_steps
        easy_acc = evaluate(model, tokenizer, val_easy, device)
        chal_acc = evaluate(model, tokenizer, val_challenge, device)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} easy={easy_acc:.4f} chal={chal_acc:.4f}")
        results["epochs"].append({
            "epoch": epoch + 1, "loss": avg_loss,
            "arc_easy": easy_acc, "arc_challenge": chal_acc,
        })

    # Final test eval
    test_chal_acc = evaluate(model, tokenizer, test_challenge, device)
    print(f"  Test challenge: {test_chal_acc:.4f}")

    results["final"] = {
        "arc_easy_val": easy_acc,
        "arc_challenge_val": chal_acc,
        "arc_challenge_test": test_chal_acc,
        "n_train": len(train_filtered),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--data_dir", default="arc")
    parser.add_argument("--result_tag", default="arc_bert")
    args = parser.parse_args()

    for condition in args.conditions:
        for seed in args.seeds:
            print(f"\n=== {condition} / seed={seed} ===")
            outdir = os.path.join(PLAN9_DIR, args.result_tag, "results")
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, f"{condition}_seed{seed}.json")
            if os.path.exists(outfile):
                print("  SKIP (already done)")
                continue

            start = time.time()
            results = train_and_eval(condition, seed, args)
            results["time_sec"] = time.time() - start
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Done in {results['time_sec']:.0f}s")

    print("\n=== All ARC BERT runs complete ===")


if __name__ == "__main__":
    main()
