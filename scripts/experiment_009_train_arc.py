"""Plan 009 Phase 3: Pythia-1B retraining on ARC with 6 conditions.
Eval on ARC-Easy dev, ARC-Challenge test.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

PLAN9_DIR = "/workspace/erase/outputs/plan9"
CONDITIONS = ["C1_full", "C2_random", "C3_self_easy", "C4_intersect", "C5_union", "C6_dedup"]

MAX_CHOICES = 5
CHOICE_LABELS = ["A", "B", "C", "D", "E"]


class ARCDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def load_arc_data():
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def parse(split_data):
        examples = []
        for item in split_data:
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]
            correct_idx = labels.index(answer_key) if answer_key in labels else 0
            examples.append({
                "id": item["id"], "question": item["question"],
                "choices": choices, "correct_idx": correct_idx,
                "num_choices": len(choices),
            })
        return examples

    train_all = parse(easy["train"]) + parse(challenge["train"])
    val_easy = parse(easy["validation"])
    val_challenge = parse(challenge["validation"])
    test_challenge = parse(challenge["test"])
    return train_all, val_easy, val_challenge, test_challenge


def format_input(item):
    lines = [f"Question: {item['question']}"]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{CHOICE_LABELS[i]}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def collate_fn(batch, tokenizer, max_length=512):
    texts = [format_input(item) for item in batch]
    labels = torch.tensor([item["correct_idx"] for item in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def evaluate_arc(model, tokenizer, data, device, label_ids):
    model.eval()
    loader = DataLoader(ARCDataset(data), batch_size=32, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :]
            choice_logits = logits[:, [label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]]
            preds = choice_logits.argmax(dim=-1)
            labels = batch["labels"].to(device)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


def train_and_eval(condition, seed, args):
    device = torch.device(f"cuda:{args.gpu}")

    # Load condition indices
    indices_path = os.path.join(PLAN9_DIR, "arc", f"{condition}_indices.json")
    with open(indices_path) as f:
        condition_ids = set(json.load(f))

    # Load ARC data
    train_all, val_easy, val_challenge, test_challenge = load_arc_data()

    # Filter train by condition
    train_filtered = [item for item in train_all if item["id"] in condition_ids]
    print(f"  Train: {len(train_filtered)} examples (from {len(train_all)} total)")

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b", torch_dtype=torch.float32).to(device)

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
                              lora_dropout=0.05, target_modules=["query_key_value"])
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    label_ids = {l: tokenizer.encode(f" {l}", add_special_tokens=False)[0] for l in CHOICE_LABELS}

    # Training
    train_loader = DataLoader(ARCDataset(train_filtered), batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=int(0.06 * total_steps),
                                                 num_training_steps=total_steps)

    results = {"condition": condition, "seed": seed, "epochs": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :]
            choice_ids = [label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
            loss = F.cross_entropy(logits[:, choice_ids], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / n_steps
        easy_acc = evaluate_arc(model, tokenizer, val_easy, device, label_ids)
        chal_acc = evaluate_arc(model, tokenizer, val_challenge, device, label_ids)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} easy={easy_acc:.4f} chal={chal_acc:.4f}")
        results["epochs"].append({
            "epoch": epoch + 1, "loss": avg_loss,
            "arc_easy": easy_acc, "arc_challenge": chal_acc,
        })

    # Final eval on test
    test_chal_acc = evaluate_arc(model, tokenizer, test_challenge, device, label_ids)
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    for condition in args.conditions:
        for seed in args.seeds:
            print(f"\n=== {condition} / seed={seed} ===")
            outdir = os.path.join(PLAN9_DIR, "arc", "results")
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

    print("\n=== All ARC runs complete ===")


if __name__ == "__main__":
    main()
