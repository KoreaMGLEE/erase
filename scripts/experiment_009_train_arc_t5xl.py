"""Plan 009: T5-v1.1-XL retraining on ARC with K=5% conditions.
Uses LoRA + Adafactor. Eval on ARC-Easy val, ARC-Challenge val/test.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Adafactor,
    get_constant_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

PLAN9_DIR = "/workspace/erase/outputs/plan9"
MAX_CHOICES = 5
CHOICE_LABELS = ["A", "B", "C", "D", "E"]


class ARCDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def format_t5_arc_input(item):
    lines = [f"Question: {item['question']}"]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{CHOICE_LABELS[i]}) {choice}")
    lines.append("answer: <extra_id_0>")
    return "\n".join(lines)


def collate_t5_arc(batch, tokenizer, max_length=512):
    inputs = [format_t5_arc_input(item) for item in batch]
    targets = [f"<extra_id_0> {CHOICE_LABELS[item['correct_idx']]}" for item in batch]
    labels_int = torch.tensor([item["correct_idx"] for item in batch])
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors="pt").input_ids
    tgt[tgt == tokenizer.pad_token_id] = -100
    return {**enc, "decoder_labels": tgt, "labels_int": labels_int}


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


def evaluate(model, tokenizer, data, device, sentinel_id, t5_label_ids):
    model.eval()
    loader = DataLoader(ARCDataset(data), batch_size=16, shuffle=False,
                        collate_fn=lambda b: collate_t5_arc(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bs = input_ids.size(0)
            dec = torch.full((bs, 2), tokenizer.pad_token_id, dtype=torch.long, device=device)
            dec[:, 1] = sentinel_id
            out = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec)
            logits = out.logits[:, 1, :]
            lid = [t5_label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
            preds = logits[:, lid].argmax(dim=-1)
            labels = batch["labels_int"].to(device)
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
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xl", legacy=True)
    model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-xl", torch_dtype=torch.bfloat16).to(device)

    # LoRA
    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
                              lora_dropout=0.05, target_modules=["q", "v"])
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M ({trainable/total_params*100:.1f}%)")

    # Label token IDs
    sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    t5_label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in CHOICE_LABELS}

    # Training
    train_loader = DataLoader(ARCDataset(train_filtered), batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_t5_arc(b, tokenizer))

    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))

    results = {"condition": condition, "seed": seed, "model": "t5-v1_1-xl", "epochs": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_steps = 0
        for batch in train_loader:
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
        easy_acc = evaluate(model, tokenizer, val_easy, device, sentinel_id, t5_label_ids)
        chal_acc = evaluate(model, tokenizer, val_challenge, device, sentinel_id, t5_label_ids)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} easy={easy_acc:.4f} chal={chal_acc:.4f}")
        results["epochs"].append({
            "epoch": epoch + 1, "loss": avg_loss,
            "arc_easy": easy_acc, "arc_challenge": chal_acc,
        })

    # Final test eval
    test_chal_acc = evaluate(model, tokenizer, test_challenge, device, sentinel_id, t5_label_ids)
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--conditions", nargs="+",
                        default=["C1_full", "C2_random", "C3_self_easy_bert", "C5_union", "C6_dedup"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--data_dir", default="arc_k5")
    parser.add_argument("--result_tag", default="arc_t5xl_k5")
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

    print("\n=== All T5-XL ARC runs complete ===")


if __name__ == "__main__":
    main()
