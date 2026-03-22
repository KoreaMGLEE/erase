"""Plan 3: Train ARC model on difficulty-based subsets."""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer, BertForMultipleChoice,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

MODEL_REGISTRY = {
    "bert-large": {"hf_id": "bert-large-uncased", "type": "encoder"},
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "type": "decoder", "lora": True},
}

MAX_CHOICES = 5
CHOICE_LABELS = ["A", "B", "C", "D", "E"]


class SubsetDataset(Dataset):
    def __init__(self, all_data, indices):
        self.data = [all_data[i] for i in indices]

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
        enc = tokenizer(
            [item["question"]] * MAX_CHOICES, choices[:MAX_CHOICES],
            padding="max_length", truncation=True, max_length=max_length, return_tensors="pt",
        )
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


def format_pythia_mc(item):
    lines = [f"Question: {item['question']}"]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{CHOICE_LABELS[i]}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def collate_pythia_mc(batch, tokenizer, max_length=512):
    texts = [format_pythia_mc(item) for item in batch]
    labels = torch.tensor([item["correct_idx"] for item in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def get_arc_label_token_ids(tokenizer):
    return {label: tokenizer.encode(f" {label}", add_special_tokens=False)[0] for label in CHOICE_LABELS}


def load_model(model_name, device, use_bf16=False):
    info = MODEL_REGISTRY[model_name]
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if info["type"] == "encoder":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMultipleChoice.from_pretrained(info["hf_id"], torch_dtype=dtype)
    else:
        tokenizer = AutoTokenizer.from_pretrained(info["hf_id"])
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(info["hf_id"], torch_dtype=dtype)

    model = model.to(device)

    if info.get("lora"):
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
                                  lora_dropout=0.05, target_modules=["query_key_value"])
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        print(f"LoRA: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    return model, tokenizer, info["type"]


def load_eval_data():
    """Load ARC Easy dev + Challenge dev."""
    from datasets import load_dataset
    easy_val = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    challenge_val = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")

    def parse(ds):
        items = []
        for item in ds:
            items.append({
                "id": item["id"], "question": item["question"],
                "choices": item["choices"]["text"],
                "choice_labels": item["choices"]["label"],
                "correct_idx": item["choices"]["label"].index(item["answerKey"]),
                "num_choices": len(item["choices"]["text"]),
            })
        return items

    easy = parse(easy_val)
    challenge = parse(challenge_val)
    overall = easy + challenge
    return overall, easy, challenge


@torch.no_grad()
def evaluate(model, data, mtype, device, tokenizer, label_token_ids=None):
    model.eval()
    correct = total = 0

    if mtype == "encoder":
        loader = DataLoader(SubsetDataset(data, list(range(len(data)))), batch_size=16, shuffle=False,
                            collate_fn=lambda b: collate_bert_mc(b, tokenizer))
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device) if "token_type_ids" in batch else None)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += len(batch["labels"])
    else:
        loader = DataLoader(SubsetDataset(data, list(range(len(data)))), batch_size=16, shuffle=False,
                            collate_fn=lambda b: collate_pythia_mc(b, tokenizer))
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            logits = out.logits[:, -1, :]
            choice_ids = [label_token_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
            preds = logits[:, choice_ids].argmax(dim=-1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += len(batch["labels"])

    model.train()
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--condition", required=True, help="Condition name for output")
    parser.add_argument("--train_indices", required=True, help="JSON file with list of indices")
    parser.add_argument("--data_file", required=True, help="matched_train.json path")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Model: {args.model}, Condition: {args.condition}, Device: {device}")

    # Load data
    with open(args.data_file) as f:
        all_data = json.load(f)
    with open(args.train_indices) as f:
        indices = json.load(f)
    print(f"Train subset: {len(indices)} examples")

    # Load eval data
    val_overall, val_easy, val_challenge = load_eval_data()
    print(f"Eval: overall={len(val_overall)}, easy={len(val_easy)}, challenge={len(val_challenge)}")

    # Load model
    model, tokenizer, mtype = load_model(args.model, device, use_bf16=args.bf16)
    label_token_ids = get_arc_label_token_ids(tokenizer) if mtype == "decoder" else None

    # DataLoader
    train_dataset = SubsetDataset(all_data, indices)
    if mtype == "encoder":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_bert_mc(b, tokenizer))
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_pythia_mc(b, tokenizer))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    # Train
    start = time.time()
    best_overall = 0
    best_results = {}
    epoch_log = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_steps = 0

        for batch in train_loader:
            optimizer.zero_grad()
            if mtype == "encoder":
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device) if "token_type_ids" in batch else None,
                            labels=batch["labels"].to(device))
                loss = out.loss
            else:
                coll = batch
                input_ids = coll["input_ids"].to(device)
                attention_mask = coll["attention_mask"].to(device)
                labels = coll["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[:, -1, :]
                choice_ids = [label_token_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
                loss = F.cross_entropy(logits[:, choice_ids], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / n_steps
        overall_acc = evaluate(model, val_overall, mtype, device, tokenizer, label_token_ids)
        easy_acc = evaluate(model, val_easy, mtype, device, tokenizer, label_token_ids)
        challenge_acc = evaluate(model, val_challenge, mtype, device, tokenizer, label_token_ids)

        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f} overall={overall_acc:.4f} easy={easy_acc:.4f} challenge={challenge_acc:.4f}")
        epoch_log.append({"epoch": epoch+1, "loss": avg_loss, "overall": overall_acc, "easy": easy_acc, "challenge": challenge_acc})

        if overall_acc > best_overall:
            best_overall = overall_acc
            best_results = {"epoch": epoch+1, "overall": overall_acc, "easy": easy_acc, "challenge": challenge_acc}

    elapsed = time.time() - start
    print(f"Best: epoch={best_results.get('epoch')}, overall={best_results.get('overall', 0):.4f}, "
          f"easy={best_results.get('easy', 0):.4f}, challenge={best_results.get('challenge', 0):.4f}")

    results = {
        "model": args.model, "condition": args.condition, "lr": args.lr,
        "train_size": len(indices), "best": best_results,
        "epoch_log": epoch_log, "time_sec": elapsed,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
