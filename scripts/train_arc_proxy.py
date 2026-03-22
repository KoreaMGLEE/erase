"""ARC Proxy model training with multiple-choice classification.

Supports: BERT (BertForMultipleChoice), Pythia (log-prob scoring).
"""
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
from datasets import load_dataset

MODEL_REGISTRY = {
    "bert-mini": {"hf_id": "prajjwal1/bert-mini", "type": "encoder", "pad_side": "right"},
    "bert-small": {"hf_id": "prajjwal1/bert-small", "type": "encoder", "pad_side": "right"},
    "bert-medium": {"hf_id": "prajjwal1/bert-medium", "type": "encoder", "pad_side": "right"},
    "bert-base": {"hf_id": "bert-base-uncased", "type": "encoder", "pad_side": "right"},
    "bert-large": {"hf_id": "bert-large-uncased", "type": "encoder", "pad_side": "right"},
    "pythia-70m": {"hf_id": "EleutherAI/pythia-70m", "type": "decoder", "pad_side": "left"},
    "pythia-160m": {"hf_id": "EleutherAI/pythia-160m", "type": "decoder", "pad_side": "left"},
    "pythia-410m": {"hf_id": "EleutherAI/pythia-410m", "type": "decoder", "pad_side": "left"},
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "type": "decoder", "pad_side": "left", "lora": True},
    "pythia-1.4b": {"hf_id": "EleutherAI/pythia-1.4b", "type": "decoder", "pad_side": "left", "lora": True},
    "pythia-2.8b": {"hf_id": "EleutherAI/pythia-2.8b", "type": "decoder", "pad_side": "left", "lora": True},
    "pythia-6.9b": {"hf_id": "EleutherAI/pythia-6.9b", "type": "decoder", "pad_side": "left", "lora": True},
}

MAX_CHOICES = 5  # ARC has mostly 4, sometimes 3 or 5


class ARCDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item


def load_arc_data():
    """Load ARC Easy + Challenge, merge train, separate dev sets."""
    easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
    challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    def parse_examples(split_data):
        examples = []
        for item in split_data:
            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]
            # Find correct index
            correct_idx = labels.index(answer_key) if answer_key in labels else 0
            examples.append({
                "question": question,
                "choices": choices,
                "correct_idx": correct_idx,
                "num_choices": len(choices),
                "id": item["id"],
            })
        return examples

    train_data = parse_examples(easy["train"]) + parse_examples(challenge["train"])
    val_overall = parse_examples(easy["validation"]) + parse_examples(challenge["validation"])
    val_challenge = parse_examples(challenge["validation"])

    return train_data, val_overall, val_challenge


def load_model_and_tokenizer(model_name, device, use_bf16=False):
    info = MODEL_REGISTRY[model_name]
    hf_id = info["hf_id"]
    mtype = info["type"]
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    if mtype == "encoder":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMultipleChoice.from_pretrained(hf_id, torch_dtype=dtype)
    elif mtype == "decoder":
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    else:
        raise ValueError(f"Unknown type: {mtype}")

    model = model.to(device)

    if info.get("lora"):
        if mtype == "decoder":
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=32, lora_alpha=64, lora_dropout=0.05,
                target_modules=["query_key_value"],
            )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total ({trainable/total*100:.1f}%)")

    return model, tokenizer, mtype


def collate_bert_mc(batch, tokenizer, max_length=128):
    """Collate for BertForMultipleChoice. Shape: (batch, num_choices, seq_len)."""
    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    labels = []

    for item in batch:
        question = item["question"]
        choices = item["choices"]
        num_choices = len(choices)

        # Pad to MAX_CHOICES if needed
        while len(choices) < MAX_CHOICES:
            choices.append("")

        choice_encodings = tokenizer(
            [question] * MAX_CHOICES,
            choices[:MAX_CHOICES],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_input_ids.append(choice_encodings["input_ids"])
        batch_attention_mask.append(choice_encodings["attention_mask"])
        if "token_type_ids" in choice_encodings:
            batch_token_type_ids.append(choice_encodings["token_type_ids"])
        labels.append(item["correct_idx"])

    result = {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.tensor(labels),
    }
    if batch_token_type_ids:
        result["token_type_ids"] = torch.stack(batch_token_type_ids)
    return result


CHOICE_LABELS = ["A", "B", "C", "D", "E"]


def format_pythia_mc(question, choices):
    """Format question + all choices into a single prompt."""
    lines = [f"Question: {question}"]
    for i, choice in enumerate(choices):
        lines.append(f"{CHOICE_LABELS[i]}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def get_arc_label_token_ids(tokenizer):
    """Get token IDs for A, B, C, D, E (with leading space)."""
    ids = {}
    for label in CHOICE_LABELS:
        token_id = tokenizer.encode(f" {label}", add_special_tokens=False)[0]
        ids[label] = token_id
    return ids


def collate_pythia_mc(batch, tokenizer, max_length=512):
    """Collate for Pythia multiple choice — all choices in one prompt."""
    texts = [format_pythia_mc(item["question"], item["choices"]) for item in batch]
    labels = torch.tensor([item["correct_idx"] for item in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def train_pythia_step(model, tokenizer, batch, device, label_token_ids):
    """Train Pythia: predict A/B/C/D at last position, CE loss over choice tokens."""
    collated = collate_pythia_mc(batch, tokenizer)
    input_ids = collated["input_ids"].to(device)
    attention_mask = collated["attention_mask"].to(device)
    labels = collated["labels"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]  # (batch, vocab)

    # Get logits for A, B, C, D, E
    choice_ids = [label_token_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
    choice_logits = logits[:, choice_ids]  # (batch, num_choices)

    loss = F.cross_entropy(choice_logits, labels)
    return loss


@torch.no_grad()
def evaluate(model, data, mtype, device, tokenizer):
    """Evaluate on ARC data, return accuracy."""
    model.eval()
    correct = 0
    total = 0

    if mtype == "encoder":
        loader = DataLoader(ARCDataset(data), batch_size=16, shuffle=False,
                            collate_fn=lambda b: collate_bert_mc(b, tokenizer))
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            # Mask out padded choices
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    elif mtype == "decoder":
        label_token_ids = get_arc_label_token_ids(tokenizer)
        loader = DataLoader(ARCDataset(data), batch_size=16, shuffle=False,
                            collate_fn=lambda b: collate_pythia_mc(b, tokenizer))
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            choice_ids = [label_token_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
            choice_logits = logits[:, choice_ids]
            preds = choice_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    model.train()
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def measure_train_confidence(model, train_data, mtype, device, tokenizer, arc_label_token_ids=None):
    """Measure confidence (P(correct choice)) for all training examples."""
    model.eval()
    confidences = {}

    if mtype == "encoder":
        loader = DataLoader(ARCDataset(train_data), batch_size=16, shuffle=False,
                            collate_fn=lambda b: (b, collate_bert_mc(b, tokenizer)))
        for raw_batch, collated in loader:
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)
            token_type_ids = collated.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            probs = F.softmax(outputs.logits, dim=-1)
            correct_probs = probs[range(len(labels)), labels]

            for item, conf in zip(raw_batch, correct_probs):
                confidences[item["id"]] = float(conf.cpu())

    elif mtype == "decoder":
        loader = DataLoader(ARCDataset(train_data), batch_size=16, shuffle=False,
                            collate_fn=lambda b: (b, collate_pythia_mc(b, tokenizer)))
        for raw_batch, collated in loader:
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            choice_ids = [arc_label_token_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
            choice_logits = logits[:, choice_ids]
            probs = F.softmax(choice_logits.float(), dim=-1)
            correct_probs = probs[range(len(labels)), labels]

            for item, conf in zip(raw_batch, correct_probs):
                confidences[item["id"]] = float(conf.cpu())

    model.train()
    return confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}, LR: {args.lr}, Epochs: {args.epochs}, Device: {device}")

    # Load data
    print("Loading ARC dataset...")
    train_data, val_overall, val_challenge = load_arc_data()
    print(f"Train: {len(train_data)}, Val overall: {len(val_overall)}, Val challenge: {len(val_challenge)}")

    # Load model
    print(f"Loading model {args.model}...")
    model, tokenizer, mtype = load_model_and_tokenizer(args.model, device, use_bf16=args.bf16)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    # Label token IDs for decoder models
    arc_label_token_ids = None
    if mtype == "decoder":
        arc_label_token_ids = get_arc_label_token_ids(tokenizer)
        print(f"ARC label tokens: {arc_label_token_ids}")

    if mtype == "encoder":
        train_dataset = ARCDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda b: collate_bert_mc(b, tokenizer))
        total_steps = len(train_loader) * args.epochs
    else:
        train_dataset = ARCDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=lambda b: b)  # raw collate
        total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps
    )

    # Training
    start_time = time.time()
    best_overall_acc = 0
    best_challenge_acc = 0
    best_epoch = 0
    best_train_conf = None
    epoch_results = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_steps = 0

        if mtype == "encoder":
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                num_steps += 1

        elif mtype == "decoder":
            for batch in train_loader:
                optimizer.zero_grad()
                loss = train_pythia_step(model, tokenizer, batch, device, arc_label_token_ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                num_steps += 1

        avg_loss = total_loss / num_steps if num_steps > 0 else 0

        # Evaluate
        overall_acc = evaluate(model, val_overall, mtype, device, tokenizer)
        challenge_acc = evaluate(model, val_challenge, mtype, device, tokenizer)

        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
              f"overall_acc={overall_acc:.4f}, challenge_acc={challenge_acc:.4f}")

        epoch_results.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "overall_acc": overall_acc,
            "challenge_acc": challenge_acc,
        })

        if overall_acc > best_overall_acc:
            best_overall_acc = overall_acc
            best_challenge_acc = challenge_acc
            best_epoch = epoch + 1

            # Measure confidence on train data at best epoch
            best_train_conf = measure_train_confidence(
                model, train_data, mtype, device, tokenizer,
                arc_label_token_ids if mtype == "decoder" else None
            )

    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Best overall_acc: {best_overall_acc:.4f}, challenge_acc: {best_challenge_acc:.4f} (epoch {best_epoch})")

    results = {
        "model": args.model,
        "lr": args.lr,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_challenge_acc": best_challenge_acc,
        "best_overall_acc": best_overall_acc,
        "training_time_sec": elapsed,
        "epoch_results": epoch_results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save confidence
    if best_train_conf is not None:
        conf_path = os.path.join(args.output_dir, "train_confidence.json")
        with open(conf_path, "w") as f:
            json.dump(best_train_conf, f)

        confs = np.array(list(best_train_conf.values()))
        print(f"Confidence: mean={confs.mean():.4f}, std={confs.std():.4f}, "
              f">0.95={np.mean(confs>0.95)*100:.1f}%, <0.25={np.mean(confs<0.25)*100:.1f}%")

    print(f"Results saved to {results_path}")
    print("Done!")


if __name__ == "__main__":
    main()
