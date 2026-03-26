"""ARC training with per-example confidence extraction on train set.
Reuses logic from train_arc_all_models.py but adds confidence measurement.
"""
import argparse, json, os, time, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, BertForMultipleChoice,
    T5Tokenizer, T5ForConditionalGeneration, Adafactor,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

MODEL_REGISTRY = {
    "bert-mini": {"hf_id": "prajjwal1/bert-mini", "family": "encoder"},
    "bert-small": {"hf_id": "prajjwal1/bert-small", "family": "encoder"},
    "bert-medium": {"hf_id": "prajjwal1/bert-medium", "family": "encoder"},
    "bert-base": {"hf_id": "bert-base-uncased", "family": "encoder"},
    "bert-large": {"hf_id": "bert-large-uncased", "family": "encoder"},
    "t5-v1_1-small": {"hf_id": "google/t5-v1_1-small", "family": "enc-dec"},
    "t5-v1_1-base": {"hf_id": "google/t5-v1_1-base", "family": "enc-dec"},
    "t5-v1_1-large": {"hf_id": "google/t5-v1_1-large", "family": "enc-dec"},
    "t5-v1_1-xl": {"hf_id": "google/t5-v1_1-xl", "family": "enc-dec", "lora": True},
    "pythia-14m": {"hf_id": "EleutherAI/pythia-14m", "family": "decoder"},
    "pythia-31m": {"hf_id": "EleutherAI/pythia-31m", "family": "decoder"},
    "pythia-70m": {"hf_id": "EleutherAI/pythia-70m", "family": "decoder"},
    "pythia-160m": {"hf_id": "EleutherAI/pythia-160m", "family": "decoder"},
    "pythia-410m": {"hf_id": "EleutherAI/pythia-410m", "family": "decoder"},
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "family": "decoder", "lora": True},
    "pythia-1.4b": {"hf_id": "EleutherAI/pythia-1.4b", "family": "decoder", "lora": True},
    "pythia-2.8b": {"hf_id": "EleutherAI/pythia-2.8b", "family": "decoder", "lora": True},
    "pythia-6.9b": {"hf_id": "EleutherAI/pythia-6.9b", "family": "decoder", "lora": True},
    "pythia-12b": {"hf_id": "EleutherAI/pythia-12b", "family": "decoder", "lora": True, "multi_gpu": True},
    "t5-v1_1-xxl": {"hf_id": "google/t5-v1_1-xxl", "family": "enc-dec", "lora": True, "multi_gpu": True},
}

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
    train = parse(easy["train"]) + parse(challenge["train"])
    val_overall = parse(easy["validation"]) + parse(challenge["validation"])
    return train, val_overall


# ============ Collate functions ============

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


def format_pythia_arc(item):
    lines = [f"Question: {item['question']}"]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{CHOICE_LABELS[i]}) {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def collate_pythia_arc(batch, tokenizer, max_length=512):
    texts = [format_pythia_arc(item) for item in batch]
    labels = torch.tensor([item["correct_idx"] for item in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


# ============ Model Loading ============

def load_model(model_name, device, use_bf16=False):
    info = MODEL_REGISTRY[model_name]
    family = info["family"]
    hf_id = info["hf_id"]
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    multi_gpu = info.get("multi_gpu", False)

    if family == "encoder":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMultipleChoice.from_pretrained(hf_id, torch_dtype=dtype).to(device)
    elif family == "enc-dec":
        tokenizer = T5Tokenizer.from_pretrained(hf_id, legacy=True)
        if multi_gpu:
            model = T5ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
        else:
            model = T5ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=dtype).to(device)
    elif family == "decoder":
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if multi_gpu:
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype).to(device)

    if info.get("lora"):
        if family == "enc-dec":
            lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
                                      lora_dropout=0.05, target_modules=["q", "v"])
        elif family == "decoder":
            lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64,
                                      lora_dropout=0.05, target_modules=["query_key_value"])
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    return model, tokenizer, family


def get_train_confidence(model, tokenizer, family, train_data, device, t5_label_ids=None, pythia_label_ids=None, sentinel_id=None):
    """Compute per-example P(correct) on the training set."""
    if family == "encoder":
        collate = lambda b: collate_bert_mc(b, tokenizer)
    elif family == "enc-dec":
        collate = lambda b: collate_t5_arc(b, tokenizer)
    elif family == "decoder":
        collate = lambda b: collate_pythia_arc(b, tokenizer)

    loader = DataLoader(ARCDataset(train_data), batch_size=32, shuffle=False, collate_fn=collate)

    model.eval()
    all_confs = []
    with torch.no_grad():
        for batch in loader:
            if family == "encoder":
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device) if "token_type_ids" in batch else None)
                # logits: (bs, num_choices)
                probs = F.softmax(out.logits, dim=-1)
                labels = batch["labels"]
                for i in range(len(labels)):
                    nc = min(MAX_CHOICES, probs.size(1))
                    p = probs[i, :nc]
                    p = p / p.sum()  # renormalize for actual choices
                    all_confs.append(p[labels[i]].item())

            elif family == "enc-dec":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                bs = input_ids.size(0)
                dec = torch.full((bs, 2), tokenizer.pad_token_id, dtype=torch.long, device=device)
                dec[:, 1] = sentinel_id
                out = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec)
                logits = out.logits[:, 1, :]
                lid = [t5_label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
                probs = F.softmax(logits[:, lid], dim=-1)
                labels = batch["labels_int"]
                for i in range(len(labels)):
                    all_confs.append(probs[i, labels[i]].item())

            elif family == "decoder":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[:, -1, :]
                choice_ids = [pythia_label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
                probs = F.softmax(logits[:, choice_ids], dim=-1)
                labels = batch["labels"]
                for i in range(len(labels)):
                    all_confs.append(probs[i, labels[i]].item())

    model.train()
    return all_confs


def train_with_confidence(model, tokenizer, family, train_data, val_data, device, args):
    if family == "encoder":
        collate = lambda b: collate_bert_mc(b, tokenizer)
    elif family == "enc-dec":
        collate = lambda b: collate_t5_arc(b, tokenizer)
    elif family == "decoder":
        collate = lambda b: collate_pythia_arc(b, tokenizer)

    train_loader = DataLoader(ARCDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(ARCDataset(val_data), batch_size=32, shuffle=False, collate_fn=collate)

    if family == "enc-dec":
        optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))
    else:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                       lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    t5_label_ids = None
    pythia_label_ids = None
    sentinel_id = None
    if family == "enc-dec":
        sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
        t5_label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in CHOICE_LABELS}
    elif family == "decoder":
        pythia_label_ids = {l: tokenizer.encode(f" {l}", add_special_tokens=False)[0] for l in CHOICE_LABELS}

    best_overall = 0
    best_epoch = 0
    epoch_results = []
    epoch_confidences = {}  # epoch -> list of P(correct) per train example

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_steps = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if family == "encoder":
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device) if "token_type_ids" in batch else None,
                            labels=batch["labels"].to(device))
                loss = out.loss
            elif family == "enc-dec":
                out = model(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            labels=batch["decoder_labels"].to(device))
                loss = out.loss
            elif family == "decoder":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[:, -1, :]
                choice_ids = [pythia_label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
                loss = F.cross_entropy(logits[:, choice_ids], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_steps += 1

        avg_loss = total_loss / n_steps

        # Eval on val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                if family == "encoder":
                    out = model(input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                token_type_ids=batch.get("token_type_ids", torch.zeros_like(batch["input_ids"])).to(device) if "token_type_ids" in batch else None)
                    preds = out.logits.argmax(dim=-1)
                elif family == "enc-dec":
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    bs = input_ids.size(0)
                    dec = torch.full((bs, 2), tokenizer.pad_token_id, dtype=torch.long, device=device)
                    dec[:, 1] = sentinel_id
                    out = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec)
                    logits = out.logits[:, 1, :]
                    lid = [t5_label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
                    preds = logits[:, lid].argmax(dim=-1)
                elif family == "decoder":
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits[:, -1, :]
                    choice_ids = [pythia_label_ids[CHOICE_LABELS[i]] for i in range(MAX_CHOICES)]
                    preds = logits[:, choice_ids].argmax(dim=-1)

                labels = batch["labels"].to(device) if "labels" in batch else batch["labels_int"].to(device)
                correct += (preds == labels).sum().item()
                total += len(labels)

        overall_acc = correct / total
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={overall_acc:.4f}")
        epoch_results.append({"epoch": epoch+1, "loss": avg_loss, "overall_acc": overall_acc})

        if overall_acc > best_overall:
            best_overall = overall_acc
            best_epoch = epoch + 1

        # Per-example confidence on train set
        confs = get_train_confidence(model, tokenizer, family, train_data, device,
                                     t5_label_ids, pythia_label_ids, sentinel_id)
        epoch_confidences[epoch + 1] = confs

    return best_overall, best_epoch, epoch_results, epoch_confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=5)
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

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Model: {args.model}, LR: {args.lr}, Epochs: {args.epochs}, Seed: {args.seed}")

    train_data, val_overall = load_arc_data()
    print(f"Train: {len(train_data)}, Val: {len(val_overall)}")

    model, tokenizer, family = load_model(args.model, device, use_bf16=args.bf16)

    start = time.time()
    best_acc, best_epoch, epoch_results, epoch_confidences = train_with_confidence(
        model, tokenizer, family, train_data, val_overall, device, args)
    elapsed = time.time() - start

    print(f"Best: val_acc={best_acc:.4f} (epoch {best_epoch}), time={elapsed:.1f}s")

    # Average confidence across epochs 1..best_epoch (overfitting prevention)
    ids = [item["id"] for item in train_data]
    avg_conf = {}
    for i, arc_id in enumerate(ids):
        vals = [epoch_confidences[ep][i] for ep in range(1, best_epoch + 1)]
        avg_conf[arc_id] = float(np.mean(vals))

    mean_conf = np.mean(list(avg_conf.values()))
    print(f"Avg confidence (epochs 1..{best_epoch}): {mean_conf:.4f}")

    # Save
    with open(os.path.join(args.output_dir, "avg_conf.json"), "w") as f:
        json.dump(avg_conf, f)

    results = {
        "model": args.model, "family": family, "lr": args.lr,
        "seed": args.seed, "epochs": args.epochs,
        "best_overall_acc": best_acc, "best_epoch": best_epoch,
        "training_time_sec": elapsed, "epoch_results": epoch_results,
        "mean_confidence": mean_conf,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
