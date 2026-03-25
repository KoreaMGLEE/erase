"""MNLI Proxy training for T5 v1.1 models (enc-dec, text-to-text).

T5 v1.1 was NOT pretrained on MNLI (C4 only), so confidence saturation should not occur.
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
    AutoTokenizer, T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

MODEL_REGISTRY = {
    "t5-v1_1-small": {"hf_id": "google/t5-v1_1-small", "params": "77M"},
    "t5-v1_1-base": {"hf_id": "google/t5-v1_1-base", "params": "248M"},
    "t5-v1_1-large": {"hf_id": "google/t5-v1_1-large", "params": "783M"},
    "t5-v1_1-xl": {"hf_id": "google/t5-v1_1-xl", "params": "3B", "lora": True},
    "t5-v1_1-xxl": {"hf_id": "google/t5-v1_1-xxl", "params": "11B", "lora": True},
}

LABEL_NAMES = ["entailment", "neutral", "contradiction"]


class MNLIDataset(Dataset):
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        item = self.ds[self.indices[idx]]
        return {
            "premise": item["premise"],
            "hypothesis": item["hypothesis"],
            "label": item["label"],
            "original_index": self.indices[idx],
        }


def collate_t5(batch, tokenizer, max_length=256):
    inputs = [f"mnli hypothesis: {b['hypothesis']} premise: {b['premise']}" for b in batch]
    targets = [LABEL_NAMES[b["label"]] for b in batch]
    labels_int = torch.tensor([b["label"] for b in batch])

    enc = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors="pt")
    tgt_ids = tgt["input_ids"]
    tgt_ids[tgt_ids == tokenizer.pad_token_id] = -100

    return {**enc, "decoder_labels": tgt_ids, "labels_int": labels_int}


def raw_collate(batch):
    return batch


def get_label_token_ids(tokenizer):
    ids = {}
    for label in LABEL_NAMES:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        ids[label] = token_ids[0]
    return ids


def load_model_and_tokenizer(model_name, device, use_bf16=False):
    info = MODEL_REGISTRY[model_name]
    hf_id = info["hf_id"]
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(hf_id, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=dtype)
    model = model.to(device)

    if info.get("lora"):
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q", "v"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")

    return model, tokenizer


@torch.no_grad()
def measure_confidence(model, dataloader, device, tokenizer, label_token_ids):
    model.eval()
    all_confs = []
    all_indices = []

    for batch_raw in dataloader:
        collated = collate_t5(batch_raw, tokenizer)
        input_ids = collated["input_ids"].to(device)
        attention_mask = collated["attention_mask"].to(device)
        labels = collated["labels_int"].to(device)
        bs = input_ids.size(0)

        decoder_input_ids = torch.full((bs, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[:, 0, :]
        label_ids = [label_token_ids[LABEL_NAMES[i]] for i in range(3)]
        label_logits = logits[:, label_ids]
        probs = F.softmax(label_logits.float(), dim=-1)
        conf = probs[range(len(labels)), labels]

        all_confs.append(conf.cpu().numpy())
        all_indices.extend([b["original_index"] for b in batch_raw])

    model.train()
    return np.concatenate(all_confs), all_indices


def evaluate_val(model, val_loader, device, tokenizer, label_token_ids):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch_raw in val_loader:
            collated = collate_t5(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels_int"].to(device)
            bs = input_ids.size(0)

            decoder_input_ids = torch.full((bs, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[:, 0, :]
            label_ids = [label_token_ids[LABEL_NAMES[i]] for i in range(3)]
            label_logits = logits[:, label_ids]
            preds = label_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    model.train()
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, train_loader, optimizer, scheduler, device, tokenizer,
                    label_token_ids=None, confidence_checkpoints=None, conf_dataloader=None):
    model.train()
    total_loss = 0
    num_steps = 0
    checkpoint_confidences = {}

    for step, batch_raw in enumerate(train_loader, 1):
        optimizer.zero_grad()
        collated = collate_t5(batch_raw, tokenizer)
        input_ids = collated["input_ids"].to(device)
        attention_mask = collated["attention_mask"].to(device)
        decoder_labels = collated["decoder_labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_steps += 1

        if step % 200 == 0:
            print(f"  Step {step}, loss={loss.item():.4f}")

        if confidence_checkpoints and step in confidence_checkpoints and conf_dataloader is not None:
            print(f"  Measuring confidence at step {step}...")
            confs, indices = measure_confidence(model, conf_dataloader, device, tokenizer, label_token_ids)
            checkpoint_confidences[step] = (confs, indices)
            model.train()

    avg_loss = total_loss / num_steps if num_steps > 0 else 0
    return avg_loss, checkpoint_confidences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--split", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--mode", choices=["lr_search", "confidence"], default="lr_search")
    parser.add_argument("--adafactor", action="store_true", help="Use Adafactor optimizer (T5 default)")
    parser.add_argument("--constant_lr", action="store_true", help="Use constant LR (no decay)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="/workspace/erase/outputs/plan0/data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}, Split: {args.split}, LR: {args.lr}, Device: {device}, Mode: {args.mode}")

    # Load data
    print("Loading MNLI dataset...")
    ds = load_dataset("glue", "mnli")
    train_ds = ds["train"]
    val_ds = ds["validation_matched"]

    with open(os.path.join(args.data_dir, f"{args.split}_indices.json")) as f:
        split_indices = json.load(f)
    print(f"Split {args.split}: {len(split_indices)} examples")

    # Load model
    print(f"Loading model {args.model}...")
    model, tokenizer = load_model_and_tokenizer(args.model, device, use_bf16=args.bf16)

    label_token_ids = get_label_token_ids(tokenizer)
    print(f"Label tokens: {label_token_ids}")

    # Datasets
    train_dataset = MNLIDataset(train_ds, split_indices)
    val_dataset = MNLIDataset(val_ds, list(range(len(val_ds))))

    eval_bs = min(64, args.batch_size * 4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=raw_collate, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False,
                            collate_fn=raw_collate, num_workers=2)

    # Optimizer
    total_steps = len(train_loader) * args.epochs
    if args.adafactor:
        from transformers import Adafactor
        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False,
        )
        print(f"Using Adafactor (fixed lr={args.lr})")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01
        )
    if args.constant_lr:
        from transformers import get_constant_schedule_with_warmup
        warmup_steps = int(0.06 * total_steps)
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        print(f"Using constant LR scheduler with warmup ({warmup_steps} steps)")
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    # Confidence checkpoints
    confidence_checkpoints = None
    conf_dataloader = None
    if args.mode == "confidence":
        steps_per_epoch = len(train_loader)
        confidence_checkpoints = {
            steps_per_epoch // 3,
            2 * steps_per_epoch // 3,
            steps_per_epoch,
        }
        print(f"Confidence checkpoints at steps: {sorted(confidence_checkpoints)}")
        conf_dataloader = DataLoader(train_dataset, batch_size=eval_bs, shuffle=False,
                                     collate_fn=raw_collate, num_workers=2)

    # Train
    start_time = time.time()
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        avg_loss, ckpt_confs = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, tokenizer,
            label_token_ids, confidence_checkpoints, conf_dataloader
        )
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        val_acc = evaluate_val(model, val_loader, device, tokenizer, label_token_ids)
        print(f"Epoch {epoch+1} val accuracy: {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save results
    results = {
        "model": args.model, "split": args.split, "lr": args.lr,
        "seed": args.seed, "mode": args.mode,
        "val_acc": val_acc, "avg_loss": avg_loss,
        "training_time_sec": elapsed,
    }

    if args.mode == "lr_search":
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "confidence" and ckpt_confs:
        all_confs_list = []
        for step in sorted(ckpt_confs.keys()):
            confs, indices = ckpt_confs[step]
            ckpt_data = {str(idx): float(c) for idx, c in zip(indices, confs)}
            ckpt_path = os.path.join(args.output_dir, f"ckpt_step{step}_conf.json")
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_data, f)
            all_confs_list.append(confs)

        avg_conf = np.mean(all_confs_list, axis=0)
        _, indices = list(ckpt_confs.values())[0]
        avg_data = {str(idx): float(c) for idx, c in zip(indices, avg_conf)}
        with open(os.path.join(args.output_dir, "avg_conf.json"), "w") as f:
            json.dump(avg_data, f)

        results["confidence_stats"] = {
            "mean": float(np.mean(avg_conf)),
            "std": float(np.std(avg_conf)),
            "gt_095_ratio": float(np.mean(avg_conf > 0.95)),
            "lt_04_ratio": float(np.mean(avg_conf < 0.4)),
        }
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
