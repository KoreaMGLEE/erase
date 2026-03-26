"""MNLI Proxy model training with confidence measurement.

Supports: BERT (encoder), T5 (enc-dec), Pythia (decoder).
Modes: lr_search (Phase 1) and confidence (Phase 2).
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
    AutoTokenizer, AutoModelForSequenceClassification,
    BertForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# Model registry
MODEL_REGISTRY = {
    "bert-mini": {"hf_id": "prajjwal1/bert-mini", "type": "encoder", "pad_side": "right"},
    "bert-small": {"hf_id": "prajjwal1/bert-small", "type": "encoder", "pad_side": "right"},
    "bert-medium": {"hf_id": "prajjwal1/bert-medium", "type": "encoder", "pad_side": "right"},
    "bert-base": {"hf_id": "bert-base-uncased", "type": "encoder", "pad_side": "right"},
    "bert-large": {"hf_id": "bert-large-uncased", "type": "encoder", "pad_side": "right"},
    "pythia-14m": {"hf_id": "EleutherAI/pythia-14m", "type": "decoder", "pad_side": "left"},
    "pythia-31m": {"hf_id": "EleutherAI/pythia-31m", "type": "decoder", "pad_side": "left"},
    "pythia-70m": {"hf_id": "EleutherAI/pythia-70m", "type": "decoder", "pad_side": "left"},
    "pythia-160m": {"hf_id": "EleutherAI/pythia-160m", "type": "decoder", "pad_side": "left"},
    "pythia-410m": {"hf_id": "EleutherAI/pythia-410m", "type": "decoder", "pad_side": "left"},
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "type": "decoder", "pad_side": "left", "lora": True},
    "pythia-1.4b": {"hf_id": "EleutherAI/pythia-1.4b", "type": "decoder", "pad_side": "left", "lora": True},
    "pythia-2.8b": {"hf_id": "EleutherAI/pythia-2.8b", "type": "decoder", "pad_side": "left", "lora": True},
    "pythia-6.9b": {"hf_id": "EleutherAI/pythia-6.9b", "type": "decoder", "pad_side": "left", "lora": True},
    "pythia-12b": {"hf_id": "EleutherAI/pythia-12b", "type": "decoder", "pad_side": "left", "lora": True, "multi_gpu": True},
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


def collate_bert(batch, tokenizer, max_length=128):
    premises = [b["premise"] for b in batch]
    hypotheses = [b["hypothesis"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(premises, hypotheses, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def collate_t5(batch, tokenizer, max_length=256):
    inputs = [f"mnli hypothesis: {b['hypothesis']} premise: {b['premise']}" for b in batch]
    targets = [LABEL_NAMES[b["label"]] for b in batch]
    labels_int = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors="pt")
    tgt_ids = tgt["input_ids"]
    tgt_ids[tgt_ids == tokenizer.pad_token_id] = -100
    return {**enc, "decoder_labels": tgt_ids, "labels_int": labels_int}


def collate_pythia(batch, tokenizer, max_length=256):
    texts = [f"Premise: {b['premise']}\nHypothesis: {b['hypothesis']}\nRelation:" for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def load_model_and_tokenizer(model_name, device, use_bf16=False):
    info = MODEL_REGISTRY[model_name]
    hf_id = info["hf_id"]
    mtype = info["type"]

    if mtype == "encoder":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        model = BertForSequenceClassification.from_pretrained(hf_id, num_labels=3, torch_dtype=dtype)
    elif mtype == "enc-dec":
        tokenizer = T5Tokenizer.from_pretrained(hf_id)
        model = T5ForConditionalGeneration.from_pretrained(hf_id, torch_dtype=torch.float32)
    elif mtype == "decoder":
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        if info.get("multi_gpu"):
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype)
    else:
        raise ValueError(f"Unknown type: {mtype}")

    if not info.get("multi_gpu"):
        model = model.to(device)

    # Apply LoRA for large decoder models
    if info.get("lora"):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["query_key_value"],
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()  # needed for gradient checkpointing with LoRA
        model.gradient_checkpointing_enable()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA applied: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total ({trainable/total*100:.1f}%) [grad_ckpt=ON]")

    return model, tokenizer, mtype


def get_t5_label_token_ids(tokenizer):
    ids = {}
    for label in LABEL_NAMES:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        ids[label] = token_ids[0]
    return ids


def get_pythia_label_token_ids(tokenizer):
    """Returns dict with 'first' (first token id per label) and 'full' (all token ids per label)."""
    first = {}
    full = {}
    for label in LABEL_NAMES:
        token_ids = tokenizer.encode(f" {label}", add_special_tokens=False)
        first[label] = token_ids[0]
        full[label] = token_ids
    return {"first": first, "full": full}


@torch.no_grad()
def measure_confidence(model, dataloader, mtype, device, tokenizer=None, label_token_ids=None):
    """Measure confidence (P(correct class)) for all examples."""
    model.eval()
    all_confs = []
    all_indices = []

    for batch_raw in dataloader:
        if mtype == "encoder":
            collated = collate_bert(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            conf = probs[range(len(labels)), labels]

        elif mtype == "enc-dec":
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
            probs = F.softmax(label_logits, dim=-1)
            conf = probs[range(len(labels)), labels]

        elif mtype == "decoder":
            collated = collate_pythia(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            first_ids = label_token_ids["first"]
            label_ids = [first_ids[LABEL_NAMES[i]] for i in range(3)]
            label_logits = logits[:, label_ids]
            probs = F.softmax(label_logits, dim=-1)
            conf = probs[range(len(labels)), labels]

        all_confs.append(conf.float().cpu().numpy())
        all_indices.extend([b["original_index"] for b in batch_raw])

    model.train()
    return np.concatenate(all_confs), all_indices


def evaluate_val(model, val_loader, mtype, device, tokenizer, label_token_ids=None):
    """Compute validation accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_raw in val_loader:
            if mtype == "encoder":
                collated = collate_bert(batch_raw, tokenizer)
                input_ids = collated["input_ids"].to(device)
                attention_mask = collated["attention_mask"].to(device)
                labels = collated["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

            elif mtype == "enc-dec":
                collated = collate_t5(batch_raw, tokenizer)
                input_ids = collated["input_ids"].to(device)
                attention_mask = collated["attention_mask"].to(device)
                labels = collated["labels_int"].to(device)
                bs = input_ids.size(0)
                decoder_input_ids = torch.full((bs, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
                label_ids_list = [label_token_ids[LABEL_NAMES[i]] for i in range(3)]
                label_logits = logits[:, label_ids_list]
                preds = label_logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

            elif mtype == "decoder":
                collated = collate_pythia(batch_raw, tokenizer)
                input_ids = collated["input_ids"].to(device)
                attention_mask = collated["attention_mask"].to(device)
                labels = collated["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]
                first_ids = label_token_ids["first"]
                label_ids_list = [first_ids[LABEL_NAMES[i]] for i in range(3)]
                label_logits = logits[:, label_ids_list]
                preds = label_logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)

    model.train()
    return correct / total if total > 0 else 0.0


def train_one_epoch(model, train_loader, optimizer, scheduler, mtype, device, tokenizer,
                    label_token_ids=None, confidence_checkpoints=None, conf_dataloader=None):
    """Train for 1 epoch. Optionally measure confidence at checkpoints."""
    model.train()
    total_loss = 0
    num_steps = 0
    checkpoint_confidences = {}

    for step, batch_raw in enumerate(train_loader, 1):
        optimizer.zero_grad()

        if mtype == "encoder":
            collated = collate_bert(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        elif mtype == "enc-dec":
            collated = collate_t5(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            decoder_labels = collated["decoder_labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_labels)
            loss = outputs.loss

        elif mtype == "decoder":
            collated = collate_pythia(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels"].to(device)
            # Append full label token sequence and compute loss on all label tokens
            full_label_ids = label_token_ids["full"]
            # Build per-example: input + label tokens
            batch_inputs = []
            batch_masks = []
            batch_targets = []
            for i in range(input_ids.size(0)):
                label_name = LABEL_NAMES[labels[i].item()]
                ltoks = torch.tensor(full_label_ids[label_name], device=device)
                inp = torch.cat([input_ids[i], ltoks])
                msk = torch.cat([attention_mask[i], torch.ones(len(ltoks), device=device, dtype=attention_mask.dtype)])
                tgt = torch.cat([torch.full((input_ids.size(1),), -100, device=device, dtype=torch.long), ltoks])
                batch_inputs.append(inp)
                batch_masks.append(msk)
                batch_targets.append(tgt)
            # Pad to same length
            max_len = max(x.size(0) for x in batch_inputs)
            for i in range(len(batch_inputs)):
                pad_len = max_len - batch_inputs[i].size(0)
                if pad_len > 0:
                    batch_inputs[i] = torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, device=device, dtype=torch.long), batch_inputs[i]])
                    batch_masks[i] = torch.cat([torch.zeros(pad_len, device=device, dtype=attention_mask.dtype), batch_masks[i]])
                    batch_targets[i] = torch.cat([torch.full((pad_len,), -100, device=device, dtype=torch.long), batch_targets[i]])
            full_input = torch.stack(batch_inputs)
            full_mask = torch.stack(batch_masks)
            target = torch.stack(batch_targets)
            outputs = model(input_ids=full_input, attention_mask=full_mask, labels=target)
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_steps += 1

        if step % 200 == 0:
            print(f"  Step {step}, loss={loss.item():.4f}")

        # Confidence checkpoint
        if confidence_checkpoints and step in confidence_checkpoints and conf_dataloader is not None:
            print(f"  Measuring confidence at step {step}...")
            confs, indices = measure_confidence(model, conf_dataloader, mtype, device, tokenizer, label_token_ids)
            checkpoint_confidences[step] = (confs, indices)
            model.train()

    avg_loss = total_loss / num_steps if num_steps > 0 else 0
    return avg_loss, checkpoint_confidences


def raw_collate(batch):
    """Just pass through raw dicts."""
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--split", required=True, help="split1, split2, split3")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float32")
    parser.add_argument("--mode", choices=["lr_search", "confidence"], default="lr_search")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="/workspace/erase/outputs/plan0/data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Seed
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

    # Load split indices
    with open(os.path.join(args.data_dir, f"{args.split}_indices.json")) as f:
        split_indices = json.load(f)
    print(f"Split {args.split}: {len(split_indices)} examples")

    # Load model
    print(f"Loading model {args.model}...")
    model, tokenizer, mtype = load_model_and_tokenizer(args.model, device, use_bf16=args.bf16)

    # Label token IDs for T5/Pythia
    label_token_ids = None
    if mtype == "enc-dec":
        label_token_ids = get_t5_label_token_ids(tokenizer)
        print(f"T5 label tokens: {label_token_ids}")
    elif mtype == "decoder":
        label_token_ids = get_pythia_label_token_ids(tokenizer)
        print(f"Pythia label tokens: {label_token_ids}")

    # Datasets
    train_dataset = MNLIDataset(train_ds, split_indices)
    val_dataset = MNLIDataset(val_ds, list(range(len(val_ds))))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=raw_collate, num_workers=2)
    eval_bs = min(64, args.batch_size * 4)  # scale eval bs with train bs
    val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False,
                            collate_fn=raw_collate, num_workers=2)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps),
                                                num_training_steps=total_steps)

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
            model, train_loader, optimizer, scheduler, mtype, device, tokenizer,
            label_token_ids, confidence_checkpoints, conf_dataloader
        )
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Validation
        val_acc = evaluate_val(model, val_loader, mtype, device, tokenizer, label_token_ids)
        print(f"Epoch {epoch+1} val accuracy: {val_acc:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Save results
    results = {
        "model": args.model,
        "split": args.split,
        "lr": args.lr,
        "seed": args.seed,
        "mode": args.mode,
        "val_acc": val_acc,
        "avg_loss": avg_loss,
        "training_time_sec": elapsed,
    }

    if args.mode == "lr_search":
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    elif args.mode == "confidence" and ckpt_confs:
        # Save per-checkpoint confidence
        all_confs_list = []
        for step in sorted(ckpt_confs.keys()):
            confs, indices = ckpt_confs[step]
            ckpt_data = {str(idx): float(c) for idx, c in zip(indices, confs)}
            ckpt_path = os.path.join(args.output_dir, f"ckpt_step{step}_conf.json")
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_data, f)
            print(f"Saved checkpoint confidence: {ckpt_path}")
            all_confs_list.append(confs)

        # Average across checkpoints
        avg_conf = np.mean(all_confs_list, axis=0)
        _, indices = list(ckpt_confs.values())[0]
        avg_data = {str(idx): float(c) for idx, c in zip(indices, avg_conf)}
        avg_path = os.path.join(args.output_dir, f"avg_conf.json")
        with open(avg_path, "w") as f:
            json.dump(avg_data, f)
        print(f"Saved average confidence: {avg_path}")

        results["confidence_stats"] = {
            "mean": float(np.mean(avg_conf)),
            "std": float(np.std(avg_conf)),
            "gt_095_ratio": float(np.mean(avg_conf > 0.95)),
            "lt_04_ratio": float(np.mean(avg_conf < 0.4)),
        }
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    print("Done!")


if __name__ == "__main__":
    main()
