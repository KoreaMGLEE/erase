"""T5 v1.1 MNLI training — FINAL version.
Sentinel prompt + Adafactor + constant LR with warmup + verbalizer (yes/maybe/no).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

MODEL_REGISTRY = {
    "t5-v1_1-small": {"hf_id": "google/t5-v1_1-small", "params": "77M"},
    "t5-v1_1-base": {"hf_id": "google/t5-v1_1-base", "params": "248M"},
    "t5-v1_1-large": {"hf_id": "google/t5-v1_1-large", "params": "783M"},
    "t5-v1_1-xl": {"hf_id": "google/t5-v1_1-xl", "params": "3B", "lora": True},
    "t5-v1_1-xxl": {"hf_id": "google/t5-v1_1-xxl", "params": "11B", "lora": True},
}

LABELS = ["yes", "maybe", "no"]  # verbalizer: entailment=yes, neutral=maybe, contradiction=no


class MNLIDataset(Dataset):
    def __init__(self, ds, indices):
        self.ds = ds
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        item = self.ds[self.indices[idx]]
        return {"premise": item["premise"], "hypothesis": item["hypothesis"],
                "label": item["label"], "original_index": self.indices[idx]}


def collate_fn(batch, tokenizer, max_length=256):
    # Sentinel prompt
    inputs = [f'mnli hypothesis: {b["hypothesis"]} premise: {b["premise"]} answer: <extra_id_0>' for b in batch]
    targets = [f'<extra_id_0> {LABELS[b["label"]]}' for b in batch]
    labels_int = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors="pt").input_ids
    tgt[tgt == tokenizer.pad_token_id] = -100
    return {**enc, "decoder_labels": tgt, "labels_int": labels_int}


def raw_collate(batch):
    return batch


def load_model(model_name, device, use_bf16=False):
    info = MODEL_REGISTRY[model_name]
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tokenizer = T5Tokenizer.from_pretrained(info["hf_id"], legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(info["hf_id"], torch_dtype=dtype).to(device)

    if info.get("lora"):
        lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=32, lora_alpha=64,
                                  lora_dropout=0.05, target_modules=["q", "v"])
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")

    return model, tokenizer


@torch.no_grad()
def measure_confidence(model, dataloader, device, tokenizer):
    model.eval()
    label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}
    sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    all_confs, all_indices = [], []

    for batch_raw in dataloader:
        collated = collate_fn(batch_raw, tokenizer)
        input_ids = collated["input_ids"].to(device)
        attention_mask = collated["attention_mask"].to(device)
        labels = collated["labels_int"].to(device)
        bs = input_ids.size(0)

        dec = torch.full((bs, 2), tokenizer.pad_token_id, dtype=torch.long, device=device)
        dec[:, 1] = sentinel_id
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec)
        logits = outputs.logits[:, 1, :]
        lid = [label_ids[l] for l in LABELS]
        label_logits = logits[:, lid]
        probs = F.softmax(label_logits.float(), dim=-1)
        conf = probs[range(len(labels)), labels]
        all_confs.append(conf.float().cpu().numpy())
        all_indices.extend([b["original_index"] for b in batch_raw])

    model.train()
    return np.concatenate(all_confs), all_indices


def evaluate(model, val_loader, device, tokenizer):
    model.eval()
    label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}
    sentinel_id = tokenizer.encode("<extra_id_0>", add_special_tokens=False)[0]
    correct = total = 0

    with torch.no_grad():
        for batch_raw in val_loader:
            collated = collate_fn(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            labels = collated["labels_int"].to(device)
            bs = input_ids.size(0)

            dec = torch.full((bs, 2), tokenizer.pad_token_id, dtype=torch.long, device=device)
            dec[:, 1] = sentinel_id
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec)
            logits = outputs.logits[:, 1, :]
            lid = [label_ids[l] for l in LABELS]
            preds = logits[:, lid].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    model.train()
    return correct / total if total > 0 else 0.0


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
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="/workspace/erase/outputs/plan0/data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Model: {args.model}, Split: {args.split}, LR: {args.lr}, Mode: {args.mode}")

    ds = load_dataset("glue", "mnli")
    train_ds, val_ds = ds["train"], ds["validation_matched"]

    with open(os.path.join(args.data_dir, f"{args.split}_indices.json")) as f:
        split_indices = json.load(f)
    print(f"Split {args.split}: {len(split_indices)} examples")

    model, tokenizer = load_model(args.model, device, use_bf16=args.bf16)

    train_dataset = MNLIDataset(train_ds, split_indices)
    val_dataset = MNLIDataset(val_ds, list(range(len(val_ds))))
    eval_bs = min(64, args.batch_size * 4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=raw_collate)
    val_loader = DataLoader(val_dataset, batch_size=eval_bs, shuffle=False, collate_fn=raw_collate)

    optimizer = Adafactor([p for p in model.parameters() if p.requires_grad],
                          lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps))

    # Confidence checkpoints
    confidence_checkpoints = None
    conf_dataloader = None
    if args.mode == "confidence":
        steps_per_epoch = len(train_loader)
        confidence_checkpoints = {steps_per_epoch // 3, 2 * steps_per_epoch // 3, steps_per_epoch}
        print(f"Confidence checkpoints: {sorted(confidence_checkpoints)}")
        conf_dataloader = DataLoader(train_dataset, batch_size=eval_bs, shuffle=False, collate_fn=raw_collate)

    start = time.time()
    checkpoint_confidences = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = n_steps = 0
        for step, batch_raw in enumerate(train_loader, 1):
            optimizer.zero_grad()
            collated = collate_fn(batch_raw, tokenizer)
            input_ids = collated["input_ids"].to(device)
            attention_mask = collated["attention_mask"].to(device)
            decoder_labels = collated["decoder_labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=decoder_labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()
            n_steps += 1
            if step % 200 == 0:
                print(f"  Step {step}, loss={outputs.loss.item():.4f}")

            if confidence_checkpoints and step in confidence_checkpoints and conf_dataloader:
                print(f"  Measuring confidence at step {step}...")
                confs, indices = measure_confidence(model, conf_dataloader, device, tokenizer)
                checkpoint_confidences[step] = (confs, indices)
                model.train()

        avg_loss = total_loss / n_steps
        val_acc = evaluate(model, val_loader, device, tokenizer)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    results = {"model": args.model, "split": args.split, "lr": args.lr,
               "seed": args.seed, "mode": args.mode, "val_acc": val_acc,
               "avg_loss": avg_loss, "training_time_sec": elapsed}

    if args.mode == "lr_search":
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "confidence" and checkpoint_confidences:
        all_confs_list = []
        for step in sorted(checkpoint_confidences.keys()):
            confs, indices = checkpoint_confidences[step]
            ckpt_data = {str(idx): float(c) for idx, c in zip(indices, confs)}
            with open(os.path.join(args.output_dir, f"ckpt_step{step}_conf.json"), "w") as f:
                json.dump(ckpt_data, f)
            all_confs_list.append(confs)
        avg_conf = np.mean(all_confs_list, axis=0)
        _, indices = list(checkpoint_confidences.values())[0]
        avg_data = {str(idx): float(c) for idx, c in zip(indices, avg_conf)}
        with open(os.path.join(args.output_dir, "avg_conf.json"), "w") as f:
            json.dump(avg_data, f)
        results["confidence_stats"] = {
            "mean": float(np.mean(avg_conf)), "std": float(np.std(avg_conf)),
            "gt_095_ratio": float(np.mean(avg_conf > 0.95)),
            "lt_04_ratio": float(np.mean(avg_conf < 0.4)),
        }
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
