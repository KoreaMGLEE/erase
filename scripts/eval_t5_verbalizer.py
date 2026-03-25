"""Test different verbalizers for T5 v1.1 base."""
import argparse
import torch
import json
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup
from datasets import load_dataset
from collections import Counter

VERBALIZERS = {
    "original": ["entailment", "neutral", "contradiction"],
    "yes_maybe_no": ["yes", "maybe", "no"],
    "true_unknown_false": ["true", "unknown", "false"],
    "ABC": ["A", "B", "C"],
}


def train_and_eval(split_name, indices, train_ds, val_ds, labels, tokenizer, device, verb_name):
    first_token_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in labels}

    model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base', torch_dtype=torch.float32).to(device)
    optimizer = Adafactor(model.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(indices) // 16
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    model.train()
    perm = np.random.RandomState(1).permutation(len(indices))
    for step in range(total_steps):
        batch_idx = [indices[perm[i]] for i in range(step * 16, min((step + 1) * 16, len(indices)))]
        inputs = [f'mnli hypothesis: {train_ds[i]["hypothesis"]} premise: {train_ds[i]["premise"]}' for i in batch_idx]
        targets = [labels[train_ds[i]['label']] for i in batch_idx]

        enc = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors='pt').input_ids.to(device)
        tgt[tgt == tokenizer.pad_token_id] = -100

        outputs = model(**enc, labels=tgt)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Eval: first token + generate + confusion matrix
    model.eval()
    correct_first = correct_gen = total = 0
    pred_counts = Counter()

    with torch.no_grad():
        for i in range(0, min(500, len(val_ds)), 16):
            batch = val_ds[i:i + 16]
            inps = [f'mnli hypothesis: {h} premise: {p}' for h, p in zip(batch['hypothesis'], batch['premise'])]
            labs = batch['label']
            enc = tokenizer(inps, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            bs = enc['input_ids'].size(0)

            # First token
            dec = torch.full((bs, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            out = model(**enc, decoder_input_ids=dec)
            logits = out.logits[:, 0, :]
            lid = [first_token_ids[l] for l in labels]
            preds1 = logits[:, lid].argmax(dim=-1)

            # Generate
            gen_ids = model.generate(**enc, max_new_tokens=8)
            for j in range(bs):
                gen_text = tokenizer.decode(gen_ids[j], skip_special_tokens=True).strip().lower()
                pred_label = -1
                for k, l in enumerate(labels):
                    if gen_text.startswith(l[:3].lower()):
                        pred_label = k
                        break
                if pred_label == labs[j]:
                    correct_gen += 1
                if preds1[j].item() == labs[j]:
                    correct_first += 1
                pred_counts[labels[preds1[j].item()]] += 1
                total += 1

    print(f'  [{verb_name}] {split_name}: first_tok={correct_first / total:.4f}, gen={correct_gen / total:.4f}, pred_dist={dict(pred_counts)}')
    del model, optimizer
    torch.cuda.empty_cache()
    return correct_first / total, correct_gen / total, dict(pred_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base', legacy=True)
    ds = load_dataset('glue', 'mnli')
    train_ds = ds['train']
    val_ds = ds['validation_matched']

    with open(f'/workspace/erase/outputs/plan0/data/{args.split}_indices.json') as f:
        indices = json.load(f)

    print(f'=== {args.split} Verbalizer Comparison (GPU {args.gpu}) ===')
    print(f'  [original] SKIP — already known: class collapse to contradiction')
    for verb_name, labels in VERBALIZERS.items():
        if verb_name == "original":
            continue
        train_and_eval(args.split, indices, train_ds, val_ds, labels, tokenizer, device, verb_name)


if __name__ == "__main__":
    main()
