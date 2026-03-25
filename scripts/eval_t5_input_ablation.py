"""T5 v1.1 base: Input ablation — normal vs shuffled vs dummy.
Also checks cross-attention patterns."""
import argparse
import torch
import json
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup
from datasets import load_dataset
from collections import Counter
import random


def make_inputs(train_ds, batch_idx, mode="normal"):
    """Generate inputs based on ablation mode."""
    inputs = []
    for i in batch_idx:
        premise = train_ds[i]["premise"]
        hypothesis = train_ds[i]["hypothesis"]
        if mode == "normal":
            inputs.append(f'mnli hypothesis: {hypothesis} premise: {premise}')
        elif mode == "shuffled":
            # Shuffle words in both premise and hypothesis
            p_words = premise.split()
            h_words = hypothesis.split()
            random.shuffle(p_words)
            random.shuffle(h_words)
            inputs.append(f'mnli hypothesis: {" ".join(h_words)} premise: {" ".join(p_words)}')
        elif mode == "dummy":
            inputs.append(f'mnli hypothesis: hello premise: world')
    return inputs


def train_and_eval(split_name, indices, train_ds, val_ds, tokenizer, device, mode, labels):
    first_token_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in labels}

    model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base', torch_dtype=torch.float32).to(device)
    optimizer = Adafactor(model.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(indices) // 16
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    random.seed(1)
    model.train()
    perm = np.random.RandomState(1).permutation(len(indices))
    for step in range(total_steps):
        batch_idx = [indices[perm[(step * 16 + k) % len(indices)]] for k in range(16)]
        inputs = make_inputs(train_ds, batch_idx, mode=mode)
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

        if (step + 1) % 500 == 0:
            print(f'    Step {step+1}, loss={outputs.loss.item():.4f}')

    # Eval with NORMAL inputs (regardless of training mode)
    model.eval()
    correct = total = 0
    pred_counts = Counter()

    with torch.no_grad():
        for i in range(0, min(500, len(val_ds)), 16):
            batch = val_ds[i:i + 16]
            # Always eval with normal inputs
            inps = [f'mnli hypothesis: {h} premise: {p}' for h, p in zip(batch['hypothesis'], batch['premise'])]
            labs = batch['label']
            enc = tokenizer(inps, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            bs = enc['input_ids'].size(0)

            dec = torch.full((bs, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            out = model(**enc, decoder_input_ids=dec)
            logits = out.logits[:, 0, :]
            lid = [first_token_ids[l] for l in labels]
            preds = logits[:, lid].argmax(dim=-1)

            for j in range(bs):
                if preds[j].item() == labs[j]:
                    correct += 1
                pred_counts[labels[preds[j].item()]] += 1
                total += 1

    acc = correct / total
    print(f'  [{mode}] {split_name}: acc={acc:.4f}, pred_dist={dict(pred_counts)}')

    # Cross-attention analysis (on a small batch)
    with torch.no_grad():
        batch = val_ds[0:8]
        inps = [f'mnli hypothesis: {h} premise: {p}' for h, p in zip(batch['hypothesis'], batch['premise'])]
        enc = tokenizer(inps, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        bs = enc['input_ids'].size(0)
        dec = torch.full((bs, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)

        out = model(**enc, decoder_input_ids=dec, output_attentions=True)

        # Decoder self-attention: out.decoder_attentions
        # Cross-attention: out.cross_attentions
        if out.cross_attentions:
            for layer_idx in [0, len(out.cross_attentions) // 2, -1]:
                ca = out.cross_attentions[layer_idx]  # (batch, heads, dec_len, enc_len)
                entropy = -(ca * (ca + 1e-10).log()).sum(dim=-1).mean()
                max_attn = ca.max(dim=-1).values.mean()
                print(f'    Cross-attn layer {layer_idx}: entropy={entropy:.4f}, max_attn={max_attn:.4f}')

    del model, optimizer
    torch.cuda.empty_cache()
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base', legacy=True)
    ds = load_dataset('glue', 'mnli')

    with open(f'/workspace/erase/outputs/plan0/data/{args.split}_indices.json') as f:
        indices = json.load(f)

    labels = ["entailment", "neutral", "contradiction"]

    print(f'=== {args.split} Input Ablation (GPU {args.gpu}) ===')
    for mode in ["normal", "shuffled", "dummy"]:
        print(f'\n--- Mode: {mode} ---')
        train_and_eval(args.split, indices, ds['train'], ds['validation_matched'],
                       tokenizer, device, mode, labels)


if __name__ == "__main__":
    main()
