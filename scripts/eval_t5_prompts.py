"""T5 v1.1 base: Prompt wording ablation — task conditioning test."""
import argparse
import torch
import json
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup
from datasets import load_dataset
from collections import Counter

LABELS = ["entailment", "neutral", "contradiction"]

PROMPTS = {
    "original": "mnli hypothesis: {h} premise: {p}",
    "nli": "nli premise: {p} hypothesis: {h}",
    "relation": "premise: {p} hypothesis: {h} relation:",
    "question": "Does the premise entail the hypothesis? premise: {p} hypothesis: {h} answer:",
    "classify": "classify: premise: {p} hypothesis: {h}",
}


def format_input(prompt_template, premise, hypothesis):
    return prompt_template.format(p=premise, h=hypothesis)


def train_and_eval(indices, train_ds, val_ds, tokenizer, device, prompt_name, prompt_template):
    first_token_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS}

    model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base', torch_dtype=torch.float32).to(device)
    optimizer = Adafactor(model.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    total_steps = len(indices) // 16
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    model.train()
    perm = np.random.RandomState(1).permutation(len(indices))
    for step in range(total_steps):
        batch_idx = [indices[perm[(step * 16 + k) % len(indices)]] for k in range(16)]
        inputs = [format_input(prompt_template, train_ds[i]["premise"], train_ds[i]["hypothesis"]) for i in batch_idx]
        targets = [LABELS[train_ds[i]['label']] for i in batch_idx]

        enc = tokenizer(inputs, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
        tgt = tokenizer(targets, padding=True, truncation=True, max_length=16, return_tensors='pt').input_ids.to(device)
        tgt[tgt == tokenizer.pad_token_id] = -100

        outputs = model(**enc, labels=tgt)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Eval
    model.eval()
    correct = total = 0
    pred_counts = Counter()

    with torch.no_grad():
        for i in range(0, min(500, len(val_ds)), 16):
            batch = val_ds[i:i + 16]
            inps = [format_input(prompt_template, p, h) for p, h in zip(batch['premise'], batch['hypothesis'])]
            labs = batch['label']
            enc = tokenizer(inps, padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            bs = enc['input_ids'].size(0)

            dec = torch.full((bs, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            out = model(**enc, decoder_input_ids=dec)
            logits = out.logits[:, 0, :]
            lid = [first_token_ids[l] for l in LABELS]
            preds = logits[:, lid].argmax(dim=-1)

            for j in range(bs):
                if preds[j].item() == labs[j]:
                    correct += 1
                pred_counts[LABELS[preds[j].item()]] += 1
                total += 1

    acc = correct / total
    print(f'  [{prompt_name}] acc={acc:.4f}, pred_dist={dict(pred_counts)}')
    del model, optimizer
    torch.cuda.empty_cache()


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

    print(f'=== {args.split} Prompt Wording Ablation (GPU {args.gpu}) ===')
    for prompt_name, prompt_template in PROMPTS.items():
        print(f'\n--- Prompt: {prompt_name} ---')
        print(f'    Format: {prompt_template}')
        train_and_eval(indices, ds['train'], ds['validation_matched'],
                       tokenizer, device, prompt_name, prompt_template)


if __name__ == "__main__":
    main()
