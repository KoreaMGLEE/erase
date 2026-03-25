"""T5 v1.1 base: 2 epoch comparison — original vs yes/maybe/no verbalizer."""
import argparse
import torch
import json
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup
from datasets import load_dataset
from collections import Counter


def train_and_eval(split_name, indices, train_ds, val_ds, labels, tokenizer, device, verb_name, epochs=2):
    first_token_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0] for l in labels}

    model = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-base', torch_dtype=torch.float32).to(device)
    optimizer = Adafactor(model.parameters(), lr=3e-4, scale_parameter=False, relative_step=False, warmup_init=False)
    steps_per_epoch = len(indices) // 16
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    perm = np.random.RandomState(1).permutation(len(indices))

    for epoch in range(epochs):
        model.train()
        for step in range(steps_per_epoch):
            idx = step % len(perm)
            batch_idx = [indices[perm[(step * 16 + k) % len(indices)]] for k in range(16)]
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

            if (step + 1) % 500 == 0:
                print(f'    Epoch {epoch+1} Step {step+1}, loss={outputs.loss.item():.4f}')

        # Eval after each epoch
        model.eval()
        correct_first = total = 0
        pred_counts = Counter()

        with torch.no_grad():
            for i in range(0, min(500, len(val_ds)), 16):
                batch = val_ds[i:i + 16]
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
                        correct_first += 1
                    pred_counts[labels[preds[j].item()]] += 1
                    total += 1

        acc = correct_first / total
        print(f'  [{verb_name}] {split_name} epoch {epoch+1}: acc={acc:.4f}, pred_dist={dict(pred_counts)}')

    del model, optimizer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--verbalizer", choices=["original", "yes_maybe_no"], required=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-base', legacy=True)
    ds = load_dataset('glue', 'mnli')

    with open(f'/workspace/erase/outputs/plan0/data/{args.split}_indices.json') as f:
        indices = json.load(f)

    if args.verbalizer == "original":
        labels = ["entailment", "neutral", "contradiction"]
    else:
        labels = ["yes", "maybe", "no"]

    print(f'=== {args.split} | {args.verbalizer} | 2 epochs (GPU {args.gpu}) ===')
    train_and_eval(args.split, indices, ds['train'], ds['validation_matched'],
                   labels, tokenizer, device, args.verbalizer, epochs=2)


if __name__ == "__main__":
    main()
