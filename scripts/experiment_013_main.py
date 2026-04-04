"""Plan 013: Subject-based easy set training experiment.
Train target models on easy sets defined by different subjects, compare OOD generalization.
"""
import argparse, json, os, time, gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

BASE = "/workspace/erase/outputs"
OUT_DIR = "/workspace/erase/outputs/plan13/main"
PLAN9_DIR = "/workspace/erase/outputs/plan9"

# ── Confidence paths ──
# Split-level: plan0/confidence/{model}_split{N}/avg_conf.json
# or plan5/confidence/{model}_split{N}/avg_conf.json (for some pythia/t5)
CONF_PATHS = {
    "bert-small":   "plan0/confidence/bert-small_split{}/avg_conf.json",
    "bert-base":    "plan0/confidence/bert-base_split{}/avg_conf.json",
    "bert-large":   "plan0/confidence/bert-large_split{}/avg_conf.json",
    "pythia-70m":   "plan0/confidence/pythia-70m_split{}/avg_conf.json",
    "pythia-160m":  "plan0/confidence/pythia-160m_split{}/avg_conf.json",
    "pythia-410m":  "plan0/confidence/pythia-410m_split{}/avg_conf.json",
    "pythia-1b":    "plan0/confidence/pythia-1b_split{}/avg_conf.json",
    "pythia-2.8b":  "plan0/confidence/pythia-2.8b_split{}/avg_conf.json",
    "pythia-6.9b":  "plan0/confidence/pythia-6.9b_split{}/avg_conf.json",
    "pythia-12b":   "plan5/confidence/pythia-12b_split{}/avg_conf.json",
}

# ── Target model configs ──
TARGETS = {
    "bert-small": {
        "hf_name": "google/bert_uncased_L-4_H-512_A-8",
        "type": "bert", "params": 29e6,
        "lr": {5: 2e-5, 10: 2e-5, 30: 3e-5},
    },
    "bert-base": {
        "hf_name": "bert-base-uncased",
        "type": "bert", "params": 110e6,
        "lr": {5: 3e-5, 10: 3e-5, 30: 3e-5},
    },
    "bert-large": {
        "hf_name": "bert-large-uncased",
        "type": "bert", "params": 335e6,
        "lr": {5: 3e-5, 10: 3e-5, 30: 3e-5},
    },
    "pythia-70m": {
        "hf_name": "EleutherAI/pythia-70m",
        "type": "pythia", "params": 70e6,
        "lr": {5: 1e-4, 10: 5e-5, 30: 1e-5},
    },
    "pythia-160m": {
        "hf_name": "EleutherAI/pythia-160m",
        "type": "pythia", "params": 160e6,
        "lr": {5: 2e-5, 10: 1e-5, 30: 1e-5},
    },
    "pythia-410m": {
        "hf_name": "EleutherAI/pythia-410m",
        "type": "pythia", "params": 410e6,
        "lr": {5: 1e-5, 10: 5e-6, 30: 5e-6},
    },
    "pythia-1b": {
        "hf_name": "EleutherAI/pythia-1b",
        "type": "pythia", "params": 1e9,
        "lr": {5: 1e-5, 10: 1e-5, 30: 5e-6},
    },
}

# ── Subject lists per target ──
# Subjects: Self + same-arch + cross-arch
SUBJECTS_FOR_TARGET = {
    "bert-small":  ["bert-small", "bert-base", "bert-large",
                    "pythia-70m", "pythia-410m", "pythia-2.8b", "pythia-6.9b", "pythia-12b"],
    "bert-base":   ["bert-base", "bert-large",
                    "pythia-70m", "pythia-410m", "pythia-2.8b", "pythia-6.9b", "pythia-12b"],
    "bert-large":  ["bert-large", "bert-base",
                    "pythia-70m", "pythia-410m", "pythia-2.8b", "pythia-6.9b", "pythia-12b"],
    "pythia-70m":  ["pythia-70m", "pythia-410m", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
                    "bert-base", "bert-large"],
    "pythia-160m": ["pythia-160m", "pythia-410m", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
                    "bert-base", "bert-large"],
    "pythia-410m": ["pythia-410m", "pythia-70m", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
                    "bert-base", "bert-large"],
    "pythia-1b":   ["pythia-1b", "pythia-70m", "pythia-410m", "pythia-2.8b", "pythia-6.9b", "pythia-12b",
                    "bert-base", "bert-large"],
}

K_EPOCHS = {5: 20, 10: 15, 30: 8}
K_SIZES = [5, 10, 30]


class NLIDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def load_conf(subject, split_num):
    path = os.path.join(BASE, CONF_PATHS[subject].format(split_num))
    with open(path) as f:
        return json.load(f)


def get_easy_ids(conf, all_ids, k_pct):
    """Get top k% easiest example IDs by confidence."""
    items = [(idx, conf.get(idx, 0)) for idx in all_ids]
    items.sort(key=lambda x: -x[1])
    n = int(len(items) * k_pct / 100)
    return [idx for idx, _ in items[:n]]


# ── BERT training ──

def bert_collate(batch, tokenizer, max_length=128):
    premises = [b["premise"] for b in batch]
    hypotheses = [b["hypothesis"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(premises, hypotheses, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def train_eval_bert(hf_name, train_examples, dev_examples, hans_examples,
                    anli_data, device, lr, epochs, batch_size=16):
    from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = BertForSequenceClassification.from_pretrained(hf_name, num_labels=3).to(device)

    loader = DataLoader(NLIDataset(train_examples), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: bert_collate(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_dev = 0
    best_hans = 0
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = n = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
            n += 1

        dev_acc = eval_bert(model, tokenizer, dev_examples, device)
        hans_acc = eval_bert(model, tokenizer, hans_examples, device)
        print(f"    Ep {epoch+1}: loss={total_loss/n:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_hans = hans_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate best model on ANLI
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    anli_results = {}
    for r_tag, r_examples in anli_data.items():
        anli_results[r_tag] = eval_bert(model, tokenizer, r_examples, device)
        print(f"    {r_tag}: {anli_results[r_tag]:.4f}")

    del model, optimizer, scheduler, best_state
    gc.collect()
    torch.cuda.empty_cache()

    return {"dev": best_dev, "hans": best_hans, "best_epoch": best_epoch, **anli_results}


def eval_bert(model, tokenizer, examples, device, batch_size=64):
    model.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: bert_collate(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            preds = out.logits.argmax(dim=-1)
            correct += (preds == batch["labels"].to(device)).sum().item()
            total += len(batch["labels"])
    return correct / total if total > 0 else 0


# ── Pythia training ──

def pythia_collate(batch, tokenizer, max_length=256):
    texts = [f"Premise: {b['premise']} Hypothesis: {b['hypothesis']} Label:" for b in batch]
    labels = torch.tensor([b["label"] for b in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {**enc, "labels": labels}


def train_eval_pythia(hf_name, train_examples, dev_examples, hans_examples,
                      anli_data, device, lr, epochs, batch_size=8, grad_accum=2):
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
    from torch import nn

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.float32).to(device)

    hidden_size = model.config.hidden_size
    classifier = nn.Linear(hidden_size, 3).to(device)

    loader = DataLoader(NLIDataset(train_examples), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: pythia_collate(b, tokenizer))
    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs // grad_accum
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    best_dev = 0
    best_hans = 0
    best_epoch = 0
    best_model_state = None
    best_cls_state = None
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        classifier.train()
        total_loss = n = 0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            seq_lens = attention_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(hidden.size(0)), seq_lens]
            logits = classifier(last_hidden)
            loss = loss_fn(logits, labels) / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum
            n += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        dev_acc = eval_pythia(model, classifier, tokenizer, dev_examples, device)
        hans_acc = eval_pythia(model, classifier, tokenizer, hans_examples, device)
        print(f"    Ep {epoch+1}: loss={total_loss/n:.4f} dev={dev_acc:.4f} hans={hans_acc:.4f}")

        if dev_acc > best_dev:
            best_dev = dev_acc
            best_hans = hans_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_cls_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    # Evaluate best model on ANLI
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        classifier.load_state_dict({k: v.to(device) for k, v in best_cls_state.items()})
    anli_results = {}
    for r_tag, r_examples in anli_data.items():
        anli_results[r_tag] = eval_pythia(model, classifier, tokenizer, r_examples, device)
        print(f"    {r_tag}: {anli_results[r_tag]:.4f}")

    del model, classifier, optimizer, scheduler, best_model_state, best_cls_state
    gc.collect()
    torch.cuda.empty_cache()

    return {"dev": best_dev, "hans": best_hans, "best_epoch": best_epoch, **anli_results}


def eval_pythia(model, classifier, tokenizer, examples, device, batch_size=32):
    model.eval()
    classifier.eval()
    loader = DataLoader(NLIDataset(examples), batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: pythia_collate(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            seq_lens = attention_mask.sum(dim=1) - 1
            last_hidden = hidden[torch.arange(hidden.size(0)), seq_lens]
            logits = classifier(last_hidden)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total if total > 0 else 0


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=K_SIZES)
    parser.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load eval data
    mnli_dev = load_dataset("glue", "mnli_matched", split="validation")
    dev_examples = [{"premise": it["premise"], "hypothesis": it["hypothesis"],
                     "label": it["label"]} for it in mnli_dev]

    with open(os.path.join(PLAN9_DIR, "hans_eval.json")) as f:
        hans_examples = json.load(f)

    anli_data = {}
    for r in ["r1", "r2", "r3"]:
        try:
            ds = load_dataset("anli", split=f"dev_{r}")
            anli_data[f"anli_{r}"] = [{"premise": it["premise"], "hypothesis": it["hypothesis"],
                                        "label": it["label"]} for it in ds]
        except Exception as e:
            print(f"  Warning: ANLI-{r} load failed: {e}")

    ds_full = load_dataset("glue", "mnli", split="train")
    all_examples = {str(i): {"premise": ds_full[i]["premise"], "hypothesis": ds_full[i]["hypothesis"],
                              "label": ds_full[i]["label"]} for i in range(len(ds_full))}

    for target_name in args.targets:
        cfg = TARGETS[target_name]
        subjects = SUBJECTS_FOR_TARGET[target_name]
        print(f"\n{'='*60}")
        print(f"  Target: {target_name} ({cfg['params']/1e6:.0f}M)")
        print(f"  Subjects: {subjects}")
        print(f"{'='*60}")

        for split_num in args.splits:
            with open(f"{BASE}/plan0/data/split{split_num}_indices.json") as f:
                split_indices = json.load(f)
            all_ids = [str(i) for i in split_indices]

            for k in args.ks:
                n_easy = int(len(all_ids) * k / 100)
                epochs = K_EPOCHS[k]
                lr = cfg["lr"][k]

                # ── Subject easy conditions ──
                for subject in subjects:
                    cond_name = f"{target_name}_sub-{subject}_k{k}_split{split_num}"
                    outfile = os.path.join(OUT_DIR, f"{cond_name}.json")
                    if os.path.exists(outfile):
                        print(f"\n  SKIP: {cond_name}")
                        continue

                    print(f"\n  --- {cond_name} (n={n_easy}, {epochs}ep, lr={lr}) ---")

                    conf = load_conf(subject, split_num)
                    easy_ids = get_easy_ids(conf, all_ids, k)
                    train_examples = [all_examples[idx] for idx in easy_ids]

                    start = time.time()
                    if cfg["type"] == "bert":
                        result = train_eval_bert(cfg["hf_name"], train_examples, dev_examples,
                                                 hans_examples, anli_data, device, lr, epochs)
                    else:
                        result = train_eval_pythia(cfg["hf_name"], train_examples, dev_examples,
                                                   hans_examples, anli_data, device, lr, epochs)

                    elapsed = time.time() - start
                    result.update({"target": target_name, "subject": subject, "k": k,
                                   "split": split_num, "n_train": len(train_examples),
                                   "epochs": epochs, "lr": lr, "time_sec": elapsed,
                                   "condition_type": "subject_easy"})

                    with open(outfile, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"    → Saved ({elapsed:.0f}s)")

                # ── Random baseline ──
                cond_name = f"{target_name}_random_k{k}_split{split_num}"
                outfile = os.path.join(OUT_DIR, f"{cond_name}.json")
                if not os.path.exists(outfile):
                    print(f"\n  --- {cond_name} (n={n_easy}, {epochs}ep, lr={lr}) ---")
                    rng = np.random.RandomState(42 + split_num + k)
                    random_ids = rng.choice(all_ids, size=n_easy, replace=False)
                    train_examples = [all_examples[idx] for idx in random_ids]

                    start = time.time()
                    if cfg["type"] == "bert":
                        result = train_eval_bert(cfg["hf_name"], train_examples, dev_examples,
                                                 hans_examples, anli_data, device, lr, epochs)
                    else:
                        result = train_eval_pythia(cfg["hf_name"], train_examples, dev_examples,
                                                   hans_examples, anli_data, device, lr, epochs)

                    elapsed = time.time() - start
                    result.update({"target": target_name, "subject": "random", "k": k,
                                   "split": split_num, "n_train": n_easy,
                                   "epochs": epochs, "lr": lr, "time_sec": elapsed,
                                   "condition_type": "random"})

                    with open(outfile, "w") as f:
                        json.dump(result, f, indent=2)
                    print(f"    → Saved ({elapsed:.0f}s)")

            # ── Full baseline (once per split) ──
            cond_name = f"{target_name}_full_split{split_num}"
            outfile = os.path.join(OUT_DIR, f"{cond_name}.json")
            if not os.path.exists(outfile):
                print(f"\n  --- {cond_name} (n={len(all_ids)}, 5ep, lr={cfg['lr'][30]}) ---")
                train_examples = [all_examples[idx] for idx in all_ids]

                start = time.time()
                if cfg["type"] == "bert":
                    result = train_eval_bert(cfg["hf_name"], train_examples, dev_examples,
                                             hans_examples, anli_data, device, cfg["lr"][30], 5)
                else:
                    result = train_eval_pythia(cfg["hf_name"], train_examples, dev_examples,
                                               hans_examples, anli_data, device, cfg["lr"][30], 5)

                elapsed = time.time() - start
                result.update({"target": target_name, "subject": "full", "k": 100,
                               "split": split_num, "n_train": len(all_ids),
                               "epochs": 5, "lr": cfg["lr"][30], "time_sec": elapsed,
                               "condition_type": "full"})

                with open(outfile, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"    → Saved ({elapsed:.0f}s)")

    print("\n=== All runs complete ===")


if __name__ == "__main__":
    main()
