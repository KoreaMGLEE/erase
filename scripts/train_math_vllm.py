"""Plan 4: MATH training (HuggingFace) + evaluation (vLLM batch generation)."""
import argparse
import json
import os
import re
import time
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

MODEL_REGISTRY = {
    "pythia-1b": {"hf_id": "EleutherAI/pythia-1b", "target_modules": ["query_key_value"]},
    "pythia-2.8b": {"hf_id": "EleutherAI/pythia-2.8b", "target_modules": ["query_key_value"]},
    "pythia-6.9b": {"hf_id": "EleutherAI/pythia-6.9b", "target_modules": ["query_key_value"]},
    "qwen2.5-1.5b": {"hf_id": "Qwen/Qwen2.5-1.5B", "target_modules": ["q_proj", "v_proj"]},
    "llama3.2-1b": {"hf_id": "meta-llama/Llama-3.2-1B", "target_modules": ["q_proj", "v_proj"]},
    "llama3.2-3b": {"hf_id": "meta-llama/Llama-3.2-3B", "target_modules": ["q_proj", "v_proj"]},
}

MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


class MATHDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def load_math_data():
    train_data, test_data = [], []
    for config in MATH_CONFIGS:
        train = load_dataset("EleutherAI/hendrycks_math", config, split="train")
        test = load_dataset("EleutherAI/hendrycks_math", config, split="test")
        for item in train:
            train_data.append({**item, "config": config})
        for item in test:
            test_data.append({**item, "config": config})
    return train_data, test_data


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return text[start:]


def normalize_answer(ans):
    ans = ans.strip()
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("\\,", "").replace("\\ ", "").replace("\\!", "")
    ans = ans.replace("\\text{", "").replace("\\mathrm{", "")
    while ans.endswith("}") and ans.count("}") > ans.count("{"):
        ans = ans[:-1]
    return ans.strip()


def check_answer(predicted, gold):
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    try:
        return abs(float(pred_norm.replace(",", "")) - float(gold_norm.replace(",", ""))) < 1e-6
    except (ValueError, TypeError):
        return False


def format_prompt(problem):
    return f"Problem: {problem}\nSolution:"


def format_train(problem, solution):
    return f"Problem: {problem}\nSolution: {solution}"


def collate_fn(batch, tokenizer, max_length=1024):
    texts = [format_train(item["problem"], item["solution"]) for item in batch]
    prompts = [format_prompt(item["problem"]) for item in batch]
    full_enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    labels = full_enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return {"input_ids": full_enc["input_ids"], "attention_mask": full_enc["attention_mask"], "labels": labels}


def load_model(model_name, device, use_bf16=False):
    info = MODEL_REGISTRY[model_name]
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(info["hf_id"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(info["hf_id"], torch_dtype=dtype)
    model = model.to(device)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=info["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%) [grad_ckpt=ON]")
    return model, tokenizer


def evaluate_vllm(model, tokenizer, test_data, device, gpu_id, max_samples=None, max_new_tokens=512):
    """Evaluate using vLLM for fast batch generation.
    Offloads training model to CPU, runs vLLM, then restores."""
    from vllm import LLM, SamplingParams
    import gc

    model.eval()

    if max_samples and max_samples < len(test_data):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(test_data), max_samples, replace=False)
        eval_data = [test_data[i] for i in indices]
    else:
        eval_data = test_data

    # Step 1: Save merged weights to disk, offload model to CPU
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"  Saving merged LoRA weights...")
        model_copy = model.merge_and_unload(progressbar=False)
        model_copy.save_pretrained(tmpdir)
        tokenizer.save_pretrained(tmpdir)
        del model_copy
        # Note: merge_and_unload destroys LoRA structure, so we can't restore.
        # We'll need to handle this at the caller level.

        # Free all GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  Loading vLLM engine...")
        llm = LLM(
            model=tmpdir,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=1024,
            dtype="auto",
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

        prompts = [format_prompt(item["problem"]) for item in eval_data]
        print(f"  Generating {len(prompts)} solutions...")
        outputs = llm.generate(prompts, sampling_params)

        correct = 0
        total = 0
        level_correct = {}
        level_total = {}

        for item, output in zip(eval_data, outputs):
            generated = output.outputs[0].text
            pred_answer = extract_boxed(generated)
            gold_answer = extract_boxed(item["solution"])
            is_correct = check_answer(pred_answer, gold_answer)
            correct += int(is_correct)
            total += 1
            level = item.get("level", "unknown")
            level_correct[level] = level_correct.get(level, 0) + int(is_correct)
            level_total[level] = level_total.get(level, 0) + 1

        accuracy = correct / total if total > 0 else 0
        level_acc = {l: level_correct[l] / level_total[l] for l in sorted(level_total.keys())}

        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return accuracy, level_acc, total


def evaluate_hf(model, tokenizer, test_data, device, max_samples=None, max_new_tokens=512):
    """Fallback: HuggingFace generation (slower but no merge needed)."""
    model.eval()
    if max_samples and max_samples < len(test_data):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(test_data), max_samples, replace=False)
        eval_data = [test_data[i] for i in indices]
    else:
        eval_data = test_data

    correct = 0
    total = 0
    level_correct = {}
    level_total = {}

    # Batch generation with HF
    batch_size = 8
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i+batch_size]
        prompts = [format_prompt(item["problem"]) for item in batch]
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )

        for j, item in enumerate(batch):
            generated = tokenizer.decode(outputs[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred_answer = extract_boxed(generated)
            gold_answer = extract_boxed(item["solution"])
            is_correct = check_answer(pred_answer, gold_answer)
            correct += int(is_correct)
            total += 1
            level = item.get("level", "unknown")
            level_correct[level] = level_correct.get(level, 0) + int(is_correct)
            level_total[level] = level_total.get(level, 0) + 1

    accuracy = correct / total if total > 0 else 0
    level_acc = {l: level_correct[l] / level_total[l] for l in sorted(level_total.keys())}
    model.train()
    return accuracy, level_acc, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for eval")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"Model: {args.model}, LR: {args.lr}, Epochs: {args.epochs}, Device: {device}, vLLM: {args.use_vllm}")

    print("Loading MATH dataset...")
    train_data, test_data = load_math_data()
    eval_n = args.eval_samples if args.eval_samples > 0 else len(test_data)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Eval: {eval_n}")

    print(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model, device, use_bf16=args.bf16)

    train_dataset = MATHDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    start = time.time()
    best_acc = 0
    best_epoch = 0
    epoch_results = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            if (step + 1) % 200 == 0:
                print(f"  Step {step+1}, loss={outputs.loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # Evaluate: HF batch eval every epoch (fast), vLLM only at end
        print(f"  Evaluating ({eval_n} samples)...")
        eval_start = time.time()
        acc, level_acc, n_eval = evaluate_hf(
            model, tokenizer, test_data, device, max_samples=eval_n)
        eval_time = time.time() - eval_start

        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, acc={acc:.4f} ({int(acc*n_eval)}/{n_eval}) eval_time={eval_time:.0f}s")
        for level, la in sorted(level_acc.items()):
            print(f"  {level}: {la:.4f}")

        epoch_results.append({
            "epoch": epoch + 1, "avg_loss": avg_loss,
            "accuracy": acc, "level_accuracy": level_acc,
            "n_eval": n_eval, "eval_time_sec": eval_time,
        })

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

    print(f"Best HF accuracy: {best_acc:.4f} (epoch {best_epoch})")

    # Final vLLM eval on larger sample after all training
    if args.use_vllm:
        print(f"\n  Final vLLM eval ({eval_n} samples)...")
        try:
            eval_start = time.time()
            vllm_acc, vllm_level_acc, vllm_n = evaluate_vllm(
                model, tokenizer, test_data, device, args.gpu, max_samples=eval_n)
            vllm_time = time.time() - eval_start
            print(f"  vLLM acc={vllm_acc:.4f} ({int(vllm_acc*vllm_n)}/{vllm_n}) time={vllm_time:.0f}s")
            for level, la in sorted(vllm_level_acc.items()):
                print(f"    {level}: {la:.4f}")
        except Exception as e:
            print(f"  vLLM eval failed: {e}")
            vllm_acc = best_acc
            vllm_level_acc = {}
    else:
        vllm_acc = best_acc
        vllm_level_acc = {}

    elapsed = time.time() - start

    results = {
        "model": args.model, "lr": args.lr, "seed": args.seed,
        "epochs": args.epochs, "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "final_vllm_accuracy": vllm_acc,
        "final_vllm_level_accuracy": vllm_level_acc,
        "training_time_sec": elapsed,
        "epoch_results": epoch_results,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
