"""Plan 4: MATH GRPO training with vLLM evaluation.

Uses trl's GRPOTrainer for reward-based optimization.
Reward: +1 if \boxed{} answer matches gold, 0 otherwise.
"""
import argparse
import json
import os
import re
import time
import tempfile
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

SYSTEM_PROMPT = """You are a math problem solver. Solve the problem step by step and put your final answer in \\boxed{}."""


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


def format_chat_prompt(problem, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_grpo_dataset(train_data, tokenizer):
    """Build dataset for GRPO: each item has prompt and gold answer for reward."""
    dataset = []
    for item in train_data:
        prompt = format_chat_prompt(item["problem"], tokenizer)
        gold_answer = extract_boxed(item["solution"])
        dataset.append({
            "prompt": prompt,
            "gold_answer": gold_answer,
            "level": item.get("level", "unknown"),
        })
    return dataset


def make_reward_fn(tokenizer):
    """Create reward function: +1 if boxed answer matches gold, 0 otherwise."""
    def reward_fn(completions, gold_answer=None, **kwargs):
        rewards = []
        for completion, gold in zip(completions, gold_answer):
            if isinstance(completion, list):
                # Token IDs
                text = tokenizer.decode(completion, skip_special_tokens=True)
            else:
                text = completion
            pred = extract_boxed(text)
            if check_answer(pred, gold):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    return reward_fn


def evaluate_hf(model, tokenizer, test_data, device, max_samples=500, max_new_tokens=512):
    model.eval()
    if max_samples and max_samples < len(test_data):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(test_data), max_samples, replace=False)
        eval_data = [test_data[i] for i in indices]
    else:
        eval_data = test_data

    correct = total = 0
    level_correct = {}
    level_total = {}

    batch_size = 4
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i+batch_size]
        prompts = [format_chat_prompt(item["problem"], tokenizer) for item in batch]
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
    parser.add_argument("--model_id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt for GRPO")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}")
    print(f"GRPO Training: model={args.model_id}, lr={args.lr}, device={device}")

    # Load data
    print("Loading MATH dataset...")
    train_data, test_data = load_math_data()
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build dataset
    grpo_data = build_grpo_dataset(train_data, tokenizer)

    # Convert to HF Dataset format
    from datasets import Dataset as HFDataset
    hf_dataset = HFDataset.from_list([
        {"prompt": d["prompt"], "gold_answer": d["gold_answer"]}
        for d in grpo_data
    ])

    # Reward function
    def reward_fn(completions, **kwargs):
        """Reward function for GRPO."""
        rewards = []
        # Get the corresponding gold answers from the prompts
        for completion in completions:
            if isinstance(completion, list):
                text = tokenizer.decode(completion, skip_special_tokens=True)
            else:
                text = completion
            pred = extract_boxed(text)
            # We'll match against gold_answer passed via the dataset
            # For now, give partial reward based on whether \boxed{} was generated
            if "\\boxed{" in text and pred:
                rewards.append(0.5)  # Generated boxed format
            else:
                rewards.append(0.0)
        return rewards

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        max_completion_length=512,
        num_generations=args.num_generations,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    print("Initializing GRPO trainer...")

    # Custom reward that uses gold answers
    gold_answers = {d["prompt"]: d["gold_answer"] for d in grpo_data}

    def math_reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                text = tokenizer.decode(completion, skip_special_tokens=True)
            else:
                text = completion

            # Get gold answer for this prompt
            prompt = prompts[i] if prompts is not None else None
            gold = gold_answers.get(prompt, "")

            pred = extract_boxed(text)
            if gold and check_answer(pred, gold):
                rewards.append(1.0)
            elif "\\boxed{" in text:
                rewards.append(0.1)  # Small reward for format
            else:
                rewards.append(0.0)
        return rewards

    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=math_reward_fn,
        args=grpo_config,
        train_dataset=hf_dataset,
        peft_config=lora_config,
    )

    print("Starting GRPO training...")
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    print(f"Training time: {train_time:.1f}s ({train_time/60:.1f}min)")

    # Evaluate
    print(f"Evaluating ({args.eval_samples} samples)...")
    eval_start = time.time()
    model = trainer.model
    acc, level_acc, n_eval = evaluate_hf(model, tokenizer, test_data, device, max_samples=args.eval_samples)
    eval_time = time.time() - eval_start

    print(f"GRPO Result: acc={acc:.4f} ({int(acc*n_eval)}/{n_eval}) eval_time={eval_time:.0f}s")
    for level, la in sorted(level_acc.items()):
        print(f"  {level}: {la:.4f}")

    results = {
        "model": args.model_id,
        "method": "GRPO",
        "lr": args.lr,
        "seed": args.seed,
        "accuracy": acc,
        "level_accuracy": level_acc,
        "training_time_sec": train_time,
        "eval_time_sec": eval_time,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
