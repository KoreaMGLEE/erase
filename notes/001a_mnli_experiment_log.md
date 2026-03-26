# Plan 001a: MNLI & ARC 세부 실험 기록

> **참조 관계**: [001_mnli_proxy_confidence.md](001_mnli_proxy_confidence.md) 및 [002_arc_proxy_confidence.md](002_arc_proxy_confidence.md)의 세부 실험 과정, 디버깅 로그, 하이퍼파라미터 탐색 기록.
> 결과 요약은 001/002를 참고하고, "왜 이 설정을 선택했는가"를 알고 싶을 때 이 문서를 참조.

---

## MNLI 추가 모델 (T5 v1.1, Pythia 14m/31m/12b)

### 목적

Plan 0에서 완료한 12개 모델(BERT 5종, Pythia 7종)에 이어,
T5 v1.1 계열 5종과 Pythia 추가 3종(14m, 31m, 12b)의 MNLI confidence를 측정한다.

Plan 0과 동일한 90K subsampled data (3 × 30K non-overlapping splits)를 사용하며,
동일한 파이프라인(LR search → Confidence 측정)을 따른다.

## 이미 완료된 모델 (Plan 0)

| 계열 | 모델 | Params | Tuning | Best LR | Val Acc | 비고 |
|------|------|--------|--------|---------|---------|------|
| Encoder | bert-mini | 11M | full | 1e-4 | 0.612 | ✅ 완료 |
| | bert-small | 29M | full | 1e-4 | 0.657 | ✅ 완료 |
| | bert-medium | 42M | full | 5e-5 | 0.698 | ✅ 완료 |
| | bert-base | 110M | full | 5e-5 | 0.761 | ✅ 완료 |
| | bert-large | 335M | full | 3e-5 | 0.795 | ✅ 완료 |
| Decoder | pythia-70m | 70M | full | 3e-5 | 0.641 | ✅ 완료 |
| | pythia-160m | 160M | full | 1e-5 | 0.713 | ✅ 완료 |
| | pythia-410m | 410M | full | 1e-5 | 0.799 | ✅ 완료 |
| | pythia-1b | 1B | LoRA r=32 | 3e-4 | 0.797 | ✅ 완료 |
| | pythia-1.4b | 1.4B | LoRA r=32 | 3e-4 | 0.829 | ✅ 완료 |
| | pythia-2.8b | 2.8B | LoRA r=32 | 3e-4 | 0.848 | ✅ 완료 |
| | pythia-6.9b | 6.9B | LoRA r=32 | 1e-4 | 0.854 | ✅ 완료 |

---

## 추가할 모델 (이번 Plan)

| 계열 | 모델 | HuggingFace ID | Params | Tuning | Padding | DeepSpeed | 비고 |
|------|------|----------------|--------|--------|---------|-----------|------|
| Decoder | pythia-14m | EleutherAI/pythia-14m | 14M | full | **left** | 불필요 | |
| | pythia-31m | EleutherAI/pythia-31m | 31M | full | **left** | 불필요 | |
| | pythia-12b | EleutherAI/pythia-12b | 12B | **LoRA r=32** | **left** | **ZeRO-2** | 2GPU 분산 |
| Enc-Dec | t5-v1_1-small | google/t5-v1_1-small | 77M | full | right | 불필요 | C4 only |
| | t5-v1_1-base | google/t5-v1_1-base | 248M | full | right | 불필요 | C4 only |
| | t5-v1_1-large | google/t5-v1_1-large | 783M | full | right | 불필요 | C4 only |
| | t5-v1_1-xl | google/t5-v1_1-xl | 3B | **LoRA r=32** | right | 불필요 (bf16 단일 GPU) | C4 only |
| | t5-v1_1-xxl | google/t5-v1_1-xxl | 11B | **LoRA r=32** | right | **ZeRO-2** | 2GPU 분산, C4 only |

> **총 8개 모델 추가** (4 full fine-tune + 4 LoRA)

### Tuning 전략

- **< 1B (pythia-14m, 31m, t5-v1_1-small/base/large)**: Full fine-tuning. 메모리 충분.
- **1B~3B (t5-v1_1-xl)**: LoRA r=32. RTX 5090 1장에 bf16으로 학습 가능. Plan 0의 Pythia 1B~2.8B와 동일 방식.
- **> 10B (pythia-12b, t5-v1_1-xxl)**: LoRA r=32 + DeepSpeed ZeRO Stage 2. RTX 5090 2장에 분산.

### LoRA 설정 (≥ 1B 모델)

Plan 0과 동일:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Pythia
    # T5의 경우: ["q", "v"]
    bias="none",
    task_type="CAUSAL_LM",  # Pythia
    # T5의 경우: "SEQ_2_SEQ_LM"
)
```

### DeepSpeed ZeRO-2 설정 (> 10B 모델)

```json
{
    "fp16": {"enabled": false},
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": true},
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": 2,
    "wall_clock_breakdown": false
}
```

> RTX 5090 2장 (각 32GB) 기준. optimizer offload to CPU를 켜서 VRAM 절약.
> micro_batch_size=2, gradient_accumulation=4 → effective batch_size=16 (2GPU × 2micro × 4accum).

---

## 데이터

### 기존 데이터 (Plan 0에서 생성, Split 1~3)

Plan 0에서 생성한 동일 데이터를 그대로 사용:

```
outputs/plan0/data/
├── split1_indices.json    # 30K 인덱스 (MNLI train)
├── split2_indices.json    # 30K 인덱스 (MNLI train)
├── split3_indices.json    # 30K 인덱스 (MNLI train)
└── all_90k_indices.json   # 90K 인덱스 (MNLI train)
```

### 신규 데이터: Split 4 (ChaosNLI 포함)

**목적**: 본문 Figure 2(모델 confidence vs 인간 난이도)에 사용할 training dynamics 기반 confidence를
ChaosNLI 예제에 대해서도 확보한다. ChaosNLI는 MNLI-matched dev set 기반이므로, 이 예제들을
training split에 포함시켜 학습 중 confidence를 측정한다.

**방법론적 정당성**: 이 연구의 "쉬움"은 "모델이 학습 과정에서 해당 예제를 얼마나 빨리/쉽게 배우는가"로
정의된다. 따라서 ChaosNLI 예제에 대해서도 동일한 training dynamics 기반 측정이 필요하다.
Dev set 예제를 train에 포함시키는 것은 일반적으로 지양되지만, 이 실험의 목적은 일반화 성능 평가가 아니라
**예제별 학습 난이도 측정**이므로 문제가 되지 않는다.
논문에서 이 점을 명시적으로 기술할 것.

**ChaosNLI 데이터**:
- 출처: [ChaosNLI](https://github.com/easonnie/ChaosNLI) — `chaosNLI_v1.0/chaosNLI_mnli.jsonl`
- MNLI-matched dev set의 일부 예제에 대해 100명의 annotator가 레이블링
- 예제 수: ~1,514개 (MNLI-matched dev에서 선택된 것)
- 각 예제에 100명의 label distribution → disagreement 계산 가능

**Split 4 구성**:
1. ChaosNLI MNLI 예제 전체 (~1,514개)를 반드시 포함
2. MNLI train set에서 나머지를 샘플링하여 총 30K로 맞춤
   - 기존 split 1~3에 사용된 90K 인덱스는 제외 (중복 방지)
   - 나머지 pool: 392,702 - 90,000 = 302,702개
   - 여기서 30,000 - 1,514 = 28,486개를 추가 샘플링
3. Label 비율(entailment/neutral/contradiction) stratified 유지

```python
import json
import numpy as np
from datasets import load_dataset
from collections import Counter

# MNLI train & dev 로드
ds_train = load_dataset("glue", "mnli")["train"]
ds_dev = load_dataset("glue", "mnli")["validation_matched"]

# ChaosNLI 로드
with open("data/chaosNLI_mnli.jsonl") as f:
    chaos_data = [json.loads(line) for line in f]
chaos_uids = {ex["uid"] for ex in chaos_data}  # ChaosNLI uid

# ChaosNLI uid → MNLI dev index 매핑
# ChaosNLI uid 형식: "mnli_dev_{idx}" 또는 실제 데이터 확인 필요
chaos_dev_indices = []
for i, ex in enumerate(ds_dev):
    # ChaosNLI uid와 매칭 (형식은 실제 데이터 확인 후 조정)
    if ex["idx"] in chaos_uids or f"mnli_dev_{i}" in chaos_uids:
        chaos_dev_indices.append(i)

print(f"ChaosNLI matched dev examples: {len(chaos_dev_indices)}")
# 기대: ~1,514개

# 기존 90K train 인덱스 로드 (제외 대상)
with open("outputs/plan0/data/all_90k_indices.json") as f:
    used_train_indices = set(json.load(f))

# 나머지 train pool에서 샘플링
remaining_train = [i for i in range(len(ds_train)) if i not in used_train_indices]
rng = np.random.RandomState(42)

n_chaos = len(chaos_dev_indices)
n_supplement = 30000 - n_chaos  # ~28,486개
supplement_indices = rng.choice(remaining_train, size=n_supplement, replace=False)

# Split 4 구성: ChaosNLI dev 예제 + 보충 train 예제
# 저장 시 출처를 구분하여 기록
split4_data = {
    "chaos_dev_indices": sorted(chaos_dev_indices),      # dev set 인덱스
    "supplement_train_indices": sorted(supplement_indices.tolist()),  # train set 인덱스
    "total_size": n_chaos + n_supplement,
    "chaos_size": n_chaos,
    "supplement_size": n_supplement
}

save_json("outputs/plan5/data/split4_info.json", split4_data)

# Label 비율 확인
chaos_labels = [ds_dev[i]["label"] for i in chaos_dev_indices]
supplement_labels = [ds_train[i]["label"] for i in supplement_indices]
all_labels = chaos_labels + supplement_labels
print(f"Split 4 label distribution: {Counter(all_labels)}")
```

**학습 시 주의사항**:
- Split 4의 데이터 로딩은 train + dev 두 소스에서 가져와야 함
- DataLoader 구성 시 source(train/dev)를 메타데이터로 유지 → confidence 저장 시 구분
- **Confidence 측정 대상은 ChaosNLI 예제(~1,514개)만 분석에 사용**
  - 보충 train 예제의 confidence는 기존 split 1~3과 중복되지 않으므로 추가 데이터로 활용 가능
- Eval(validation)은 기존과 동일하게 MNLI dev-matched 전체를 사용하되,
  **Split 4에 포함된 ChaosNLI 예제가 eval set에도 포함됨을 인지** (eval accuracy에 약간의 bias 가능, 그러나 LR search는 split 1에서 하므로 무관)

**저장 구조**:
```
outputs/plan5/data/
├── split4_info.json              # split 4 구성 정보
├── split4_chaos_dev_indices.json # ChaosNLI dev 인덱스
└── split4_train_indices.json     # 보충 train 인덱스
```

---

## Phase 1: LR Search

### T5 v1.1 입력/출력 형식 (Text-to-Text)

```python
# 입력: "mnli hypothesis: {hypothesis} premise: {premise}"
# 타겟: "entailment" / "neutral" / "contradiction"

def format_mnli_t5(example):
    return {
        "input_text": f"mnli hypothesis: {example['hypothesis']} premise: {example['premise']}",
        "target_text": ["entailment", "neutral", "contradiction"][example["label"]]
    }
```

### Pythia 14m/31m 입력 형식 (Causal LM)

Plan 0의 Pythia와 동일:
```python
# 입력: "Premise: {p}\nHypothesis: {h}\nRelation:"
# 다음 토큰: " entailment" / " neutral" / " contradiction" (앞에 공백!)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### LR 후보

**Pythia < 110M (pythia-14m, pythia-31m)**: Plan 0의 소규모 Pythia와 동일

| 모델 | LR 후보 |
|------|---------|
| pythia-14m | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 |
| pythia-31m | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 |

**Pythia ≥ 1B LoRA (pythia-12b)**: Plan 0의 대규모 Pythia LoRA와 동일

| 모델 | LR 후보 |
|------|---------|
| pythia-12b | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 |

**T5 v1.1 < 1B (small, base, large)**: Enc-dec full fine-tune. BERT/Pythia ≥ 110M 범위 참고하되, T5는 일반적으로 약간 더 높은 LR을 선호하므로 상한을 넓힘.

| 모델 | LR 후보 |
|------|---------|
| t5-v1_1-small (77M) | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 |
| t5-v1_1-base (248M) | 1e-5, 3e-5, 5e-5, 1e-4, 3e-4 |
| t5-v1_1-large (783M) | 1e-5, 2e-5, 3e-5, 5e-5, 1e-4 |

**T5 v1.1 ≥ 1B LoRA (xl, xxl)**: Pythia LoRA와 동일 범위.

| 모델 | LR 후보 |
|------|---------|
| t5-v1_1-xl (3B) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 |
| t5-v1_1-xxl (11B) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 |

### 실행

**Split 1, Seed=1**로만 LR search:

```bash
# === Pythia 추가 (14m, 31m): full fine-tune ===
declare -A LR_CANDIDATES
LR_CANDIDATES[pythia-14m]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[pythia-31m]="3e-5 5e-5 1e-4 3e-4 1e-3"

for model in pythia-14m pythia-31m; do
    for lr in ${LR_CANDIDATES[$model]}; do
        python scripts/train_mnli_proxy.py \
            --model $model \
            --split split1 \
            --lr $lr \
            --epochs 1 \
            --batch_size 16 \
            --seed 1 \
            --eval_during_training \
            --output_dir outputs/plan1/lr_search/${model}_lr${lr}
    done
done

# === T5 v1.1 (small, base, large): full fine-tune ===
LR_CANDIDATES[t5-v1_1-small]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[t5-v1_1-base]="1e-5 3e-5 5e-5 1e-4 3e-4"
LR_CANDIDATES[t5-v1_1-large]="1e-5 2e-5 3e-5 5e-5 1e-4"

for model in t5-v1_1-small t5-v1_1-base t5-v1_1-large; do
    for lr in ${LR_CANDIDATES[$model]}; do
        python scripts/train_mnli_t5_proxy.py \
            --model $model \
            --split split1 \
            --lr $lr \
            --epochs 1 \
            --batch_size 16 \
            --seed 1 \
            --eval_during_training \
            --output_dir outputs/plan1/lr_search/${model}_lr${lr}
    done
done

# === T5 v1.1-xl (3B): LoRA, 단일 GPU, bf16 ===
LR_CANDIDATES[t5-v1_1-xl]="5e-5 1e-4 3e-4 1e-3 3e-3"

for lr in ${LR_CANDIDATES[t5-v1_1-xl]}; do
    python scripts/train_mnli_t5_proxy.py \
        --model t5-v1_1-xl \
        --split split1 \
        --lr $lr \
        --epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --seed 1 \
        --use_lora --lora_r 32 --lora_alpha 64 \
        --bf16 \
        --eval_during_training \
        --output_dir outputs/plan1/lr_search/t5-v1_1-xl_lr${lr}
done

# === T5 v1.1-xxl (11B): LoRA + DeepSpeed ZeRO-2, 2GPU ===
LR_CANDIDATES[t5-v1_1-xxl]="5e-5 1e-4 3e-4 1e-3 3e-3"

for lr in ${LR_CANDIDATES[t5-v1_1-xxl]}; do
    deepspeed --num_gpus=2 scripts/train_mnli_t5_proxy.py \
        --model t5-v1_1-xxl \
        --split split1 \
        --lr $lr \
        --epochs 1 \
        --seed 1 \
        --use_lora --lora_r 32 --lora_alpha 64 \
        --deepspeed configs/ds_zero2.json \
        --eval_during_training \
        --output_dir outputs/plan1/lr_search/t5-v1_1-xxl_lr${lr}
done

# === Pythia-12B: LoRA + DeepSpeed ZeRO-2, 2GPU ===
LR_CANDIDATES[pythia-12b]="5e-5 1e-4 3e-4 1e-3 3e-3"

for lr in ${LR_CANDIDATES[pythia-12b]}; do
    deepspeed --num_gpus=2 scripts/train_mnli_proxy.py \
        --model pythia-12b \
        --split split1 \
        --lr $lr \
        --epochs 1 \
        --seed 1 \
        --use_lora --lora_r 32 --lora_alpha 64 \
        --deepspeed configs/ds_zero2.json \
        --eval_during_training \
        --output_dir outputs/plan1/lr_search/pythia-12b_lr${lr}
done
```

### LR 선택 기준

Plan 0과 동일:
- Validation accuracy (MNLI matched dev set) 기준 best LR 선택
- 1 epoch만 학습, epoch 끝의 val_acc로 비교
- 동률이면 더 작은 LR 선택 (안정성)

### Run 수

8 models × 5 LRs = **40 runs** (각 1 epoch, 30K 데이터)

### 시간 추정 (RTX 5090 기준)

| 모델 | Tuning | GPU | 30K × 1ep 예상 시간 |
|------|--------|-----|-------------------|
| pythia-14m | full | 1 | ~3분 |
| pythia-31m | full | 1 | ~4분 |
| t5-v1_1-small (77M) | full | 1 | ~6분 |
| t5-v1_1-base (248M) | full | 1 | ~15분 |
| t5-v1_1-large (783M) | full | 1 | ~35분 |
| t5-v1_1-xl (3B) | LoRA | 1 (bf16) | ~40분 |
| t5-v1_1-xxl (11B) | LoRA+ZeRO-2 | 2 | ~90분 |
| pythia-12b | LoRA+ZeRO-2 | 2 | ~80분 |

**Phase 1 총: ~24시간** (순차 기준, 40 runs)

> 병렬 전략: 소규모 모델(14m, 31m, t5-small/base/large) LR search를 GPU 0에서 순차 실행하면서,
> 대규모 모델(xxl, 12b)은 2GPU를 잡아야 하므로 소규모 완료 후 실행.
> xl은 1GPU에서 실행 가능하므로 소규모와 번갈아 실행 가능.
> 예상 총 시간 (효율적 스케줄링): **~12시간**

---

## Phase 2: Confidence 측정 (3 Splits)

Phase 1에서 찾은 best LR로, 3개 split 각각에 대해 학습하면서 confidence 측정.

### 설정

Plan 0과 동일:

| 항목 | 값 |
|------|-----|
| Data | split1 (30K), split2 (30K), split3 (30K) |
| Epochs | 1 |
| LR | Phase 1에서 찾은 best LR (모델별) |
| Batch size | 모델별 상이 (아래 참조) |
| Confidence 측정 시점 | **1/3, 2/3, 3/3 지점** (step 기준) |

### T5 v1.1 Confidence 측정 방법

```python
# T5 v1.1 text-to-text confidence 측정

# 3개 레이블 토큰 ID
label_token_ids = {
    "entailment": tokenizer.encode("entailment", add_special_tokens=False)[0],
    "neutral": tokenizer.encode("neutral", add_special_tokens=False)[0],
    "contradiction": tokenizer.encode("contradiction", add_special_tokens=False)[0],
}

# T5의 decoder start token
decoder_start = tokenizer.pad_token_id

model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=torch.full((batch_size, 1), decoder_start, device=device)
    )
    logits = outputs.logits[:, 0, :]  # 첫 생성 위치
    label_logits = logits[:, list(label_token_ids.values())]  # 3개 레이블만
    probs = F.softmax(label_logits, dim=-1)
    # labels: 0=entailment, 1=neutral, 2=contradiction
    confidence = probs[range(batch_size), labels]
```

> **주의**: T5 v1.1은 원본 T5와 달리 supervised pre-training을 하지 않았으므로,
> 초기 confidence가 낮을 수 있음 (원본 T5에서 발생했던 0.95+ 포화 문제가 없을 것으로 기대).
> 만약 confidence가 여전히 포화되면 사용자에게 보고.

### Pythia 14m/31m Confidence 측정 방법

Plan 0의 Pythia와 완전히 동일:

```python
# ⚠️ Left padding 필수
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

label_token_ids = {
    "entailment": tokenizer.encode(" entailment", add_special_tokens=False)[0],
    "neutral": tokenizer.encode(" neutral", add_special_tokens=False)[0],
    "contradiction": tokenizer.encode(" contradiction", add_special_tokens=False)[0],
}

model.eval()
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, -1, :]  # left padding → 마지막 위치가 실제 마지막 토큰
    label_logits = logits[:, list(label_token_ids.values())]
    probs = F.softmax(label_logits, dim=-1)
    confidence = probs[range(batch_size), labels]
```

### 실행

```bash
# === Pythia 소규모 (14m, 31m): full fine-tune ===
for model in pythia-14m pythia-31m; do
    BEST_LR=$(cat outputs/plan1/lr_search/lr_selection_summary.json | jq -r ".${model}")
    for split in split1 split2 split3; do
        python scripts/train_mnli_proxy.py \
            --model $model \
            --split $split \
            --lr $BEST_LR \
            --epochs 1 \
            --batch_size 16 \
            --seed 1 \
            --measure_confidence \
            --confidence_checkpoints 3 \
            --output_dir outputs/plan1/confidence/${model}_${split}
    done
done

# === T5 v1.1 소규모 (small, base, large): full fine-tune ===
for model in t5-v1_1-small t5-v1_1-base t5-v1_1-large; do
    BEST_LR=$(cat outputs/plan1/lr_search/lr_selection_summary.json | jq -r ".${model}")
    for split in split1 split2 split3; do
        python scripts/train_mnli_t5_proxy.py \
            --model $model \
            --split $split \
            --lr $BEST_LR \
            --epochs 1 \
            --batch_size 16 \
            --seed 1 \
            --measure_confidence \
            --confidence_checkpoints 3 \
            --output_dir outputs/plan1/confidence/${model}_${split}
    done
done

# === T5 v1.1-xl (3B): LoRA, 단일 GPU, bf16 ===
BEST_LR=$(cat outputs/plan1/lr_search/lr_selection_summary.json | jq -r '.["t5-v1_1-xl"]')
for split in split1 split2 split3; do
    python scripts/train_mnli_t5_proxy.py \
        --model t5-v1_1-xl \
        --split $split \
        --lr $BEST_LR \
        --epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --seed 1 \
        --use_lora --lora_r 32 --lora_alpha 64 \
        --bf16 \
        --measure_confidence \
        --confidence_checkpoints 3 \
        --output_dir outputs/plan1/confidence/t5-v1_1-xl_${split}
done

# === T5 v1.1-xxl (11B): LoRA + DeepSpeed ZeRO-2, 2GPU ===
BEST_LR=$(cat outputs/plan1/lr_search/lr_selection_summary.json | jq -r '.["t5-v1_1-xxl"]')
for split in split1 split2 split3; do
    deepspeed --num_gpus=2 scripts/train_mnli_t5_proxy.py \
        --model t5-v1_1-xxl \
        --split $split \
        --lr $BEST_LR \
        --epochs 1 \
        --seed 1 \
        --use_lora --lora_r 32 --lora_alpha 64 \
        --deepspeed configs/ds_zero2.json \
        --measure_confidence \
        --confidence_checkpoints 3 \
        --output_dir outputs/plan1/confidence/t5-v1_1-xxl_${split}
done

# === Pythia-12B: LoRA + DeepSpeed ZeRO-2, 2GPU ===
BEST_LR=$(cat outputs/plan1/lr_search/lr_selection_summary.json | jq -r '.["pythia-12b"]')
for split in split1 split2 split3; do
    deepspeed --num_gpus=2 scripts/train_mnli_proxy.py \
        --model pythia-12b \
        --split $split \
        --lr $BEST_LR \
        --epochs 1 \
        --seed 1 \
        --use_lora --lora_r 32 --lora_alpha 64 \
        --deepspeed configs/ds_zero2.json \
        --measure_confidence \
        --confidence_checkpoints 3 \
        --output_dir outputs/plan1/confidence/pythia-12b_${split}
done
```

### Run 수

8 models × 3 splits = **24 runs**

### 시간 추정

| 모델 | Tuning | GPU | 30K × 1ep + 3회 inference |
|------|--------|-----|--------------------------|
| pythia-14m | full | 1 | ~5분 |
| pythia-31m | full | 1 | ~6분 |
| t5-v1_1-small (77M) | full | 1 | ~9분 |
| t5-v1_1-base (248M) | full | 1 | ~22분 |
| t5-v1_1-large (783M) | full | 1 | ~50분 |
| t5-v1_1-xl (3B) | LoRA | 1 (bf16) | ~60분 |
| t5-v1_1-xxl (11B) | LoRA+ZeRO-2 | 2 | ~135분 |
| pythia-12b | LoRA+ZeRO-2 | 2 | ~120분 |

**Phase 2 총: ~24시간** (순차, 24 runs)
**효율적 스케줄링 시: ~14시간**

---

## 출력 파일 구조

```
outputs/plan5/
├── lr_search/
│   ├── {model}_lr{lr}_results.json   # val_acc, train_loss 등
│   └── lr_selection_summary.json     # 모델별 best LR
├── confidence/
│   ├── {model}_split{1,2,3}_ckpt{1,2,3}_conf.json  # checkpoint별 raw (기존 split)
│   ├── {model}_split{1,2,3}_avg_conf.json            # split별 3-ckpt 평균
│   ├── {model}_90k_avg_conf.json                     # 90K 전체 confidence
│   ├── {model}_split4_ckpt{1,2,3}_conf.json          # Split 4 checkpoint별 raw
│   ├── {model}_split4_avg_conf.json                   # Split 4 전체 평균
│   └── {model}_split4_chaos_conf.json                 # Split 4 중 ChaosNLI 예제만 추출
├── data/
│   ├── split4_info.json              # Split 4 구성 정보
│   ├── split4_chaos_dev_indices.json # ChaosNLI dev 인덱스
│   └── split4_train_indices.json     # 보충 train 인덱스
├── configs/
│   └── ds_zero2.json                                  # DeepSpeed 설정 파일
└── logs/
    └── {model}_split{split}_training.log
```

---

## 실행 우선순위

MNLI를 우선 완료한 후 ARC로 넘어감.

### 순서 (GPU 효율 고려)

1. **[GPU 0] Pythia-14m, 31m LR search** (~35분) — 빠르게 끝남
2. **[GPU 0] T5 v1.1-small, base LR search** (~1.5시간) — 1과 병렬 불가 시 순차
3. **[GPU 0] T5 v1.1-large LR search** (~3시간)
4. **[GPU 0] T5 v1.1-xl LR search** (~3.5시간) — 1GPU bf16
5. **[GPU 0+1] T5 v1.1-xxl LR search** (~7.5시간) — 2GPU DeepSpeed
6. **[GPU 0+1] Pythia-12B LR search** (~6.5시간) — 2GPU DeepSpeed
7. Phase 1 결과 정리 → best LR 선택
8. Phase 2 (Split 1~3): 위와 동일 순서로 confidence 측정
9. **Phase 2.5 (Split 4): 전체 20개 모델 × Split 4 confidence 측정**

> **5와 6은 2GPU를 점유하므로**, 1~4를 먼저 완료한 뒤 순차 실행.
> 1~4는 1GPU이므로, 여유 GPU가 있으면 일부 병렬 실행 가능.

---

## Phase 2.5: Split 4 Confidence 측정 (ChaosNLI 포함)

Phase 2 완료 후, **Split 4에 대해 전체 20개 모델의 confidence를 측정**한다.
Split 4는 ChaosNLI dev 예제(~1,514개) + 보충 train 예제(~28,486개)로 구성된 30K split이다.

### 대상 모델

**Plan 001에서 완료한 12개 모델** (기존 best LR 그대로 사용):
- BERT: mini, small, medium, base, large
- Pythia: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b

**Plan 005에서 추가한 8개 모델** (Phase 1에서 결정된 best LR 사용):
- T5 v1.1: small, base, large, xl, xxl
- Pythia: 14m, 31m, 12b

### 실행 방식

- Split 1~3과 동일한 학습 스크립트를 사용하되, `--split split4` 지정
- **DataLoader 수정 필요**: Split 4는 MNLI train + dev 두 소스에서 로딩
  - DataLoader가 source(train/dev) + index를 메타데이터로 유지
  - Confidence 저장 시 ChaosNLI 예제만 별도 파일로 추출
- 학습 중 3 checkpoint(1/3, 2/3, 3/3 epoch)에서 confidence 측정
- ChaosNLI 예제의 3-checkpoint 평균 confidence를 `{model}_split4_chaos_conf.json`으로 저장

### 스크립트 수정 사항

기존 `scripts/train_mnli_proxy.py` (또는 T5용 `train_mnli_t5_proxy.py`)에 다음 추가:

```python
# Split 4 데이터 로딩
if args.split == "split4":
    split4_info = load_json("outputs/plan5/data/split4_info.json")
    chaos_dev_indices = split4_info["chaos_dev_indices"]
    supplement_train_indices = split4_info["supplement_train_indices"]

    # MNLI dev에서 ChaosNLI 예제 로드
    ds_dev = load_dataset("glue", "mnli")["validation_matched"]
    chaos_examples = [ds_dev[i] for i in chaos_dev_indices]

    # MNLI train에서 보충 예제 로드
    ds_train = load_dataset("glue", "mnli")["train"]
    supplement_examples = [ds_train[i] for i in supplement_train_indices]

    # 합치기 (source 태깅)
    all_examples = []
    for i, ex in enumerate(chaos_examples):
        ex["_source"] = "chaos_dev"
        ex["_source_idx"] = chaos_dev_indices[i]
        all_examples.append(ex)
    for i, ex in enumerate(supplement_examples):
        ex["_source"] = "train"
        ex["_source_idx"] = supplement_train_indices[i]
        all_examples.append(ex)

    # shuffle
    random.shuffle(all_examples)
    train_dataset = all_examples  # 30K

# Confidence 저장 시 ChaosNLI만 추출
if args.split == "split4":
    chaos_conf = {
        str(ex["_source_idx"]): conf
        for ex, conf in zip(all_examples, confidences)
        if ex["_source"] == "chaos_dev"
    }
    save_json(f"outputs/plan5/confidence/{model}_split4_chaos_conf.json", chaos_conf)
```

### 시간 추정

Split 4는 기존 split과 동일한 30K 크기이므로, **기존 Phase 2와 동일한 시간이 소요**.
단, 전체 20개 모델을 돌려야 하므로:

| 모델 그룹 | 모델 수 | 예상 시간 (순차) |
|-----------|--------|----------------|
| BERT 5종 (full, Plan 001 LR) | 5 | ~2.5시간 |
| Pythia 소규모 (14m~410m, full) | 5 | ~1시간 |
| Pythia 대규모 (1b~6.9b, LoRA) | 4 | ~4시간 |
| Pythia-12b (LoRA+ZeRO-2) | 1 | ~2시간 |
| T5 v1.1 소규모 (small~large, full) | 3 | ~1.5시간 |
| T5 v1.1 대규모 (xl, LoRA) | 1 | ~1시간 |
| T5 v1.1-xxl (LoRA+ZeRO-2) | 1 | ~2.5시간 |
| **총 (Split 4 only)** | **20** | **~14.5시간** |

효율적 스케줄링(2 GPU 병렬 활용) 시 **~8시간**.

### Plan 001 모델의 Split 4 실행 시 주의사항

- Plan 001에서 학습한 모델들은 checkpoint가 저장되어 있지 않으므로, **Split 4에 대해 처음부터 학습을 실행**해야 함
- Plan 001에서 확정된 best LR을 그대로 사용
- 학습 스크립트, 하이퍼파라미터 등 모든 설정은 Plan 001 때와 동일하게 유지 (split만 변경)

---

## 전체 Run 계산

| Phase | Runs | 시간 (순차) | 시간 (효율적 스케줄링) |
|-------|------|-----------|---------------------|
| Phase 1: LR Search | 40 | ~24시간 | ~12시간 |
| Phase 2: Confidence (Split 1~3) | 24 | ~24시간 | ~14시간 |
| Phase 2.5: Confidence (Split 4, 전체 20모델) | 20 | ~14.5시간 | ~8시간 |
| **총** | **84 runs** | **~62.5시간** | **~34시간** |

---

## T5 v1.1 학습 인사이트 (실험으로 확인)

### 문제: Class Collapse
T5 v1.1 base는 기본 설정(AdamW, 기존 프롬프트)에서 **decoder prior collapse** 발생.
모델이 encoder 입력을 무시하고 특정 class만 출력. Input ablation(normal/shuffled/dummy)에서 동일한 결과로 확인.

### 원인 분석
1. **Optimizer**: AdamW로는 학습 불가. **Adafactor 필수**.
2. **프롬프트**: 기존 `"mnli hypothesis: {h} premise: {p}"` → collapse 빈번.
   T5 v1.1은 span corruption으로 pretrain되어 `<extra_id_0>` sentinel 패턴에 익숙.
   **`"mnli hypothesis: {h} premise: {p} answer: <extra_id_0>"`** 프롬프트 + decoder target `"<extra_id_0> label"` 형식이 pretrain과 일치하여 안정적.
3. **Verbalizer**: `"entailment"` (4토큰) 대신 **`"yes"/"maybe"/"no"` (각 1토큰)** 사용.
   Multi-token target은 teacher forcing에서 loss 희석 (step 2+가 trivially 쉬워짐).
4. **Scheduler**: Linear decay는 1 epoch에서 후반 LR이 너무 떨어져 회복 불가.
   **Constant LR + warmup 6%**가 가장 안정적.
5. **LR**: 5e-5가 안정적. 너무 크면 (≥1e-4) split에 따라 collapse.

### 최종 T5 v1.1 설정
```
Optimizer: Adafactor(lr=5e-5, scale_parameter=False, relative_step=False, warmup_init=False)
Scheduler: constant_schedule_with_warmup(warmup=6%)
Encoder input: "mnli hypothesis: {h} premise: {p} answer: <extra_id_0>"
Decoder target: "<extra_id_0> yes" / "<extra_id_0> maybe" / "<extra_id_0> no"
Eval: decoder_input=[<pad>, <extra_id_0>] → position 1 logits에서 yes/maybe/no 비교
Epochs: 1
```

### 검증 결과 (1 epoch, 4 splits)
| split1 | split2 | split3 | split4 | mean | std |
|--------|--------|--------|--------|------|-----|
| 0.625 | 0.542 | 0.630 | 0.549 | 0.587 | 0.042 |

Collapse 없이 4/4 split 모두 안정적.

---

## 주의사항

### T5 v1.1 관련
1. **T5 v1.1은 원본 T5와 다른 모델**: `google/t5-v1_1-*`을 사용. `t5-small` 등 원본 T5를 사용하지 말 것.
2. **T5 v1.1 tokenizer**: `T5Tokenizer.from_pretrained("google/t5-v1_1-*", legacy=True)` 사용. AutoTokenizer는 에러 발생.
3. **Sentinel 프롬프트 필수**: encoder input에 `<extra_id_0>`, decoder target에 `<extra_id_0> label`.
4. **T5 v1.1은 MNLI를 본 적 없으므로**, 원본 T5의 포화 문제가 없음 확인됨.

### DeepSpeed 관련
5. **DeepSpeed ZeRO-2 사용 시 confidence 측정**: inference 시에는 DeepSpeed를 비활성화하거나, model.module로 접근하여 측정. DeepSpeed wrapped model에서 직접 inference하면 partition 문제가 생길 수 있음.
6. **DeepSpeed config**: `configs/ds_zero2.json`에 저장. 위의 JSON 참조.
7. **2GPU 실행**: `deepspeed --num_gpus=2` 사용. CUDA_VISIBLE_DEVICES 설정 필요 시 명시.

### 기존 Plan 0 주의사항 유지
8. **Left padding (Decoder 모델)**: Pythia 계열은 반드시 `tokenizer.padding_side = "left"`.
9. **fp16 사용 금지**: Pythia는 fp32, OOM 시 bf16. T5 v1.1도 bf16 권장.
10. **Pythia label tokenization**: " entailment" (공백 포함!) 등 첫 토큰 사용.
11. **Eval 시 gradient 끄기**: `torch.no_grad()` + `model.eval()` 필수.
12. **Checkpoint 미저장**: 학습 중 현장에서 confidence 측정, 모델 checkpoint는 저장하지 않음 (디스크 절약).

---

## Phase 3: 분석 (Plan 001 + Plan 005 전체 모델 통합)

Phase 2 완료 후, Plan 001(BERT 5종 + Pythia 7종)과 Plan 005(T5 v1.1 5종 + Pythia 3종)의
confidence 데이터를 통합하여 **총 20개 모델**에 대한 분석을 수행한다.

> 모든 분석 스크립트는 `scripts/`에, 결과는 `outputs/plan5/analysis/`에 저장.

Framework v4 기준으로, 본문에는 **3개의 Figure**만 포함하고 나머지는 Appendix로.

---

## 본문 분석 (3 Figures)

### 본문 분석 1: Pairwise Jaccard Heatmap (Figure 1 — §3 RQ1)

**목적**: 서로 다른 모델이 식별하는 "쉬운 예제" 집합이 얼마나 다른지를 한눈에 보여줌.
계열 내 유사도 vs 교차 계열 유사도를 동시에 비교하여, "주체마다 쉬운 예제가 다르다"를 직접 시각화.

**방법**:
1. 각 모델의 90K confidence에서 상위 k% (k=30) 예제를 "쉬운 예제"로 정의
2. 3계열 × 3규모 = 9개 대표 모델을 선정
   - Encoder: BERT-small (29M), BERT-base (110M), BERT-large (335M)
   - Enc-Dec: T5v1.1-small (77M), T5v1.1-base (248M), T5v1.1-large (783M)
   - Decoder: Pythia-70M, Pythia-410M, Pythia-2.8B
3. 9×9 = 36쌍에 대해 Jaccard 유사도 계산
4. 9×9 대칭 heatmap으로 시각화, 계열 경계에 검은 선 표시

```python
def jaccard(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0

representative_models = [
    "bert-small", "bert-base", "bert-large",
    "t5-v1_1-small", "t5-v1_1-base", "t5-v1_1-large",
    "pythia-70m", "pythia-410m", "pythia-2.8b"
]

# 9×9 Jaccard matrix
for m1 in representative_models:
    for m2 in representative_models:
        jaccard_matrix[m1][m2] = jaccard(easy_set[m1], easy_set[m2])
```

**시각화**: (a) MNLI, (b) ARC를 나란히 배치한 2-panel figure.
- 9×9 대칭 heatmap, 각 셀에 Jaccard 값 표기
- 계열 경계(BERT | T5 v1.1 | Pythia)에 검은 선 → 블록 구조가 한눈에 보임
- 색상: YlOrRd (낮으면 노랑, 높으면 빨강)

**기대 결과**:
- 대각선 블록(계열 내): 높은 유사도 (0.4~0.6+)
- 비대각선 블록(교차 계열): 낮은 유사도 (0.2~0.3)
- 특히 Encoder↔Decoder가 가장 낮음
- 블록 구조 자체가 "같은 계열은 비슷, 다른 계열은 다르다"를 직관적으로 전달

**출력**: `outputs/plan5/analysis/jaccard_matrix_9x9_k30.json`, `figures/fig1_jaccard_heatmap_9x9.pdf`

---

### 본문 분석 2: 모델↔인간 난이도 방향성 비교 — Spearman (Figure 2 — §3 RQ1)

**목적**: 모델이 학습 과정에서 느끼는 난이도의 **순서(방향성)**가 인간이 느끼는 난이도 순서와
얼마나 일치하는지를 Spearman 순위 상관으로 정량화.

**왜 Spearman인가**: Raw confidence는 모델 규모에 따라 절대값이 크게 다르다 —
대규모 모델은 전반적으로 높은 confidence, 소규모 모델은 낮은 confidence를 보인다.
따라서 raw confidence vs 인간 난이도를 비교하면, "큰 모델이 인간과 더 일치"가 아니라
단순히 "큰 모델이 전부 높다"를 보게 된다. Spearman은 순위만 비교하므로 이 confound를 제거하고,
"난이도 순서의 방향성이 얼마나 일치하는가"만 순수하게 측정한다.

**핵심 설계**: ChaosNLI 예제를 **Split 4의 학습 데이터에 포함**시켜, 기존 split 1~3과
동일한 training dynamics 기반 confidence를 측정한다. 이를 통해 "모델이 학습 시 얼마나
쉽게 배우는가"라는 본 연구의 "쉬움" 정의와 일관된 비교가 가능해진다.

**데이터**:
- **MNLI**: ChaosNLI 활용. Split 4에서 측정한 ChaosNLI 예제의 training confidence 사용.
  - ChaosNLI: MNLI-matched dev의 ~1,514개 예제에 대해 100명 annotator가 레이블링
  - Annotator disagreement (entropy)를 인간 난이도 proxy로 사용
  - **Human easiness** = 1 - normalized_entropy (또는 majority proportion)로 변환하여
    모델 confidence와 방향을 맞춤 (둘 다 높을수록 "쉬움")
- **ARC**: 기존 인간 난이도 annotation (Low=1/Medium=2/High=3) → 인간 easiness로 반전.
  - 2,151개 예제에 난이도 라벨 존재

**방법**:

1. 각 모델에 대해 ChaosNLI 예제의 training confidence 순위와 human easiness 순위 간 Spearman ρ 계산
2. x축: model params (log scale), y축: Spearman ρ
3. 3계열을 각각 라인으로 표시

```python
import json
import numpy as np
from scipy.stats import entropy, spearmanr

# ChaosNLI → human easiness 계산
with open("data/chaosNLI_mnli.jsonl") as f:
    chaos_data = [json.loads(line) for line in f]

human_easiness = {}
for example in chaos_data:
    label_counts = example["label_counter"]  # e.g., {"e": 72, "n": 20, "c": 8}
    total = sum(label_counts.values())
    probs = [count / total for count in label_counts.values()]
    h = entropy(probs)
    max_entropy = np.log(3)  # 3-class uniform
    human_easiness[example["uid"]] = 1 - (h / max_entropy)  # 높을수록 인간에게 쉬움

# Split 4 training confidence 로드
for model in all_models:
    split4_conf = load(f"outputs/plan5/confidence/{model}_split4_chaos_conf.json")
    model_chaos_conf[model] = split4_conf

# 각 모델별 Spearman 계산
uids = sorted(human_easiness.keys())
human_scores = [human_easiness[uid] for uid in uids]

for model in all_models:
    model_scores = [model_chaos_conf[model][uid] for uid in uids]
    rho, pval = spearmanr(model_scores, human_scores)
    spearman_results[model] = {"rho": rho, "pval": pval}
```

**시각화**: (a) MNLI, (b) ARC를 나란히 배치한 2-panel line plot.
- x축: model params (log scale), y축: Spearman ρ (model confidence rank vs human easiness rank)
- 3계열(BERT, T5 v1.1, Pythia)을 각각 다른 색/마커의 라인으로
- Figure 1(heatmap: 모델↔모델)과 자연스럽게 대비되는 형태

**핵심 서술 포인트**:
- 모든 모델에서 ρ < 0.5 → 모델과 인간의 난이도 순서가 약~중간 수준으로만 일치
- 규모가 커질수록 ρ가 소폭 증가하지만, 여전히 불일치가 큼
- 계열마다 ρ의 절대값과 증가 패턴이 다름 → 아키텍처도 인간과의 일치도에 영향
- **핵심 메시지**: "모델의 쉬움 ≠ 인간의 쉬움"이 모든 모델에서 성립하며, 그 불일치의 정도는 주체(모델)에 따라 다름

**기대 결과**: 모든 모델이 인간과 약한 양의 상관을 보이지만(ρ ≈ 0.15~0.45),
완전한 일치(ρ → 1)에는 크게 미달. 모델 규모/아키텍처에 따라 인간과의 거리가 다름.

**출력**: `outputs/plan5/analysis/human_model_spearman.json`, `figures/fig2_human_model_spearman.pdf`

**주의사항**:
- ChaosNLI 데이터를 사전에 다운로드해야 함: `data/chaosNLI_mnli.jsonl`
- ChaosNLI uid와 MNLI dev-matched의 매칭이 필요 — uid 형식은 데이터 확인 후 조정
- Split 4 학습 시 ChaosNLI 예제의 confidence를 별도로 저장하도록 스크립트 수정 필요
- ARC 인간 난이도 annotation의 예제 수가 2,151개로 제한적 → Spearman p-value 확인 필수
- Human easiness의 정의(1 - normalized_entropy vs majority proportion)에 따라 ρ 값이 달라질 수 있음 → 둘 다 계산하여 robust한 쪽 사용
- **논문에 명시할 사항**: ChaosNLI dev 예제를 학습 데이터에 포함시킨 이유와 방법론적 정당성
  (목적이 일반화 평가가 아닌 학습 난이도 측정이므로 허용됨)

---

### 본문 분석 3: Model Size vs Shortcut 비율 (Figure 3 — §4 RQ2)

**목적**: 각 주체(모델)가 쉬워하는 예제의 언어적 특성이 규모/아키텍처에 따라 질적으로 다름을 보여줌.
"소규모 모델의 쉬움 = shortcut, 대규모 모델의 쉬움 ≠ shortcut"을 직접 시각화.

**⚠️ Class별 측정 (핵심)**:
MNLI의 annotation artifact는 class-specific하다:
- **부정어(negation)** → **contradiction** 예측의 shortcut
  - Hypothesis에 부정어가 있으면 contradiction으로 판단하는 표면적 단서
  - 따라서 **쉬운 예제 중 contradiction 예제만** 대상으로 부정어 비율을 측정
- **어휘 중복(lexical overlap)** → **entailment** 예측의 shortcut
  - Premise와 hypothesis의 단어 겹침이 크면 entailment로 판단하는 표면적 단서
  - 따라서 **쉬운 예제 중 entailment 예제만** 대상으로 어휘 중복 비율을 측정

쉬운 예제 전체에서 비율을 재면 class 분포 차이에 의해 신호가 희석된다.
해당 shortcut이 실제로 작동하는 class 안에서 측정해야 shortcut 의존도를 정확히 포착할 수 있다.

**방법**:
1. 각 모델의 쉬운 예제 집합(상위 30%)을 class별로 분리
2. Contradiction 쉬운 예제 중 부정어 비율 계산
3. Entailment 쉬운 예제 중 고어휘중복(overlap ≥ 0.6) 비율 계산
4. x축: model params (log scale), y축: shortcut 비율

```python
NEGATION_WORDS = {"no", "not", "never", "nothing", "nobody", "neither",
                  "nor", "none", "cannot", "can't", "don't", "doesn't",
                  "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
                  "isn't", "aren't", "wasn't", "weren't"}

def has_negation(hypothesis):
    tokens = set(hypothesis.lower().split())
    return len(tokens & NEGATION_WORDS) > 0

def lexical_overlap(premise, hypothesis):
    p_tokens = set(premise.lower().split())
    h_tokens = set(hypothesis.lower().split())
    if len(h_tokens) == 0:
        return 0.0
    return len(p_tokens & h_tokens) / len(h_tokens)

for model in all_models:
    easy_indices = easy_set[model]

    # Contradiction 쉬운 예제 중 부정어 비율
    easy_contradiction = [idx for idx in easy_indices if labels[idx] == 2]  # contradiction
    if len(easy_contradiction) > 0:
        negation_by_model[model] = sum(
            has_negation(dataset[idx]["hypothesis"]) for idx in easy_contradiction
        ) / len(easy_contradiction)

    # Entailment 쉬운 예제 중 어휘 중복 비율
    easy_entailment = [idx for idx in easy_indices if labels[idx] == 0]  # entailment
    if len(easy_entailment) > 0:
        overlap_by_model[model] = sum(
            lexical_overlap(dataset[idx]["premise"], dataset[idx]["hypothesis"]) >= 0.6
            for idx in easy_entailment
        ) / len(easy_entailment)
```

**Global baseline도 동일하게 class별로 계산**:
```python
# 전체 데이터셋의 contradiction 예제 중 부정어 비율
all_contradiction = [idx for idx in all_90k_indices if labels[idx] == 2]
global_negation = sum(has_negation(dataset[idx]["hypothesis"]) for idx in all_contradiction) / len(all_contradiction)

# 전체 데이터셋의 entailment 예제 중 어휘 중복 비율
all_entailment = [idx for idx in all_90k_indices if labels[idx] == 0]
global_overlap = sum(lexical_overlap(dataset[idx]["premise"], dataset[idx]["hypothesis"]) >= 0.6
                     for idx in all_entailment) / len(all_entailment)
```

**시각화**: 2-panel line plot. Figure 2와 동일한 x축(model params, log scale).
- **(a) Negation ratio in contradiction easy examples**: 3계열 각각 라인. Global baseline을 수평 점선으로.
- **(b) Lexical overlap ratio in entailment easy examples**: 동일 구성.

**기대 결과**:
- 부정어 (contradiction 내): 모든 계열에서 규모 ↑ → 비율 ↓ (**scale-sensitive**)
  - 소규모 모델은 contradiction을 맞출 때 부정어에 크게 의존
  - 대규모 모델은 부정어 없이도 contradiction을 맞추는 능력이 커짐
- 어휘 중복 (entailment 내): 규모에 거의 반응 없음, 계열 간 차이가 지배적 (**architecture-sensitive**)
  - Encoder 계열이 decoder 계열보다 일관되게 높은 overlap 의존도
  - 규모를 키워도 architecture 특성이 유지됨
- 이 차이가 기존 불일치를 설명: 소규모 encoder → shortcut-heavy한 "쉬운 예제" → "해롭다",
  대규모 decoder → 다른 "쉬운 예제" → "유익하다"

**출력**: `outputs/plan5/analysis/shortcut_by_model.json`, `figures/fig3_scale_vs_shortcut.pdf`

---

## Appendix 분석

### Appendix 분석 A: Pairwise Jaccard Heatmap (20×20)

**목적**: 본문 Figure 1의 line plot을 보완하여 전체 모델 쌍의 Jaccard를 한눈에 보여줌.

**방법**:
1. 모든 모델 쌍(20C2 = 190쌍)에 대해 Jaccard 유사도 계산
2. 20×20 대칭 heatmap으로 시각화
3. 계열 경계에 검은 선 표시 (BERT | T5 v1.1 | Pythia)

**출력**: `outputs/plan5/analysis/jaccard_matrix_k30.json`, `figures/fig_app_jaccard_20x20.pdf`

---

### Appendix 분석 B: k ∈ {10, 20, 30, 40, 50} 강건성 분석

**목적**: "상위 30%" 기준이 결과에 미치는 영향을 확인. k에 따라 본문 결과가 유지되는지 검증.

**방법**:
1. k ∈ {10, 20, 30, 40, 50}에 대해 Figure 1(scale vs Jaccard)과 Figure 3(scale vs shortcut)를 재생산
2. 결과의 정성적 패턴이 k에 robust한지 확인

**출력**: `outputs/plan5/analysis/jaccard_matrix_k{k}.json`, `figures/fig_app_k_robustness.pdf`

---

### Appendix 분석 C: Cross-Model Confidence Variance

**목적**: Jaccard(이진)를 보완하는 연속 지표. 동일 예제에 대한 모델 간 confidence 분산.

**방법**:
1. 90K 예제 각각에 대해 20개 모델의 confidence 값 수집
2. 각 예제의 cross-model variance 계산
3. Variance 분포를 히스토그램으로 시각화
4. Variance 상위/하위 예제의 특성 비교

```python
for idx in all_90k_indices:
    confidences_across_models = [model_conf[model][idx] for model in all_models]
    variance[idx] = np.var(confidences_across_models)
```

**출력**: `outputs/plan5/analysis/cross_model_variance.json`, `figures/fig_app_variance_hist.pdf`

---

### Appendix 분석 D: Spearman 순위 상관

**목적**: Jaccard(집합)와 variance(예제 단위)를 보완하는 전체 순위 상관 지표.

**방법**:
1. 모든 모델 쌍에 대해 Spearman rank correlation 계산
2. 20×20 heatmap으로 시각화

```python
from scipy.stats import spearmanr

for m1, m2 in combinations(all_models, 2):
    confs_m1 = [model_conf[m1][idx] for idx in all_90k_indices]
    confs_m2 = [model_conf[m2][idx] for idx in all_90k_indices]
    spearman_matrix[m1][m2], _ = spearmanr(confs_m1, confs_m2)
```

**출력**: `outputs/plan5/analysis/spearman_matrix.json`, `figures/fig_app_spearman_20x20.pdf`

---

### Appendix 분석 E: Sharing Degree × Shortcut 비율

**목적**: 여러 모델이 공통적으로 "쉽다"고 보는 예제(높은 공유도)가 shortcut인지 분석.

**방법** (본문 Figure 3과 동일하게 class별 측정):
1. 각 예제의 공유도(sharing degree) 계산: 20개 모델 중 해당 예제를 상위 30%에 넣은 모델 수 (d ∈ [0, 20])
2. 공유도 구간별로:
   - **Contradiction 예제 중** 부정어 비율 측정
   - **Entailment 예제 중** 어휘 중복 비율 측정
3. x축: sharing degree, y축: shortcut 비율

```python
for idx in all_90k_indices:
    sharing_degree[idx] = sum(1 for model in all_models if idx in easy_set[model])

for d in range(0, len(all_models) + 1):
    examples_at_d = [idx for idx in all_90k_indices if sharing_degree[idx] == d]
    # Class별 측정
    contradiction_at_d = [idx for idx in examples_at_d if labels[idx] == 2]
    entailment_at_d = [idx for idx in examples_at_d if labels[idx] == 0]
    negation_ratio[d] = compute_negation_ratio(contradiction_at_d)
    overlap_ratio[d] = compute_overlap_ratio(entailment_at_d)
```

**기대 결과**: 공유도가 높을수록 shortcut 비율 증가 → 모델 간 합의 = 편향에 대한 합의.

**출력**: `outputs/plan5/analysis/sharing_degree_stats.json`, `figures/fig_app_sharing_shortcut.pdf`

---

### Appendix 분석 F: 모델 부분집합 교집합의 편향 식별 효율

**목적**: 소수의 이질적 모델 조합의 교집합이 편향 식별에 효율적인지 확인.

**방법**:
1. 2개 모델 조합(20C2 = 190쌍)의 교집합에서 shortcut 비율 계산
2. 이질적 계열 조합 vs 동일 계열 조합 비교

**출력**: `outputs/plan5/analysis/pair_intersection_bias.json`

---

### Appendix 분석 G: Dev Inference 기반 모델↔인간 비교 (방법 A 보완)

**목적**: 본문 Figure 2(방법 B, Split 4 training confidence)의 보완 분석.
Training에 포함시키지 않은 상태에서의 inference confidence와 인간 난이도의 관계를 확인.
방법 B의 결과가 "학습 데이터에 포함시켰기 때문에 편향된 것은 아닌지"를 검증.

**방법**:
1. 기존 Split 1~3으로 학습한 모델의 checkpoint 시점(또는 최종 모델)에서
   ChaosNLI dev 예제에 대해 inference-only confidence를 측정
2. 본문 Figure 2와 동일한 분석 수행 (disagreement 구간 vs inference confidence)
3. 방법 B(training confidence)와 방법 A(inference confidence)의 결과 패턴이 일치하는지 비교

**구현**: 기존 Split 1~3 학습 스크립트에 dev inference 단계를 추가하거나,
Split 4 학습 시 중간 checkpoint에서 별도로 ChaosNLI dev에 대해 inference 실행.

```python
# Split 1~3 학습 중, checkpoint 시점에서 dev inference 추가
# (학습 스크립트 수정 필요 — checkpoint 시점에서 model.eval() → dev forward)
model.eval()
with torch.no_grad():
    for batch in chaos_dev_dataloader:
        outputs = model(**batch)
        # ... confidence 계산 (모델 타입별 기존 방식과 동일)
        dev_inference_conf[batch_indices] = confidence_values
```

**시각화**: 본문 Figure 2와 동일 형식의 figure를 appendix에 배치.
방법 B와 나란히 비교할 수 있도록 구성.

**기대 결과**: 방법 A와 방법 B의 정성적 패턴이 일치 → 방법 B의 결과가 robust함을 확인.
만약 패턴이 다르면, 그 차이 자체가 "학습 과정에서의 쉬움"과 "추론 시 쉬움"의 차이를 보여주는
추가적 발견이 될 수 있음.

**출력**: `outputs/plan5/analysis/dev_inference_vs_human.json`, `figures/fig_app_dev_inference_human.pdf`

---

### 출력 파일 구조 (분석)

```
outputs/plan5/analysis/
├── scale_convergence.json             # 본문 Figure 1
├── human_vs_model_difficulty.json     # 본문 Figure 2 (방법 B, training confidence)
├── shortcut_by_model.json             # 본문 Figure 3
├── jaccard_matrix_k{k}.json          # Appendix A, B
├── cross_model_variance.json          # Appendix C
├── spearman_matrix.json               # Appendix D
├── sharing_degree_stats.json          # Appendix E
├── pair_intersection_bias.json        # Appendix F
└── dev_inference_vs_human.json        # Appendix G (방법 A 보완)

figures/
├── fig1_scale_vs_jaccard.pdf          # 본문 Figure 1 (§3 RQ1)
├── fig2_human_vs_model.pdf            # 본문 Figure 2 (§3 RQ1)
├── fig3_scale_vs_shortcut.pdf         # 본문 Figure 3 (§4 RQ2)
├── fig_app_jaccard_20x20.pdf          # Appendix A
├── fig_app_k_robustness.pdf           # Appendix B
├── fig_app_variance_hist.pdf          # Appendix C
├── fig_app_spearman_20x20.pdf         # Appendix D
├── fig_app_sharing_shortcut.pdf       # Appendix E
├── (pair_intersection은 table로 제공)  # Appendix F
└── fig_app_dev_inference_human.pdf    # Appendix G
```

---

## 검증 체크리스트

Phase 완료 후 확인 사항:

### Phase 1 (LR Search) 후
1. 모든 모델의 val_acc가 chance (0.333) 이상인가?
2. T5 v1.1 모델의 val_acc가 합리적인가? (원본 T5 대비 낮을 수 있으나, chance보다는 높아야 함)
3. DeepSpeed 모델(xxl, 12b)이 정상 학습되었는가? (loss가 감소하는지 확인)

### Phase 2 (Confidence, Split 1~3) 후
4. T5 v1.1의 confidence 분포: >0.95 비율이 50% 미만인가? (포화 없음 확인)
5. Pythia-14m/31m: chance 수준(~0.33)보다 높은 mean confidence를 보이는가?
6. Pythia-12B: 기존 6.9B와 유사하거나 더 높은 confidence 패턴을 보이는가?
7. 3 split 간 confidence 분포(mean, std, >0.95 비율)가 유사한가?
8. DeepSpeed 모델의 confidence 측정이 non-DeepSpeed 모델과 동일한 방식으로 이루어졌는가?

### Phase 2.5 (Confidence, Split 4) 후
9. Split 4의 전체 confidence 분포가 Split 1~3과 유사한 패턴을 보이는가? (보충 train 예제 기준)
10. ChaosNLI 예제(~1,514개)의 confidence가 정상적으로 추출되었는가? (결측 없음 확인)
11. ChaosNLI 예제의 confidence 범위가 합리적인가? (모두 0.33이거나 모두 1.0이면 문제)
12. 20개 모델 전부 Split 4 실행 완료되었는가? (Plan 001 모델 12개 + Plan 005 모델 8개)
13. ChaosNLI uid ↔ MNLI dev index 매칭이 올바른가? (샘플 10개 수동 확인)

---

## 실험 결과 (완료된 모델)

### Phase 1 & 2 결과

| 모델 | Params | Tuning | Optimizer | Best LR | Val Acc | 90K Mean | >0.95 |
|------|--------|--------|-----------|---------|---------|----------|-------|
| pythia-14m | 14M | full | AdamW | 3e-5 | 0.635 | 0.533 | 0.1% |
| pythia-31m | 31M | full | AdamW | 1e-4 | 0.647 | 0.538 | 0.4% |
| t5-v1_1-small | 77M | full | Adafactor | 1e-4 | 0.634 | 0.484 | 1.1% |
| t5-v1_1-base | 248M | full | Adafactor | 1e-4 | 0.788 | 0.523 | 3.8% |
| t5-v1_1-large | 783M | full | Adafactor | 5e-5 | 0.869 | 0.832 | 57.1% |
| t5-v1_1-xl | 3B | LoRA | Adafactor | 1e-4 | 0.886 | 0.834 | 60.2% |
| pythia-12b | 12B | LoRA | — | — | — | — | 미완료 |
| t5-v1_1-xxl | 11B | LoRA | — | — | — | — | 미완료 |

> T5 v1.1은 sentinel 프롬프트 + Adafactor + constant LR with warmup 사용.
> t5-v1_1-large, xl의 >0.95 비율이 57~60%로 포화 경계.
