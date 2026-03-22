# Plan 0: MNLI Proxy Confidence 측정 (90K Subsampled, 3 Non-Overlapping Splits × 30K)

## 목적

MNLI train (392,702개)에서 90K개를 비복원 추출하여 3개의 겹치지 않는 하위 집합(각 30K)을 생성. 이 3개 split에 대해 proxy 모델을 각각 학습하고, 학습 과정에서 confidence를 측정하여 각 예제의 "모델 난이도"를 산출.

**왜 30K × 3 split인가?**
- MNLI 전체(393K)로 학습하면 큰 모델은 confidence가 거의 1.0으로 포화
- 30K (전체의 ~7.6%)는 충분히 학습 가능하면서도 포화를 방지
- 3개 비중복 split은 seed 역할: 서로 다른 학습 데이터 → confidence의 robustness 측정
- 각 split은 독립적으로 학습 → confidence를 따로 측정 → 겹치는 예제가 없으므로 90K 예제 각각에 대해 1개 split에서의 confidence를 가짐

## 데이터 준비

### Step 1: 90K 샘플링 (비중복 3 × 30K)

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("glue", "mnli")["train"]

# 392,702개 중 90,000개를 비복원 추출
rng = np.random.RandomState(42)
all_indices = rng.permutation(len(ds))[:90000]

# 3개 비중복 split (각 30,000개)
split1_indices = sorted(all_indices[:30000])
split2_indices = sorted(all_indices[30000:60000])
split3_indices = sorted(all_indices[60000:90000])

# 레이블 비율(entailment/neutral/contradiction) stratified 확인
for name, idx in [("split1", split1_indices), ("split2", split2_indices), ("split3", split3_indices)]:
    labels = [ds[int(i)]["label"] for i in idx]
    print(f"{name}: {Counter(labels)}")
    # 약 33/33/33% 기대

# 저장
save_json("data/split1_indices.json", split1_indices.tolist())
save_json("data/split2_indices.json", split2_indices.tolist())
save_json("data/split3_indices.json", split3_indices.tolist())
save_json("data/all_90k_indices.json", all_indices.tolist())
```

> **Stratified sampling 권장**: label 비율을 유지하면서 샘플링. sklearn의 `train_test_split`에 `stratify` 옵션 사용.

> **중요**: 3개 split은 서로 겹치지 않아야 함. `split1 ∩ split2 = ∅`, `split1 ∩ split3 = ∅`, `split2 ∩ split3 = ∅`.

---

## Phase 1: LR Search

### 모델 목록

| 계열 | 모델 | Params | HuggingFace ID | Padding | Tuning |
|------|------|--------|---------------|---------|--------|
| **Encoder** | bert-mini | 11M | prajjwal1/bert-mini | right | full |
| | bert-small | 29M | prajjwal1/bert-small | right | full |
| | bert-medium | 42M | prajjwal1/bert-medium | right | full |
| | bert-base | 110M | bert-base-uncased | right | full |
| | bert-large | 335M | bert-large-uncased | right | full |
| **Decoder** | pythia-70m | 70M | EleutherAI/pythia-70m | **left** | full |
| | pythia-160m | 160M | EleutherAI/pythia-160m | **left** | full |
| | pythia-410m | 410M | EleutherAI/pythia-410m | **left** | full |
| | pythia-1b | 1B | EleutherAI/pythia-1b | **left** | **LoRA r=32** |
| | pythia-1.4b | 1.4B | EleutherAI/pythia-1.4b | **left** | **LoRA r=32** |
| | pythia-2.8b | 2.8B | EleutherAI/pythia-2.8b | **left** | **LoRA r=32** |
| | pythia-6.9b | 6.9B | EleutherAI/pythia-6.9b | **left** | **LoRA r=32** |

> **총 12개 모델** (8 full fine-tune + 4 LoRA). T5 계열은 pretrain 시 MNLI를 multi-task로 학습하여 confidence 포화(>0.95가 69%) 발생 → 제외.
>
> **LoRA 설정 (≥ 1B 모델)**: `peft` 라이브러리 사용. `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, target modules: query/value projection layers. LoRA는 학습 파라미터가 적어 더 큰 LR이 필요하므로 별도 LR search space 적용.

### LR 후보

모델 크기(< 110M vs ≥ 110M)에 따라 LR search space를 구분. 작은 모델은 더 큰 LR이 필요하므로, 가장 작은 LR 2개를 3e-4, 1e-3으로 대체.

**≥ 110M (bert-base, bert-large, pythia-160m, pythia-410m)**

| 계열 | LR 후보 | 근거 |
|------|---------|------|
| BERT | 1e-5, 2e-5, 3e-5, 5e-5, 1e-4 | BERT 논문 권장 범위 2e-5~5e-5, 양쪽 확장 |
| Pythia | 1e-5, 2e-5, 3e-5, 5e-5, 1e-4 | Decoder 모델 일반 범위 |

**< 110M (bert-mini, bert-small, bert-medium, pythia-70m)**

| 계열 | LR 후보 | 근거 |
|------|---------|------|
| BERT | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 | 작은 모델은 높은 LR 필요, 낮은 LR 2개 제거 후 상위 확장 |
| Pythia | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 | 작은 decoder도 높은 LR 탐색 필요 |

**≥ 1B LoRA (pythia-1b, pythia-1.4b, pythia-2.8b, pythia-6.9b)**

LoRA는 학습 파라미터가 전체의 ~1% 수준이므로 full fine-tune 대비 훨씬 큰 LR이 필요.

| 계열 | LR 후보 | 근거 |
|------|---------|------|
| Pythia (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | LoRA adapter만 학습, 5e-5부터 탐색 |

### 실행

**Split 1, Seed=1**로만 LR search:

```bash
# LR candidates by model size
declare -A LR_CANDIDATES
# < 110M: 높은 LR 범위
LR_CANDIDATES[bert-mini]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[bert-small]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[bert-medium]="3e-5 5e-5 1e-4 3e-4 1e-3"
LR_CANDIDATES[pythia-70m]="3e-5 5e-5 1e-4 3e-4 1e-3"
# >= 110M: 기본 LR 범위
LR_CANDIDATES[bert-base]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_CANDIDATES[bert-large]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_CANDIDATES[pythia-160m]="1e-5 2e-5 3e-5 5e-5 1e-4"
LR_CANDIDATES[pythia-410m]="1e-5 2e-5 3e-5 5e-5 1e-4"
# >= 1B LoRA: 큰 LR 범위
LR_CANDIDATES[pythia-1b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-1.4b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-2.8b]="5e-5 1e-4 3e-4 1e-3 3e-3"
LR_CANDIDATES[pythia-6.9b]="5e-5 1e-4 3e-4 1e-3 3e-3"

for model in bert-mini bert-small bert-medium bert-base bert-large \
             pythia-70m pythia-160m pythia-410m \
             pythia-1b pythia-1.4b pythia-2.8b pythia-6.9b; do
    for lr in ${LR_CANDIDATES[$model]}; do
        python train_mnli_proxy.py \
            --model $model \
            --split split1 \
            --lr $lr \
            --epochs 1 \
            --batch_size 16 \
            --seed 1 \
            --eval_during_training \
            --output_dir outputs/plan0/lr_search/${model}_lr${lr}
    done
done
```

### LR 선택 기준

- **Validation accuracy** (MNLI matched dev set) 기준 best LR 선택
- 1 epoch만 학습하므로 epoch 끝의 val_acc로 비교
- 동률이면 더 작은 LR 선택 (안정성)

### Run 수

12 models × 5 LRs = **60 runs** (각 1 epoch, 30K 데이터)

### 시간 추정

| 모델 | 30K × 1ep 예상 시간 |
|------|-------------------|
| bert-mini/small/medium | ~5분 |
| bert-base | ~12분 |
| bert-large | ~25분 |
| pythia-70m/160m | ~8분 |
| pythia-410m | ~20분 |
| pythia-1b (LoRA) | ~15분 |
| pythia-1.4b (LoRA) | ~20분 |
| pythia-2.8b (LoRA) | ~35분 |
| pythia-6.9b (LoRA) | ~60분 |

**총: ~8시간** (순차), 4 GPU 병렬 시 ~2시간

---

## Phase 2: Confidence 측정 (3 Splits)

Phase 1에서 찾은 best LR로, 3개 split 각각에 대해 학습하면서 confidence 측정.

### 학습 + Confidence 측정 설정

| 항목 | 값 |
|------|-----|
| Data | split1 (30K), split2 (30K), split3 (30K) |
| Epochs | 1 |
| LR | Phase 1에서 찾은 best LR (모델별) |
| Batch size | 16 (effective) |
| Confidence 측정 시점 | **1/3, 2/3, 3/3 지점** (step 기준) |

### Confidence 측정 방법

1 epoch = 30,000 / 16 = 1,875 steps. 측정 지점: step 625, 1250, 1875.

```python
checkpoints = [total_steps // 3, 2 * total_steps // 3, total_steps]
confidences = []

for step in training:
    # ... 학습 ...

    if step in checkpoints:
        # 현재 checkpoint에서 전체 학습 데이터(해당 split)에 대해 inference
        conf = measure_confidence(model, split_data)
        confidences.append(conf)

# 3개 checkpoint의 confidence 평균
avg_confidence = np.mean(confidences, axis=0)  # shape: (30000,)
```

### Confidence 측정 — 모델별 방법

#### BERT (Encoder): Sequence Classification
```python
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
probs = F.softmax(outputs.logits, dim=-1)  # (batch, 3)
confidence = probs[range(batch_size), labels]  # 정답 class의 확률
```

#### T5 (Enc-Dec): Text-to-Text
```python
# 입력: "mnli hypothesis: {h} premise: {p}"
# 타겟: "entailment" / "neutral" / "contradiction"
# → 타겟의 첫 토큰 확률을 confidence로 사용

# 3개 레이블 토큰의 logit 비교
label_tokens = {
    "entailment": tokenizer.encode("entailment", add_special_tokens=False)[0],
    "neutral": tokenizer.encode("neutral", add_special_tokens=False)[0],
    "contradiction": tokenizer.encode("contradiction", add_special_tokens=False)[0],
}
decoder_start = tokenizer.pad_token_id  # T5 decoder start token

outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=torch.tensor([[decoder_start]] * batch_size))
logits = outputs.logits[:, 0, :]  # 첫 번째 생성 위치

# 3개 레이블 토큰에 대해서만 softmax
label_logits = logits[:, list(label_tokens.values())]
probs = F.softmax(label_logits, dim=-1)
# labels: 0=entailment, 1=neutral, 2=contradiction
confidence = probs[range(batch_size), labels]
```

#### Pythia (Decoder): Causal LM
```python
# ⚠️ Left padding 필수
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 입력 형식: "Premise: {p}\nHypothesis: {h}\nRelation:"
# → 다음 토큰으로 "entailment" / "neutral" / "contradiction" 예측

# 3개 레이블의 첫 토큰
label_tokens = {
    "entailment": tokenizer.encode(" entailment", add_special_tokens=False)[0],
    "neutral": tokenizer.encode(" neutral", add_special_tokens=False)[0],
    "contradiction": tokenizer.encode(" contradiction", add_special_tokens=False)[0],
}

outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# 마지막 non-pad 토큰의 logit (left padding이므로 항상 마지막 위치)
logits = outputs.logits[:, -1, :]

label_logits = logits[:, list(label_tokens.values())]
probs = F.softmax(label_logits, dim=-1)
confidence = probs[range(batch_size), labels]
```

### 최종 Confidence 계산

```python
# 각 split에서 3 checkpoint 평균 → split별 confidence
for split_id in [1, 2, 3]:
    confs_at_checkpoints = []
    for ckpt in [1/3, 2/3, 3/3]:
        conf = load_confidence(model, split_id, ckpt)  # (30000,)
        confs_at_checkpoints.append(conf)
    split_conf = np.mean(confs_at_checkpoints, axis=0)  # (30000,)
    save(f"{model}_split{split_id}_avg_conf.json", split_conf)

# 90K 전체에 대한 confidence (3 split 합침, 각 예제는 1개 split에만 속함)
all_conf = {}  # index → confidence
for split_id, indices in enumerate([split1_indices, split2_indices, split3_indices]):
    split_conf = load(f"{model}_split{split_id+1}_avg_conf.json")
    for i, idx in enumerate(indices):
        all_conf[idx] = split_conf[i]
save(f"{model}_90k_avg_conf.json", all_conf)
```

### Run 수

12 models × 3 splits = **36 runs** (각 1 epoch, 30K + 3회 inference)

### 시간 추정

- 학습: Phase 1과 유사 (모델별 5~50분)
- Inference: 학습 데이터 30K에 대해 3회 → 추가 ~50% 시간
- **총: ~8시간** (순차), 4 GPU 병렬 시 ~2시간

---

## 출력 파일 구조

```
outputs/plan0/
├── data/
│   ├── split1_indices.json          # 30K 인덱스
│   ├── split2_indices.json
│   ├── split3_indices.json
│   └── all_90k_indices.json         # 90K 인덱스
├── lr_search/
│   ├── {model}_lr{lr}_results.json  # val_acc, train_loss 등
│   └── lr_selection_summary.json    # 모델별 best LR
├── confidence/
│   ├── {model}_split{1,2,3}_ckpt{1,2,3}_conf.json  # checkpoint별 raw
│   ├── {model}_split{1,2,3}_avg_conf.json           # split별 3-ckpt 평균
│   └── {model}_90k_avg_conf.json                    # 90K 전체 confidence
└── logs/
    └── {model}_split{split}_training.log
```

---

## 전체 Run 계산

| Phase | Runs | 시간 (순차) |
|-------|------|-----------|
| Phase 1: LR Search | 60 | ~5시간 |
| Phase 2: Confidence | 36 | ~9시간 |
| **총** | **96 runs** | **~14시간** |

4 GPU 병렬 시: **~3.5시간**

---

## 검증 체크리스트

학습 및 confidence 측정 후 아래 항목을 확인:

1. **Val accuracy**: 모든 모델이 chance(0.333) 이상인지 확인. 미만이면 LR 재탐색.
2. **Confidence 분포**: 모든 예제가 0.33 근처면 학습 안 된 것. 분포가 넓어야 정상.
3. **Confidence 포화**: >0.95인 예제가 50% 이상이면 변별력 부족. 30K split이므로 포화 가능성 있음 — 포화 시 사용자에게 보고.
4. **3 split 간 일관성**: 3개 split은 서로 다른 예제이므로 직접 비교 불가. 대신 모델별 confidence 분포(mean, std, >0.95 비율)가 split 간 유사한지 확인.
5. **Decoder 모델 padding**: Pythia의 left padding이 제대로 적용됐는지 확인.
6. **fp32 필수**: Pythia는 반드시 fp32로 로딩. fp16에서 logit overflow → NaN 발생 확인됨. 메모리 부족 시 bf16 사용.

---

## 주의사항

1. **Left padding (Decoder 모델)**: Pythia 계열은 반드시 `tokenizer.padding_side = "left"`. 이것을 빠뜨리면 confidence가 완전히 틀림.
2. **BERT 입력 형식**: "[CLS] premise [SEP] hypothesis [SEP]". AutoTokenizer가 자동 처리.
3. **Pythia 입력 형식**: "Premise: {p}\nHypothesis: {h}\nRelation:" — pad_token = eos_token 설정 필수.
4. **Pythia label tokenization**: "entailment"은 2개 토큰(`entail`+`ment`)으로 분리됨. 학습 시 전체 label 토큰에 loss를 주고, confidence 측정 시 첫 토큰만 사용.
5. **fp16 사용 금지**: Pythia의 logit scale이 크므로(~800) fp16에서 수치 overflow 발생. 반드시 fp32, OOM 시 bf16.
5. **Eval 시 gradient 끄기**: `torch.no_grad()` 또는 `model.eval()` 필수. Confidence 측정 시 메모리 절약.
6. **Checkpoint 저장 vs 현장 측정**: checkpoint를 디스크에 저장하고 나중에 inference하면 디스크 사용량이 큼 (모델 크기 × 체크포인트 수). 가능하면 학습 중 현장에서 confidence를 측정하고 모델은 저장하지 않는 방식 권장.
7. **Random seed**: split 생성과 학습 seed를 구분. Split은 seed=42로 고정, 학습은 seed=1 (LR search) / 따로 지정 (Phase 2).
8. **MNLI dataset 로딩**: `load_dataset("glue", "mnli")`의 `train` split 사용. Matched dev는 eval용.

---

## 최종 실험 결과

### T5 계열 제외 결정

T5는 pretrain 시 MNLI를 multi-task로 학습하여, fine-tuning 시작 시점부터 confidence가 포화됨 (>0.95가 69%). 상위 30% 쉬운 예제의 변별력이 부족하여 제외.

### Phase 1: LR Search 결과 (전체 모델)

| 모델 | Params | Tuning | Best LR | Val Acc |
|------|--------|--------|---------|---------|
| bert-mini | 11M | full | 1e-4 | 0.612 |
| bert-small | 29M | full | 1e-4 | 0.657 |
| bert-medium | 42M | full | 5e-5 | 0.698 |
| bert-base | 110M | full | 5e-5 | 0.761 |
| bert-large | 335M | full | 3e-5 | 0.795 |
| pythia-70m | 70M | full | 3e-5 | 0.641 |
| pythia-160m | 160M | full | 1e-5 | 0.713 |
| pythia-410m | 410M | full | 1e-5 | 0.799 |
| pythia-1b | 1B | LoRA r=32 | 3e-4 | 0.797 |
| pythia-1.4b | 1.4B | LoRA r=32 | 3e-4 | 0.829 |
| pythia-2.8b | 2.8B | LoRA r=32 | 3e-4 | 0.848 |
| pythia-6.9b | 6.9B | LoRA r=32 | 1e-4 | 0.854 |

> LR 경향: BERT는 모델이 커질수록 작은 LR 선호 (1e-4 → 3e-5). Pythia full도 동일 (3e-5 → 1e-5). LoRA 모델은 3e-4가 최적이나, 6.9B는 1e-4로 더 보수적.

### Phase 2: 90K Confidence (전체 모델)

| 모델 | Params | 90K Mean | Std | >0.95 | <0.40 |
|------|--------|----------|------|-------|-------|
| bert-mini | 11M | 0.498 | 0.208 | 0.0% | 34.9% |
| bert-small | 29M | 0.572 | 0.240 | 0.1% | 26.8% |
| bert-medium | 42M | 0.615 | 0.253 | 0.8% | 23.1% |
| bert-base | 110M | 0.697 | 0.251 | 10.5% | 15.6% |
| **bert-large** | **335M** | **0.615** | **0.282** | **12.1%** | **41.2%** |
| pythia-70m | 70M | 0.546 | 0.244 | 1.6% | 31.2% |
| pythia-160m | 160M | 0.626 | 0.263 | 7.3% | 22.7% |
| pythia-410m | 410M | 0.767 | 0.240 | 23.9% | 10.7% |
| pythia-1b | 1B | 0.732 | 0.252 | 18.9% | 13.6% |
| pythia-1.4b | 1.4B | 0.774 | 0.242 | 27.0% | 10.5% |
| pythia-2.8b | 2.8B | 0.815 | 0.257 | 49.2% | 10.3% |
| pythia-6.9b | 6.9B | 0.851 | 0.264 | 64.9% | 9.8% |

### 주요 관찰

1. **모델 크기 ↑ → val_acc ↑, confidence mean ↑, >0.95 ↑**: 예상대로 큰 모델이 더 자신 있게 예측.
2. **bert-large 특이 현상**: val_acc(0.795)는 높은데 confidence mean(0.615)이 낮고 <0.40이 41.2%. 높은 accuracy에도 불구하고 confidence 분포가 넓어 **변별력이 가장 좋음**.
3. **포화 경계**: pythia-2.8b(49.2%), pythia-6.9b(64.9%)는 >0.95 비율이 높아 상위 구분이 어려움.
4. **proxy confidence 최적 모델**: bert-base~bert-large, pythia-160m~pythia-410m이 포화 없이 변별력 좋음.
5. **BERT vs Pythia 상관**: bert-base와 pythia-160m의 Pearson r = 0.7065. 두 계열이 유사한 난이도 판단.

### Lexical Overlap 분석 (BERT-base 기준)

BERT confidence 상위 30% (쉬운 예제, conf >= 0.8825) 중 lexical overlap >= 0.6: **29.6%** (7,991/27,000).
- entailment: 57.8%가 high overlap → shallow heuristic 의존 패턴 확인
- neutral: 5.8%, contradiction: 16.0%

### 기술 노트

- **fp16 사용 금지**: Pythia는 logit scale이 크므로(~800) fp16에서 overflow → NaN. fp32 필수, OOM 시 bf16.
- **gradient checkpointing**: pythia-2.8b, 6.9b는 bf16 + gradient checkpointing + batch_size 4로 실행.
- **Pythia label tokenization**: "entailment"은 2개 토큰으로 분리. 학습 시 전체 label 토큰에 loss, confidence 측정 시 첫 토큰만 사용.
