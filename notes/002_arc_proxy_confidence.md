# Plan 2: ARC Proxy Confidence 측정

## 목적

ARC (AI2 Reasoning Challenge) 데이터셋에 대해 proxy 모델을 학습하고 confidence를 측정. Plan 0 (MNLI)과 동일한 프레임워크를 ARC에 적용.

## 데이터셋 정보

```
AI2 ARC (allenai/ai2_arc)
├── ARC-Easy:      2,251 train / 570 val / 2,376 test
├── ARC-Challenge: 1,119 train / 299 val / 1,172 test
```

**학습**: ARC 전체 train set 사용 (Easy 2,251 + Challenge 1,119 = **3,370개**)
**평가**: ARC overall dev (869개) + ARC-Challenge dev (299개)

## Feasibility Check 결과

### 결과 요약

| 모델 | 방식 | Best LR | Challenge Acc | Overall Acc | 판정 |
|------|------|---------|--------------|-------------|------|
| bert-base | BertForMultipleChoice | 3e-5 | 0.401 | 0.482 | ✅ |
| bert-large | BertForMultipleChoice | 1e-5 | 0.448 | 0.543 | ✅ |
| pythia-410m | A/B/C/D token (full) | 2e-5 | 0.314 | 0.290 | ❌ overfitting 심함 |
| pythia-1b | A/B/C/D token (LoRA) | 5e-5 | 0.334 | 0.443 | ✅ overall 기준 |

### 결론

- **BERT**: BertForMultipleChoice로 잘 작동. 모든 사이즈 포함 가능.
- **Pythia full fine-tune (≤410m)**: ARC에서 overfitting 심함 (loss 0.0으로 떨어지나 val_acc 향상 없음). **제외**.
- **Pythia LoRA (≥1b)**: overall 기준으로 학습 가능. pythia-1b overall 0.443으로 bert-base에 근접. **포함**.

---

## Phase 1: LR Search (확장)

### 모델 목록 (9개)

| 계열 | 모델 | Params | Tuning | LR Search 상태 |
|------|------|--------|--------|---------------|
| **Encoder** | bert-mini | 11M | full | 새로 실행 |
| | bert-small | 29M | full | 새로 실행 |
| | bert-medium | 42M | full | 새로 실행 |
| | bert-base | 110M | full | ✅ 완료 (best=3e-5) |
| | bert-large | 335M | full | ✅ 완료 (best=1e-5) |
| **Decoder** | pythia-1b | 1B | LoRA r=32 | ✅ 완료 (best=5e-5) |
| | pythia-1.4b | 1.4B | LoRA r=32 | 새로 실행 |
| | pythia-2.8b | 2.8B | LoRA r=32 | 새로 실행 (bf16) |
| | pythia-6.9b | 6.9B | LoRA r=32 | 새로 실행 (bf16) |

### 입력 형식

**BERT**: `BertForMultipleChoice` — [CLS] question [SEP] choice [SEP] × 4, CE loss over 4 logits.

**Pythia**: 모든 선택지를 하나의 prompt에 넣고, 마지막 위치에서 A/B/C/D 토큰 확률로 예측.
```
Question: {question}
A) {choice_A}
B) {choice_B}
C) {choice_C}
D) {choice_D}
Answer:
```
→ CE loss over P(" A"), P(" B"), P(" C"), P(" D") at last position.

### LR 후보

| 모델 | LR 후보 | 근거 |
|------|---------|------|
| bert-mini/small/medium | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 | MNLI 소형 모델 범위 |
| pythia-1.4b (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | MNLI LoRA 범위 |
| pythia-2.8b (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | MNLI LoRA 범위 |
| pythia-6.9b (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | MNLI LoRA 범위 |

### 학습 설정

| 항목 | 값 |
|------|-----|
| Data | ARC 전체 train (3,370) |
| Epochs | 10 |
| Batch size | 8 (큰 모델은 4) |
| Eval | 매 epoch, ARC overall dev + Challenge dev |
| LR 선택 | best overall_acc epoch 기준 |

### Run 수

새로 실행: 6 models × 5 LRs = **30 runs** (각 10 epochs)

---

## Phase 2: Confidence 측정

LR search 완료 후, 전체 9개 모델에 대해 confidence 측정.

ARC train이 3,370개로 작으므로 MNLI처럼 3 split은 불가 → **전체 train으로 학습, seed 3개로 robustness 확인**.

### 설정

| 항목 | 값 |
|------|-----|
| Data | ARC 전체 train (3,370) |
| Seeds | 1, 2, 3 |
| Epochs | 10 |
| Confidence 측정 시점 | 매 epoch (10 checkpoints) |
| Best epoch 기준 | overall_acc가 가장 높은 epoch의 confidence 사용 |

### Confidence 측정 방법

**BERT**: softmax(4개 choice logits) → 정답 choice의 확률.
**Pythia**: softmax(A/B/C/D token logits) → 정답 choice의 확률.

### Run 수

9 models × 3 seeds = **27 runs** (각 10 epochs)

---

## 전체 Run 계산

| Phase | Runs |
|-------|------|
| Phase 1: LR Search (신규) | 30 |
| Phase 2: Confidence | 27 |
| **총** | **57 runs** |

---

## 주의사항

1. **Pythia full fine-tune 제외**: ≤410m은 ARC에서 overfitting 심함. LoRA(≥1b)만 사용.
2. **fp32 기본, OOM 시 bf16**: pythia-2.8b, 6.9b는 bf16 + gradient checkpointing.
3. **데이터 크기**: 3,370개로 작음. 10 epoch에서도 overfitting 가능 → best epoch 기준 평가.
4. **선택지 수**: 대부분 4개, 소수 3/5개 존재. MAX_CHOICES=5로 padding 처리.

---

## 최종 실험 결과

### Phase 1: LR Search

| 모델 | Params | Tuning | Best LR | Best Overall | Best Challenge |
|------|--------|--------|---------|-------------|----------------|
| bert-mini | 11M | full | 5e-5 | 0.398 | 0.291 |
| bert-small | 29M | full | 5e-5 | 0.427 | 0.337 |
| bert-medium | 42M | full | 5e-5 | 0.438 | 0.362 |
| bert-base | 110M | full | 3e-5 | 0.492 | 0.401 |
| bert-large | 335M | full | 1e-5 | 0.543 | 0.448 |
| pythia-1b | 1B | LoRA | 1e-4 | 0.473 | 0.343 |
| pythia-1.4b | 1.4B | LoRA | 1e-4 | 0.520 | 0.371 |
| pythia-2.8b | 2.8B | LoRA | 1e-4 | 0.593 | 0.391 |
| pythia-6.9b | 6.9B | LoRA | 5e-5 | 0.603 | 0.465 |

### Phase 2: Confidence (3 seeds 평균)

| 모델 | Overall (3seed) | Challenge (3seed) | 안정성 |
|------|----------------|-------------------|--------|
| bert-mini | 0.390 ± 0.007 | 0.291 ± 0.010 | ✅ 안정 |
| bert-small | 0.422 ± 0.004 | 0.337 ± 0.004 | ✅ 안정 |
| bert-medium | 0.442 ± 0.012 | 0.362 ± 0.014 | ✅ 안정 |
| bert-base | 0.471 ± 0.023 | 0.352 ± 0.010 | ✅ 안정 |
| bert-large | 0.516 ± 0.001 | 0.406 ± 0.010 | ✅ 매우 안정 |
| pythia-1b | 0.479 ± 0.005 | 0.343 ± 0.020 | ✅ 안정 |
| pythia-1.4b | 0.450 ± 0.123 | 0.338 ± 0.057 | ❌ 불안정 |
| pythia-2.8b | 0.493 ± 0.154 | 0.391 ± 0.105 | ❌ 불안정 |
| pythia-6.9b | 0.500 ± 0.153 | 0.401 ± 0.076 | ❌ 불안정 |

### 주요 관찰

1. **BERT가 안정적**: seed 분산이 0.001~0.023으로 매우 작음. 모델 크기 비례 성능 향상.
2. **Pythia LoRA 불안정**: seed 분산 0.05~0.15. LoRA + 작은 데이터(3.3K)에서 seed 민감도가 높음.
3. **bert-large가 최고 성능 + 최고 안정성**: overall 0.516 ± 0.001.
4. **Confidence 포화**: best epoch 시점 train confidence >0.95가 84~99%. 데이터가 작아서 모델이 train set 암기.
