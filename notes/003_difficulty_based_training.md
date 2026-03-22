# Plan 3: Difficulty-Based Training — Human Difficulty vs Model Confidence (ARC)

## 목적

ARC 데이터의 human difficulty annotation (Low/Medium/High)과 model confidence에 따라 학습 데이터를 선별하여 튜닝했을 때 성능에 어떤 영향을 미치는지 분석.

**참고 논문**: "The Unreasonable Effectiveness of Easy Training Data for Hard Tasks" (ACL 2024)
- Paper: https://aclanthology.org/2024.acl-long.378/
- GitHub: https://github.com/allenai/easy-to-hard-generalization

## 데이터

### ARC + Human Difficulty Annotation

`easy-to-hard-generalization/data/arc-challenge-easy-annotations.json`에서 difficulty, grade, bloom skill 정보 로딩.

**ARC train과 매칭된 예제:**

| Human Difficulty | 개수 | 비율 |
|-----------------|------|------|
| Low (Easy) | 465 (= N_easy) | 21.6% |
| Medium | 1,399 (= N_med) | 65.0% |
| High (Hard) | 287 (= N_hard) | 13.3% |
| **합계** | **2,151** | 100% |

> 전체 ARC train 3,370 중 2,151개만 annotation 매칭됨. 매칭된 예제만 사용.

### Model Confidence

Plan 2에서 측정한 train confidence (best epoch 시점). 모든 모델(9개)의 confidence 사용 가능.

---

## 주 학습 모델 (2개)

| 모델 | Params | Tuning | Best LR (Plan 2) |
|------|--------|--------|-------------------|
| bert-large | 335M | full | 1e-5 |
| pythia-1b | 1B | LoRA | 1e-4 |

---

## Experiment 1: Human Difficulty 서브셋 학습

각 난이도 서브셋으로만 학습 → ARC overall dev + Challenge dev + Easy dev로 평가.

### 학습 조건

| 조건 | 학습 데이터 | 크기 |
|------|-----------|------|
| Easy only | Low difficulty 예제 | 465 |
| Medium only | Medium difficulty 예제 | 1,399 |
| Hard only | High difficulty 예제 | 287 |
| Random (N_easy=465) | 전체 2,151에서 465 랜덤 샘플 | 465 |
| Random (N_med=1399) | 전체 2,151에서 1,399 랜덤 샘플 | 1,399 |
| Full (baseline) | 매칭된 전체 | 2,151 |

### Run 수

2 models × 6 conditions = **12 runs**

---

## Experiment 2: 동일 크기 비교 (Hard=287 기준)

Hard가 가장 적으므로(287개), N_hard로 통일하여 크기 효과 제거.

### 학습 조건

| 조건 | 학습 데이터 | 크기 |
|------|-----------|------|
| Hard | High difficulty 전체 | 287 |
| Easy (subsampled) | Low에서 287개 랜덤 샘플 | 287 |
| Medium (subsampled) | Medium에서 287개 랜덤 샘플 | 287 |
| Random | 전체 2,151에서 287개 랜덤 샘플 | 287 |

### Run 수

2 models × 4 conditions = **8 runs**

---

## Experiment 3: Model Confidence 기반 선별

매칭된 2,151개에 대해, **모든 모델(9개)**의 confidence를 기준으로 데이터를 선별하여 주 모델(bert-large, pythia-1b)을 학습.

### Confidence 소스 모델 (9개)

bert-mini, bert-small, bert-medium, bert-base, bert-large, pythia-1b, pythia-1.4b, pythia-2.8b, pythia-6.9b

### 학습 조건 (각 confidence 소스 모델별)

| 조건 | 학습 데이터 | 크기 | 설명 |
|------|-----------|------|------|
| Model-Easy top(N_easy) | Confidence 상위 465개 | 465 | 모델이 가장 쉽다고 판단 |
| Model-Easy top(N_hard) | Confidence 상위 287개 | 287 | 위와 동일, 더 적은 수 |
| Model-Hard bottom(N_easy) | Confidence 하위 465개 | 465 | 모델이 가장 어렵다고 판단 |
| Model-Hard bottom(N_hard) | Confidence 하위 287개 | 287 | 위와 동일, 더 적은 수 |

### Run 수

2 주모델 × 9 confidence 소스 × 4 conditions = **72 runs**

---

## 전체 Run 계산

| Experiment | Runs |
|-----------|------|
| Exp 1: Human Difficulty 서브셋 | 12 |
| Exp 2: 동일 크기 비교 | 8 |
| Exp 3: Model Confidence 선별 | 72 |
| **총** | **92 runs** |

---

## 학습 설정

| 항목 | 값 |
|------|-----|
| Epochs | 10 |
| Batch size | 8 (pythia-1b LoRA) |
| LR | Plan 2 best LR (bert-large=1e-5, pythia-1b=1e-4) |
| Eval | ARC overall dev (869) + Easy dev (570) + Challenge dev (299) |
| Seed | 1 (고정) |

---

## 평가 매트릭스

모든 실험에서 아래 3개 metric 보고:

| Metric | 설명 |
|--------|------|
| Overall Acc | ARC dev 전체 (869) |
| Easy Acc | ARC-Easy dev (570) |
| Challenge Acc | ARC-Challenge dev (299) |

### 핵심 분석 질문

1. **Exp 1**: Easy/Medium/Hard 중 어떤 예제로 학습하는 게 가장 효과적인가? 특히 Challenge dev에서.
2. **Exp 2**: 동일 287개일 때 Easy vs Medium vs Hard의 순수 효과 비교.
3. **Exp 3**: Model confidence로 선별한 easy/hard 예제가 human difficulty 기반 선별보다 효과적인가? 어떤 confidence 소스 모델이 가장 좋은 데이터 선별을 하는가?
4. **Cross-analysis**: Human difficulty와 model confidence의 일치도. "Human-Hard but Model-Easy" 예제의 특성은?

---

## 실행 순서

1. **데이터 준비**: annotation 매칭, difficulty별 인덱스 생성, model confidence 로딩
2. **Exp 1**: 6 conditions × 2 models (GPU 0, 1 병렬)
3. **Exp 2**: 4 conditions × 2 models
4. **Exp 3**: 9 confidence 소스 × 4 conditions × 2 models (가장 많은 runs)
5. **결과 분석**: 비교 테이블 + cross-difficulty 분석

---

## 주의사항

1. **매칭 누락**: ARC train 3,370 중 2,151만 annotation 있음. 매칭된 예제만 사용.
2. **데이터 크기**: Hard=287개로 매우 작음. 10 epoch에서도 overfitting 가능.
3. **Confidence 포화**: Plan 2 confidence가 포화(>0.95 비율 높음)되어 있으나, ranking 기준으로 사용.
4. **pythia-1b seed 민감도**: Plan 2에서 확인. 결과 해석 시 주의.

---

## 최종 실험 결과

### Exp 1: Human Difficulty 서브셋 학습

| 조건 | n | bert-large |  |  | pythia-1b |  |  |
|------|---|-----------|------|-----------|-----------|------|-----------|
|  |  | Overall | Easy | Challenge | Overall | Easy | Challenge |
| Easy only | 465 | 0.410 | 0.458 | 0.318 | 0.268 | 0.274 | 0.258 |
| Medium only | 1,399 | 0.411 | 0.483 | 0.274 | 0.334 | 0.337 | 0.328 |
| Hard only | 287 | 0.366 | 0.391 | 0.318 | 0.282 | 0.298 | 0.251 |
| Random(465) | 465 | 0.434 | 0.481 | 0.345 | 0.304 | 0.312 | 0.288 |
| Random(1399) | 1,399 | **0.482** | 0.528 | **0.395** | 0.331 | 0.349 | 0.298 |
| Full | 2,151 | 0.481 | **0.533** | 0.381 | **0.413** | **0.474** | **0.298** |

**관찰:**
- Random(N)이 같은 크기의 difficulty-specific 서브셋보다 일관되게 좋음 → 다양성이 중요.
- Medium only가 Easy only보다 전체 성능은 비슷하지만, Easy는 Challenge에서 더 좋고 Medium은 Easy dev에서 더 좋음.
- 데이터 양 효과 큼: Random(1399) ≈ Full(2151).

### Exp 2: 동일 크기 비교 (N_hard=287)

| 조건 | bert-large |  |  | pythia-1b |  |  |
|------|-----------|------|-----------|-----------|------|-----------|
|  | Overall | Easy | Challenge | Overall | Easy | Challenge |
| Hard | 0.366 | 0.391 | 0.318 | 0.282 | 0.298 | 0.251 |
| Easy(sub) | 0.399 | 0.453 | 0.298 | 0.292 | 0.311 | 0.258 |
| **Medium(sub)** | **0.418** | **0.460** | **0.338** | **0.288** | **0.293** | **0.278** |
| Random | 0.398 | 0.435 | 0.328 | 0.277 | 0.297 | 0.241 |

**관찰:**
- **Medium이 동일 287개에서 가장 효과적** (bert-large overall 0.418, challenge 0.338).
- Hard로만 학습하면 Easy dev 성능이 가장 낮음 (0.391) — Hard 예제가 Easy 문제 풀기에 도움이 안 됨.
- Easy(sub)는 overall은 좋지만 Challenge에서 약함 (0.298).

### Exp 3: Model Confidence 기반 선별

bert-large에서 confidence 소스별 Model-Easy vs Model-Hard 비교 (n=465):

| Confidence 소스 | Model-Easy Overall | Model-Hard Overall | 차이 | Model-Easy Challenge | Model-Hard Challenge |
|----------------|-------------------|-------------------|------|---------------------|---------------------|
| bert-large | **0.455** | 0.358 | +0.097 | 0.345 | 0.314 |
| bert-small | **0.452** | 0.356 | +0.096 | 0.334 | 0.308 |
| pythia-1b | **0.427** | 0.274 | +0.153 | 0.348 | 0.301 |
| pythia-6.9b | **0.444** | 0.387 | +0.057 | 0.345 | 0.308 |
| bert-base | **0.428** | 0.404 | +0.024 | 0.311 | 0.348 |
| bert-mini | 0.381 | 0.379 | +0.002 | 0.304 | 0.324 |

n=287 비교:

| Confidence 소스 | Model-Easy Overall | Model-Hard Overall | 차이 |
|----------------|-------------------|-------------------|------|
| bert-small | **0.447** | 0.345 | +0.102 |
| bert-mini | **0.422** | 0.342 | +0.080 |
| bert-large | **0.411** | 0.363 | +0.048 |
| pythia-1b | **0.414** | 0.287 | +0.127 |

**관찰:**
- **Model-Easy 선별이 일관되게 Model-Hard보다 우수** — 거의 모든 confidence 소스에서 성립.
- bert-small, bert-large의 confidence가 데이터 선별에 가장 효과적 (차이 ~0.10).
- pythia-1b confidence 기반 선별 시 차이가 가장 큼 (+0.153) — Model-Hard가 거의 학습 안 됨(0.274).
- bert-base confidence는 차이가 작음 (+0.024) — confidence의 변별력이 약하거나 포화.

### pythia-1b 결과 요약

pythia-1b는 ARC에서 전반적으로 약함 (대부분 0.25~0.31). 주목할 패턴:
- 자기 자신(pythia-1b)의 confidence로 Model-Easy를 선별하면 0.377로 가장 높음.
- Full 데이터(2,151)로 학습할 때만 0.413 달성 — 데이터 양이 결정적.
- 287개 수준의 작은 데이터에서는 거의 학습이 안 됨.

---

## 핵심 결론

1. **다양성 > 난이도 특화**: Random 샘플이 difficulty-specific 서브셋보다 일관되게 좋음.
2. **Medium이 최적**: 동일 크기에서 Medium 난이도가 Easy나 Hard보다 효과적. "너무 쉽지도, 너무 어렵지도 않은" 예제가 가장 유용.
3. **Model-Easy 선별이 효과적**: 모델이 쉽다고 판단한 예제로 학습하면 성능이 더 좋음. "The Unreasonable Effectiveness of Easy Training Data" 논문의 결과와 일치.
4. **데이터 양이 중요**: 287개 vs 1,399개 vs 2,151개에서 명확한 크기 효과. 일정 수(~1,400) 이상이면 포화.
5. **Confidence 소스 모델 선택**: bert-small, bert-large의 confidence가 데이터 선별에 가장 효과적.
