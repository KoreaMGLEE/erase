# Plan 002: ARC Proxy Confidence 측정

> 📋 세부 실험 과정 및 디버깅 기록 (seed 불안정성, LR 탐색 과정 등)은 [001a_mnli_experiment_log.md](001a_mnli_experiment_log.md) 참조.

## 목적

ARC (AI2 Reasoning Challenge) 데이터셋에 대해 다양한 모델의 proxy confidence를 측정.
MNLI (Plan 001)과 동일한 모델 세트를 사용하여 cross-task 비교 가능.

## 데이터

```
ARC Train: Easy=2,251 / Challenge=1,119 / Total=3,370
ARC Val:   Easy=570   / Challenge=299   / Total=869
```

- 학습: ARC 전체 train (3,370)
- 평가: ARC overall dev (869)
- Confidence: 3 seeds (seed 1, 2, 3) — 데이터가 작아서 MNLI의 3 splits 대신 seed로 robustness 확인
- **5 epochs** — 데이터가 작으므로 multi-epoch 필요
- **Confidence**: best epoch 이하 checkpoint 평균 (overfitting 방지)

---

## 모델 (17개)

Plan 001과 동일한 3계열. t5-xxl, pythia-12b는 미포함.

### Encoder: BERT (5종)

- BertForMultipleChoice 사용
- Optimizer: AdamW + linear decay + warmup 6%
- 입력: `[CLS] question [SEP] choice [SEP]` × 4 choices

### Enc-Dec: T5 v1.1 (4종)

- **Sentinel 프롬프트** (MNLI와 동일 방식):
  - Encoder: `"Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nanswer: <extra_id_0>"`
  - Decoder target: `"<extra_id_0> A"` (or B/C/D)
- Optimizer: Adafactor + constant warmup 6%
- Eval: decoder position 1에서 A/B/C/D logits 비교

### Decoder: Pythia (8종)

- 입력: `"Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:"`
- 학습: 마지막 위치에서 A/B/C/D 토큰 CE loss
- ≤410m full fine-tune, ≥1b LoRA
- **주의**: Pythia full fine-tune (14m~410m)은 ARC에서 random 수준 (~0.27). LoRA(1b+)부터 유의미.

---

## LR 후보

| 모델 그룹 | LRs |
|----------|-----|
| BERT small (<110M) | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 |
| BERT large (≥110M) | 1e-5, 2e-5, 3e-5, 5e-5, 1e-4 |
| T5 v1.1 small/base | 3e-5, 5e-5, 1e-4, 3e-4 |
| T5 v1.1 large/xl | 3e-5, 5e-5, 1e-4 |
| Pythia small (<110M) | 3e-5, 5e-5, 1e-4, 3e-4, 1e-3 |
| Pythia medium (≥110M) | 1e-5, 2e-5, 3e-5, 5e-5, 1e-4 |
| Pythia LoRA (≥1B) | 5e-5, 1e-4, 3e-4, 1e-3 |

---

## ARC-specific 인사이트

### 1. Pythia full fine-tune 실패
Pythia 14m~410m은 ARC에서 학습이 안 됨 (~0.27, random). 이유:
- ARC는 multiple choice (4-way) → 단순 next-token prediction으로는 학습 어려움
- Full fine-tune + 작은 데이터(3.3K)에서 overfitting without generalization
- LoRA (1b+)부터 유의미한 성능 → regularization 효과

### 2. T5 v1.1 small/base도 random
77M, 248M 모델은 ARC를 못 배움. Large(783M)부터 유의미, xl(3B)에서 0.784로 최고.

### 3. 5 epoch + linear decay 불안정성
10 epoch → 5 epoch 변경 시:
- Warmup이 절반으로 줄어듦
- Linear decay가 2배 빠름
- 큰 모델(bert-large, t5-large)에서 seed 불안정성 유발
- **해결**: 더 작은 LR 사용 (bert-large: 3e-5 → 1e-5에서 안정화)

### 4. LoRA seed 민감도
Pythia LoRA + 작은 데이터에서 seed에 따라 학습 실패 발생 (pythia-6.9b seed3: 0.258).
LoRA 초기화가 "dead zone"에 빠지면 5 epoch 내 회복 못 함.

---

## 전체 결과

### LR Search (5 epochs)

| 모델 | Params | Family | Best LR | Best Acc |
|------|--------|--------|---------|---------|
| bert-mini | 11M | BERT | 5e-5 | 0.387 |
| bert-small | 29M | BERT | 5e-5 | 0.414 |
| bert-medium | 42M | BERT | 5e-5 | 0.433 |
| bert-base | 110M | BERT | 5e-5 | 0.486 |
| bert-large | 335M | BERT | 3e-5 | 0.517 |
| t5-v1_1-small | 77M | T5 | 3e-5 | 0.270 |
| t5-v1_1-base | 248M | T5 | 1e-4 | 0.281 |
| t5-v1_1-large | 783M | T5 | 5e-5 | 0.447 |
| **t5-v1_1-xl** | **3B** | **T5** | **1e-4** | **0.784** |
| **t5-v1_1-xxl** | **11B** | **T5** | **1e-4** | **0.852** |
| pythia-14m | 14M | Pythia | 1e-3 | 0.270 |
| pythia-31m | 31M | Pythia | 1e-4 | 0.297 |
| pythia-70m | 70M | Pythia | 3e-4 | 0.280 |
| pythia-160m | 160M | Pythia | 2e-5 | 0.277 |
| pythia-410m | 410M | Pythia | 1e-5 | 0.280 |
| pythia-1b | 1B | Pythia | 1e-4 | 0.473 |
| pythia-1.4b | 1.4B | Pythia | 5e-5 | 0.524 |
| pythia-2.8b | 2.8B | Pythia | 5e-5 | 0.592 |
| pythia-6.9b | 6.9B | Pythia | 5e-5 | 0.604 |
| pythia-12b | 12B | Pythia | 5e-5 | 0.615 |

### Confidence (3 seeds 평균)

| 모델 | Mean ± Std | 안정성 |
|------|-----------|--------|
| bert-mini | 0.375 ± 0.017 | ✅ |
| bert-small | 0.410 ± 0.004 | ✅ |
| bert-medium | 0.442 ± 0.006 | ✅ |
| bert-base | 0.471 ± 0.014 | ✅ |
| bert-large | 0.510 ± 0.005 (lr=1e-5) | ✅ (lr=3e-5는 불안정) |
| t5-v1_1-small | 0.267 ± 0.003 | ✅ (random) |
| t5-v1_1-base | 0.269 ± 0.009 | ✅ (random) |
| t5-v1_1-large | 0.561 ± 0.042 (lr=3e-5, 3/4 seed 성공) | ⚠️ seed-sensitive |
| **t5-v1_1-xl** | **0.766 ± 0.014** | **✅** |
| pythia-14m~410m | 0.26~0.28 | ✅ (random) |
| pythia-1b | 0.472 ± 0.012 | ✅ |
| pythia-1.4b | 0.520 ± 0.011 | ✅ |
| pythia-2.8b | 0.566 ± 0.027 | ✅ |
| pythia-6.9b | 0.606 ± 0.003 (3/4 seed) | ✅ (seed3 제외) |

### 불안정 모델 추가 탐색

**bert-large**: lr=1e-5에서 안정화 (0.505~0.517, 3/3 seed 성공)

| LR | seed1 | seed2 | seed3 |
|------|-------|-------|-------|
| 1e-5 | 0.509 | 0.505 | 0.517 |
| 2e-5 | 0.459 | 0.504 | 0.522 |
| 3e-5 | 0.517 | 0.222 | 0.257 |

**t5-v1_1-large**: lr=3e-5가 최선 (4 seed 중 3개 성공). 성공 평균 0.561 ± 0.042.

| LR | seed1 | seed2 | seed3 | seed4 |
|------|-------|-------|-------|-------|
| 3e-5 | 0.259 | **0.621** | **0.521** | **0.541** |
| 5e-5 | 0.447 | 0.260 | 0.261 | — |
| 1e-4 | 0.262 | **0.585** | 0.256 | — |
| 2e-4 | 0.267 | 0.259 | 0.256 | — |

lr=2e-4는 전부 실패. lr=1e-4는 seed2만 성공. t5-v1_1-large는 ARC에서 매우 seed-sensitive.

**pythia-6.9b**: 3/4 seed 성공 (seed3만 실패)

| seed1 | seed2 | seed3 | seed4 |
|-------|-------|-------|-------|
| 0.602 | 0.609 | 0.258 | 0.608 |

### Per-example Confidence (train set, best epoch 이하 평균)

19개 전체 모델에 대해 측정 완료. 실패 seed 제외, 성공 seed 3개 사용.

| 모델 | Params | Seeds | Val Acc (mean) | Train Conf (mean) |
|------|--------|-------|----------------|-------------------|
| bert-mini | 11M | 1,2,3 | 0.377 ± 0.016 | 0.379 ± 0.010 |
| bert-small | 29M | 1,2,3 | 0.402 ± 0.006 | 0.577 ± 0.014 |
| bert-medium | 42M | 1,2,3 | 0.438 ± 0.006 | 0.674 ± 0.008 |
| bert-base | 110M | 1,2,3 | 0.486 ± 0.016 | 0.754 ± 0.013 |
| bert-large | 335M | 1,2,3 | 0.513 ± 0.005 | 0.609 ± 0.054 |
| t5-v1_1-small | 77M | 1,2,3 | 0.265 ± 0.004 | 0.249 ± 0.001 |
| t5-v1_1-base | 248M | 1,2,3 | 0.265 ± 0.004 | 0.251 ± 0.002 |
| t5-v1_1-large | 783M | 2,3,4 | 0.576 ± 0.023 | 0.372 ± 0.031 |
| t5-v1_1-xl | 3B | 1,2,3 | 0.748 ± 0.005 | 0.598 ± 0.019 |
| **t5-v1_1-xxl** | **11B** | **1,2,3** | **0.846 ± 0.005** | **0.870 ± 0.015** |
| pythia-14m | 14M | 1,2,3 | 0.264 ± 0.004 | 0.251 ± 0.000 |
| pythia-31m | 31M | 1,2,3 | 0.278 ± 0.007 | 0.313 ± 0.034 |
| pythia-70m | 70M | 1,2,3 | 0.262 ± 0.008 | 0.298 ± 0.041 |
| pythia-160m | 160M | 1,2,3 | 0.267 ± 0.007 | 0.293 ± 0.054 |
| pythia-410m | 410M | 1,3,4 | 0.288 ± 0.008 | 0.480 ± 0.062 |
| pythia-1b | 1B | 1,2,3 | 0.464 ± 0.018 | 0.428 ± 0.033 |
| pythia-1.4b | 1.4B | 1,2,3 | 0.517 ± 0.013 | 0.463 ± 0.010 |
| pythia-2.8b | 2.8B | 1,2,3 | 0.551 ± 0.013 | 0.450 ± 0.002 |
| pythia-6.9b | 6.9B | 1,4,5 | 0.594 ± 0.004 | 0.524 ± 0.072 |
| pythia-12b | 12B | 1,3,4 | 0.650 ± 0.007 | 0.621 ± 0.035 |

**참고**:
- t5-large seed1 실패 → seed 2,3,4 사용.
- pythia-410m seed2 이상치 → seed 1,3,4 사용.
- pythia-6.9b seed2,3 실패 → seed 1,4,5 사용.
- pythia-12b seed2 실패 → seed 1,3,4 사용.
- t5-xxl 3 seeds 전부 안정적 (0.838~0.853).

### 주요 관찰

1. **t5-v1_1-xxl이 ARC 최강**: 0.846 ± 0.005 (t5-xl 0.748 대비 큰 향상).
2. **BERT 안정적**: std < 0.015. bert-large는 lr=1e-5에서 안정 (0.510 ± 0.005).
3. **T5/Pythia small은 ARC에서 random**: 모델이 작으면 ARC를 못 배움.
4. **Pythia LoRA(1b+)부터 유의미**: full fine-tune은 overfitting.
5. **큰 모델 + 작은 데이터 = seed 불안정**: 5 epoch에서 warmup 짧고 decay 빨라 민감.
6. **ARC Jaccard 전체적으로 낮음**: MNLI 대비 모델 간 "쉬운 예제" 합의가 약함. Random 모델(t5-small/base, pythia-70m/410m) 간 Jaccard ~0.15.

---

## 분석 결과

### Figure 1: Pairwise Jaccard Heatmap (9×9)
`fig1_jaccard_heatmap_9x9_arc.png` — seed별 평균 ± std.
- ARC Jaccard 전체적으로 낮음 (0.13~0.41 vs MNLI 0.30~0.54)
- BERT 내부 일관성 (0.40~0.41), 큰 모델끼리 cross-family 합의 (0.34~0.37)
- Random 모델 간 Jaccard ~0.15

### Figure 2: Model↔인간 난이도
`easy-to-hard-generalization` annotation 활용 (2,151개 매칭).

**(a) Continuous Spearman** (`fig2_arc_continuous.png`):
- Difficulty: BERT ~0.09 안정, **T5-xl 최고 (0.121)**, Pythia 소형 음수→대형 양수 반전
- Grade (G3~G8): 전반적으로 약함 (|ρ| < 0.07)
- 모든 계열에서 모델 크기 ↑ → alignment ↑

**(b) Jaccard Human vs Model** (`fig2_arc_jaccard_human.png`):
- Easy (Low 465개 vs Top-465): BERT 최고 (~0.17), random baseline(0.121) 대비 35%↑
- Hard (High 287개 vs Bottom-287): T5-xl 최고 (0.100), random(0.071) 대비 40%↑
- **쉬운 예제 식별이 어려운 예제 식별보다 강함** — 학습 시 쉬운 예제 우선 학습과 일치

---

## 주의사항

1. **ARC 선택지 수**: 대부분 4개, 소수 3/5개. MAX_CHOICES=5로 padding.
2. **T5 v1.1 sentinel**: MNLI와 동일. `answer: <extra_id_0>` + Adafactor + constant warmup.
3. **Pythia full fine-tune 제외**: ≤410m은 random. 분석에서 제외하거나 참고용.
4. **fp32 필수, OOM 시 bf16**: pythia-2.8b, 6.9b, t5-xl.
5. **Confidence 측정**: best epoch 이하 checkpoint 평균 (overfitting 방지).
6. **불안정 모델 처리**: 실패한 seed 제외하고 성공한 seed만으로 confidence 계산. 또는 추가 seed로 교체.
