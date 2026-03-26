# Plan 001: MNLI Proxy Confidence 측정

> 📋 세부 실험 과정 및 디버깅 기록 (T5 v1.1 class collapse 진단, optimizer/scheduler 비교 등)은 [001a_mnli_experiment_log.md](001a_mnli_experiment_log.md) 참조.

## 목적

MNLI train (392,702개)에서 90K개를 비복원 추출하여 3개의 겹치지 않는 하위 집합(각 30K)을 생성.
다양한 모델(Encoder/Enc-Dec/Decoder, 20개)을 각각 학습하고, 학습 과정에서 confidence를 측정하여
각 예제의 "모델 난이도"를 산출.

## 데이터

### 90K 샘플링 (비중복 3 × 30K)

MNLI train에서 stratified sampling으로 90K 추출 → 3개 비중복 split (각 30K).
Split 생성 seed=42.

```
outputs/plan0/data/
├── split1_indices.json    # 30K
├── split2_indices.json    # 30K
├── split3_indices.json    # 30K
├── split4_indices.json    # 30K (추가, ChaosNLI 분석용)
└── all_90k_indices.json   # 90K
```

---

## 모델 (20개)

### Encoder: BERT (5종)

| 모델 | Params | Tuning | HuggingFace ID |
|------|--------|--------|---------------|
| bert-mini | 11M | full | prajjwal1/bert-mini |
| bert-small | 29M | full | prajjwal1/bert-small |
| bert-medium | 42M | full | prajjwal1/bert-medium |
| bert-base | 110M | full | bert-base-uncased |
| bert-large | 335M | full | bert-large-uncased |

- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: linear decay + warmup 6%
- 입력: `[CLS] premise [SEP] hypothesis [SEP]` (AutoTokenizer 자동 처리)
- Confidence: softmax(3-class logits) → 정답 class 확률

### Enc-Dec: T5 v1.1 (5종)

| 모델 | Params | Tuning | HuggingFace ID |
|------|--------|--------|---------------|
| t5-v1_1-small | 77M | full | google/t5-v1_1-small |
| t5-v1_1-base | 248M | full | google/t5-v1_1-base |
| t5-v1_1-large | 783M | full | google/t5-v1_1-large |
| t5-v1_1-xl | 3B | LoRA r=32 | google/t5-v1_1-xl |
| t5-v1_1-xxl | 11B | LoRA r=32 | google/t5-v1_1-xxl (미완료) |

- **Optimizer: Adafactor** (lr 고정, scale_parameter=False, relative_step=False)
- **Scheduler: constant LR + warmup 6%** (linear decay는 class collapse 유발)
- **Sentinel 프롬프트** (pretrain 패턴과 일치):
  - Encoder input: `"mnli hypothesis: {h} premise: {p} answer: <extra_id_0>"`
  - Decoder target: `"<extra_id_0> yes"` / `"<extra_id_0> maybe"` / `"<extra_id_0> no"`
- Confidence: decoder position 1에서 yes/maybe/no logits → softmax → 정답 확률
- LoRA target modules: `["q", "v"]`

### Decoder: Pythia (10종)

| 모델 | Params | Tuning | HuggingFace ID |
|------|--------|--------|---------------|
| pythia-14m | 14M | full | EleutherAI/pythia-14m |
| pythia-31m | 31M | full | EleutherAI/pythia-31m |
| pythia-70m | 70M | full | EleutherAI/pythia-70m |
| pythia-160m | 160M | full | EleutherAI/pythia-160m |
| pythia-410m | 410M | full | EleutherAI/pythia-410m |
| pythia-1b | 1B | LoRA r=32 | EleutherAI/pythia-1b |
| pythia-1.4b | 1.4B | LoRA r=32 | EleutherAI/pythia-1.4b |
| pythia-2.8b | 2.8B | LoRA r=32 | EleutherAI/pythia-2.8b |
| pythia-6.9b | 6.9B | LoRA r=32 | EleutherAI/pythia-6.9b |
| pythia-12b | 12B | LoRA r=32 | EleutherAI/pythia-12b (미완료) |

- Optimizer: AdamW (weight_decay=0.01)
- Scheduler: linear decay + warmup 6%
- 입력: `"Premise: {p}\nHypothesis: {h}\nRelation:"` (left padding 필수)
- 학습: 전체 label 토큰 시퀀스에 loss ("entailment" = [▁, en, tail, ment])
- Confidence: 마지막 위치 logits에서 첫 토큰 확률 비교
- LoRA target modules: `["query_key_value"]`

---

## T5 v1.1 학습 인사이트

### 문제: Class Collapse
T5 v1.1 base는 기본 설정(AdamW, 기존 프롬프트)에서 **decoder prior collapse** 발생.
모델이 encoder 입력을 무시하고 특정 class만 출력.

### 원인 진단
1. **Input ablation**: normal/shuffled/dummy 입력에서 동일한 결과 → encoder를 안 보고 있음
2. **Teacher forcing 함정**: loss가 내려가도 step 2+ (토큰 시퀀스 암기)에서만 줄어들고, step 1 (분류 결정)은 개선 안 됨
3. **Verbalizer 영향 확인**: yes/maybe/no (1토큰)로 바꿔도 collapse → verbalizer 문제만은 아님
4. **Prompt conditioning**: 다양한 prompt wording에서 모두 collapse → prompt 문제만도 아님

### 해결: Sentinel 프롬프트
T5 v1.1은 span corruption pretrain에서 `<extra_id_0>` sentinel token을 사용.
Encoder input에 `<extra_id_0>`를 넣으면 pretrain 패턴과 일치하여 안정적으로 학습됨.

### 최종 설정
```python
# Encoder input
input_text = f"mnli hypothesis: {h} premise: {p} answer: <extra_id_0>"
# Decoder target
target_text = f"<extra_id_0> yes"  # or maybe / no

# Optimizer & Scheduler
optimizer = Adafactor(lr=best_lr, scale_parameter=False, relative_step=False, warmup_init=False)
scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps=6%)
```

---

## Phase 1: LR Search

Split 1, Seed=1로만 LR search. 1 epoch, 30K 데이터.

### LR 후보

**BERT/Pythia < 110M**: `3e-5, 5e-5, 1e-4, 3e-4, 1e-3`
**BERT/Pythia ≥ 110M**: `1e-5, 2e-5, 3e-5, 5e-5, 1e-4`
**Pythia LoRA (≥ 1B)**: `5e-5, 1e-4, 3e-4, 1e-3, 3e-3`
**T5 v1.1 (sentinel)**: `3e-5, 5e-5, 1e-4` (모델별 조정)

---

## Phase 2: Confidence 측정 (3 Splits)

Best LR로 3개 split 각각 학습. 학습 중 1/3, 2/3, 3/3 지점에서 confidence 측정.
3 checkpoint 평균 → split별 confidence → 90K 전체 합산.

---

## 전체 결과

### LR Search

| 모델 | Params | Best LR | Val Acc |
|------|--------|---------|---------|
| bert-mini | 11M | 1e-4 | 0.612 |
| bert-small | 29M | 1e-4 | 0.657 |
| bert-medium | 42M | 5e-5 | 0.698 |
| bert-base | 110M | 5e-5 | 0.761 |
| bert-large | 335M | 3e-5 | 0.795 |
| pythia-14m | 14M | 3e-5 | 0.635 |
| pythia-31m | 31M | 1e-4 | 0.647 |
| pythia-70m | 70M | 3e-5 | 0.641 |
| pythia-160m | 160M | 1e-5 | 0.713 |
| pythia-410m | 410M | 1e-5 | 0.799 |
| pythia-1b | 1B | 3e-4 | 0.797 |
| pythia-1.4b | 1.4B | 3e-4 | 0.829 |
| pythia-2.8b | 2.8B | 3e-4 | 0.848 |
| pythia-6.9b | 6.9B | 1e-4 | 0.854 |
| pythia-12b | 12B | 1e-4 | 0.860 |
| t5-v1_1-small | 77M | 1e-4 | 0.634 |
| t5-v1_1-base | 248M | 1e-4 | 0.788 |
| t5-v1_1-large | 783M | 5e-5 | 0.869 |
| t5-v1_1-xl | 3B | 1e-4 | 0.886 |
| **t5-v1_1-xxl** | **11B** | **1e-4** | **0.904** |

### 90K Confidence

| 모델 | 90K Mean | Std | >0.95 | <0.40 |
|------|----------|------|-------|-------|
| bert-mini | 0.498 | 0.208 | 0.0% | 34.9% |
| bert-small | 0.572 | 0.240 | 0.1% | 26.8% |
| bert-medium | 0.615 | 0.253 | 0.8% | 23.1% |
| bert-base | 0.697 | 0.251 | 10.5% | 15.6% |
| bert-large | 0.615 | 0.282 | 12.1% | 41.2% |
| pythia-14m | 0.533 | 0.237 | 0.1% | 32.2% |
| pythia-31m | 0.538 | 0.228 | 0.4% | 30.2% |
| pythia-70m | 0.546 | 0.244 | 1.6% | 31.2% |
| pythia-160m | 0.626 | 0.263 | 7.3% | 22.7% |
| pythia-410m | 0.767 | 0.240 | 23.9% | 10.7% |
| pythia-1b | 0.732 | 0.252 | 18.9% | 13.6% |
| pythia-1.4b | 0.774 | 0.242 | 27.0% | 10.5% |
| pythia-2.8b | 0.815 | 0.257 | 49.2% | 10.3% |
| pythia-6.9b | 0.851 | 0.264 | 64.9% | 9.8% |
| pythia-12b | 0.812 | — | — | — |
| t5-v1_1-small | 0.484 | 0.225 | 1.1% | — |
| t5-v1_1-base | 0.523 | 0.239 | 3.8% | — |
| t5-v1_1-large | 0.832 | 0.264 | 57.1% | — |
| t5-v1_1-xl | 0.834 | 0.277 | 60.2% | — |
| **t5-v1_1-xxl** | **0.878** | — | — | — |

### 주요 관찰

1. **모델 크기 ↑ → val_acc ↑, confidence mean ↑**: 예상대로.
2. **bert-large 특이**: val_acc 높은데 confidence mean이 낮고 <0.40이 41.2% → 변별력 최고.
3. **포화 경계**: pythia-2.8b+(49~65%), t5-large/xl(57~60%)은 >0.95 높아 상위 변별력 약화.
4. **T5 v1.1 vs 원본 T5**: v1.1은 MNLI pretrain 안 해서 포화 문제 없음 (원본 T5는 69% 포화로 제외).
5. **3계열 비교**: BERT↔Pythia 유사도 높음 (Pearson r=0.71), T5 v1.1은 독자적 패턴.

### Shortcut 분석 (class-specific)

- **Negation (contradiction 내)**: 모든 계열에서 규모 ↑ → 부정어 의존 ↓
- **Lexical overlap (entailment 내)**: 규모에 덜 민감, 아키텍처 차이가 지배적

---

## 주의사항

1. **fp16 사용 금지**: Pythia logit scale이 크므로(~800) fp16에서 overflow → NaN. fp32 필수, OOM 시 bf16.
2. **Left padding (Decoder)**: Pythia는 반드시 `tokenizer.padding_side = "left"`.
3. **T5 v1.1 sentinel 필수**: encoder input에 `<extra_id_0>`, decoder target에 `<extra_id_0> label`.
4. **T5 v1.1 tokenizer**: `T5Tokenizer.from_pretrained(hf_id, legacy=True)` 사용. AutoTokenizer는 에러.
5. **T5 v1.1 Adafactor**: AdamW 사용 시 class collapse. Adafactor + constant warmup 필수.
6. **gradient checkpointing**: pythia-2.8b, 6.9b, t5-xl은 bf16 + gradient checkpointing 필요.

---

## 분석 계획

### Figure 1: Pairwise Jaccard Heatmap (9×9 대표 모델)
BERT-small/base/large × T5v1.1-small/base/large × Pythia-70m/410m/1b
각 모델의 상위 30% "쉬운 예제" 집합 간 Jaccard 유사도.
- MNLI: `fig1_jaccard_heatmap_9x9_v2.png` (split별 평균 ± std)
- ARC: `fig1_jaccard_heatmap_9x9_arc.png` (seed별 평균 ± std)

### Figure 2: Model↔인간 난이도 (ARC)
`easy-to-hard-generalization` annotation (difficulty, grade, bloom) 활용.
- **(a) Continuous Spearman**: model confidence (연속값) vs difficulty ordinal (Low=0/Med=1/High=2) + grade ordinal (G3~G8)
  - `fig2_arc_continuous.png`
- **(b) Jaccard Human vs Model**: Human Easy(Low, 465개) vs Model Top-465 / Human Hard(High, 287개) vs Model Bottom-287
  - `fig2_arc_jaccard_human.png`
- 2,151개 annotation 매칭 (ARC train 3,370 중). Grade는 G3~G8만 (Biology, Earth Science 제외).
- 주요 결과:
  - Difficulty Spearman: BERT ~0.09 안정, T5-xl 최고 (0.121), Pythia 소형 음수→대형 양수
  - Grade Spearman: 전반적으로 약함 (|ρ| < 0.07), difficulty보다 낮음
  - Easy Jaccard: BERT 최고 (~0.17), random baseline(0.121) 대비 35%↑
  - Hard Jaccard: T5-xl 최고 (0.100), random(0.071) 대비 40%↑, Easy보다 margin 작음

### Figure 3: Model Size vs Shortcut 비율
Negation (contradiction 내) + Lexical overlap (entailment 내). Class-specific 측정.

### Appendix: 20×20 Jaccard, k-robustness, variance 등.
