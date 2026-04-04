# 011: 주체별 Easy Set 학습 실험 설계

> §3 "주체-의존적 데이터 큐레이션"을 위한 실험 계획
> 작성: 2026-03-31

---

## 0. 핵심 질문

**"같은 수의 쉬운 예제로 학습하더라도, 누구의 쉬움인지에 따라 일반화 성능이 달라지는가?"**

§2에서 "쉬움이 주체-의존적"임을 보였으므로, §3에서는 이 차이가 실제 학습에 어떤 practical consequence를 갖는지 통제된 비교 실험으로 보인다.

---

## 1. 실험 A: 주체별 Easy Set 비교

### 1.1 목적

동일한 수(k개)의 학습 예제를 주체별 easy 기준으로 선택했을 때, target model의 Dev 및 OOD 성능을 비교.

### 1.2 조건 설계

**주체별 easy set (세분화 포함):**

"큰 모델"의 정의에 따라 3가지 variant를 실험:

| "큰 모델" 정의 | 설명 |
|---------------|------|
| L = Self | Target model 자체 (e.g., BERT-large가 target이면 BERT-large의 easy) |
| L = T5-XL | 외부 큰 모델 T5-XL의 easy |
| L = Pythia-2.8B | 외부 큰 모델 Pythia-2.8B의 easy |

> 세 variant를 비교하면 "자기 자신의 easy vs 외부 큰 모델의 easy" 효과를 분리 가능.
> Phase 1 C3 (Self-easy 제거)는 해로웠으므로, Self vs 외부 모델의 차이가 핵심 비교축.

**조건 테이블 (L의 정의별 각각 수행):**

| ID | 조건 | 설명 | 크기 |
|----|------|------|------|
| A0 | Random | 랜덤 (크기 통제 baseline) | 자연 크기 매칭* |
| A1 | Full | 전체 데이터 (성능 상한 baseline) | 30,000 (MNLI) / 3,370 (ARC) |
| **큰 모델 기준** | | | |
| A2 | L-easy (전체) | 큰 모델 easy 전체 | 자연 크기 |
| A3 | L-easy ∩ S-hard | 큰 모델 easy & 작은 모델 hard (= L-only) | 자연 크기 |
| A4 | L-easy ∩ S-easy | 큰 모델 easy & 작은 모델도 easy (= Shared) | 자연 크기 |
| **작은 모델 기준** | | | |
| A5 | S-easy (전체) | 작은 모델 easy 전체 | 자연 크기 |
| A6 | S-easy ∩ L-hard | 작은 모델 easy & 큰 모델 hard (= S-only) | 자연 크기 |
| **인간 기준** | | | |
| A7 | H-easy (전체) | 인간 easy 전체 | 자연 크기 |
| A8 | H-easy ∩ L-hard | 인간 easy & 큰 모델 hard | 자연 크기 |
| A9 | H-easy ∩ L-easy | 인간 easy & 큰 모델도 easy | 자연 크기 |

> A9/A10에서 "M-hard"는 큰 모델(L) hard로 정의 (작은 모델 판단은 포함하지 않음).

### 1.3 크기 통제 방법

**각 조건의 자연 크기(교집합 크기)를 그대로 사용.**

- 교집합 조건(A3, A4, A6, A8, A9)은 각각 자연 크기가 다름 → 그대로 사용
- A0 (Random)은 각 비교 대상 조건의 크기에 매칭한 random baseline을 제공
  - 예: A3이 1,700개이면 A0도 1,700개로 random 추출
  - A6이 2,500개이면 A0도 2,500개로 random 추출
- **논리**: 안 좋은 예제를 제거해서 2,000개에서 1,700개가 됐으면, 2,000개보다 좋아야 의미가 있음. 크기를 억지로 맞출 필요 없음.
- 크기 차이가 결과에 미치는 영향은 동일 크기 Random baseline과의 비교로 통제

### 1.4 k(easy threshold) 결정

Phase 2 zone 비율 (MNLI, k=30% threshold 기준):
- S-all: ~7% (~2,100개)
- S-partial: ~19% (~5,700개)
- Shared: ~19% (~5,700개)
- L-only: ~11% (~3,300개)
- Neither: ~44% (~13,200개)

→ Easy threshold는 Phase 2와 동일하게 상위 30%로 설정.
→ ARC의 경우 전체가 3,370개이므로 교집합이 작을 수 있음. 사전에 크기 확인 필요.

### 1.5 "쉬움"의 정의

- **모델 easy**: confidence 상위 30% (Phase 2와 동일 threshold)
  - Confidence 측정: 학습 과정 중 1/3, 2/3, 3/3 지점에서 측정한 confidence의 평균
    - MNLI: 30K개 각 split으로 1 epoch 학습, 3개 checkpoint의 confidence 평균
    - ARC: 5 epoch 학습, 3개 checkpoint의 confidence 평균
  - 큰 모델: Self (target model), T5-XL, Pythia-2.8B — 3가지 variant
  - 작은 모델: **데이터셋별로 다름** (§1.6 참고)
  - **모든 모델의 confidence 데이터는 이미 확보 완료**
- **인간 easy**: ChaosNLI 인간 일치도(agreement) 상위, 또는 ARC의 경우 Easy split
- **교집합의 "hard"**: 큰 모델(L)의 easy threshold(상위 30%) 밖에 있는 예제

### 1.6 소형 모델 (Detector) 구성

**MNLI:**

| 모델 | 파라미터 | 비고 |
|------|---------|------|
| BERT-mini | 11M | Phase 2 Det B와 동일 |
| T5-v1.1-small | 77M | Phase 1-2와 동일 |
| Pythia-70M | 70M | Phase 2 Det B와 동일 |

> Phase 1 (BERT-small, T5-small, Pythia-14M)에서 Phase 2 Det B로 변경.
> Det A/B가 동일 패턴을 보였으므로 하나로 통일.

**ARC — Detector Set 2개로 비교:**

| Set | 모델 | 이유 |
|-----|------|------|
| Det-Small | BERT-mini, T5-v1.1-small, Pythia-70M | MNLI와 동일 (but ARC에서 거의 random 수준일 수 있음) |
| Det-Mid | (사전 확인 필요: ARC에서 random 이상 성능을 보이는 최소 모델 3개) | ARC의 4지선다에서 의미있는 예측을 하는 최소 규모 |

> **ARC 사전 확인 필수**: Det-Small 모델들의 ARC 성능 확인.
> 만약 ARC accuracy ≈ 25% (random)이면, 이 모델들의 "easy" 판단은 무의미.
> Det-Mid 후보: BERT-base (~340M), T5-base (250M), Pythia-410M 등 — ARC에서 30%+ 달성하는 최소 모델.
> Det-Small이 random이면 그 자체가 흥미로운 발견: "너무 작은 모델의 easy 판단은 정보가 없다"

### 1.7 Target Model & 데이터셋

| Target Model | 데이터셋 | Detector Set | 평가 |
|-------------|---------|-------------|------|
| T5-XL (3B) | MNLI (30K) | BERT-mini, T5-small, Pythia-70M | Dev-matched + HANS + ANLI-r1,r2,r3 |
| BERT-large (335M) | MNLI (30K) | BERT-mini, T5-small, Pythia-70M | Dev-matched + HANS + ANLI-r1,r2,r3 |
| T5-XL (3B) | ARC | Det-Small + Det-Mid (2세트 비교) | ARC-Easy val + ARC-Challenge test |
| BERT-large (335M) | ARC | Det-Small + Det-Mid (2세트 비교) | ARC-Easy val + ARC-Challenge test |

> Phase 1-2 결과에서 T5-XL만 실용적 성공을 보였으나, BERT-large도 비교를 위해 포함.
> ARC도 T5-XL, BERT-large 모두 실험.
> 모델별 confidence 및 ARC accuracy는 이미 확보 완료.

---

## 2. 실험 B: 연속 가중치 기반 큐레이션 비교

### 2.1 목적

Phase 2 B3 Graded (discrete zone 가중치)를 발전시켜, confidence 기반 연속 가중치의 design choice들을 병렬 비교.

### 2.2 가중치 공식

**기본 공식:**

$$w_i = \text{conf}_{\text{large}}(x_i) \times (1 - \text{agg}_{\text{small}}(x_i))$$

- conf_large: target model의 confidence (softmax probability)
- agg_small: 소형 모델 3개 (MNLI: BERT-mini, T5-v1.1-small, Pythia-70M)의 confidence 집계

**Design choice 1 — 집계 함수 (agg_small):**

| ID | 방식 | 설명 |
|----|------|------|
| B-mean | Mean | 소형 3개 confidence 평균. 부드러운 신호. |
| B-max | Max | 소형 3개 중 최대 confidence. 공격적 억제. Phase 1 C5 Union과 대응. |

**Design choice 2 — 가중치 함수:**

| ID | 공식 | 특성 |
|----|------|------|
| B-prod | conf_L × (1 - agg_S) | 곱 기반. agg_S 높으면 ≈ 0. |
| B-diff | σ(conf_L - agg_S) | 차이 기반 + sigmoid. Shared zone도 중간 weight. |
| B-ratio | conf_L / (agg_S + ε) | 비율 기반. agg_S가 0에 가까우면 weight 폭발 → clipping 필요. |

**Design choice 3 — Confidence scale:**

| ID | 방식 |
|----|------|
| B-raw | Raw confidence (softmax probability) 직접 사용 |
| B-rank | Percentile rank로 변환 후 사용 (모델 간 scale 차이 보정) |

### 2.3 비교 조건 (최종)

| ID | 공식 | agg | scale | 비고 |
|----|------|-----|-------|------|
| B0 | Baseline (uniform weight) | — | — | 전체 데이터, w=1 |
| B1 | Phase 2 B3 Graded (discrete) | zone | — | 기존 결과 재현/비교 기준 |
| B2 | prod | mean | raw | 기본 연속 가중치 |
| B3 | prod | max | raw | 공격적 소형 억제 |
| B4 | prod | mean | rank | scale 보정 |
| B5 | prod | max | rank | scale 보정 + 공격적 |
| B6 | diff | mean | raw | 차이 기반 |
| B7 | diff | max | raw | 차이 기반 + 공격적 |
| B8 | Random reweight | — | — | 랜덤 weight (통제) |

> 총 9개 조건 × 3 splits = 27 runs per target model.
> ratio 방식은 clipping 등 추가 hyperparameter가 필요하므로 우선 제외.

### 2.4 가중치 적용 방법

**Weighted sampling**: 각 example의 sampling probability를 w_i에 비례하게 설정.
- epoch당 원래 N개를 sampling (replacement 가능)
- 또는 loss에 w_i를 직접 곱하는 방식 (weighted loss)

> Phase 2에서는 upsampling/downsampling으로 구현했으므로, weighted sampling이 기존과 일관적.

---

## 3. 실험 인프라

### 3.1 Learning Rate 탐색

**중요**: 실험 A에서 데이터 크기가 조건마다 크게 다름 (2,000개 vs 30,000개).

| 데이터 크기 | LR 후보 (T5-XL LoRA) | LR 후보 (BERT-large full FT) |
|-----------|----------------------|------------------------------|
| ~2,000개 | {3e-5, 5e-5, 1e-4, 2e-4, 3e-4} | {5e-6, 1e-5, 2e-5, 3e-5, 5e-5} |
| ~5,000개 | {3e-5, 5e-5, 1e-4, 2e-4, 3e-4} | {5e-6, 1e-5, 2e-5, 3e-5, 5e-5} |
| ~30,000개 | {3e-5, 5e-5, 1e-4, 2e-4, 3e-4} | {5e-6, 1e-5, 2e-5, 3e-5, 5e-5} |

> 모든 크기에서 LR 후보 5개로 탐색. 기존 설정(T5-XL: 1e-4, BERT-lg: 2e-5)이 최적이 아닐 수 있으므로.

**절차:**
1. A0 Random 조건으로 각 데이터 크기별 LR sweep (5 LR × 1 split = 5 runs per 크기)
2. 최적 LR을 해당 크기의 모든 조건에 적용
3. 실험 B는 전체 데이터 사용이므로 30K 크기의 최적 LR 적용

**Epoch:**
- ≤ 5,000개: **10 epochs**
- 30,000개 (A1 Full): **5 epochs**

> 소규모 데이터는 일찍 수렴할 가능성 높음. 실제로는 중간에 수렴하겠지만, 충분한 epoch을 줘서 수렴 확인.

### 3.2 평가 메트릭 & 리포팅

| 메트릭 | 용도 |
|-------|------|
| Dev accuracy | In-distribution 성능 유지 확인 |
| HANS accuracy | Shortcut 의존도 평가 (MNLI) |
| ANLI-r1,r2,r3 accuracy | OOD 일반화 (MNLI) |
| OOD avg | HANS + ANLI 3개 평균 |
| ARC-Easy val accuracy | In-distribution (ARC) |
| ARC-Challenge test accuracy | OOD 일반화 (ARC) |

**리포팅 방식:**
- **저장**: seed(split)별 개별 결과 전부 저장 (재현성 및 추후 분석용)
- **리포팅**: 각 데이터셋·메트릭마다 **평균 ± 표준편차** (예: 90.16 ± 0.34)
- 테이블 형식: `Dev: 90.16±0.34 | HANS: 70.51±2.58 | ANLI avg: 44.46±1.12 | OOD avg: 50.97±1.35`

### 3.3 CV 설계

- 3-fold split (Phase 1-2와 동일)
- 가능하면 5-fold로 확장 (T5-XL B3의 통계적 유의성 이슈)

### 3.4 단계적 실행 계획

전체 ~540 runs(~5일)을 한 번에 수행하는 대신, 핵심 결과를 먼저 확보하고 결과를 보고 확장 여부를 결정하는 단계적 접근.

---

#### Phase 3a: 사전 확인 + LR sweep (~60 runs, ~반나절)

| 항목 | 내용 | runs |
|------|------|------|
| 교집합 크기 확인 | 모든 조건의 자연 크기 계산 (코드만, 학습 불필요) | 0 |
| ARC Det-Small 성능 확인 | 이미 확보된 accuracy 데이터 확인 | 0 |
| LR sweep (MNLI) | T5-XL: 3 크기 × 5 LR, BERT-lg: 동일 | 30 |
| LR sweep (ARC) | 동일 | 30 |
| **소계** | | **~60** |

**사전 확인 항목:**
- [ ] 각 교집합(A3, A6, A8, A9 등)의 실제 크기 확인
- [ ] ARC Det-Small 성능 확인 → random 수준이면 Det-Mid 후보 결정
- [ ] ARC 교집합 크기 확인 → 너무 작으면 ARC 실험 범위 조정
- [ ] ChaosNLI 인간 난이도 데이터 가용성 확인
- [ ] Weighted sampling 구현
- [x] 모든 모델의 confidence 데이터 확보 완료

---

#### Phase 3b: 핵심 실험 — T5-XL × MNLI, L=Self/T5-XL (~60 runs, ~2일)

**왜 이것부터?**
- T5-XL은 Phase 1-2에서 유일하게 실용적 성공(B3 Graded: OOD +2.35%p)을 보인 모델
- MNLI는 데이터가 충분하고(30K) 교집합 크기도 실험 가능한 수준
- L-variant는 Self vs T5-XL 2개로 시작 — "자기 자신의 easy vs 외부 모델의 easy"의 핵심 비교를 먼저 확인
- Pythia-2.8B는 T5-XL과 유사한 패턴이면 redundant → Phase 3b 결과 보고 결정

| 항목 | 조건 수 | Splits | runs |
|------|---------|--------|------|
| L 무관: A0(Random 크기별 ~5개), A1, A5, A7 | ~8 | 3 | ~24 |
| L=Self: A2, A3, A4, A6, A8, A9 | 6 | 3 | 18 |
| L=T5-XL: A2, A3, A4, A6, A8, A9 | 6 | 3 | 18 |
| **소계** | ~20 | | **~60** |

**Phase 3b에서 답할 질문:**
1. L-only(A3)로만 학습해도 일반화되는가? → "genuine easy"의 가치
2. S-only(A6)로 학습하면 일반화가 안 되는가? → shortcut의 해로움
3. Self easy(L=Self) vs 외부 모델 easy(L=T5-XL)는 어떤 차이가 나는가?
4. 인간 easy(A7) vs 모델 easy(A2)는 어떤 차이가 나는가?

---

#### Phase 3c: 결과 기반 확장 판단 (학습 불필요)

Phase 3b 결과를 보고 다음을 결정:

| 판단 | 조건 | 액션 |
|------|------|------|
| **L=Pythia-2.8B 추가?** | L=Self과 L=T5-XL의 결과가 크게 다르면 → 제3의 외부 모델로 일반성 검증 필요. 비슷하면 → 불필요 | 추가 시 +18 runs |
| **BERT-large 추가?** | Phase 1-2와 동일하게 BERT-lg에서 효과가 미미할 것으로 예상. 하지만 "큰 모델에서만 작동" 자체가 발견이므로, 핵심 조건(A1, A2, A3, A5, A6 + Random)만 돌려서 확인 | +~30 runs (축소) |
| **ARC 실험?** | Det-Small 성능 확인 결과에 따라: (a) random 수준이면 Det-Mid만, (b) 유의미하면 둘 다 | +60-105 runs |
| **실험 B 진행?** | Phase 3b에서 주체별 차이가 유의미하면 → 가중치 실험으로 확장. 차이가 미미하면 → 실험 B의 실익 없음 | 진행 시 Phase 3d로 |

---

#### Phase 3d: 실험 B (가중치 큐레이션) — 결과에 따라 진행 (~30-50 runs, ~1일)

**왜 나중에?**
- 실험 B는 실험 A에서 "주체별 easy의 차이가 학습에 유의미한 영향을 준다"가 확인된 후에야 의미가 있음
- 실험 A에서 차이가 미미하면, 연속 가중치로 세밀하게 조절해도 효과가 없을 가능성 높음
- 또한 실험 A 결과를 보고 가중치 함수 design choice를 사전에 좁힐 수 있음:
  - 예: A6(S-only)이 확실히 해롭다면 → prod 방식(소형 confidence 높으면 weight ≈ 0)이 적합
  - 예: A4(Shared)가 중립이면 → diff 방식(공유 easy에 중간 weight)이 적합

**Pilot 후 축소:**
- 9개 조건 중 prod/diff × mean/max = 4개 핵심 조건 + baseline 3개(B0, B1, B8) = **7개**로 시작
- raw vs rank는 pilot 1 split으로 먼저 비교 → 유망한 쪽만 3-split

| 항목 | 조건 수 | Splits | runs |
|------|---------|--------|------|
| raw vs rank pilot | 4 조건 | 1 | 4 |
| 핵심 조건 (유망 scale만) | 7 조건 | 3 | 21 |
| **소계 (MNLI T5-XL)** | | | **~25** |
| ARC T5-XL (진행 시) | 7 조건 | 3 | 21 |

---

#### Phase 3e: 보완 실험 — 필요 시 (~50-100 runs)

Phase 3b-3d 결과를 종합한 후, 논문에 필요한 보완 실험:

| 항목 | 조건 | 이유 |
|------|------|------|
| BERT-large 핵심 조건 | A1-A6 + Random, MNLI + ARC | "큰 모델에서만 작동"을 보이려면 BERT-lg 대조군 필요 |
| L=Pythia-2.8B | Phase 3c 판단에 따라 | 외부 큰 모델의 일반성 검증 |
| ARC Det-Mid | Phase 3a 결과에 따라 | Det-Small이 random이면 필수 |
| 실험 B BERT-lg | 핵심 조건만 | 가중치 접근이 BERT-lg에서도 안 되는지 확인 |
| 추가 seed/split | 통계적 유의성 확보 | T5-XL 핵심 결과에 대해 5-fold로 확장 |

---

### 3.5 총 예상 실험량

| 단계 | runs | 시간 (GPU 2장 병렬) | 상태 |
|------|------|-------------------|------|
| **Phase 3a** (사전 확인 + LR sweep) | 20 | ~8시간 | **완료** |
| **Phase 3b** (T5-XL × MNLI 핵심) | 60 | ~15시간 | **완료** (11 collapsed, 7 unrecoverable) |
| **Phase 3c** (판단) | 0 | — | **진행 중** |
| **Phase 3d** (실험 B) | ~25-46 | ~1일 | 3c 판단 후 |
| **Phase 3e** (보완) | ~50-100 | ~1-2일 | 논문 제출 전 |

---

## 4. Phase 3a 결과: LR Sweep

### 최적 LR

| 크기 | T5-XL (LoRA) | BERT-large (Full FT) |
|------|:---:|:---:|
| ~3,000 | 1e-4 | 3e-5 |
| ~8,000 | 2e-4 | 5e-5 |
| 30,000 | 1e-4 (기존) | 2e-5 (기존) |

> 3e-4 이상은 두 모델 모두 학습 붕괴.

### 사전 확인 결과

**ARC Det-Small 성능:**
- BERT-mini: 39.0% (OK)
- T5-v1.1-small: 26.6% (RANDOM)
- Pythia-70M: 26.6% (RANDOM)
→ ARC에서 Det-Small 사용 불가. Det-Mid (BERT-base 47%, Pythia-1B 47%, T5-large 45%) 필요.

**ChaosNLI:** 미확보. A7-A9 보류.

---

## 5. Phase 3b 결과: T5-XL × MNLI 주체별 Easy Set

### 5.1 결과 요약 (dev≥0.70인 split만, size-matched random 대비)

**L=Self (T5-XL) — target model 자신의 easy 기준:**

| 조건 | 설명 | Splits | N | Dev | OOD | Rand OOD | **ΔOOD** |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **A1 Full** | 전체 데이터 | 3 | 30,000 | 90.29 | 50.58 | — | ref |
| **A3 L-only** | Self-easy ∩ S-hard | 2 | 2,822 | 83.66 | **46.04** | 43.59 | **+2.46** |
| A2 L-easy | Self-easy 전체 | 2 | 9,000 | 84.23 | 43.74 | 48.14 | -4.40 |
| A4 Shared | Self-easy ∩ S-easy | 2 | 6,178 | 81.70 | 39.67 | 45.21 | -5.54 |
| A5 S-easy | S-easy union | 3 | 13,845 | 81.64 | 38.12 | 49.28 | **-11.15** |
| A6 S-only | S-easy ∩ Self-hard | 3 | 7,918 | 77.88 | 36.81 | 45.96 | **-9.15** |

> A3 L-only(Self)만 random 대비 OOD 우수. S-easy/S-only는 random보다 크게 악화.

**L=Pythia-2.8B — 외부 큰 모델의 easy 기준 (T5-XL target에 적용):**

| 조건 | 설명 | Splits | N | Dev | OOD | Rand OOD | **ΔOOD** |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| A3 L-only | P2.8B-easy ∩ S-hard | 1* | 1,916 | 84.43 | 40.00 | 44.40 | -4.40 |
| A2 L-easy | P2.8B-easy 전체 | 2 | 9,000 | 80.88 | 37.23 | 46.89 | -9.66 |
| A4 Shared | P2.8B-easy ∩ S-easy | 2 | 6,480 | 80.31 | 36.83 | 45.25 | -8.42 |
| A6 S-only | S-easy ∩ P2.8B-hard | 3 | 7,249 | 80.53 | 37.55 | 48.20 | -10.65 |

> *A3는 유효 split 1개만. 모든 조건에서 random 대비 나쁨 → 외부 모델의 easy는 target에 전이되지 않음.
>
> **핵심 비교: A3 L-only(Self) OOD 46.04 vs A3 L-only(P-2.8B) OOD 40.00**
> → 같은 "L-only" 정의라도, L이 **자기 자신이냐 외부 모델이냐**에 따라 6%p 차이. "누구의 easy인지"가 결정적.

### 5.2 핵심 발견

**1. L-only(Self)가 유일하게 random 대비 OOD 우수 (+2.46%p)**
- T5-XL 자신이 easy하고, 소형 모델은 hard하다고 판단한 예제 ~2,800개
- 이 예제만으로 OOD 46.0 달성 (Full 30K의 91% 수준을 1/10 데이터로)

**2. S-easy/S-only로 학습하면 OOD 크게 악화 (-9~11%p vs random)**
- Shortcut 예제만으로 학습 → 일반화 심각하게 손상
- 같은 수의 random보다 훨씬 나쁨 → **shortcut의 해로움 직접 입증**

**3. Shared(L∩S easy)도 해로움 (-5.5%p vs random)**
- 대형+소형 모두 easy → shortcut 성분이 지배적

**4. L=Pythia-2.8B의 easy는 T5-XL에 전이되지 않음**
- 모든 조건에서 random보다 나쁨 → **"자기 자신의 easy만 유효"**

**5. A2 L-easy(Self) 전체도 random보다 나쁨 (-4.4%p)**
- L-easy = L-only + Shared인데, Shared의 shortcut이 L-only의 이점을 상쇄

### 5.3 학습 불안정성

11/60 runs (18%)이 dev<0.70으로 붕괴. 2회 시드 변경 재실행 후에도 7개 미회복.

| 붕괴 패턴 | 수 | 해석 |
|----------|:---:|------|
| A0 Random (복구됨) | 4/4 | seed 문제 → 재실행으로 해결 |
| A2/A3/A4 condition (미복구) | 7/7 | **데이터 특성 문제** — 편향된 분포로 학습 자체가 어려움 |

→ 편향된 subset으로 학습 시 T5-XL LoRA가 불안정. split1에 집중(4/7).
→ 분석 시 dev≥0.70 split만 사용. 이로 인해 일부 조건은 1-2 split만 유효.

### 5.4 한계 및 논의

1. **유효 split 수 부족**: A3 L-only(Self)이 2 splits, A3(P-2.8B)이 1 split만 유효 → 통계적 신뢰도 낮음
2. **Dev 하락**: 모든 subset 조건이 Full 대비 dev 크게 하락 (6-13%p) → 실용적 "A3만으로 학습" 시나리오는 한계
3. **Random baseline도 불안정**: 일부 random split이 붕괴 → 소규모 T5-XL 학습 자체의 문제
4. **교집합 크기 변동**: split 간 L-only 크기가 1,900~3,600으로 큰 편차

---

## 6. Phase 3c: 확장 판단

Phase 3b 결과를 바탕으로:

| 판단 | 결론 | 근거 |
|------|------|------|
| **L=Pythia-2.8B 추가?** | **불필요** | 모든 조건에서 random 대비 나쁨. Self만 유효. |
| **BERT-large target?** | **보류** | Phase 1-2에서 이미 비실용적 확인. 비교용으로만 핵심 조건 축소 실행 가능 |
| **ARC 실험?** | **보류** | Det-Small 2/3이 random. Det-Mid 구성 필요 + ARC 교집합 작음(430~600개) |
| **실험 B (가중치)?** | **조건부 진행** | A3 L-only의 OOD 우위가 확인됨 → 연속 가중치로 정교화 가치 있음 |
| **A7-A9 (인간)?** | **보류** | ChaosNLI 다운로드 필요. 우선순위 낮음 |

---

## 7. 논문에서의 제시 방식

§3은 학회 논문 스타일로 **통제된 비교 실험의 결과 테이블**을 중심으로 구성:

**Table 1: 주체별 Easy Set 학습 결과 (실험 A)**
- 행: A1(Full), A3(L-only), A2(L-easy), A4(Shared), A5(S-easy), A6(S-only), A0(Random)
- 열: N_train / Dev / HANS / ANLI avg / OOD avg / ΔOOD vs Random
- L=Self만 메인, L=Pythia-2.8B는 appendix

**핵심 메시지:** "같은 수의 easy 예제라도, 누구의 easy인지에 따라 OOD 성능이 극적으로 달라진다"
- L-only(genuine easy): random 대비 OOD +2.5%p
- S-only(shortcut): random 대비 OOD -9.2%p
- 차이: **11.6%p** → 주체-의존성의 practical consequence

**Table 2: 가중치 기반 큐레이션 (실험 B, 진행 시)**

**본문 구조:**
- §3.1 실험 설계
- §3.2 주체별 easy set의 일반화 차이 (Table 1 + L-only vs S-only 핵심 비교)
- §3.3 주체-의존적 가중치 큐레이션 (Table 2, 진행 시)
- §3.4 종합

---

## 8. 파일 위치

| 항목 | 경로 |
|------|------|
| 교집합 indices | `outputs/plan11_expA/L_{self,pythia2.8b}/split{1,2,3}/` |
| LR sweep 결과 | `outputs/plan11_expA/lr_sweep/results_{bert,t5xl}/` |
| T5-XL 결과 | `outputs/plan11_expA/results_t5xl/` (60 files) |
| 스크립트 | `scripts/experiment_011a_*.py` |

---

## 9. 실험 B 결과: 연속 가중치 기반 큐레이션 (T5-large)

### 9.1 실험 설정

- **Target**: T5-v1.1-large (770M), Full FT, 5 epochs, 3 splits
- **L**: T5-large (Self) confidence
- **S**: 3개 소형 모델 (bert-mini, t5-small, pythia-70m)의 confidence
- **WeightedRandomSampler** 기반 학습 (weight로 sampling 확률 조절)

### 9.2 결과 (norm tag, lr=5e-5, 3 splits 평균)

| Condition | Formula | Dev | HANS | ANLI1 | ANLI2 | ANLI3 | OOD avg |
|-----------|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| B2_prod_mean_raw | L×(1-mean_S), raw | 0.877 | 0.612 | 0.384 | 0.323 | 0.357 | 0.419 |
| B3_prod_max_raw | L×(1-max_S), raw | 0.880 | 0.595 | 0.388 | 0.321 | 0.358 | 0.416 |
| B4_prod_mean_rank | L×(1-mean_S), rank | 0.874 | 0.588 | 0.384 | 0.318 | 0.367 | 0.414 |
| B5_prod_max_rank | L×(1-max_S), rank | 0.872 | 0.623 | 0.391 | 0.327 | 0.377 | **0.429** |
| B6_diff_mean_raw | sigmoid(L-mean_S), raw | 0.880 | 0.633 | 0.365 | 0.318 | 0.352 | 0.417 |
| B7_diff_max_raw | sigmoid(L-max_S), raw | 0.879 | 0.627 | 0.400 | 0.334 | 0.355 | 0.429 |
| B8_random | random weights | 0.879 | 0.669 | 0.383 | 0.326 | 0.347 | **0.431** |
| **F5_max_raw** | **(1-S²)×(0.5+L), max** | **0.878** | **0.619** | **0.398** | **0.321** | **0.357** | **0.424** |
| **F5_mean_raw** | **(1-S²)×(0.5+L), mean** | **0.881** | **0.614** | **0.378** | **0.321** | **0.349** | **0.416** |

> B6/B7/B8は norm tag 기준 2 splits만 완료 (split3 진행 중). 최종 결과는 3 splits 완료 후 업데이트 필요.

### 9.3 해석

1. **Random이 가장 높은 OOD (0.431)** — 가중치 조건들이 random을 이기지 못함
2. **B5_prod_max_rank (0.429)와 B7_diff_max_raw (0.429)**가 random에 근접
3. **F5는 중간 수준** (0.416~0.424) — lr=5e-5 고정이므로 LR sweep 후 재평가 필요
4. **Dev는 모든 조건이 비슷** (0.872~0.881) — weight이 ID 성능에는 영향 미미
5. **HANS에서 random (0.669)이 가장 높음** — weight 조건들은 HANS를 0.59~0.63으로 낮춤

### 9.4 TODO

- [ ] 011B split3 완료 후 최종 3-split 평균 업데이트
- [ ] F5에 대한 LR sweep (Plan 013 완료 후)
- [ ] lr=1e-4 tag 결과도 동일하게 정리

### 9.5 파일 위치

| 항목 | 경로 |
|------|------|
| norm 결과 | `outputs/plan11_expB/results_t5large_norm/` |
| lr1e-4 결과 | `outputs/plan11_expB/results_t5large_lr1e-4/` |
| F5 결과 | `outputs/plan11_expB/results_t5large_f5_norm/` |
| 스크립트 | `scripts/experiment_011b_train_t5large.py`, `scripts/experiment_011b_f5.py` |
