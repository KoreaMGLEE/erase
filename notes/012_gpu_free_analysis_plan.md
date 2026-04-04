# 012: GPU-Free 분석 계획 — 어떤 주체의 쉬운 예제가 도움/해가 되는가?

> §3 학습 실험에 앞서, 이미 확보된 confidence 데이터만으로 수행하는 사전 분석
> 작성: 2026-04-02

---

## 0. 동기 & 목표

Phase 3b 결과:
- L-only(Self)만 OOD에서 Random보다 좋음 (+2.46%p)
- S-easy, S-only는 Random보다 크게 나쁨 (-9~-11%p)
- 외부 모델(Pythia-2.8B)의 easy는 전혀 전이되지 않음

→ **핵심 질문**: "쉬운 예제"의 주체를 바꾸면 학습 효과가 달라진다. 그렇다면 **어떤 주체의 쉬운 예제가 도움이 되고, 어떤 주체의 쉬운 예제가 해가 되는가?**

**주체의 스펙트럼** (target model 기준):
1. 본인 (Self)
2. 본인보다 약간 큰 모델
3. 본인보다 훨씬 큰 모델
4. 본인보다 약간 작은 모델
5. 본인보다 훨씬 작은 모델
6. 비슷한 크기의 다른 계열 모델
7. 인간

→ **이 분석의 목표**: 학습 실험(GPU) 없이, confidence 데이터의 통계적 특성(overlap, shortcut ratio, zone 구성)만으로 "어떤 주체의 easy가 좋은 easy일 가능성이 높은가"를 사전 예측하고, 학습 실험 대상을 좁히는 것.

---

## 1. 사용 가능한 데이터

### 1.1 모델 목록 (20개)

| 계열 | 모델 | 파라미터 | MNLI | ARC |
|------|------|---------|------|-----|
| BERT | BERT-mini | 11M | ✓ | ✓ |
| BERT | BERT-small | 29M | ✓ | ✓ |
| BERT | BERT-base | 110M | ✓ | ✓ |
| BERT | BERT-large | 335M | ✓ | ✓ |
| T5 | T5-v1.1-small | 77M | ✓ | ✓ |
| T5 | T5-v1.1-base | 250M | ✓ | ✓ |
| T5 | T5-v1.1-large | 770M | ✓ | ✓ |
| T5 | T5-v1.1-xl | 3B | ✓ | ✓ |
| T5 | T5-v1.1-xxl | 11B | ✓ | ✓ |
| Pythia | Pythia-14M | 14M | ✓ | ✓ |
| Pythia | Pythia-70M | 70M | ✓ | ✓ |
| Pythia | Pythia-160M | 160M | ✓ | ✓ |
| Pythia | Pythia-410M | 410M | ✓ | ✓ |
| Pythia | Pythia-1B | 1B | ✓ | ✓ |
| Pythia | Pythia-2.8B | 2.8B | ✓ | ✓ |
| Pythia | Pythia-6.9B | 6.9B | ✓ | ✓ |
| Pythia | Pythia-12B | 12B | ✓ | ✓ |

> 정확한 모델 수/이름은 실제 confidence 데이터 파일 확인 후 보정 필요

### 1.2 Confidence 측정 방식

- MNLI: 30K 각 split으로 1 epoch 학습, 1/3·2/3·3/3 checkpoint에서 측정한 confidence 평균
- ARC: 5 epoch 학습, 동일하게 3 checkpoint 평균
- 모든 20개 모델에 대해 확보 완료

### 1.3 인간 난이도

- MNLI: **ChaosNLI** (미확보 — 이 분석에서 확보 필요)
  - ChaosNLI는 SNLI/MNLI-matched 중 ~1,500개에 대해 100명의 annotator 판단 제공
  - Agreement rate를 인간 confidence로 사용
- ARC: ARC-Easy/ARC-Challenge 구분을 인간 난이도 proxy로 활용 가능

---

## 2. Target Model 선정

**원칙**: 모든 20개를 하지 않되, 리뷰어가 "cherry-picking" 비판을 할 수 없을 만큼 다양하게.

### 2.1 선정 기준

1. **3개 계열 모두 포함** (BERT, T5, Pythia)
2. **크기 다양성**: 소형, 중형, 대형 각 1개 이상
3. **실험 연속성**: Phase 1-2에서 사용한 T5-XL, BERT-large 포함
4. **대칭성**: 각 target에 대해 "더 큰/작은 모델"이 존재해야 함

### 2.2 선정된 Target Model (6개)

| Target | 파라미터 | 계열 | 선정 이유 |
|--------|---------|------|----------|
| **BERT-base** | 110M | BERT | 중간 크기, BERT 계열 대표. 위로 BERT-large, 아래로 BERT-mini/small 존재 |
| **BERT-large** | 335M | BERT | Phase 1-2 target. 같은 계열 내 가장 큰 모델 (위로 갈 때 cross-arch 필요) |
| **T5-v1.1-base** | 250M | T5 | 중간 크기, T5 계열 대표. 위아래 모두 동일 계열 모델 풍부 |
| **T5-v1.1-xl** | 3B | T5 | Phase 1-2 target, 유일한 실용적 성공 모델. 위로 T5-xxl, 아래로 T5-large/base |
| **Pythia-410M** | 410M | Pythia | 중간 크기, Pythia 계열 대표. 위아래 모두 동일 계열 모델 풍부 |
| **Pythia-2.8B** | 2.8B | Pythia | 대형 Pythia. 위로 6.9B/12B, 아래로 다양한 소형 모델 |

> 6개 target × 다양한 subject = 충분한 패턴 파악 가능
> 필요 시 Pythia-70M(소형 대표) 추가 가능

---

## 3. Subject 매핑 (각 Target별)

각 target model T에 대해, 다음 7가지 유형의 "주체"를 매핑:

### 3.1 Subject 유형 정의

| 유형 | 정의 | 예시 (T = T5-XL, 3B 기준) |
|------|------|---------------------------|
| **Self** | T 자신 | T5-XL |
| **Same-arch-slightly-larger** | 동일 계열, T보다 한 단계 큰 모델 | T5-xxl (11B) |
| **Same-arch-much-larger** | 동일 계열, T보다 훨씬 큰 모델 | (T5-xxl이 최대이므로 해당 없음) |
| **Cross-arch-larger** | 다른 계열, T보다 큰 모델 | Pythia-6.9B, Pythia-12B |
| **Same-arch-slightly-smaller** | 동일 계열, T보다 한 단계 작은 모델 | T5-large (770M) |
| **Same-arch-much-smaller** | 동일 계열, T보다 훨씬 작은 모델 | T5-small (77M) |
| **Cross-arch-similar-size** | 다른 계열, T와 비슷한 크기 | Pythia-2.8B, BERT-large(335M) |
| **Cross-arch-smaller** | 다른 계열, T보다 작은 모델 | Pythia-70M, BERT-mini(11M) |
| **Human** | 인간 | ChaosNLI / ARC-Easy split |

### 3.2 구체적 매핑 테이블

#### Target: BERT-base (110M)

| Subject 유형 | 모델 | 파라미터 | 크기 비율 (Subject/Target) |
|-------------|------|---------|--------------------------|
| Self | BERT-base | 110M | 1.0× |
| Same-arch-slightly-larger | BERT-large | 335M | 3.0× |
| Cross-arch-larger | T5-base (250M), T5-large (770M), T5-XL (3B), Pythia-410M, Pythia-1B, Pythia-2.8B | 250M-3B | 2.3-27× |
| Same-arch-slightly-smaller | BERT-small | 29M | 0.26× |
| Same-arch-much-smaller | BERT-mini | 11M | 0.10× |
| Cross-arch-similar-size | T5-small (77M), Pythia-70M, Pythia-160M | 70-160M | 0.64-1.45× |
| Cross-arch-smaller | Pythia-14M | 14M | 0.13× |
| Human | ChaosNLI / ARC | — | — |

#### Target: BERT-large (335M)

| Subject 유형 | 모델 | 파라미터 |
|-------------|------|---------|
| Self | BERT-large | 335M |
| Cross-arch-larger (동일 계열 larger 없음) | T5-large (770M), T5-XL (3B), Pythia-1B, Pythia-2.8B | 770M-3B |
| Same-arch-slightly-smaller | BERT-base | 110M |
| Same-arch-much-smaller | BERT-mini (11M), BERT-small (29M) | 11-29M |
| Cross-arch-similar-size | T5-base (250M), Pythia-410M | 250-410M |
| Cross-arch-smaller | T5-small (77M), Pythia-70M, Pythia-160M | 70-160M |
| Human | ChaosNLI / ARC | — |

#### Target: T5-v1.1-base (250M)

| Subject 유형 | 모델 | 파라미터 |
|-------------|------|---------|
| Self | T5-base | 250M |
| Same-arch-slightly-larger | T5-large | 770M |
| Same-arch-much-larger | T5-XL (3B), T5-xxl (11B) | 3-11B |
| Cross-arch-larger | Pythia-1B, Pythia-2.8B, BERT-large (335M) | 335M-2.8B |
| Same-arch-slightly-smaller | T5-small | 77M |
| Cross-arch-similar-size | BERT-base (110M), Pythia-160M, Pythia-410M | 110-410M |
| Cross-arch-smaller | BERT-mini (11M), Pythia-14M, Pythia-70M | 11-70M |
| Human | ChaosNLI / ARC | — |

#### Target: T5-v1.1-xl (3B)

| Subject 유형 | 모델 | 파라미터 |
|-------------|------|---------|
| Self | T5-XL | 3B |
| Same-arch-slightly-larger | T5-xxl | 11B |
| Cross-arch-larger | Pythia-6.9B, Pythia-12B | 6.9-12B |
| Same-arch-slightly-smaller | T5-large | 770M |
| Same-arch-much-smaller | T5-base (250M), T5-small (77M) | 77-250M |
| Cross-arch-similar-size | Pythia-2.8B, BERT-large (335M) | 335M-2.8B |
| Cross-arch-smaller | BERT-mini (11M), Pythia-70M, Pythia-160M | 11-160M |
| Human | ChaosNLI / ARC | — |

#### Target: Pythia-410M

| Subject 유형 | 모델 | 파라미터 |
|-------------|------|---------|
| Self | Pythia-410M | 410M |
| Same-arch-slightly-larger | Pythia-1B | 1B |
| Same-arch-much-larger | Pythia-2.8B (2.8B), Pythia-6.9B, Pythia-12B | 2.8-12B |
| Cross-arch-larger | T5-large (770M), T5-XL (3B), BERT-large (335M) | 335M-3B |
| Same-arch-slightly-smaller | Pythia-160M | 160M |
| Same-arch-much-smaller | Pythia-70M (70M), Pythia-14M (14M) | 14-70M |
| Cross-arch-similar-size | T5-base (250M), BERT-base (110M), BERT-large (335M) | 110-335M |
| Human | ChaosNLI / ARC | — |

#### Target: Pythia-2.8B

| Subject 유형 | 모델 | 파라미터 |
|-------------|------|---------|
| Self | Pythia-2.8B | 2.8B |
| Same-arch-slightly-larger | Pythia-6.9B | 6.9B |
| Same-arch-much-larger | Pythia-12B | 12B |
| Cross-arch-larger | T5-XL (3B), T5-xxl (11B) | 3-11B |
| Same-arch-slightly-smaller | Pythia-1B | 1B |
| Same-arch-much-smaller | Pythia-410M, Pythia-160M, Pythia-70M, Pythia-14M | 14-410M |
| Cross-arch-similar-size | T5-XL (3B), BERT-large (335M) | 335M-3B |
| Cross-arch-smaller | T5-small (77M), BERT-mini (11M) | 11-77M |
| Human | ChaosNLI / ARC | — |

---

## 4. 분석 항목 (Analysis Metrics)

각 (target T, subject S) 쌍에 대해 다음을 계산:

### 4.1 Easy Set 기본 통계

- **Easy set 크기**: |Easy_S| (S가 easy로 판단하는 예제 수)
- **Easy set overlap (Jaccard)**: J(Easy_T, Easy_S) = |Easy_T ∩ Easy_S| / |Easy_T ∪ Easy_S|
  - 높을수록 두 주체의 "쉬움"이 유사
  - 이미 §2 Figure 1에서 일부 계산되어 있음
- **비대칭 overlap**: |Easy_T ∩ Easy_S| / |Easy_T| (T의 easy 중 S도 easy로 보는 비율)

### 4.2 Zone 분석 (T 기준, S를 "작은 모델"로 취급)

T를 "큰 모델", S를 "작은 모델"로 놓고 zone 계산:

| Zone | 정의 | 의미 |
|------|------|------|
| **T-only** | Easy_T ∩ Hard_S | T에게만 쉬운 예제 (S 관점의 "genuine easy" 후보) |
| **Shared** | Easy_T ∩ Easy_S | 둘 다 쉬운 예제 |
| **S-only** | Hard_T ∩ Easy_S | S에게만 쉬운 예제 (shortcut 의심) |
| **Neither** | Hard_T ∩ Hard_S | 둘 다 어려운 예제 |

> **중요**: 여기서 T를 "큰 모델"로 고정하는 것이 아니라, 매번 T를 target, S를 subject로 놓고 zone을 계산함.
> 예: T=BERT-base, S=T5-xxl이면 "T-only"는 BERT-base에게만 쉽고 T5-xxl에게 어려운 것 → 이 zone은 shortcut이 많을 가능성.
> 반대로 T=BERT-base, S=BERT-mini이면 "T-only"는 L-only와 동일 → genuine easy 후보.

**Zone 크기 비율**: 각 zone이 전체 데이터에서 차지하는 비율 (%)

### 4.3 Shortcut Ratio 분석

**MNLI 전용** (HANS heuristic label 활용):

각 zone 내 shortcut 비율 = (해당 zone 내 HANS heuristic에 부합하는 예제 수) / (해당 zone 크기)

| 분석 대상 | 의미 |
|----------|------|
| Shortcut ratio in Easy_S | S의 easy set에 shortcut이 얼마나 많은가? |
| Shortcut ratio in T-only zone | T에게만 쉬운 예제의 shortcut 비율 → 낮으면 "genuine easy" |
| Shortcut ratio in S-only zone | S에게만 쉬운 예제의 shortcut 비율 → 높으면 shortcut-dominated |
| Shortcut ratio in Shared zone | 공유 easy의 shortcut 비율 |

> **핵심 가설**: S가 T보다 훨씬 작을수록, S-only zone의 shortcut ratio가 높을 것.
> S가 T보다 클수록, S-only zone (= 큰 모델에게만 쉬운) 의 shortcut ratio가 낮을 것.

### 4.4 Confidence 분포 분석

- **T의 confidence 분포**: Easy_S에 속한 예제들의 T confidence 평균/중앙값
  - "S의 easy를 T가 학습하면 T에게 얼마나 쉬운/어려운 데이터인가"
- **Confidence gap**: mean(conf_T | Easy_S) - mean(conf_T | Random)
  - 양수: S의 easy가 T에게도 비교적 쉬움
  - 음수: S의 easy가 T에게는 어려움

### 4.5 "유망한 Easy Set" 예측 지표

Phase 3b 결과에서 관찰된 패턴:
- L-only(Self)는 OOD +2.46%p → **성공**
- S-easy, S-only는 OOD -9~-11%p → **실패**
- L=Pythia-2.8B 조건은 전부 Random 이하 → **실패**

이 패턴과 일치하는 통계적 지표를 찾는 것이 목표:

**후보 지표 (각 subject S의 easy set에 대해):**

1. **Shortcut ratio**: 낮을수록 좋은 easy
2. **T-only zone 비율**: 높을수록 T에게만 새로운 정보 → 좋은 easy?
3. **Jaccard with Self-easy**: Self-easy와 overlap이 높을수록 좋은 easy?
4. **T confidence gap**: T에게 적당히 쉬운 게 좋은가, 어려운 게 좋은가?
5. **크기 비율 (S/T)**: 크기 차이와 학습 효과의 관계

---

## 5. Threshold 민감도 분석

### 5.1 다중 Threshold 테스트

현재 "Easy = 상위 30%"만 사용 중. 이 threshold가 적절한지 검증 필요.

**Threshold 후보**: {10%, 20%, 30%, 50%}

각 threshold t에 대해:
- Easy_S(t) = confidence 상위 t%인 예제 집합
- 4절의 모든 분석을 threshold별로 반복

### 5.2 Threshold별 비교 항목

| 비교 항목 | 확인 사항 |
|----------|----------|
| Easy set 크기 변화 | threshold에 따른 자연 크기 (특히 교집합) |
| Jaccard 안정성 | threshold 바꿔도 overlap 패턴이 유지되는가? |
| Shortcut ratio 변화 | 더 strict한 threshold(10%)에서 shortcut ratio가 어떻게 변하는가? |
| Zone 비율 안정성 | T-only, S-only 비율이 threshold에 따라 급변하는가? |

### 5.3 최적 Threshold 결정 기준

- **안정성**: 20-30% 근처에서 패턴이 안정적이면 30% 유지
- **차별력**: threshold에 따라 주체 간 차이가 가장 뚜렷한 지점
- **실용성**: 교집합 크기가 학습에 충분한 수준 (최소 ~500개 이상)

---

## 6. 구체적 분석 스크립트 설계

### 6.1 Input 데이터 형식

```
# 필요한 데이터 (이미 확보):
# conf[model_name][example_id] = float (average confidence across 3 checkpoints)
# 모든 20개 모델 × MNLI 30K + ARC 3,370

# HANS heuristic labels (MNLI only):
# hans_heuristic[example_id] = bool (HANS heuristic에 부합하는지)
```

### 6.2 분석 파이프라인

```
Phase A: 기본 데이터 로드 & Easy Set 계산
├── A1: 모든 모델의 confidence 로드
├── A2: 각 threshold {10, 20, 30, 50}별 easy set 계산
│   easy_set[model][threshold] = {example_ids with conf in top-t%}
└── A3: HANS heuristic label 로드 (MNLI)

Phase B: Pairwise 분석 (6 targets × ~10 subjects × 4 thresholds)
├── B1: Jaccard overlap 계산 → heatmap
├── B2: Zone 계산 (T-only, Shared, S-only, Neither)
├── B3: Zone별 shortcut ratio (MNLI only)
├── B4: Confidence 분포 (T의 conf on Easy_S)
└── B5: 비대칭 overlap

Phase C: 패턴 분석 & 시각화
├── C1: 크기 비율(S/T) vs Jaccard 산점도
├── C2: 크기 비율(S/T) vs shortcut ratio in S-easy 산점도
├── C3: 동일 계열 vs 교차 계열 비교
├── C4: Threshold 민감도 비교 차트
└── C5: Phase 3b 실험결과와의 대조 (지표 vs 실제 OOD 성능)

Phase D: 인간 난이도 포함 분석 (ChaosNLI 확보 후)
├── D1: 인간 agreement → confidence 변환
├── D2: 인간 easy set 계산
└── D3: Phase B의 모든 분석을 인간도 subject로 포함하여 반복
```

### 6.3 핵심 출력물

1. **Jaccard Heatmap (확장판)**: 6 target × 모든 subject, threshold별
2. **Zone 비율 테이블**: 각 (target, subject) 쌍의 T-only / Shared / S-only / Neither 비율
3. **Shortcut Ratio 히트맵**: Zone별 shortcut ratio → 어떤 zone이 위험한지
4. **"유망도 점수" 테이블**: 각 subject의 easy set에 대한 복합 지표
5. **Threshold 안정성 차트**: 주요 패턴이 threshold에 robust한지

---

## 7. Phase 3b 결과와의 사후 검증

이미 확보된 Phase 3b 결과(T5-XL × MNLI)를 이 분석의 **검증 데이터**로 사용:

### 7.1 검증 방법

| Phase 3b 조건 | 실제 OOD 결과 | 대응되는 분석 지표 |
|--------------|-------------|-----------------|
| A3 L-only(Self) | +2.46%p (성공) | Self-easy의 shortcut ratio, T-only zone 비율 |
| A5 S-easy | -11.15%p (실패) | Det-small easy의 shortcut ratio |
| A6 S-only | -9.15%p (실패) | S-only zone의 shortcut ratio |
| L=Pythia-2.8B 전조건 | 전부 실패 | Pythia-2.8B easy의 Jaccard with Self-easy, confidence gap |

### 7.2 검증 질문

1. **Shortcut ratio가 실제 OOD 성능을 예측하는가?**
   - L-only(Self)의 shortcut ratio < S-only의 shortcut ratio → 예상과 일치하는지?
2. **Jaccard with Self-easy가 높을수록 좋은가?**
   - Pythia-2.8B easy의 Self-easy와의 Jaccard가 낮은지?
3. **크기 비율이 핵심인가, 아니면 계열(architecture)이 핵심인가?**
   - T5-XL target에서: T5-xxl(동일계열 큰) vs Pythia-2.8B(다른계열 비슷) vs Pythia-12B(다른계열 큰) 비교

### 7.3 검증 성공 기준

- Phase 3b에서 성공한 조건(A3)의 지표 값이 실패한 조건(A5, A6, L=Pythia)보다 일관되게 좋으면 → 해당 지표를 이후 실험 설계의 가이드로 사용
- 어떤 지표도 Phase 3b를 설명하지 못하면 → 지표 재설계 또는 추가 학습 실험 필요

---

## 8. ChaosNLI 확보 계획

### 8.1 ChaosNLI 데이터

- **출처**: Nie et al. (2020), "What Can We Learn from Collectively Generated Datasets?"
- **내용**: SNLI/MNLI dev 중 ~1,514개 예제에 대해 100명 annotator의 label 분포
- **다운로드**: https://github.com/easonnie/ChaosNLI
- **사용법**: 100명 중 gold label과 일치하는 비율 = human agreement rate → 이를 "인간 confidence"로 사용
  - High agreement (≥90%) = 인간에게 쉬운 예제
  - Low agreement (<60%) = 인간에게 어려운 예제

### 8.2 한계

- MNLI dev 1,514개만 커버 → 30K 학습 데이터와 직접 대응 불가
- **대안 1**: ChaosNLI가 커버하는 예제만으로 인간 분석 수행 (N이 작지만 패턴 확인 가능)
- **대안 2**: AmbiNLI (Meissner et al., 2021) 등 추가 인간 판단 데이터 탐색
- **대안 3**: 학습 데이터에 대한 인간 annotation은 별도 진행 (비용 발생)

### 8.3 ARC 인간 난이도

- ARC-Easy / ARC-Challenge 구분을 proxy로 사용
- ARC-Easy에 속하면 "인간 easy", Challenge에 속하면 "인간 hard"로 이진 분류
- 더 세밀한 인간 난이도는 별도 annotation 없이는 불가

---

## 9. 실행 현황 & 결과

### Step 1-2: 데이터 준비 + 기본 분석 — 완료

- [x] 17개 모델의 90K avg confidence 로드 (T5-xxl, Pythia-12B는 split 기반만 있어 제외)
- [x] HANS heuristic 구현 (3가지: lexical overlap subset, subsequence, high overlap ≥70% + negation)
  - SC strict (full subset/subseq + entailment OR neg + contradiction)
  - SC soft (70% overlap + entailment OR neg + contradiction)
- [x] **Hypothesis-only baseline** 구현 및 학습 완료
  - BERT-mini로 hypothesis만 입력, 1 epoch 학습 (62초)
  - Hypothesis-only accuracy: 56.5% (chance 33.3%) → hypothesis만으로도 상당히 예측 가능
  - Per-example confidence를 shortcut proxy로 활용
- [ ] ChaosNLI 미확보
- [x] 4개 threshold(10,20,30,50%) × 17개 모델의 easy set 계산
- [x] 6 target × 17 subject pairwise 분석 (Jaccard, Zone, SC ratio, Confidence gap)
- [x] 결과 저장: `outputs/plan12_analysis/pairwise_results_v2.json`, `hyp_only_confidence.json`

### Step 3: 패턴 분석 — 완료

#### 핵심 발견 1: Subject 크기에 따른 SC(S-only) 패턴 — Fig 2, 5

S-only zone = Subject만 easy하고 Target에게 hard한 예제. 이 zone의 shortcut ratio가 높으면, Subject의 easy로 학습 시 shortcut을 주입하게 됨.

| Subject vs Target | SC(S-only) HANS | SC(S-only) Hyp-only | 해석 |
|-------------------|:---:|:---:|------|
| **Subject << Target** (훨씬 작은 모델) | **0.39-0.56** | **0.38-0.48** | 작은 모델의 고유 easy = shortcut 가득 |
| **Subject ≈ Target** (비슷한 크기) | 0.24-0.37 | 0.30-0.36 | 중간 |
| **Subject >> Target** (훨씬 큰 모델) | **0.22-0.31** | **0.29-0.34** | 큰 모델의 고유 easy = shortcut 적음 |

→ **두 shortcut 측정법(HANS heuristic, hyp-only) 모두 동일 패턴**: Subject가 작을수록 S-only에 shortcut 오염이 심함
→ 이는 "작은 모델의 easy ≈ shortcut"이라는 핵심 가설을 지지

#### 핵심 발견 2: 동일 계열 vs 교차 계열 — Fig 2, 3

- Jaccard: 동일 계열이 약간 높음 (0.39-0.50 vs 0.34-0.44)
- **SC(S-only): 큰 차이 없음** — **계열보다 크기 비율이 더 중요한 요인**
- 예외: T5-v1.1-small은 target과 다른 계열이어도 SC(S-only)가 매우 높음 (0.52-0.60)

→ **시나리오 A ("크기 비율이 핵심")에 가장 부합**. 동일/교차 계열 여부는 부차적.

#### 핵심 발견 3: Threshold 민감도 — Fig 4

- k=10%→50%: SC:Easy_S는 감소 (strict easy = more shortcut), Jaccard는 증가
- **패턴 방향은 k=10-50% 전체에서 유지** → threshold에 robust
- k=10%에서 가장 극단적 차이 (SC:S-only 0.88 vs 0.36)

#### 핵심 발견 4: Shortcut ratio만으로는 OOD 예측 불충분 — Fig 5, 6

Phase 3b 실험 결과와 대조:

| Condition | ΔOOD | HypSC% | HANS SC | T-XL conf | Det conf |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **A3 L-only(Self)** | **+2.46** | **18.0%** | 0.113 | **0.999** | **0.377** |
| A3 L-only(P-2.8B) | -4.40 | 19.4% | 0.100 | 0.899 | 0.418 |
| A4 Shared(Self) | -5.54 | **63.1%** | **0.668** | 0.999 | 0.734 |
| A6 S-only(Self) | -9.15 | 40.0% | 0.360 | 0.866 | 0.644 |
| A5 S-easy | -11.15 | 50.0% | 0.493 | 0.924 | 0.683 |

**A3 L-only(Self) vs A3 L-only(P-2.8B)**: SC는 거의 동일(18% vs 19%)인데 OOD 결과는 정반대(+2.46 vs -4.40)
→ Shortcut ratio는 "이 subset이 전반적으로 해로운가"는 예측하지만, "Self의 L-only가 왜 유일하게 도움이 되는가"는 설명 못함

**핵심 구분 요인: Target confidence + Detector confidence의 조합**
- A3 L-only(Self): Target conf **0.999** + Det conf **0.377** → target이 확신하는데 소형이 못 맞추는 예제 = genuine easy
- A3 L-only(P-2.8B): Target conf 0.899 + Det conf 0.418 → target 확신이 상대적으로 낮음

→ **"좋은 학습 예제" = (1) 낮은 shortcut + (2) target 자신이 높은 확신 + (3) 소형 모델이 낮은 확신**
→ 세 조건을 모두 만족하는 예제는 **Self의 L-only**에만 존재

### Step 4: 인간 분석 (ChaosNLI) — 완료

- [x] ChaosNLI 다운로드 (MNLI matched dev 1,599개 × 100 annotators)
- [x] ChaosNLI는 MNLI dev 예제 → train 90K와 매칭 불가 (겹치는 예제 0개)
- [x] Dev 예제에 대해 HANS heuristic + lexical overlap 분석

**결과:**
- 인간 Easy/Medium/Hard 모두 SC ratio 0.09-0.12 → **인간 난이도와 shortcut 무관**
- Lexical overlap과 agreement 상관 r=-0.038 → **인간의 "쉬움" ≠ 모델의 "쉬움"**
- 인간의 쉬움은 의미적 명확성에 기반, 모델의 쉬움은 표면적 패턴에 기반

### Figures

| Figure | 내용 | 경로 |
|--------|------|------|
| Fig 1 | Jaccard Heatmap (6T × 17S, k=30%) | `figures/fig1_jaccard_heatmap.png` |
| Fig 2 | S/T ratio vs SC(S-only) — HANS heuristic scatter | `figures/fig2_sizeratio_vs_sc_sonly.png` |
| Fig 2b | 3-zone SC 비교 (T-only / Shared / S-only × 6 targets) | `figures/fig2b_three_zone_sc.png` |
| Fig 3 | S/T ratio vs Jaccard scatter | `figures/fig3_sizeratio_vs_jaccard.png` |
| Fig 4 | Threshold sensitivity (T5-XL target) | `figures/fig4_threshold_sensitivity.png` |
| Fig 5 | HANS vs Hypothesis-only SC 비교 (T5-XL target) | `figures/fig5_hans_vs_hyponly.png` |
| Fig 6 | Phase 3b 검증: ΔOOD vs SC/TargetConf/DetConf | `figures/fig6_phase3b_validation.png` |
| Fig 7 | ChaosNLI 인간 난이도 분석 | `figures/fig7_chaosnli_human.png` |
| Fig 8 | ARC Jaccard overlap | `figures/fig8_arc_jaccard.png` |
| Fig 9 | 다수 큰 모델 합의 분석 — consensus 수 vs SC/크기 | `figures/fig9_consensus_analysis.png` |
| Fig 10 | ChaosNLI: 모델 크기별 인간 agreement vs confidence | `figures/fig10_chaosnli_modelsize.png` |
| Fig 11 | Weight score formula 비교 (6 formulas × 5 categories) | `figures/fig11_weight_score_compare.png` |

> Phase 3b 검증(Fig 6) 결과는 Plan 011 note에도 기록.

### Step 5: 종합 결론 — 완료

**"어떤 주체의 easy가 좋은 easy인가?"에 대한 답:**

1. **Target 자신의 L-only(Self-easy ∩ Detector-hard)가 유일하게 도움이 되는 easy**
   - 낮은 shortcut (HANS 11%, Hyp-only 18%) + 높은 target confidence (0.999) + 낮은 detector confidence (0.377)
   - 이 세 조건의 조합이 "좋은 easy"를 정의

2. **작은 모델의 easy는 해로움** — 크기 비율(S/T)이 핵심 요인
   - Subject가 작을수록 S-only zone의 shortcut 비율이 높음 (Fig 2, 5)
   - 계열(architecture)보다 크기가 더 중요 (Fig 2 same vs cross arch)

3. **외부 큰 모델의 easy도 도움 안 됨** — Self만 유효
   - Pythia-2.8B(비슷한 크기, 다른 계열)의 L-only: shortcut은 낮지만(10%), target confidence가 낮음(0.899) → OOD -4.40%p
   - Target confidence가 구분 요인이므로, 외부 모델의 판단은 target에 전이 불가

4. **인간의 "쉬움"은 모델의 "쉬움"과 독립적**
   - ChaosNLI: 인간 Easy/Hard 모두 SC ratio 0.09-0.12, overlap 상관 r=-0.04
   - 인간은 의미적 명확성으로 판단, 모델은 표면적 패턴으로 판단

5. **Threshold에 robust** — k=10-50%에서 모든 패턴 유지

---

## 10. 시나리오 판정 결과

분석 결과를 사전 시나리오와 대조:

| 시나리오 | 판정 | 근거 |
|---------|------|------|
| **A: 크기 비율이 핵심** | **가장 부합** | SC(S-only)가 S/T 크기 비율과 일관된 역상관. 계열보다 크기가 더 중요. |
| B: 동일 계열이 핵심 | 부분 지지 | Jaccard는 동일 계열이 약간 높지만, SC(S-only)는 계열 차이 미미. |
| C: Self-easy가 최선 | **강하게 지지** | Phase 3b에서 Self L-only만 OOD 양수. Target confidence가 핵심 구분 요인. |
| D: Threshold에 의존 | 기각 | 패턴이 k=10-50%에서 robust. |
| E: 혼합 패턴 | 부분 해당 | SC ratio만으로는 불충분하고, target confidence가 추가 필요 → 다차원적. |

**종합 결론**: **시나리오 A+B+C 조합 (v3 업데이트)**

T5-xxl(11B)과 Pythia-12B(12B) 추가 후 시나리오 판정 수정:

| 비교 | SC:S-only | Jaccard | 해석 |
|------|:---------:|:-------:|------|
| T5-xxl (same-arch, 3.7×) | **0.151** | **0.530** | same-arch larger → shortcut 최저, overlap 최고 |
| Pythia-12B (cross-arch, 4×) | **0.415** | 0.449 | cross-arch larger → shortcut 여전히 높음 |

→ **크기만으로는 불충분 (시나리오 A 수정)**: Pythia-12B는 T5-XL의 4배인데도 SC:S-only 0.415
→ **같은 계열 + 큰 모델이 핵심 (시나리오 A+B)**: T5-xxl만 SC:S-only 0.151 달성
→ **Self가 최선 (시나리오 C 유지)**: Self는 SC:S-only가 0 (정의상), 외부 모델 중에는 same-arch-larger가 차선

**최종 결론**:
1. **같은 계열의 더 큰 모델**의 easy가 가장 shortcut-free → 학습에 도움될 가능성 높음
2. **다른 계열은 크기와 무관하게** shortcut 비율이 높음 → easy가 전이되지 않음
3. **Self-easy**가 가장 신뢰할 수 있는 easy (target confidence 최고 + shortcut 0)
4. 따라서 실용적 가이드라인: **Self-easy 기반 가중치 큐레이션** 또는 **same-arch-larger의 easy를 활용**

#### 추가 분석: 다수 큰 모델 합의 ∩ 소형 hard (Fig 9)

"여러 큰 모델이 공통으로 easy하다고 합의한 예제 중, 작은 모델은 어려워하는 예제"는 더 순수한 genuine easy일 수 있다는 가설을 검증.

큰 모델 4개: T5-XL(3B), T5-XXL(11B), Pythia-6.9B, Pythia-12B
작은 모델 3개: BERT-mini, T5-small, Pythia-70M (union)

| 합의 조건 | N | SC(HANS) | SC(Hyp) | T-XL conf | Det conf |
|----------|:---:|:---:|:---:|:---:|:---:|
| S-easy (소형 union) | 41,473 | **42.6%** | **50.0%** | 0.924 | 0.683 |
| A3 L-only (T5-XL 1개) | 9,053 | 7.0% | 18.0% | 0.999 | 0.377 |
| 2-large (XL∩XXL) | 6,080 | 5.2% | 20.0% | 0.999 | 0.372 |
| 3-large (XL∩XXL∩P-6.9B) | 3,052 | 3.9% | 23.9% | 0.999 | 0.387 |
| **4-large (all consensus)** | **2,197** | **3.3%** | 27.3% | 0.999 | 0.390 |
| 7-model (large+mid 전부) | 598 | 4.7% | 37.6% | 0.999 | 0.412 |

**발견:**
1. **HANS shortcut은 합의 수에 따라 단조 감소**: 7.0% → 5.2% → 3.9% → **3.3%**. 4개 큰 모델 합의 시 shortcut 3.3%로 거의 0에 근접
2. **Hyp-only SC는 오히려 증가** (18% → 27%): HANS heuristic에 잡히지 않는 표면적 패턴이 남거나, 진짜 쉬운 예제는 hypothesis만으로도 판단 가능한 것일 수 있음
3. **크기-순도 trade-off**: 합의가 엄격할수록 shortcut이 줄지만 예제 수도 급감 (9K → 2.2K)
4. **Sweet spot**: 2-3개 큰 모델 합의 (3K-6K개, SC 4-5%) — 학습에 충분한 크기 + 낮은 shortcut

### Step 6: ChaosNLI × 모델 크기별 Confidence 비교 — 완료 (Fig 10)

**목적**: 인간 난이도(ChaosNLI agreement)와 모델 confidence가 모델 크기에 따라 어떻게 달라지는지 확인.

**방법**: BERT-mini(11M), BERT-small(29M), BERT-base(110M)을 MNLI train 1 epoch 학습 후, ChaosNLI 1,599개 예제에 대한 confidence 측정. (BERT-large는 OOM으로 보류 — 나중에 batch size 조정 후 추가)

**스크립트**: `scripts/analysis_012_chaosnli_model_conf.py`
**출력**: `outputs/plan12_analysis/chaosnli_model_confidence.json`

**결과:**

| 모델 | Params | Pearson r | Acc | Conf spread (VHigh-Low) |
|------|:---:|:---:|:---:|:---:|
| bert-mini | 11M | 0.075 | 50.0% | 0.058 |
| bert-small | 29M | 0.104 | 51.5% | 0.093 |
| bert-base | 110M | **0.131** | **58.5%** | **0.183** |

- **모델이 클수록 인간 agreement와의 상관이 높아짐** (0.075 → 0.131)
- **Conf spread**: 인간 VHigh(≥90%) vs Low(<50%) 간 모델 confidence 차이. bert-mini는 0.058 (거의 flat), bert-base는 0.183 (뚜렷한 기울기)
- → **큰 모델일수록 인간의 "쉬움"에 더 align** — Weight score에서 L(large model conf)을 사용하는 것이 인간 기준과도 방향 일치
- 단, 전체적으로 상관은 약함 (r=0.13) → 인간 난이도와 모델 난이도는 상당히 독립적

---

### Step 7: Weight Score 설계 — Fig 1-9 해석 기반

#### 7.1 핵심 관찰 (Fig별 요약)

**Fig 4b (Binned SC by Subject Confidence)** — Weight Score의 핵심 근거:
- **작은 모델 (BERT-mini 등)**: confidence 상위 bin(0-5%)에서 SC ratio ~0.8-0.9로 급증 → **비선형(exponential-like) 관계**
- **큰 모델 (Pythia-2.8B, T5-large 등)**: SC ratio가 confidence bin에 따라 상대적으로 **flat** (~0.3-0.4)
- → 작은 모델의 confidence가 shortcut의 주된 신호, 큰 모델의 confidence는 약한 신호

**Fig 2b (3-zone SC 비교)**:
- T-only zone (큰 모델만 easy): SC ratio **낮음** → genuine easy 후보
- S-only zone (작은 모델만 easy): SC ratio **높음** → shortcut
- Shared zone: 중간
- → 큰 모델만 easy하고 작은 모델이 hard한 예제 = 가장 순수한 학습 데이터

**Fig 2 (S/T ratio vs SC(S-only))**:
- Subject가 작을수록 S-only zone의 SC ratio가 일관되게 높음
- → 작은 모델의 confidence를 **비선형으로** penalty에 반영해야 함

**Fig 4 (Threshold sensitivity)**:
- k=10%→50%: SC:Easy_S 감소, 패턴 방향 유지
- → threshold에 robust하므로 weight score도 robust할 것으로 기대

**Fig 6 (Phase 3b 검증)**:
- A3 L-only(Self)만 OOD +2.46%p 성공
- **SC ratio만으로는 불충분** (L-only(Self) 18% vs L-only(P-2.8B) 19% — SC 비슷한데 결과 반대)
- **Target confidence가 구분 요인**: Self의 L-only는 target conf 0.999, Pythia의 L-only는 0.899
- → Weight score에 target(large) confidence 반영 필요

#### 7.2 Weight Score 설계 논리

**목표**: 각 학습 예제에 [0, 1] 범위의 가중치를 부여하여:
- Genuine easy (큰 모델 확신 + 작은 모델 불확신) → **높은 weight** (upweight)
- Shortcut (작은 모델 확신) → **낮은 weight** (downweight)
- Hard for both → 중간~낮은 weight (하지만 loss가 높으므로 실효 gradient는 충분)

**핵심 논리**:
1. 작은 모델 confidence(S)가 높으면 shortcut 확률이 **비선형으로** 급증 (Fig 4b) → S²이나 exp(S) penalty
2. 큰 모델 confidence(L)가 높으면 genuine easy → upweight 필요
3. Hard-for-both 예제는 weight이 낮더라도, 학습 시 **loss가 높아서 gradient가 자연히 큼** → weight × loss의 곱이 적정 수준 유지
4. 음수 weight은 피하고, 모든 예제가 최소한 참여 → [0, 1] 정규화

#### 7.3 Weight Score 후보

**Option A: Additive (Shift & Scale)**
```
r(x) = L(x) - α · S(x)²
w(x) = (r(x) - r_min) / (r_max - r_min)     # min-max → [0, 1]
```

- 장점: 직관적, α 하나만 튜닝
- 단점: outlier에 민감, α에 따라 분포 모양 변화

예시 (α=1):

| 유형 | L | S | r | w (approx) |
|------|:-:|:-:|:-:|:-:|
| Genuine easy | 0.99 | 0.3 | 0.90 | **0.95** |
| Hard for both | 0.3 | 0.3 | 0.21 | 0.61 |
| Random | 0.6 | 0.5 | 0.35 | 0.68 |
| Shortcut shared | 0.95 | 0.95 | 0.05 | 0.53 |
| Shortcut S-only | 0.4 | 0.9 | -0.41 | **0.27** |

**Option B: Multiplicative**
```
w(x) = L(x) · (1 - S(x)^γ)     # γ ≥ 2
```

- 장점: 자연스럽게 [0, 1] 범위, 음수 없음, S가 높으면 L과 무관하게 억제
- 단점: hard-for-both (L=0.3)의 weight이 더 낮아짐 → loss와의 곱으로 보상

예시 (γ=2):

| 유형 | L | S | w |
|------|:-:|:-:|:-:|
| Genuine easy | 0.99 | 0.3 | **0.90** |
| Hard for both | 0.3 | 0.3 | 0.27 |
| Random | 0.6 | 0.5 | 0.45 |
| Shortcut shared | 0.95 | 0.95 | **0.09** |
| Shortcut S-only | 0.4 | 0.9 | **0.08** |

#### 7.4 실제 90K 데이터 검증 (Fig 11) — 완료

실제 confidence 데이터(L=T5-large, S=mean(bert-mini, t5-small, pythia-70m))에서 6가지 formula를 비교.

**실제 데이터 분포**:
- L: mean=0.83, >0.9이 66% → 대부분 confident
- S: mean=0.51, <0.5이 48% → 고르게 분포
- Hard-for-both (L<0.5, S<0.5): 9,641개 (10.7%), SC=**0.040** ← 거의 shortcut 아님

**테스트한 Formula:**
1. F1: `L × (1-S)` — 011B에서 사용 중 (linear)
2. F2: `L × (1-S²)` — quadratic penalty
3. F3: `L × (1-S³)` — cubic penalty
4. F4: `1-S²` — L 제거 (quality-only)
5. **F5: `(1-S²) × (0.5+L)`** — hybrid: quality base + L bonus with floor
6. F6: `max(L,0.3) × (1-S²)` — L에 floor 적용

**Category별 Mean Weight 비교:**

| Formula | Genuine easy | SC-shared | SC-Sonly | Hard-both | Separation |
|---------|:---:|:---:|:---:|:---:|:---:|
| F1: L×(1-S) | 0.647 | 0.140 | 0.047 | **0.155** | 0.507 |
| F2: L×(1-S²) | 0.833 | 0.250 | 0.084 | **0.193** | 0.583 |
| F4: 1-S² | 0.854 | 0.167 | 0.210 | **0.893** | 0.645 |
| **F5: (1-S²)×(0.5+L)** | **0.829** | **0.169** | **0.061** | **0.371** | **0.660** |
| F6: max(L,0.3)×(1-S²) | 0.819 | 0.189 | 0.028 | **0.248** | 0.631 |

> Separation = Genuine easy - max(SC-shared, SC-Sonly). 높을수록 좋음.

**Hard-for-both 문제 분석:**

순수 multiplicative (F1~F3)는 L이 곱해지므로 hard-for-both가 심하게 억제됨 (0.15~0.21).
이 예제들의 SC ratio는 0.040으로 shortcut이 거의 아닌데도 낮은 weight을 받음.
loss가 높아서 w×loss는 보상되지만, 불필요한 억제.

L을 완전히 빼면 (F4) hard-both가 0.89로 보존되지만, S-only shortcut도 0.21로 높아져서 분리 실패.

**F5 `(1-S²) × (ε+L)` (ε=0.5)가 최적 균형:**
- `(1-S²)` = shortcut quality filter (S가 높으면 확실히 억제)
- `(ε+L)` = L이 낮아도 ε=0.5가 floor 역할 → hard-both가 살아남음
- Separation 최고 (0.660), hard-both 0.37 (F1 대비 2.4배)
- SC-shared 0.17, SC-Sonly 0.06 — 확실한 억제

**w×loss 관점 (F5):**

| 유형 | w | loss (approx) | w × loss |
|------|:-:|:-:|:-:|
| Genuine easy | 0.83 | ~0.1 | **0.08** (upweight 효과) |
| Hard for both | 0.37 | ~1.2 | **0.44** (충분한 gradient) |
| Shortcut shared | 0.17 | ~0.1 | **0.02** (억제됨) |
| SC S-only | 0.06 | ~0.8 | **0.05** (억제됨) |

→ 모든 카테고리에서 의도한 대로 작동.

**스크립트**: `scripts/analysis_012_weight_score_compare.py`
**Figure**: `figures/fig11_weight_score_compare.png`

#### 7.5 최종 선정: F5

```
w(x) = (1 - S(x)²) × (ε + L(x))
```

- **S**: 소형 모델 confidence의 mean aggregation (bert-mini, t5-small, pythia-70m)
- **L**: target model (또는 same-arch larger) confidence
- **ε = 0.5**: hard-for-both 보호를 위한 floor
- 정규화: min-max → [0, 1] 또는 mean=1 normalize

#### 7.6 F5 예비 실험 결과 (T5-large, lr=5e-5 고정, normalize, 3 splits)

| Condition | Formula | Dev | HANS | OOD avg |
|-----------|---------|:---:|:---:|:---:|
| **F5_max_raw** | (1-S²)×(0.5+L), max | **0.878** | 0.619 | **0.424** |
| **F5_mean_raw** | (1-S²)×(0.5+L), mean | **0.881** | 0.614 | 0.416 |
| B2_prod_mean_raw (참고) | L×(1-S), mean | 0.876 | 0.642 | 0.429 |
| B8_random (참고) | random | 0.880 | 0.675 | 0.435 |

> **주의**: F5는 lr=5e-5 고정으로 LR sweep 없이 실행. 011B 조건들은 아직 2 splits만 완료.
> F5의 weight 분포가 011B와 다르므로 최적 LR이 다를 수 있음 → **LR sweep 필요** (Plan 013 완료 후).

**해석**: Dev는 높은 편이나, HANS/OOD는 011B linear 조건이나 random보다 약간 낮음. LR이 최적이 아닐 가능성이 있으므로, LR sweep 후 재비교 필요.

#### 7.7 다음 단계

1. [x] ChaosNLI 모델 크기별 confidence 비교 완료 (Step 6, Fig 10)
2. [x] Weight score를 실제 90K confidence 데이터에 적용 → 분포 확인 (Fig 11)
3. [x] F5 예비 실험 완료 (lr 고정, 3 splits)
4. [ ] Plan 013 완료 후 F5 LR sweep 실행
5. [ ] Plan 011B 전체 완료 후 F5와 정밀 비교

---

## 11. 이 분석이 불충분할 경우의 대안

이 GPU-free 분석으로 결론이 나지 않으면:

1. **소규모 학습 실험 추가**: 6 target 중 2-3개만, 가장 유망한 subject 3-4개만 → ~30-50 runs
2. **Probing 분석**: 각 easy set으로 학습한 representation의 probing accuracy 비교 (GPU 필요하지만 lightweight)
3. **Influence function**: easy set 예제들의 학습 데이터 기여도 추정 (computationally heavy)

→ 우선 이 분석을 완료하고, 결과를 보고 판단.
