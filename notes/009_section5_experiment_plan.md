# Plan 009: §5 Retraining Experiment — Proof-of-Concept Guideline Validation

## 목적

§3-4에서 발견한 "쉬움의 주체-의존성"에 기반한 데이터 필터링 가이드라인을 proof-of-concept 실험으로 검증한다.

**핵심 메시지**: 쉬운 예제를 무조건 제거하는 것은 과도하다. 자신이 쉽다고 한 예제는 대체로 유용하지만, 소형 모델들이 공통으로 쉽다고 한 예제는 shortcut에 의존할 가능성이 높다. 이때 shortcut 예제의 **존재 자체**가 아니라 **과잉(redundancy)**이 문제이므로, 중복을 줄이되 일부는 보존하는 전략이 가장 효과적이다.

---

## 데이터 제약: 30K Subset 기반 실험

### 기존 실험 인프라
- §3-4의 분석은 MNLI train에서 **30K 랜덤 subset × 3 splits**으로 수행됨 (전체 393K가 아님)
- 각 split에서 20개 모델 × 3 체크포인트의 confidence가 이미 계산되어 있음
- ARC는 train set 자체가 작으므로 (~2.5K) 전체 사용

### §5 실험 적용
- **MNLI**: 기존 30K subset 3개를 그대로 활용. 각 split = 1 seed.
  - 30K 내에서 필터링 → target model 재학습 → 평가
  - 3 splits 결과의 평균 ± 표준편차로 보고
  - 장점: 추가 confidence 계산 불필요, 3-split 일관성 = robustness 증거
- **ARC**: 전체 train set 사용 (원래 전체에 대해 계산했으므로). 3 seeds로 반복.

---

## Target Models (2개)

| Model | Size | Family | 선정 이유 |
|---|---|---|---|
| **BERT-large** | 335M | Encoder | 기존 "easy=harmful" 연구의 주력 계열. HANS 평가와 직접 대응 |
| **Pythia-1B** | 1B | Decoder | Appendix에서 보인 "수렴 경계". 대형 decoder의 대표 |

> T5-large는 2차 실험으로 확장 가능. 우선 2개로 proof-of-concept.

---

## 데이터셋별 실험 구성 (2 models × 2 datasets = 4 combinations)

### MNLI
**학습**: 30K subset × 3 splits (기존 인프라 활용)
**평가**: MNLI-matched dev, HANS, ANLI R1/R2/R3
**Target models**: BERT-large (full FT), Pythia-1B (LoRA)

### ARC
**학습**: ARC-Easy + ARC-Challenge train 합산 (~3.4K), 3 seeds
**평가**: ARC-Easy dev, ARC-Challenge val/test
**Target models**: BERT-large (BertForMultipleChoice), Pythia-1B (LoRA)

### 실행 우선순위
1. BERT-large × ARC (빠름, ~5min/run)
2. Pythia-1B × ARC (중간, ~20min/run)
3. BERT-large × MNLI (느림, ~15min/run)
4. Pythia-1B × MNLI (느림, ~30min/run)

---

## Shortcut Detector: "소형 모델" 정의

각 계열에서 가장 작은 모델 1개를 shortcut detector로 사용한다.

| Family | Shortcut Detector | Size | 근거 |
|---|---|---|---|
| Encoder | BERT-small | 29M | §4.1에서 negation 의존 가장 높음 |
| Enc-Dec | T5 v1.1-small | 77M | 가장 소형 |
| Decoder | Pythia-14M | 14M | 가장 소형 |

**"소형 모델 easy" 합집합**: 위 3개 모델 각각이 쉽다고 한 예제(top-30%)의 합집합(union).

> 교집합이 아닌 합집합을 사용하는 이유: §4.1에서 shortcut 유형이 아키텍처에 민감함을 보였다 (예: lexical overlap은 T5-small이 잘 못 잡음). 교집합으로 하면 특정 계열에서만 포착되는 편향이 누락된다. 합집합은 recall을 높이고, 이후 Step 3(dedup)에서 precision을 보완한다.

### 합집합 크기 예상 (MNLI 30K 기준)
- 각 소형 모델 easy: 30K × 30% = 9K
- 세 모델 간 Jaccard < 0.5이므로, 합집합 ≈ 15K~20K (30K의 50~67%)
- 이는 상당량이므로, Condition 4(전부 제거)는 학습 데이터를 절반 가까이 줄임
- → Condition 5(dedup으로 일부 보존)의 필요성이 더 강해짐

---

## 필터링 조건 (6개)

모든 조건에서 제거된 예제 수를 기록하고, 조건 간 비교가 가능하도록 제거 비율을 보고한다.

### Condition 1: Full (baseline)
- 30K 전체(MNLI) 또는 전체 train(ARC) 사용
- 비교 기준점

### Condition 2: Random-K%
- 전체 데이터에서 랜덤으로 K% 제거
- K = Condition 4에서 실제 제거되는 비율과 동일하게 맞춤
- **역할**: 데이터 양 감소 자체의 효과를 control

### Condition 3: Remove-Self-Easy
- Target model 자신이 쉽다고 한 예제(top-30%) 전부 제거
- **역할**: 기존 "easy examples are harmful" 진영의 접근법 재현
- **예상**: HANS/ARC-C 개선, but MNLI-dev/ARC-Easy 하락 (과도한 제거)

### Condition 4: Remove-Intersection-All
- 소형 모델 3개 **교집합** easy 전부 제거
- 세 계열이 모두 쉽다고 동의한 것 = **가장 보편적인 shortcut**
- **역할**: 가장 보수적인 필터링. "확실한 shortcut만 제거"
- **예상 크기**: Jaccard가 낮으므로 교집합은 작음 (30K의 ~10~15%?)
- **예상**: HANS 소폭 개선, dev 하락 최소. but 특정 아키텍처에서만 포착되는 shortcut은 누락

### Condition 5: Remove-Union-All
- 소형 모델 3개 **합집합** easy 전부 제거
- 어느 계열이든 하나라도 쉽다고 한 것 = **넓은 범위 shortcut 후보**
- **역할**: "소형 모델이 쉽다고 한 것 = shortcut" 가설의 aggressive 적용
- **예상 크기**: 합집합 ≈ 15K~20K (30K의 50~67%)
- **예상**: HANS 큰 개선, but dev 하락도 큼 (과도한 제거)

### Condition 6: Remove-Union-Dedup (제안 방법)
- 소형 모델 3개 합집합 easy를 식별
- 이 중 **redundant 예제만 제거**, 각 클러스터에서 대표 예제 보존 (10~20%)
- **역할**: "shortcut의 과잉이 문제이지 존재 자체가 아니다"를 검증
- **예상**: challenge set 개선 + broad performance 최소 하락 = 최적 균형

---

## Deduplication 방법 (Condition 5)

### Confidence Profile Clustering

**Confidence 스케일 문제**: 소형 모델과 대형 모델의 raw confidence 분포가 크게 다름 (소형은 전반적으로 낮고, 대형은 높음). Raw confidence를 concatenate하면 대형 모델 차원이 클러스터링을 지배함.

**해결: Binary Easy Vector (추천)**

각 예제에 대해 20개 모델 각각이 "easy(top-30%)"로 분류했는지 여부를 binary로 인코딩:
```
x_i = [easy_BERT-small, easy_BERT-base, easy_BERT-large,
       easy_T5-sm, ..., easy_Pythia-12B]   ∈ {0, 1}^20
```

- 스케일 문제 완전 해소 (binary이므로)
- 해석이 명확: "어떤 모델 조합이 이 예제를 쉽다고 봤는가"가 feature
- 같은 shortcut 유형에 의존하는 예제들은 유사한 binary pattern → 같은 클러스터
  - 예: negation shortcut 예제 → 소형 모델 전반에서 1, 대형에서 0
  - 예: lexical overlap 예제 → Pythia 계열에서 1, BERT에서 0
- 추가 계산 비용 = 0 (이미 있는 confidence에서 threshold만 적용)

**대안: Rank-Normalized Confidence Vector**

각 모델별로 confidence를 percentile rank [0, 1]로 변환 후 concatenate:
```
x_i = [rank_BERT-small(c_i), rank_BERT-base(c_i), ..., rank_Pythia-12B(c_i)]   ∈ [0,1]^20
```

- Binary보다 세밀한 정보 보존 (borderline easy vs extremely easy 구분)
- 스케일 통일됨 (percentile이므로)
- 3-checkpoint 활용 시 60차원으로 확장 가능

**현재 결정**: **Binary vector (20차원)로 우선 실험**. 이유:
1. 해석 가능성이 가장 높음
2. 본문 §2의 easy 정의(top-30% binary)와 일관적
3. 결과가 약하면 rank-normalized로 확장

### 클러스터링 파라미터

| 항목 | 기본값 | 근거 |
|---|---|---|
| 방법 | k-means on binary vectors | 간결성 |
| 클러스터 수 | Silhouette score로 결정 (탐색 범위: 10~100) | 데이터 주도적 |
| 대표 보존 비율 | 10% per cluster | max(1, cluster_size × 0.1) |
| 거리 함수 | Hamming distance (binary에 적합) | binary vector |

> 클러스터 수가 결과에 민감한지 robustness check 필요. 20, 50, 100으로 비교.

---

## 평가 메트릭

### Primary metrics

| Dataset | Metric | 의미 |
|---|---|---|
| MNLI-matched dev | Accuracy | Broad performance (하락하면 안 됨) |
| HANS | Accuracy | Shortcut robustness (개선되어야 함) |
| ANLI R1/R2/R3 | Accuracy | Adversarial robustness |
| ARC-Easy dev | Accuracy | Broad performance |
| ARC-Challenge test | Accuracy | 어려운 추론 |

### 보고 방식
- 각 조건 × 3 splits(MNLI) 또는 3 seeds(ARC) 평균 ± 표준편차
- **Δ (Full 대비 변화)**를 주요 비교 지표로 사용
- Broad performance 하락 대비 challenge set 개선의 trade-off를 scatter plot으로 시각화 가능

---

## 실험 결과 (진행 중)

### BERT-large × ARC (10 epochs, lr=1e-5, 3 seeds 평균)

| Condition | N_train | Easy Val | Chal Test | Δeasy | Δchal |
|-----------|---------|----------|-----------|-------|-------|
| C1 Full (baseline) | 3370 | 0.575±.013 | 0.379±.003 | — | — |
| C2 Random | 1136 | 0.475±.001 | 0.301±.019 | -0.100 | -0.078 |
| C3 Self-easy | 2359 | 0.477±.010 | 0.331±.002 | -0.099 | -0.048 |
| C4 Intersect | 3291 | 0.567±.013 | 0.374±.003 | -0.009 | -0.005 |
| C5 Union | 1136 | 0.461±.014 | 0.308±.007 | -0.115 | -0.071 |
| C6 Dedup | 1326 | 0.481±.007 | 0.314±.001 | -0.094 | -0.065 |

**ARC 관찰 (데이터 3.4K로 소규모)**:
1. **C4 Intersect만 baseline 유지** (교집합 79개만 제거 → 영향 거의 없음)
2. **C2/C5 동일 데이터량(1136)에서 C5가 더 나쁨** → union 제거가 random보다 해로움
3. **C3/C6 모두 큰 하락** → ARC 규모에서는 어떤 필터링이든 데이터 부족이 지배적
4. **예상과 달리 challenge가 개선되지 않음** → ARC에서는 shortcut 제거 효과 < 데이터 감소 효과
5. **핵심 문제**: ARC train이 3.4K로 너무 작아서 필터링 실험에 부적합할 수 있음

### BERT-large × MNLI (3 epochs, lr=3e-5, 3 splits 평균) — 진행 중

일부 완료된 결과 (11/18):

| Condition | 완료 splits |
|-----------|-----------|
| C1 Full | 3/3 |
| C4 Intersect | 3/3 |
| C5 Union | 3/3 |
| C6 Dedup | 2/3 |
| C2 Random | 0/3 |
| C3 Self-easy | 0/3 |

### Pythia-1B × ARC — 미실행
### Pythia-1B × MNLI — 미실행

## 예상 결과 및 스토리 (업데이트 필요)

ARC 결과에서 예상이 빗나감. 핵심 원인:
- ARC train = 3.4K → union이 66%를 차지 → C5/C6에서 데이터가 1/3로 줄어 학습 자체가 어려움
- MNLI (30K)에서는 다른 패턴이 나올 가능성 있음 (데이터 충분)
- **플랜 재검토 필요**: ARC에서는 필터링 대신 다른 접근 (weighting, curriculum 등) 고려

---

## 실험 실행 순서

### Phase 1: 데이터 준비
1. 기존 30K × 3 split의 20개 모델 confidence score 확인 (이미 계산됨)
2. 각 split에서 소형 모델 3개(BERT-small, T5-small, Pythia-14M)의 easy set(top-30%) 추출
3. 합집합 구성 → 각 split에서 합집합 크기 확인 및 보고
4. 합집합 예제의 20차원 binary easy vector 구성
5. k-means 클러스터링 수행 → 각 클러스터에서 10% 대표 예제 선택
6. 5개 조건의 학습 데이터 준비 (각 split별로)

### Phase 2: BERT-large 실험 (MNLI)
1. 6개 조건 × 3 splits = 18 runs
2. 각 run: 30K split에서 필터링 → BERT-large fine-tuning → eval on MNLI-dev, HANS, ANLI
3. 예상 GPU time: 18 × ~1h = ~18h (A100 기준)

### Phase 3: Pythia-1B 실험 (ARC)
1. 6개 조건 × 3 seeds = 18 runs
2. 각 run: ARC train에서 필터링 → Pythia-1B fine-tuning → eval on ARC-Easy, ARC-Challenge
3. 예상 GPU time: 18 × ~2h = ~36h (A100 기준)

### Phase 4: 분석 및 시각화
1. 6-조건 비교 테이블 (main result)
2. Broad performance vs Robustness trade-off scatter plot
3. 교집합 vs 합집합 크기 비교 (각 split에서)
4. 합집합 내 클러스터 분석 (어떤 binary pattern이 잡혔는지 → shortcut 유형 해석)
5. 클러스터 수 민감도 분석 (20 vs 50 vs 100)

---

## 하이퍼파라미터 결정 사항

| 항목 | 기본값 | 근거 |
|---|---|---|
| Easy threshold (k%) | 30% | 본문 §2와 동일 |
| Dedup 보존 비율 | 10% | 클러스터당 max(1, size×0.1) |
| k-means 클러스터 수 | Silhouette (10~100) | 데이터 주도적 |
| Binary vector 차원 | 20 (모델 수) | 1 체크포인트 기준 |
| MNLI splits | 3 × 30K | 기존 인프라 |
| ARC seeds | 3 | 학습 seed 변경 |

---

## Discussion 포인트 (§5 끝 또는 §6에)

- **일반화**: 본 실험은 20개 모델의 binary easy profile을 활용했지만, 실전에서는 소형 모델 1개 + target model만으로 유사한 효과를 달성할 수 있다. 소형 모델의 easy set을 shortcut 후보로 삼고, target model의 early checkpoint confidence를 기준으로 redundancy를 판별하면 된다.
- **TracIn 확장**: Gradient 기반 dedup은 binary profile보다 더 직접적으로 "같은 이유로 쉬운" 예제를 식별할 수 있다. 향후 연구에서 검증이 필요하다.
- **커리큘럼 학습과의 관계**: 본 가이드라인은 일괄 필터링이지만, shortcut 예제를 학습 후반에 제시하는 커리큘럼 전략으로 확장 가능하다.
- **보존 비율의 최적화**: 10%는 heuristic 값이며, 과제와 데이터셋에 따라 최적값이 달라질 수 있다. 5%, 10%, 20% 비교를 appendix에 포함 가능.

---

## Reference

- Pruthi, G., et al. (2020). Estimating Training Data Influence by Tracing Gradient Descent. NeurIPS.
- Swayamdipta, S., et al. (2020). Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics. EMNLP.
