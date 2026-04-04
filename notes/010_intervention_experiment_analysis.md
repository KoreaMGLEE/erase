# Intervention (Retraining) 실험 분석

> 소형 모델이 "쉽다"고 판단한 예제를 제거/조정하여 target model의 OOD 성능을 개선하는 실험.
> 마지막 업데이트: 2026-03-31

---

## 0. Executive Summary

```
핵심 발견 (Dev-OOD trade-off 관점):

1. T5-XL에서만 실용적 개선 달성:
   - Phase 2 B3 Graded: Dev -0.5%p, OOD +2.35%p ← 유일하게 좋은 trade-off
   - Phase 1 C5 Union:  Dev ±0%p,  OOD +0.77%p  ← dev 무손상, 소폭 개선

2. BERT-large에서는 실용적 개선 없음:
   - OOD 큰 개선 조건 (B5/B7): Dev -5%p 이상 하락 → 실용적이지 않음
   - Dev 유지 조건 (B1/B2/B4): OOD +0.15~0.63%p → 노이즈 범위 내
   - k% 올려도 동일 딜레마: k=30% C5는 OOD +3.94%p이나 Dev -16%p

3. 필터링 메커니즘 자체는 유효:
   - Random 제거 대비 targeted 제거가 일관되게 우수
   - Detector 모델 선택에 robust
   - 다만 BERT-large는 효과 크기가 노이즈를 넘지 못함

4. 모델 규모/능력이 핵심 요인:
   - T5-XL (3B, Dev 90%, HANS 72%): 여유가 있어 필터링 효과 흡수 가능
   - BERT-large (335M, Dev 81%, HANS 52%): 기본 학습도 불안정, 필터링 효과 관찰 어려움
```

### 전체 실험 맵

| 실험 | Target Model | 전략 | k% | Runs | 상태 |
|------|-------------|------|-----|------|------|
| Phase 1 (Plan 009) | T5-XL | C1-C6 제거 | 5% | 18 | 완료 |
| Phase 1 | BERT-large | C1-C6 제거 | 5% | 18 | 완료 |
| Phase 1 | BERT-large | C1-C6 제거 | 10% | 18 | 완료 |
| Phase 1 | BERT-large | C1-C6 제거 | 30% | 12* | 부분완료 |
| Phase 1 | BERT-large | C1-C6 ARC | 30% | 18 | 완료 |
| **Phase 2 (Plan 011)** | BERT-large | B0-B9 zone 기반 | 30% | 30 | 완료 |
| Phase 2 | BERT-large | B0,B3 k-sweep | 10,20% | 12 | 완료 |
| Phase 2 | BERT-large | Det B 검증 | 30% | 9 | 완료 |
| Phase 2 | T5-XL | B0-B5 zone 기반 | 30% | 18 | 완료 |
| | | | **총** | **153+** | |

> *C2, C3 미실행. C4 split1 실패, C6 2 splits만 완료.

---

## 1. 실험 설계

### Phase 1: 단방향 필터링 (Plan 009)

소형 모델이 easy라고 판단한 예제를 **제거만** 하는 전략.

| Condition | 설명 | 핵심 질문 |
|-----------|------|----------|
| C1 Full | 전체 데이터 (baseline) | — |
| C2 Random | C5와 동일 수만큼 랜덤 제거 | 데이터 양 효과 통제 |
| C3 Self-easy | Target model 자신의 easy 제거 | 자기 참조 vs 외부 참조 |
| C4 Intersect | 소형 3모델 교집합 easy 제거 | 보수적 필터링 |
| C5 Union | 소형 3모델 합집합 easy 제거 | 공격적 필터링 |
| C6 Dedup | Union 중 중복만 제거, 대표 10% 보존 | Redundancy vs 존재 자체 |

소형 모델: BERT-small, T5-v1.1-small, Pythia-14M

### Phase 2: 양방향 zone 기반 개입 (Plan 011)

소형 easy와 대형 easy를 **교차**하여 5개 zone을 정의하고, zone별 가중치를 조절.

**5-Zone 정의** (k=30% threshold):

| Zone | 조건 | 비율 | 의미 |
|------|------|------|------|
| S-all | 소형 3개 모두 easy, 대형 아님 | ~7% | 고확신 shortcut |
| S-partial | 소형 1-2개 easy, 대형 아님 | ~19% | 부분 shortcut |
| Shared | 소형+대형 모두 easy | ~19% | 혼합 |
| L-only | 대형만 easy | ~11% | Genuine easy |
| Neither | 양쪽 모두 어려움 | ~44% | 어려운 예제 |

**10가지 조건:**

| Condition | S-all | S-partial | Shared | L-only | Neither | 목적 |
|-----------|-------|-----------|--------|--------|---------|------|
| B0 Baseline | 1× | 1× | 1× | 1× | 1× | 기준선 |
| B1 Down-S-all | **0×** | 1× | 1× | 1× | 1× | S-all만 제거 |
| B2 Up-L-only | 1× | 1× | 1× | **2×** | 1× | L-only만 강화 |
| B3 Graded | **0×** | **0.5×** | 1× | **2×** | 1× | 점진적 조절 |
| B4 Graded-soft | 0.5× | 0.75× | 1× | 1.5× | 1× | 완만한 조절 |
| B5 Binary | **0×** | **0×** | 1× | **2×** | 1× | S 전면 제거 + L 강화 |
| B6 Random-matched | B3 크기로 랜덤 | | | | | 데이터 양 통제 |
| B7 Down-all-S | **0×** | **0×** | 1× | 1× | 1× | S 전면 제거만 (강화 없이) |
| B8 Up-S-all (neg) | **2×** | 1× | 1× | 1× | 1× | 역방향 (shortcut 강화) |
| B9 Down-L-only (neg) | 1× | 1× | 1× | **0×** | 1× | 역방향 (genuine 제거) |

---

## 2. Phase 1 결과: 단방향 필터링

### 2.1 Dev-OOD Trade-off 종합 (baseline 대비 Δ, %p)

| Condition | T5-XL k=5% | | BERT-lg k=5% | | BERT-lg k=10% | | BERT-lg k=30% | |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | Δ Dev | Δ OOD | Δ Dev | Δ OOD | Δ Dev | Δ OOD | Δ Dev | Δ OOD |
| C2 Random | +0.08 | -0.02 | +0.03 | -0.75 | -0.51 | -0.13 | — | — |
| C3 Self-easy | +0.06 | -0.48 | -0.07 | -0.68 | -0.45 | +0.41 | — | — |
| C4 Intersect | +0.26 | -0.40 | -0.16 | -0.44 | +0.17 | +0.06 | -1.18 | +0.18 |
| **C5 Union** | **+0.05** | **+0.77** | -0.43 | +0.29 | **-1.85** | **+1.15** | **-15.98** | **+3.94** |
| C6 Dedup | -0.07 | -0.96 | -0.37 | +0.31 | -1.38 | +0.46 | -4.89 | +2.09 |

> **핵심**: T5-XL C5만 dev 무손상 + OOD 개선. BERT-large는 OOD 개선이 클수록 dev 하락도 큼.

### 2.2 T5-XL (k=5%, LoRA, 3 epochs, lr=1e-4)

| Condition | N_train | Dev | HANS | ANLI-r1 | ANLI-r2 | ANLI-r3 | OOD avg |
|-----------|---------|-----|------|---------|---------|---------|---------|
| C1 Full | 30,000 | 90.11 | 69.84 | 52.20 | 39.10 | 39.67 | 50.20 |
| C2 Random | ~27,374 | 90.19 | 70.31 | 51.83 | 39.30 | 39.28 | 50.18 |
| C3 Self-easy | 28,500 | 90.17 | 70.06 | 52.50 | 37.83 | 38.50 | 49.72 |
| C4 Intersect | ~29,361 | 90.37 | 67.04 | 52.53 | 38.97 | 40.64 | 49.80 |
| **C5 Union** | ~27,374 | **90.16** | **70.51** | **54.30** | 38.93 | 40.14 | **50.97** |
| C6 Dedup | ~27,634 | 90.04 | 67.92 | 51.40 | 37.57 | 40.08 | 49.24 |

> Dev 90.16 (손상 없음) + OOD +0.77%p. C5=C2 동일 크기인데 C5 우수 → 필터링 효과.

### 2.3 BERT-large × MNLI (k=5%, full FT, 3 epochs, lr=3e-5)

| Condition | N_train | Dev | HANS | ANLI avg | OOD avg | Δ OOD |
|-----------|---------|-----|------|----------|---------|-------|
| C1 Full | 30,000 | 80.62 | 54.73 | 26.89 | 33.85 | — |
| C2 Random | ~27,411 | 80.65 | 52.91 | 26.49 | 33.10 | -0.75 |
| C3 Self-easy | 28,500 | 80.55 | 52.19 | 26.83 | 33.17 | -0.68 |
| C4 Intersect | ~29,341 | 80.46 | 52.11 | 27.17 | 33.41 | -0.44 |
| C5 Union | ~27,411 | 80.19 | 54.03 | 27.51 | 34.14 | +0.29 |
| C6 Dedup | ~27,634 | 80.25 | 54.34 | 27.43 | 34.16 | +0.31 |

> Dev 유지되나 OOD 개선이 +0.3%p로 노이즈 범위 내.

### 2.4 BERT-large × MNLI (k=10%)

| Condition | N_train | Dev | HANS | ANLI avg | OOD avg | Δ OOD |
|-----------|---------|-----|------|----------|---------|-------|
| C1 Full | 30,000 | 80.57 | 53.55 | 27.05 | 33.68 | — |
| C2 Random | ~24,495 | 80.06 | 52.65 | 27.19 | 33.55 | -0.13 |
| C3 Self-easy | 27,000 | 80.12 | 53.22 | 27.71 | 34.09 | +0.41 |
| C4 Intersect | ~28,800 | 80.74 | 53.61 | 27.11 | 33.74 | +0.06 |
| C5 Union | ~24,495 | 78.72 | 54.86 | 28.16 | 34.83 | +1.15 |
| C6 Dedup | ~25,041 | 79.19 | 53.63 | 27.64 | 34.14 | +0.46 |

> C5: OOD +1.15%p이나 Dev -1.85%p. 개선 대비 비용이 큼.

### 2.5 BERT-large × MNLI (k=30%, 부분 완료)

| Condition | N_train | Dev | HANS | OOD avg | Δ Dev | Δ OOD |
|-----------|---------|-----|------|---------|-------|-------|
| C1 Full | 30,000 | 80.84 | 53.76 | 33.50 | — | — |
| C4 Intersect | ~25,023 | 79.66 | 53.72 | 33.68 | -1.18 | +0.18 |
| C5 Union | ~16,332 | 64.86 | 54.41 | 37.44 | **-15.98** | +3.94 |
| C6 Dedup | ~17,917 | 75.95 | 55.01 | 35.59 | -4.89 | +2.09 |

> OOD +3.94%p이나 Dev 65% → 모델로서 쓸 수 없는 수준.

### 2.6 BERT-large × ARC (k=30%)

| Condition | N_train | Easy Val | Chal Test |
|-----------|---------|----------|-----------|
| C1 Full | 3,370 | 57.5±1.3 | 37.9±0.3 |
| C5 Union | 1,136 | 46.1±1.4 | 30.8±0.7 |

> 데이터 3.4K → 필터링 시 1.1K만 남음. **데이터 부족으로 실험 부적합.**

### 2.7 Phase 1 결론

1. **T5-XL C5 Union만 실용적 성공** — Dev 손상 없이 OOD +0.77%p
2. **BERT-large는 dev-OOD 딜레마** — k% 올리면 OOD↑ but Dev↓↓, k% 낮추면 효과 미미
3. **필터링 메커니즘은 유효** — C2 Random vs C5 Union 비교에서 일관되게 확인
4. **C3 Self-easy는 해로움** — 자기 참조 제거는 유의미한 정보도 제거
5. **C6 Dedup < C5 Union** — shortcut의 존재 자체가 해로움, redundancy만의 문제가 아님

---

## 3. Phase 2 결과: 양방향 zone 기반 개입

### 3.1 BERT-large (k=30%, 5 epochs, lr=2e-5, 3 splits 평균)

| Condition | Dev | Δ Dev | HANS | ANLI-r1 | ANLI-r2 | ANLI-r3 | OOD avg | Δ OOD |
|-----------|-----|-------|------|---------|---------|---------|---------|-------|
| **B0 Baseline** | 80.52 | — | 52.34 | 23.20 | 28.20 | 30.31 | 33.51 | — |
| B1 Down-S-all | 80.08 | -0.44 | 55.34 | 23.03 | 27.87 | 30.33 | 34.14 | +0.63 |
| B2 Up-L-only | 80.49 | -0.03 | 54.02 | 22.20 | 28.77 | 30.67 | 33.92 | +0.41 |
| B3 Graded | 79.58 | -0.94 | 52.84 | 24.07 | 28.07 | 30.97 | 33.99 | +0.48 |
| B4 Graded-soft | 80.37 | -0.15 | 53.11 | 23.40 | 27.53 | 30.58 | 33.66 | +0.15 |
| B5 Binary | 75.35† | **-5.17** | 53.54 | 27.87 | 29.83 | 32.64 | 35.97 | +2.46 |
| B6 Random-matched | 80.57 | +0.05 | 55.09 | 22.17 | 27.93 | 29.94 | 33.78 | +0.27 |
| B7 Down-all-S | 75.44† | **-5.08** | 55.47 | 26.33 | 30.40 | 32.47 | 36.17 | +2.66 |
| B8 Up-S-all (neg) | 80.98 | +0.46 | 54.87 | 22.90 | 28.53 | 29.86 | 34.04 | +0.53 |
| B9 Down-L-only (neg) | 79.15 | -1.37 | 53.95 | 22.17 | 28.33 | 29.97 | 33.61 | +0.10 |

> † B5 split3=67.94, B7 split3=69.43 불안정. split1,2만 평균 시에도 dev ~79 수준.

**BERT-large Dev-OOD Trade-off 요약:**

| 그룹 | 조건 | Δ Dev | Δ OOD | 판정 |
|------|------|-------|-------|------|
| Dev 유지 | B1, B2, B4, B8 | -0.4~+0.5 | +0.15~+0.63 | OOD 개선 미미 (노이즈 범위) |
| Dev 하락 | B3 | -0.94 | +0.48 | 비용 대비 효과 부족 |
| Dev 대폭 하락 | B5, B7 | **-5.1~-5.2** | +2.5~+2.7 | **OOD 개선되나 dev 손실 과대** |
| 통제 | B6 Random | +0.05 | +0.27 | Random과 구별 안 됨 |

> **BERT-large 결론: dev를 유지하면서 OOD를 유의미하게 개선하는 조건이 없음.**

### 3.2 T5-XL (k=30%, LoRA, 5 epochs, lr=1e-4, 3 splits 평균)

| Condition | Dev | Δ Dev | HANS | ANLI-r1 | ANLI-r2 | ANLI-r3 | OOD avg | Δ OOD |
|-----------|-----|-------|------|---------|---------|---------|---------|-------|
| **B0 Baseline** | 90.29±0.11 | — | 71.58±2.58 | 52.50±1.35 | 39.13±1.40 | 39.11±1.04 | 50.58 | — |
| B1 Down-S-all | 90.42±0.13 | +0.13 | 70.23±3.78 | 53.97±1.77 | 40.10±1.55 | 41.64±0.72 | 51.49 | +0.91 |
| B2 Up-L-only | 90.23±0.64 | -0.06 | 68.39±4.52 | 53.23±0.38 | 39.00±0.82 | 39.53±0.97 | 50.04 | -0.54 |
| **B3 Graded** | **89.79±0.63** | **-0.50** | **74.69±3.64** | **54.90±1.00** | **40.73±0.71** | **41.39±1.62** | **52.93** | **+2.35** |
| B4 Graded-soft | 90.03±0.19 | -0.26 | 68.64±1.02 | 53.30±1.80 | 39.43±1.97 | 40.33±1.23 | 50.43 | -0.15 |
| B5 Binary | 88.15±3.68† | -2.14 | 65.79±11.81† | 53.80±0.89 | 39.30±0.78 | 40.97±2.26 | 49.97 | -0.61 |

> † B5 split3 불안정 (dev=83.90, HANS=52.16). split1,2만: dev=90.28, HANS=72.61, OOD=51.48 (+0.90).

**T5-XL Dev-OOD Trade-off 요약:**

| 그룹 | 조건 | Δ Dev | Δ OOD | 판정 |
|------|------|-------|-------|------|
| **실용적 성공** | **B3 Graded** | **-0.50** | **+2.35** | **Dev 거의 유지 + OOD 의미있는 개선** |
| Dev 유지, 소폭 개선 | B1 Down-S-all | +0.13 | +0.91 | 양호하나 효과 작음 |
| 무효과 | B2, B4 | -0.1~-0.3 | -0.5~-0.2 | 효과 없음 |
| 불안정 | B5 Binary | -2.14 | -0.61 | split3 붕괴, 해로움 |

> **T5-XL B3 Graded: Dev -0.50%p로 OOD +2.35%p 달성. 전 실험 통틀어 유일한 실용적 성공.**

### 3.3 Robustness 검증

**Detector Set 비교 (BERT-large, k=30%)**

| Detector Set | 소형 모델 | B0 OOD | B3 Δ | B5 Δ |
|-------------|----------|--------|------|------|
| A (기본) | BERT-small, T5-small, Pythia-14M | 33.51 | +0.48 | +2.46 |
| B | BERT-mini, T5-small, Pythia-70M | 33.40 | +0.42 | +2.68 |

> 두 세트 모두 동일 패턴. Detector 선택에 robust. (단, 두 세트 모두 BERT-large의 dev-OOD 딜레마는 동일.)

**k% Sweep (B3 Graded, BERT-large)**

| k% | B0 Dev | B3 Dev | Δ Dev | B0 OOD | B3 OOD | Δ OOD |
|----|--------|--------|-------|--------|--------|-------|
| 10% | 80.59 | 80.07 | -0.52 | 33.69 | 34.42 | +0.73 |
| 20% | 80.59 | 79.81 | -0.78 | 34.12 | 33.55 | -0.57 |
| 30% | 80.52 | 79.58 | -0.94 | 33.51 | 33.99 | +0.48 |

> k%에 무관하게 효과 미미하고 불안정. BERT-large에서 B3 Graded는 작동하지 않음.

---

## 4. 가설 검증

### BERT-large

| 가설 | 결과 | 판정 |
|------|------|------|
| H1: L-only upweight 도움 | B2: Dev -0.03, OOD +0.41 | **약한 지지** (노이즈 범위) |
| H2: S-all 제거 도움 | B1: Dev -0.44, OOD +0.63 | **약한 지지** (노이즈 범위) |
| H3: 양방향 > 단방향 | B5(+2.46) > B1(+0.63), B2(+0.41) | OOD만 보면 지지, **but dev -5%p** |
| H4: Graded > Binary | B3(+0.48) < B5(+2.46) | OOD만 보면 기각, **but trade-off로는 둘 다 실패** |
| H5: 효과 ≠ 데이터 감소 | B7(+2.66) >> B6(+0.27) | **지지** — targeted > random |
| H6: S-partial 보존 가치 | B3(+0.48) < B7(+2.66) | OOD만 보면 기각, **but B7도 dev -5%p** |

> BERT-large에서 H3, H4, H6은 dev 비용을 무시할 때만 성립. Trade-off 고려 시 모든 전략이 비실용적.

### T5-XL

| 가설 | 결과 | 판정 |
|------|------|------|
| H2: S-all 제거 도움 | B1: Dev +0.13, OOD +0.91 | **지지** |
| H4: Graded > Binary | B3: Dev -0.50, OOD +2.35 vs B5: Dev -2.14, OOD -0.61 | **강한 지지** |
| H1: L-only upweight 도움 | B2: Dev -0.06, OOD -0.54 | **기각** |

> T5-XL에서 B3 Graded만이 dev-OOD 양면에서 성공.

---

## 5. 종합 해석

### 왜 T5-XL에서만 작동하는가?

| | BERT-large (335M) | T5-XL (3B) |
|--|-------------------|------------|
| Baseline Dev | 80.5% | 90.3% |
| Baseline HANS | 52.3% (≈random) | 71.6% (양호) |
| 학습 안정성 | 불안정 (실패 runs 다수) | 안정 (0 실패) |
| 데이터 제거 시 | Dev 급락 | Dev 소폭 하락 |
| **실용적 개선** | **없음** | **B3 Graded: Dev -0.5, OOD +2.35** |

**해석:**
1. **여유 용량(headroom) 가설**: T5-XL은 Dev 90%로 학습이 충분히 되어 있어, 일부 데이터를 제거/조절해도 broad performance를 유지할 여유가 있음. BERT-large는 Dev 81%로 학습 자체가 빠듯해서 데이터 조작에 취약.

2. **Shortcut 의존도 차이**: T5-XL HANS 72%는 shortcut을 부분적으로만 사용한다는 뜻. 점진적 조절(B3 Graded)로 최악의 shortcut만 줄이면 됨. BERT-large HANS 52%는 shortcut에 크게 의존하므로 전면 제거(B5/B7)가 필요하지만, 그러면 데이터가 너무 줄어 Dev가 망가짐.

3. **학습 불안정성**: BERT-large full FT는 본질적으로 불안정 (Plan 009에서 5/36 실패, Plan 011에서 B5/B7 split3 붕괴). 이 노이즈가 condition 간 차이(+0.3~0.6%p)를 mask.

### 모델 간 공통 발견

1. **Targeted > Random**: C5 vs C2, B7 vs B6 모두 targeted 제거가 우수 → 필터링 메커니즘 자체는 유효
2. **Detector robust**: Set A, Set B 동일 패턴 → 소형 모델 세부 선택은 중요하지 않음
3. **Binary 불안정**: B5 split3가 두 모델 모두에서 붕괴 → 과도한 데이터 제거의 위험

### Phase 1 vs Phase 2 비교

| | Phase 1 최선 (C5 Union) | Phase 2 최선 (B3 Graded) |
|--|------------------------|-------------------------|
| **T5-XL** | Dev ±0, OOD +0.77 | Dev -0.50, **OOD +2.35** |
| **BERT-large** | Dev -0.4~-16 (k% 의존), OOD +0.3~+3.9 | Dev -0.9, OOD +0.48 |

> T5-XL: Phase 2 B3 Graded가 Phase 1 C5 Union의 3배 효과. Zone 기반 점진적 조절이 단순 제거보다 우수.
> BERT-large: Phase 1이든 Phase 2든 실용적 개선 없음.

---

## 6. 논의: 한계 및 미해결 문제

### 6.1 BERT-large의 근본적 한계
- Dev 81%는 일반적 BERT-large MNLI 성능 (~86%)보다 낮음 → 30K 데이터 + 학습 설정 문제 가능
- Full FT의 고유한 불안정성 → LoRA로 전환하면 개선될 수 있음
- 3-split CV의 분산이 커서 +0.5%p 수준의 차이를 감지하기 어려움

### 6.2 C6 Dedup이 기대 이하
- 가설: shortcut의 과잉(redundancy)이 문제 → dedup으로 해결
- 실제: C5 Union (전부 제거) > C6 Dedup (중복만 제거)
- 결론: **shortcut의 존재 자체가 해로움**. redundancy만의 문제가 아님

### 6.3 Negative Control 불일치
- B8 (S-all 2×): 해롭지 않음 (+0.53%p). S-all이 ~7%로 너무 작아서 영향 미미?
- B9 (L-only 0×): 거의 무효과 (+0.10%p). L-only도 ~11%로 작음

### 6.4 T5-XL B3의 해석 주의
- B3 Graded의 OOD +2.35%p는 의미있지만, 3 splits의 표준편차를 고려하면 통계적 유의성은 제한적
- HANS 74.69±3.64 vs baseline 71.58±2.58 → 분포가 겹침
- 더 많은 splits 또는 seed로 검증 필요

---

## 7. 논문 반영 방향

### §2.4 Analysis 확장

> **From diagnosis to prescription.**
> Subject-dependence 분석은 기존 방법의 한계를 지적하는 데 그치지 않고, 학습 가이드라인을 제시한다:
>
> (1) 소형 모델이 공통으로 쉽다고 판단하는 예제(S-all)는 아키텍처/스케일 초월 보편적 shortcut → downweight
> (2) 소형 모델 일부만 쉽다고 판단하는 예제(S-partial)는 아키텍처-특이적 shortcut → 약하게 downweight
> (3) 대형 target 모델만 쉽다고 판단하는 예제(L-only)는 genuine easy → upweight
>
> 단, 이 차등 양방향 개입(B3 Graded)은 **충분한 headroom이 있는 모델(T5-XL)에서만 실용적**이며,
> 기본 학습이 불안정한 모델(BERT-large)에서는 dev-OOD trade-off가 성립하지 않음.

### Chapter 3과의 연결

- **Curriculum Debiasing** (Ch.3): 시간 축에서의 bidirectional (biased→unbiased 스케줄)
- **Bidirectional Intervention** (이 실험): 데이터 축에서의 bidirectional (영역별 차등 가중)
- 두 접근이 상보적이며, 조합도 가능 (future work)

---

## 8. 실험 인프라

| 항목 | 값 |
|------|-----|
| GPU | RTX 5090 32GB × 2 |
| BERT-large run | ~7.5분 (5 epochs, 30K) |
| T5-XL run | ~97분 (5 epochs, 30K, LoRA+bf16) |
| 총 완료 runs | Plan 009: 101 + Plan 011: 69 = **170 runs** |

---

## 9. 파일 위치

| 항목 | 경로 |
|------|------|
| **Plan 009 (Phase 1)** | |
| T5-XL 결과 | `outputs/plan9/mnli_t5xl_k5/results/` |
| BERT-lg k=5% | `outputs/plan9/mnli_bert_k5/results/` |
| BERT-lg k=10% | `outputs/plan9/mnli_bert_k10/results/` |
| BERT-lg k=30% | `outputs/plan9/mnli/results/` |
| ARC | `outputs/plan9/arc_bert/results/` |
| Indices | `outputs/plan9/{mnli,mnli_bert_k5,mnli_bert_k10}/split{1,2,3}/` |
| 스크립트 | `scripts/experiment_009_*.py` |
| **Plan 011 (Phase 2)** | |
| BERT-lg B0-B9 | `outputs/plan11/results/` (30 files) |
| BERT-lg k-sweep | `outputs/plan11/results_ksweep/` (12 files) |
| Det B 검증 | `outputs/plan11_detB/results/` (9 files) |
| T5-XL B0-B5 | `outputs/plan11/results_t5xl/` (18 files) |
| Zone 데이터 | `outputs/plan11/split{1,2,3}/zones.json` |
| 스크립트 | `scripts/experiment_011_*.py` |
