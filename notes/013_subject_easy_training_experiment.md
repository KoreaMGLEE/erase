# 013: 주체별 Easy Set 학습 실험 — "누구의 쉬움으로 학습해야 하는가?"

> §3 핵심 실험: 다양한 주체가 쉽다고 판단한 예제로 target model을 학습하고 일반화 성능 비교
> 작성: 2026-04-02

---

## 0. 논문 구조에서의 위치

```
§2: 주체마다 쉬워하는 예제가 다르다 (Jaccard, zone 분석)
    ↓
§3: 그래서 다른 주체의 easy로 학습하면 일반화가 어떻게 달라지는가? [← 이 실험]
    ↓
§4: 왜 그런가? (shortcut ratio, confidence 분포, consensus 분석 — Plan 012 결과)
```

**§3의 핵심 질문**: "주체 S가 쉽다고 판단한 상위 k%만으로 target T를 학습하면, S와 k에 따라 일반화 성능이 어떻게 달라지는가?"

---

## 1. 실험 설계

### 1.1 Target Models (8개)

BERT 3개 + Pythia 5개:

| 계열 | Target | 파라미터 | 학습 방식 | 비고 |
|------|--------|---------|----------|------|
| BERT (encoder-only) | BERT-small | 29M | Full FT | 소형 |
| | BERT-base | 110M | Full FT | 중형 |
| | BERT-large | 335M | Full FT | 대형 |
| Pythia (decoder-only) | Pythia-70M | 70M | Full FT | 소형 |
| | Pythia-160M | 160M | Full FT | |
| | Pythia-410M | 410M | Full FT | 중형 |
| | Pythia-1B | 1B | Full FT (or LoRA) | 대형 |
| | Pythia-6.9B | 6.9B | LoRA | 초대형 |

> **T5 계열**: BERT/Pythia에서 패턴 확인 후 보충 실험으로 추가 (encoder-decoder 대표)
> **T5-XL(3B)**: Plan 012 결과 검증이 필요한 경우 핵심 조건만 추가

### 1.2 Subject Models (easy set 제공 주체, 9개)

| 크기 | BERT 계열 | Pythia 계열 |
|------|----------|------------|
| 소형 | — | Pythia-70M |
| 중형 | BERT-base (110M) | Pythia-410M |
| 대형 | BERT-large (335M) | Pythia-2.8B |
| 초대형 | — | Pythia-6.9B, Pythia-12B |

+ **Self** (target 자기 자신) → target별로 자동 포함

> BERT 계열은 base(110M)와 large(335M) 2개만 subject로 사용. BERT-mini는 너무 작아 subject로서 의미가 제한적.
> Subject별 easy set은 이미 Plan 012에서 계산 완료 (confidence 상위 k%).

### 1.3 Easy Threshold (3개)

| k | 의미 | MNLI 크기 (per split 30K) |
|---|------|---------------------------|
| 5% | 매우 쉬운 예제만 | **1,500개** |
| 10% | 쉬운 예제 | **3,000개** |
| 30% | 넓은 easy set | **9,000개** |

> Plan 012 Fig 4b에서 편향 예제 비율이 confidence 상위로 갈수록 exponential하게 증가.
> k=5%는 편향이 극심할 것으로 예상 (특히 소형 subject), k=30%는 상대적으로 완화.
> 세 지점이면 "k가 작아질수록(더 strict한 easy일수록) 어떤 패턴"인지 확인 가능.

### 1.4 비교 구조 & Baseline

**핵심 비교축: 쉬운 예제들 간의 비교**

이 실험의 목적은 "쉬운 예제 vs 랜덤"이 아니라, **"누구의 쉬운 예제가 더 좋은 쉬운 예제인가"**를 밝히는 것.
따라서 같은 k% 내에서 subject 간 비교가 본문의 핵심이 됨.

| ID | 조건 | 역할 |
|----|------|------|
| S_i-k% | Subject S_i의 confidence 상위 k% | **핵심 비교 대상** (subject 간 비교) |
| R-k% | Random k% (크기 매칭) | 참고 baseline (점선으로만 표시) |
| Full | 전체 30K | 성능 상한 참고 |

> **본문 서술 방향**: "동일한 수(k%)의 쉬운 예제로 학습하더라도, 누구의 기준으로 쉬운 예제인지에 따라 일반화가 크게 달라진다."
> Random은 "다양한 난이도가 섞인 데이터"이므로 easy set보다 좋을 수 있지만, 이는 비교의 주축이 아님.
> 핵심은 **가장 좋은 easy set과 가장 나쁜 easy set의 격차**, 그리고 그것이 target 크기에 따라 어떻게 변하는가.

### 1.5 데이터셋

**1차: MNLI (30K)**
- 평가: Dev-matched, HANS, ANLI-r1/r2/r3
- OOD avg = (HANS + ANLI-r1 + ANLI-r2 + ANLI-r3) / 4

**2차 (보충): ARC**
- BERT/Pythia 소형은 ARC에서 random 수준일 수 있으므로, MNLI 결과 확인 후 결정

---

## 2. 실험 조건 매트릭스

### 2.1 각 Target에 대한 조건

Target T에 대해:

| 조건 | 학습 데이터 | k값 |
|------|-----------|-----|
| Self-5% | T 자신의 confidence 상위 5% | 5% |
| Self-10% | T 자신의 confidence 상위 10% | 10% |
| Self-30% | T 자신의 confidence 상위 30% | 30% |
| S_i-5% | Subject S_i의 confidence 상위 5% | 5% |
| S_i-10% | Subject S_i의 confidence 상위 10% | 10% |
| S_i-30% | Subject S_i의 confidence 상위 30% | 30% |
| R-5% | Random 5% (크기 매칭) | 5% |
| R-10% | Random 10% | 10% |
| R-30% | Random 30% | 30% |
| Full | 전체 30K | — |

### 2.2 Subject 조건 수

Target별로 subject 수가 다름 (self는 별도):

| Target | Self | Same-arch subjects | Cross-arch subjects | Total subjects (Self 포함) |
|--------|:----:|:------------------:|:-------------------:|:--------------------------:|
| BERT-small (29M) | 1 | 2 (base, large) | 5 (P-70M~P-12B) | 8 |
| BERT-base (110M) | 1 | 1 (large) | 5 | 7 |
| BERT-large (335M) | 1 | 1 (base) | 5 | 7 |
| Pythia-70M | 1 | 4 (410M~12B) | 2 (B-base, B-large) | 7 |
| Pythia-160M | 1 | 4 (70M제외, 410M~12B) | 2 | 7 |
| Pythia-410M | 1 | 4 (70M, 2.8B~12B) | 2 | 7 |
| Pythia-1B | 1 | 4 (70M, 410M, 2.8B~12B) | 2 | 7 |
| Pythia-6.9B | 1 | 4 (70M, 410M, 2.8B, 12B) | 2 | 7 |

> Subject에 Self가 포함되고, 나머지 subject 중 자기 자신은 제외.
> Subject 목록에 없는 모델이 target인 경우(BERT-small, Pythia-160M, Pythia-1B, Pythia-6.9B), Self만 별도 추가.

### 2.3 총 Run 수 계산

Target별 평균: (~7.5 subjects × 3 thresholds + 3 random + 1 full) × 3 splits ≈ (22.5 + 3 + 1) × 3 ≈ **80 runs**

전체: 8 targets × ~80 runs ≈ **640 runs**

| Target 크기 | 예상 1-run 시간 | Target 수 | 총 시간 (병렬 2 GPU) |
|------------|:-------------:|:---------:|:------------------:|
| 소형 (29M-70M) | ~3-5분 | 2 | ~4시간 |
| 중소형 (110M-160M) | ~5-10분 | 2 | ~7시간 |
| 중형 (335M-410M) | ~10-20분 | 2 | ~13시간 |
| 대형 (1B) | ~20-40분 | 1 | ~13시간 |
| 초대형 (6.9B, LoRA) | ~15-30분 | 1 | ~10시간 |
| **총계** | | **8** | **~47시간 (≈2일)** |

> GPU 4장이면 ~1일, 8장이면 ~반나절.
> Pythia-6.9B는 LoRA로 학습하므로 Full FT 대비 시간 절감.

---

## 3. 학습 인프라

### 3.1 Learning Rate

**Target별 + 데이터 크기별 LR sweep 필요.**

| Target 계열 | LR 후보 |
|------------|---------|
| BERT (Full FT) | {5e-6, 1e-5, 2e-5, 3e-5, 5e-5} |
| Pythia (Full FT) | {5e-6, 1e-5, 2e-5, 5e-5, 1e-4} |
| Pythia-1B (LoRA, if applicable) | {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} |
| Pythia-6.9B (LoRA) | {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} |

**LR sweep 절차:**
1. 각 target × 각 데이터 크기 대표(5%, 10%, 30%)에서 Random 조건으로 LR sweep
2. 5 LR × 3 크기 × 1 split = 15 runs per target
3. 8 targets × 15 = 120 runs (소형은 빠르므로 실질적으로 반나절)
4. 최적 LR을 해당 (target, 크기) 조합의 모든 조건에 적용

### 3.2 Epoch 설정

| 데이터 크기 | 예제 수 | Epoch |
|-----------|:------:|:-----:|
| 5% | 1,500개 | 20 |
| 10% | 3,000개 | 15 |
| 30% | 9,000개 | 8 |
| Full | 30,000개 | 5 |

### 3.3 평가 메트릭

| 메트릭 | 용도 |
|-------|------|
| Dev accuracy | In-distribution 성능 |
| HANS accuracy | Shortcut 의존도 (MNLI) |
| ANLI-r1,r2,r3 accuracy | OOD 일반화 |
| OOD avg | (HANS + ANLI-r1 + r2 + r3) / 4 |

### 3.4 CV 설계

- 3-fold split (기존과 동일)
- 결과 리포팅: 평균 ± 표준편차

---

## 4. 단계적 실행 계획

### Phase 1: LR Sweep — ✅ 완료 (105/105 runs)

원격 서버(RTX 5090 × 2)에서 실행. 7 targets × 3 k% × 5 LRs = 105 runs, split1 기준.

**최적 LR 결과:**

| Target | k=5% | k=10% | k=30% | 대표 LR |
|--------|:---:|:---:|:---:|:---:|
| bert-small (29M) | 2e-5 | 2e-5 | 3e-5 | 2e-5 |
| bert-base (110M) | 3e-5 | 3e-5 | 3e-5 | 3e-5 |
| bert-large (335M) | 3e-5 | 3e-5 | 3e-5 | 3e-5 |
| pythia-70m (70M) | 1e-4 | 5e-5 | 1e-5 | k별 상이 |
| pythia-160m (160M) | 2e-5 | 1e-5 | 1e-5 | 1e-5 |
| pythia-410m (410M) | 1e-5 | 5e-6 | 5e-6 | 5e-6 |
| pythia-1b (1B) | 1e-5 | 1e-5 | 5e-6 | 1e-5 |

> 결과 위치: `outputs/plan13/lr_sweep/`
> 스크립트: `scripts/experiment_013_lr_sweep.py`

### Phase 2+3: 소형+중형 Target — 대기 중

**스크립트 준비 완료**: `scripts/experiment_013_main.py`

**GPU 분배 계획 (로컬 2 + 원격 2 = 4장):**

| GPU | Target | Runs | 예상 시간 |
|-----|--------|:---:|:---:|
| 로컬 0 | bert-small (29M) | 84 | ~2.3h |
| 로컬 1 | bert-base (110M) | 75 | ~4.8h |
| 원격 0 | pythia-70m + pythia-160m | 150 | ~9.6h |
| 원격 1 | pythia-410m (410M) | 75 | ~10.4h |

> 총 384 runs, 병목 ~10시간. 로컬 GPU는 011B 완료 후 투입.

**실행 명령 (준비됨):**
```bash
# 로컬
python3 scripts/experiment_013_main.py --gpu 0 --targets bert-small
python3 scripts/experiment_013_main.py --gpu 1 --targets bert-base

# 원격
python3 /workspace/erase/scripts/experiment_013_main.py --gpu 0 --targets pythia-70m pythia-160m
python3 /workspace/erase/scripts/experiment_013_main.py --gpu 1 --targets pythia-410m
```

### Phase 4: 대형 Target — 대기

BERT-large(335M), Pythia-1B.
Phase 2+3 결과 확인 후 실행. ~13시간 예상.

### Phase 4.5: 초대형 Target — 대기

Pythia-6.9B (LoRA). 별도 스크립트 필요.

### Phase 5: 검증 & 보충

- Phase 2-4 결과 기반 패턴 확인 후:
  - [ ] T5 계열 보충 필요 시: T5-small, T5-base, T5-large
  - [ ] T5-XL 핵심 조건 재현
  - [ ] ARC 데이터셋 실험

---

## 5. 예상 결과 & Figure 설계

> **비교 원칙**: 모든 figure에서 핵심 비교는 **같은 k% 내에서 subject 간 비교**. Random은 참고선(점선)으로만 표시.

### Figure A: Subject별 OOD 성능 비교 (Target 크기별 패널)

**구조**: 4개 대표 패널 (BERT-small, BERT-large, Pythia-70M, Pythia-6.9B), x축 = subject (크기순), y축 = OOD avg (절대값), k=10% 고정. 색상 = same-arch(초록) / cross-arch(주황) / self(파랑). Random은 점선.

**핵심 메시지**: 같은 수의 "쉬운 예제"인데, 누구 기준이냐에 따라 OOD가 이만큼 달라진다.

**예상 패턴**:
- 소형 target: subject 간 OOD 격차가 크지 않음 (전부 비슷하게 낮음)
- 대형 target: subject 간 격차가 큼. Self와 same-arch-larger의 easy가 최상위, cross-arch-smaller의 easy가 최하위.

### Figure B: "Best easy vs Worst easy" 격차 × Target 크기 (핵심 figure)

**구조**: x축 = target 크기 (29M → 6.9B), y축 = OOD. 각 target에서 best subject의 OOD(상단), worst subject의 OOD(하단), self의 OOD를 연결. k별 3개 선(5%, 10%, 30%).

**핵심 메시지**: 모델이 커질수록 "좋은 easy"와 "나쁜 easy"의 격차가 벌어지고, self-easy의 위치가 worst에서 best 쪽으로 이동한다.

**예상 패턴**:
- 소형: best와 worst의 격차 작음, self는 worst 근처
- 대형: 격차가 크게 벌어짐, self는 best 근처
- k=5%에서 패턴이 가장 극단적

### Figure C: Same-arch vs Cross-arch Subject 비교

**구조**: x축 = subject 크기 / target 크기 (log scale), y축 = OOD (절대값), 색상 = same-arch vs cross-arch. 대형 target(BERT-large, Pythia-1B)에서만.

**핵심 메시지**: 같은 크기비의 subject여도, 같은 계열이면 OOD가 높고 다른 계열이면 낮다.

**예상 패턴**:
- same-arch larger: OOD 최고 수준
- cross-arch: 크기와 무관하게 OOD가 낮음
- Plan 012의 SC 분석(T5-xxl SC=0.15 vs Pythia-12B SC=0.42)과 일치 예상

### Figure D: Threshold(k%)에 따른 subject 간 격차 변화

**구조**: x축 = k% (5, 10, 30), y축 = OOD avg, 선 = subject 그룹 (self / same-arch-larger / same-arch-smaller / cross-arch-smaller). Target = Pythia-6.9B (초대형).

**핵심 메시지**: k가 작을수록(더 strict한 easy일수록) 좋은 easy와 나쁜 easy의 격차가 벌어진다. → §4에서 "왜?"로 연결.

**예상 패턴**:
- k↓: 좋은 subject(self, same-arch-larger)의 OOD는 유지/소폭 하락, 나쁜 subject(cross-arch-smaller)의 OOD는 급락
- 이는 Plan 012 Fig 4b의 "작은 모델일수록 high-confidence에서 shortcut이 exponential 증가" 패턴과 직접 대응

---

## 6. 논문에서의 제시 방식

### §3 구조 (예상)

**§3.1 실험 설계** (1단락)
- Target 8개 (BERT 3 + Pythia 5), Subject 9개, Threshold 3개
- "주체 S가 쉽다고 판단한 상위 k%만으로 학습" 설명

**§3.2 주체별 학습 결과** (Figure A + 주요 결과 서술)
- 핵심 발견: "주체에 따라 같은 easy set이 도움이 되기도, 해가 되기도 한다"
- Subject 크기별 패턴, same-arch vs cross-arch 차이

**§3.3 Self-easy의 이중성: 모델 크기에 따른 전환** (Figure B — 논문의 핵심)
- "작은 모델의 self-easy는 해롭지만, 큰 모델의 self-easy는 도움이 된다"
- 전환점 분석

**§3.4 Threshold와 일반화의 관계** (Figure D)
- "k가 작을수록 주체 간 차이가 극단적으로 벌어진다"
- → §4의 "왜?" 분석으로 연결되는 브릿지

### §4에서 연결

§3의 결과를 받아서:
- "왜 작은 모델의 easy가 해로운가?" → shortcut ratio (Plan 012 Fig 4b)
- "왜 cross-arch가 안 되는가?" → Jaccard/SC 분석 (Plan 012 Fig 2, 5)
- "왜 self-easy가 대형에서만 유효한가?" → target confidence 분석 (Plan 012 Fig 6)
