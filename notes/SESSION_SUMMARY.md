# ERASE 프로젝트 세션 요약

> 이 문서는 여러 세션에 걸친 논의 흐름과 실험 결과를 기록합니다.
> 새 세션 시작 시 이 파일을 먼저 읽으면 컨텍스트를 파악할 수 있습니다.
> 마지막 업데이트: 2026-03-29

---

## 1. 프로젝트 개요

**연구 질문**: "쉬운 예제"를 정의하는 것은 모델/아키텍처에 따라 달라지는가? (proxy confidence 기반 난이도 측정)

**핵심 메시지**: 쉬운 예제를 무조건 제거하는 것은 과도하다. 자기 모델이 쉽다고 한 예제는 대체로 유용하지만, 소형 모델들이 공통으로 쉽다고 한 예제는 shortcut 의존 가능성이 높다. 이때 shortcut 예제의 **과잉(redundancy)**이 문제이므로, 중복을 줄이되 일부는 보존하는 전략이 가장 효과적이다.

**사용 모델 (총 20개)**:
- BERT: mini, small, medium, base, large
- T5 v1.1: small, base, large, xl, xxl
- Pythia: 14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b

**데이터셋**: MNLI (30K subset × 3 splits), ARC (전체 train set, 3 seeds)

**평가**: MNLI→HANS/ANLI-r1,r2,r3 (OOD), ARC→Easy/Challenge dev

---

## 2. 논의 흐름 (시간순)

### Phase 1: 초기 셋업 및 BERT/Pythia LR Search (Plan 001)
- MNLI에서 BERT (mini~large), Pythia (14m~12b) 모델들의 LR search 수행
- 작은 모델은 큰 LR (3e-4, 1e-3), 큰 모델은 작은 LR 필요
- 1B 이상 모델은 LoRA (r=32)로 튜닝
- **fp16 사용 금지** → fp32 default, OOM시 bf16
- Pythia에서 padding=left, logits 스케일 이슈 확인/해결
- checkpoint 저장 비활성화 (100GB 저장공간 제한)

### Phase 2: T5 모델 도입 및 문제 해결 (Plan 005)
- T5는 seq2seq 특성상 evaluation이 까다로움
  - 초기: 첫 decoder token logits로 eval → 성능 안 나옴
  - confidence 포화 문제 (T5가 pre-training에서 이미 분류 학습)
  - **해결: sentinel token 방식** — `mnli premise: ... hypothesis: ... answer: <extra_id_0>` 형태로 입력, decoder target으로 `<extra_id_0> entailment` 등 사용
  - Optimizer: AdamW → **Adafactor**로 변경 (T5 원래 방식)
  - warmup 필수, linear decay 사용
  - T5-base: lr=5e-5, T5-large: lr=5e-5, T5-xl: lr=1e-4

### Phase 3: Confidence 측정 (Plan 001, 005)
- 20개 모델 × MNLI 3 splits에서 proxy confidence (avg over epochs up to best dev) 측정
- 상위 30%를 "easy"로 분류
- Jaccard heatmap으로 모델 간 easy set 중첩도 시각화 (Fig 1)

### Phase 4: ARC 실험 확장 (Plan 002)
- ARC에서도 동일하게 LR search → confidence
- BERT: small, base, large / T5: small, base, large, xl / Pythia: 70m~12b
- 일부 seed 학습 불안정 → 대체 seed 사용
- ARC에서도 Jaccard heatmap 생성 (Fig 2)
- **인간 난이도와 모델 난이도 비교**: Spearman correlation, human-easy vs model-easy Jaccard

### Phase 5: 정성적 분석 (Plan 006, 007, 008)
- ARC에서 human easy-model hard, human hard-model easy 예제 분석
- Knowledge type annotation (Procedural / Factual / Situational)
  - 465개 예제에 수동 annotation
  - **발견**: model-hard subset에서 Procedural knowledge 비율이 base rate 대비 +13pp 높고, Factual은 -18pp 낮음
- Fig 4: diverging bar chart로 knowledge type 편차 시각화

### Phase 6: Figure 다듬기
- Fig 1: 9×9 Jaccard heatmap → 20×20 full heatmap (appendix)
- Fig 2: ARC human-easy overlap + difficulty/grade alignment (3-panel combined figure)
- Fig 4: knowledge type diverging bar chart
- 학술 논문 스타일로 반복 수정 (font, grid, marker, spacing 등)

### Phase 7: §5 Retraining 실험 (Plan 009) — **현재 진행 중**
- T5-XL × MNLI (k=30%) 완료 → C5_union이 OOD 최고
- BERT-large × MNLI k=5%, k=10% 완료 → condition 간 차이 미미
- BERT-large 학습 불안정 이슈 (4/36 run 실패, 재실행 중)
- BERT-large × ARC 완료 → 데이터 부족으로 필터링 효과 없음

### Phase 8: Jaccard Robustness Figure (k% sweep)
- 20개 모델 전체 pairwise Jaccard를 k=1~50%에서 계산
- 3×3 grid figure (rows=anchor sizes, cols=target families)
- BERT/T5/Pythia × MNLI/ARC = 6 figure → 하나의 9×6 mega figure로 통합
- 주요 발견: same-family 유사도 > cross-family, 작은 anchor일수록 cross-family 유사도 상승

---

## 3. Plan 009 실험 결과

### 필터링 조건 설명
| Condition | 설명 | Train 크기 |
|-----------|------|-----------|
| C1_full | 30K 전체 사용 (baseline) | 30000 |
| C2_random | C5와 동일한 수만큼 랜덤 제거 | ~27374 |
| C3_self_easy | T5-XL 자신의 top-30% easy 제거 | 28500 |
| C4_intersect | 소형 3모델 교집합 easy 제거 | ~29361 |
| C5_union | 소형 3모델 합집합 easy 제거 | ~27374 |
| C6_dedup | 합집합 easy 중 redundant만 제거, 대표 보존 | ~27634 |

### T5-XL × MNLI (k=30%, 3-split 평균)

| Condition | Dev | HANS | ANLI-r1 | ANLI-r2 | ANLI-r3 | OOD avg |
|-----------|-----|------|---------|---------|---------|---------|
| **C1_full** | 0.9011 | 0.6984 | 0.5220 | 0.3910 | 0.3967 | 0.5020 |
| **C2_random** | 0.9019 | 0.7031 | 0.5183 | 0.3930 | 0.3928 | 0.5018 |
| **C3_self_easy** | 0.9017 | 0.7006 | 0.5250 | 0.3783 | 0.3850 | 0.4972 |
| **C4_intersect** | 0.9037 | 0.6704 | 0.5253 | 0.3897 | 0.4064 | 0.4980 |
| **C5_union** | 0.9016 | **0.7051** | **0.5430** | 0.3893 | 0.4014 | **0.5097** |
| **C6_dedup** | 0.9004 | 0.6792 | 0.5140 | 0.3757 | 0.4008 | 0.4924 |

### Split별 상세

**C1_full**:
- split1: dev=0.8999 hans=0.7262 r1=0.5250 r2=0.4060 r3=0.3917
- split2: dev=0.9017 hans=0.6852 r1=0.4990 r2=0.3710 r3=0.3850
- split3: dev=0.9016 hans=0.6837 r1=0.5420 r2=0.3960 r3=0.4133

**C2_random**:
- split1: dev=0.9037 hans=0.7496 r1=0.5220 r2=0.3920 r3=0.3717
- split2: dev=0.9031 hans=0.6769 r1=0.5240 r2=0.4020 r3=0.4067
- split3: dev=0.8988 hans=0.6829 r1=0.5090 r2=0.3850 r3=0.4000

**C3_self_easy_t5xl**:
- split1: dev=0.9009 hans=0.7002 r1=0.5140 r2=0.3750 r3=0.3933
- split2: dev=0.9013 hans=0.7279 r1=0.5190 r2=0.3680 r3=0.3558
- split3: dev=0.9029 hans=0.6738 r1=0.5420 r2=0.3920 r3=0.4058

**C4_intersect**:
- split1: dev=0.9047 hans=0.6940 r1=0.5360 r2=0.4090 r3=0.4150
- split2: dev=0.9040 hans=0.6631 r1=0.5060 r2=0.3700 r3=0.3842
- split3: dev=0.9025 hans=0.6541 r1=0.5340 r2=0.3900 r3=0.4200

**C5_union**:
- split1: dev=0.9005 hans=0.7110 r1=0.5180 r2=0.3770 r3=0.3850
- split2: dev=0.9022 hans=0.6491 r1=0.5380 r2=0.3880 r3=0.3967
- split3: dev=0.9022 hans=0.7553 r1=0.5730 r2=0.4030 r3=0.4225

**C6_dedup**:
- split1: dev=0.8994 hans=0.6869 r1=0.5340 r2=0.3890 r3=0.4025
- split2: dev=0.8999 hans=0.7384 r1=0.5090 r2=0.3790 r3=0.3933
- split3: dev=0.9018 hans=0.6124 r1=0.4990 r2=0.3590 r3=0.4067

### BERT-large × MNLI k=5% (3 epochs, lr=3e-5, 실패 run 제외)

| Condition | Dev | HANS | ANLI avg | OOD avg |
|-----------|------|------|----------|---------|
| C1 Full | 0.8062 | 0.5473 | 0.2689 | 0.3385 |
| C2 Random | 0.8064 | 0.5315 | 0.2683 | 0.3341 |
| C3 Self-easy | 0.8055 | 0.5219 | 0.2683 | 0.3317 |
| C4 Intersect | 0.8045 | 0.5171 | 0.2768 | 0.3369 |
| C5 Union | 0.8019 | 0.5403 | 0.2751 | 0.3414 |
| C6 Dedup | 0.8025 | 0.5434 | 0.2743 | 0.3416 |

### BERT-large × MNLI k=10% (3 epochs, lr=3e-5, 실패 run 제외)

| Condition | Dev | HANS | ANLI avg | OOD avg |
|-----------|------|------|----------|---------|
| C1 Full | 0.8057 | 0.5355 | 0.2705 | 0.3368 |
| C2 Random | 0.8006 | 0.5265 | 0.2719 | 0.3355 |
| C3 Self-easy | 0.8012 | 0.5322 | 0.2771 | 0.3409 |
| C4 Intersect | 0.8074 | 0.5361 | 0.2711 | 0.3374 |
| C5 Union | 0.7878 | 0.5372 | 0.2800 | 0.3443 |
| C6 Dedup | 0.7901 | 0.5381 | 0.2765 | 0.3419 |

> **BERT-large 관찰**: T5-XL과 달리 condition 간 차이가 매우 작음. BERT-large의 학습 불안정성도 문제 (4/36 run 실패).

---

## 4. 주요 인사이트 & 결정 사항

### 실험 관련
- T5 모델은 sentinel token 방식 + Adafactor + warmup이 필수
- BERT/Pythia는 AdamW 사용
- best dev epoch 기준으로 HANS/ANLI 측정
- MNLI 5 epoch, ARC 5 epoch (이전에는 10이었으나 축소)
- confidence = best dev epoch 이하 epoch들의 average

### 분석 관련
- 소형 모델 shortcut detector: BERT-small, T5-small, Pythia-14m
- easy threshold: top-30% (기존), 5%/10%도 실험함
- Jaccard heatmap: 같은 family 내 유사도가 높고, cross-family는 낮음
- 인간 난이도와 모델 난이도는 moderate correlation (Spearman ~0.3-0.4)

### 인프라
- 로컬 GPU 2장 (RTX 5090 32GB)
- 저장공간 100GB → HF 캐시 정리, checkpoint 미저장

---

## 5. 파일 구조

### Plans (notes/)
- `001_mnli_proxy_confidence.md` — MNLI 실험 메인 계획
- `001a_mnli_experiment_log.md` — MNLI 세부 실험 기록
- `002_arc_proxy_confidence.md` — ARC 실험 계획 및 결과
- `003_difficulty_based_training.md` — 난이도 기반 학습 실험
- `004_math_training.md` — MATH 학습 실험 (pythia/qwen/llama)
- `006_qualitative_analysis.md` — 정성적 분석
- `007_fig4_knowledge_type_spec.md` — knowledge type figure spec
- `008_knowledge_type_annotation_guide.md` — annotation 가이드
- `009_section5_experiment_plan.md` — §5 retraining 실험 계획

### 주요 코드 (scripts/)
- 학습/confidence 측정 스크립트들
- 분석/figure 생성 스크립트 (`analysis_fig*.py`, `analysis_appendix_*.py`)

### Outputs (outputs/)
- `plan0/` — BERT/Pythia MNLI confidence
- `plan5/` — T5 sentinel 방식 MNLI confidence + Pythia 일부
- `plan2_v2/` — ARC confidence
- `plan9/` — §5 retraining 결과

### Figures (figures/)
- `fig1_*` — Jaccard heatmap
- `fig2_*` — ARC human-model alignment
- `fig4_*` — Knowledge type diverging bar chart
- `fig5_*` — Jaccard overlap vs k% threshold (anchor별 3×3 grid, mega figure)
- `appendix_*` — 20×20 full heatmaps

---

## 6. 다음 할 일 (TODO)

- [ ] Plan 009: BERT-large k=5%/10% 실패 run 재실행 (4 runs, 진행 중)
- [ ] Plan 009: BERT-large × MNLI k=30% 나머지 (C2, C3, C6_split1)
- [ ] Plan 009: Pythia-1B 실험 여부 결정 (BERT-large 결과가 약해서 재검토 필요)
- [ ] Fig 5 mega figure 레이아웃 최종 다듬기
- [ ] 논문 figure 최종 다듬기
- [ ] GitHub 업데이트 (git@github.com:KoreaMGLEE/erase.git)
