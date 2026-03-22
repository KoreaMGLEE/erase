# Plan 4: MATH 데이터셋 훈련 및 평가

## 목적

MATH (Hendrycks et al.) 데이터셋에서 Pythia 대형 모델과 Qwen2.5-1.5B를 fine-tune하여 수학 문제 풀이 성능을 측정. 생성 기반(generation) 평가로 정확한 정답 추출이 필요.

## 데이터셋 정보

```
EleutherAI/hendrycks_math
├── algebra:                  train=1,744 / test=1,187
├── counting_and_probability: train=771   / test=474
├── geometry:                 train=870   / test=479
├── intermediate_algebra:     train=1,295 / test=903
├── number_theory:            train=869   / test=540
├── prealgebra:               train=1,205 / test=871
└── precalculus:              train=746   / test=546
Total:                        train=7,500 / test=5,000
```

**Level 분포 (train):**

| Level | 개수 | 비율 |
|-------|------|------|
| Level 1 (쉬움) | 564 | 7.5% |
| Level 2 | 1,348 | 18.0% |
| Level 3 | 1,592 | 21.2% |
| Level 4 | 1,690 | 22.5% |
| Level 5 (어려움) | 2,304 | 30.7% |

> Level이 곧 human difficulty. Plan 3과 유사하게 difficulty-based 분석 가능.

## 모델

| 모델 | Params | HuggingFace ID | Tuning | 근거 |
|------|--------|---------------|--------|------|
| pythia-1b | 1B | EleutherAI/pythia-1b | LoRA r=32 | Plan 0/2에서 검증 |
| pythia-2.8b | 2.8B | EleutherAI/pythia-2.8b | LoRA r=32 | 더 큰 Pythia |
| pythia-6.9b | 6.9B | EleutherAI/pythia-6.9b | LoRA r=32 | 최대 Pythia |
| Qwen2.5-1.5B | 1.5B | Qwen/Qwen2.5-1.5B | LoRA r=32 | 최신 모델, 수학에 강점 |

> Qwen2.5는 수학 특화 학습이 포함되어 있어 Pythia 대비 높은 성능 기대.

## 학습 방식

### 입력/출력 형식

```
Problem: {problem}
Solution: {solution}
```

- Causal LM next-token prediction: Problem 부분은 loss mask (-100), Solution 부분에만 loss 계산
- 이전 Plan과 동일한 방식

### 평가 방식: Generation + Answer Extraction

MATH는 multiple-choice가 아닌 **자유 형식 정답**이므로, 모델이 solution을 생성하고 최종 답을 추출해야 함.

```python
# 생성
prompt = f"Problem: {problem}\nSolution:"
generated = model.generate(prompt, max_new_tokens=512)

# 정답 추출: \boxed{...} 패턴
import re
match = re.search(r'\\boxed\{(.+?)\}', generated)
predicted_answer = match.group(1) if match else ""

# 비교: 정답의 \boxed{} 내용과 비교
gold_answer = extract_boxed(solution)
correct = normalize_answer(predicted_answer) == normalize_answer(gold_answer)
```

> MATH의 정답은 `\boxed{답}` 형식으로 solution 끝에 포함. 정규화(분수, 소수 등) 필요.

### 정답 정규화

```python
def normalize_answer(ans):
    """LaTeX 수식 정규화."""
    ans = ans.strip()
    # 분수: \frac{a}{b} → a/b
    # 공백 제거
    # 소수점 표현 통일
    # sympy를 활용한 수학적 동치 비교 고려
    return ans
```

## Phase 1: LR Search

### LR 후보

| 모델 | LR 후보 | 근거 |
|------|---------|------|
| pythia-1b (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | Plan 0/2 LoRA 범위 |
| pythia-2.8b (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | 동일 |
| pythia-6.9b (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | 동일 |
| Qwen2.5-1.5B (LoRA) | 5e-5, 1e-4, 3e-4, 1e-3, 3e-3 | LoRA 표준 범위 |

### 학습 설정

| 항목 | 값 |
|------|-----|
| Data | MATH train 전체 (7,500) |
| Epochs | 3 |
| Batch size | 4 (grad accumulation 4 → effective 16) |
| Max length | 1024 (수학 풀이가 길 수 있음) |
| Eval | MATH test (5,000) — generation + \boxed{} 추출 |
| Metric | Accuracy (정답 일치 비율) |

> 7,500개 × 3 epoch = 22,500 steps (effective bs=16 → ~1,406 steps). ARC(3,370)보다 크므로 3 epoch이면 충분.

### Run 수

4 models × 5 LRs = **20 runs**

### 시간 추정

| 모델 | 학습 (3ep) | Eval (5K 생성) | 총 |
|------|-----------|--------------|-----|
| pythia-1b | ~30분 | ~60분 | ~90분 |
| pythia-2.8b | ~45분 | ~90분 | ~2.3시간 |
| pythia-6.9b | ~60분 | ~120분 | ~3시간 |
| Qwen2.5-1.5B | ~30분 | ~60분 | ~90분 |

> Eval이 generation 기반이라 학습보다 오래 걸림. 5,000문제 × max 512 tokens.

**총: ~60시간** (순차), 2 GPU 병렬 시 **~30시간**

---

## Phase 2: Confidence 측정 (LR search 후)

Best LR로 3 seed 학습, confidence 측정.

### Confidence 정의

MATH는 생성 기반이므로 confidence 정의가 ARC/MNLI와 다름:

1. **Token-level confidence**: 생성된 토큰들의 평균 log-probability
2. **Answer confidence**: `\boxed{}`를 생성할 때의 토큰 확률
3. **Binary correctness**: 정답 여부 자체를 confidence proxy로 사용

> 가장 실용적인 것은 (3) 정답 여부. 여러 seed에서의 정답 비율 = confidence.

### Run 수

4 models × 3 seeds = **12 runs**

---

## 전체 Run 계산

| Phase | Runs | 예상 시간 |
|-------|------|---------|
| Phase 1: LR Search | 20 | ~30시간 (2 GPU) |
| Phase 2: Confidence | 12 | ~18시간 (2 GPU) |
| **총** | **32 runs** | **~48시간 (2 GPU)** |

---

## 주의사항

1. **Generation 평가 비용**: 5,000문제 × 512 tokens 생성은 시간이 많이 걸림. eval 빈도를 줄이거나 test subset 사용 고려.
2. **정답 정규화**: LaTeX 수식 비교가 까다로움. `\frac{1}{2}` vs `0.5` vs `1/2` 등. sympy 활용 권장.
3. **Qwen2.5 토크나이저**: Pythia와 다른 토크나이저. chat template이 있을 수 있으니 base model 사용.
4. **메모리**: pythia-6.9b는 bf16 + gradient checkpointing + bs=4 필요. Qwen2.5-1.5B는 fp32 가능.
5. **Max length**: MATH solution이 매우 길 수 있음 (특히 Level 5). 1024 토큰으로 truncation.
6. **LoRA target modules**: Pythia는 `query_key_value`, Qwen2.5는 `q_proj`, `v_proj` (확인 필요).

---

## 실행 순서

1. **환경 준비**: Qwen2.5 모델 다운로드, sympy 설치 (정답 비교용)
2. **학습 스크립트 작성**: generation + \boxed{} 추출 로직 포함
3. **Phase 1**: LR search (4 models × 5 LRs, 2 GPU 병렬)
4. **Phase 2**: Confidence 측정 (4 models × 3 seeds)
5. **분석**: Level별 accuracy, 모델 간 비교, Pythia vs Qwen2.5
