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

1. **Generation 평가 비용**: 5,000문제 × 512 tokens 생성은 시간이 많이 걸림. eval 500개 subset 사용.
2. **정답 정규화**: LaTeX 수식 비교가 까다로움. `\frac{1}{2}` vs `0.5` vs `1/2` 등. sympy 활용 권장.
3. **메모리**: pythia-6.9b는 bf16 + gradient checkpointing + bs=2 필요.
4. **Max length**: MATH solution이 매우 길 수 있음 (특히 Level 5). 1024 토큰으로 truncation.
5. **LoRA target modules**: Pythia는 `query_key_value`, LLaMA/Qwen은 `q_proj`, `v_proj`.
6. **Instruct 모델**: chat template 적용 필수. system prompt로 `\boxed{}` 형식 유도.

---

## 실험 결과

### Base Model LR Search (SFT)

| 모델 | Params | LR | Ep1 Acc | Ep2 Acc | Ep3 Acc | Best |
|------|--------|-----|---------|---------|---------|------|
| **pythia-1b** | 1B | 5e-5 | 0.4% | 0.8% | 0.4% | **0.8%** |
| **pythia-6.9b** | 6.9B | 5e-5 | 2.6% | 1.8% | 3.0% | **3.0%** |
| **llama3.2-3b** | 3B | 5e-5 | 6.8% | 8.0% | 9.4% | **9.4%** |
| **qwen2.5-1.5b** | 1.5B | 5e-5 | **30.0%** | 28.8% | 27.2% | **30.0%** |
| qwen2.5-1.5b | 1.5B | 1e-4 | 29.0% | **30.0%** | 28.8% | **30.0%** |

### 주요 관찰

1. **Pythia는 MATH에서 학습 불가**: 1B~6.9B 모두 loss가 수렴하지 않고 acc < 3%. LaTeX/수학 토큰을 pretrain에서 학습한 적 없음.
2. **Qwen2.5-1.5B**: 모든 LR에서 ~30%로 동일. **fine-tuning 효과 없음 — pretrain 성능 그대로.** Epoch 진행 시 오히려 성능 하락.
3. **LLaMA3.2-3B**: 9.4%로 Pythia보다 나으나 Qwen에 크게 못 미침.
4. **SFT의 한계**: reference 풀이를 토큰 단위로 따라하는 방식이 수학 문제에서 효과적이지 않음. 풀이 경로가 다양하므로 loss가 높게 유지.

### 방향 전환: Instruct 모델 + GRPO

Base model SFT로는 한계. 다음 단계:
1. **Instruct 모델** (LLaMA3.2-3B-Instruct, Qwen2.5-1.5B-Instruct)로 SFT 재시도 — chat template 적용, instruction following 활용
2. **GRPO**: 정답만 맞추면 reward → 풀이 경로 자유도 허용. Instruct 모델이 기본 policy로 적합.

---

## Instruct Model LR Search

### 모델

| 모델 | Params | HuggingFace ID |
|------|--------|---------------|
| LLaMA3.2-3B-Instruct | 3B | meta-llama/Llama-3.2-3B-Instruct |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen/Qwen2.5-1.5B-Instruct |

### 학습 형식

Chat template 적용:
```
<system> You are a math problem solver. Solve step by step and put final answer in \boxed{}.
<user> {problem}
<assistant> {solution}
```
Loss는 assistant response (풀이) 부분에만 계산.

### LLaMA3.2-3B-Instruct LR Search

| LR | Ep1 Acc | Ep2 Acc | Ep3 Acc | Best | 경향 |
|------|---------|---------|---------|------|------|
| 5e-5 | 30.2% | 29.0% | 29.6% | 30.2% | 정체 |
| 1e-4 | 29.8% | 28.4% | 27.6% | 29.8% | 하락 |
| **3e-4** | 29.6% | 31.0% | **31.8%** | **31.8%** | **상승 ↑** |
| 1e-3 | 22.6% | 27.8% | 25.8% | 27.8% | 불안정 |
| 3e-3 | 3.2% | 12.6% | 21.6% | 21.6% | 회복 중 |

**Best: LR=3e-4, acc=31.8%** — epoch마다 성능 상승하는 유일한 LR. SFT 효과 있음.

### Qwen2.5-1.5B-Instruct LR Search

| LR | Ep1 Acc | Ep2 Acc | Ep3 Acc | Best |
|------|---------|---------|---------|------|
| **5e-5** | 22.4% | 22.8% | **23.6%** | **23.6%** |
| 1e-4 | 22.4% | 21.2% | 22.4% | 22.4% |
| 3e-4 | 21.8% | 21.2% | 23.6% | 23.6% |
| 1e-3 | 20.6% | 19.8% | 22.4% | 22.4% |
| 3e-3 | 9.6% | 14.8% | 18.0% | 18.0% |

**Best: LR=5e-5, acc=23.6%** — SFT 효과 거의 없음. Base 모델(30.0%)보다 오히려 낮음.

### Instruct 모델 비교 종합

| 모델 | Type | Best Acc | SFT 효과 | GRPO 적합성 |
|------|------|---------|---------|------------|
| Qwen2.5-1.5B | base | 30.0% | ❌ | 낮음 (이미 포화) |
| Qwen2.5-1.5B-Instruct | instruct | 23.6% | ❌ | 낮음 |
| LLaMA3.2-3B | base | 9.4% | 약간 | 낮음 (성능 부족) |
| **LLaMA3.2-3B-Instruct** | **instruct** | **31.8%** | **✅ 상승 중** | **높음** |

### 주요 관찰

1. **LLaMA3.2-3B-Instruct가 최고**: SFT가 실제로 작동하고, epoch마다 성능 상승 (LR=3e-4).
2. **Qwen2.5-1.5B-Instruct는 base보다 나쁨**: chat template이 오히려 방해. Base 모델의 pretrain 성능(30%)을 instruct가 못 따라감.
3. **Instruct 모델 ≠ 항상 좋음**: 모델에 따라 SFT 효과가 달라짐. LLaMA instruct는 SFT에 잘 반응, Qwen instruct는 안 됨.
4. **GRPO 후보**: LLaMA3.2-3B-Instruct가 base policy로 가장 적합 — SFT로 31.8%, GRPO로 추가 향상 기대.

---

## 다음 단계

1. **LLaMA3.2-3B-Instruct + GRPO**: 정답만 맞추면 reward, 풀이 경로 자유. `trl` 라이브러리 사용.
2. **More epochs**: LR=3e-4에서 epoch 5~10으로 늘려서 성능 상한 확인.
3. **Level별 분석**: Level 1(57%) vs Level 5(10~14%) 성능 격차를 GRPO가 줄일 수 있는지.
