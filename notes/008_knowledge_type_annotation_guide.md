# Knowledge Type Annotation Guide for Human-Easy Examples (ARC)

## Task Overview

ARC 데이터셋의 Human-Easy 예제 465개 각각에 대해, de Jong & Ferguson-Hessler (1996)의 지식 유형 분류에 따라 **S / F / P** 중 하나를 부여한다.

**입력**: 각 예제의 `question`, `choices`, `answer_key` (+ `grade`, `bloom` 참고용)
**출력**: 각 예제에 `knowledge_type` 필드 추가 ("S", "F", "P" 중 하나)

---

## 분류 기준

### S — Situational Knowledge (경험적/상황적 지식)

**정의**: 일상 경험, 감각적 관찰, 신체적 체험만으로 답을 알 수 있는 문제. Formal education 없이도 생활 속에서 자연스럽게 알게 되는 지식.

**판별 기준** (하나라도 해당하면 S):
- 직접 보거나 만져본 경험으로 답할 수 있는가?
- 주방, 놀이터, 야외 등 일상 환경에서 관찰 가능한가?
- 과학 용어 없이 설명 가능한가?
- "어린아이도 경험으로 알 수 있는가?"에 Yes이면 S

**예시**:

| Q (요약) | Answer | Why S |
|----------|--------|-------|
| 말벌 독침 용도 | defend itself | 벌에 쏘여본 경험 |
| 거북이 껍데기 기능 | protection | 동물 관찰 |
| 얼음 녹은 이유 | heat | 일상 체험 |
| 자석에 끌리는 것 | iron nail | 놀이 경험 |
| 매끈한 바닥에서 공이 빨리 구름 | smooth surface | 체감 가능 |
| 고무 손잡이는 뜨겁지 않음 | insulator | 주방 경험 |
| 웅덩이 사라짐 | evaporation | 관찰 가능 |
| 레몬즙+우유 응고 | chemical change | 주방 실험 |
| 얼음/돌/알루미늄 = 고체 | solid | 촉감으로 판단 |
| 플루트 소리 원리 | vibrating | 악기 불어본 경험 |
| 철가루와 모래 분리 | magnet | 자석 놀이 |
| 종이 재활용 이유 | reduces cutting trees | 일상 환경 실천 |
| 블록 느낌 기록 | observation | 직접 만져보는 행위 |
| 식물 성장 기록 | observation | 관찰 행위 |

**주의 — S로 분류하지 말아야 할 경우**:
- 일상에서 관찰 가능하더라도 **정답이 과학 전문 용어**인 경우 → F
  - 예: "웅덩이가 사라졌다" → 답이 "evaporation"이어도, 현상 자체가 일상 관찰이면 **S**
  - 예: "세포막이 물질 이동을 조절" → 답이 "cell membrane"이면 **F** (세포막은 관찰 불가)
- 핵심 판별: **"답에 도달하는 추론 과정"이 경험적인가, 교과서적인가?**

---

### F — Factual/Conceptual Knowledge (사실적/개념적 지식)

**정의**: 명시적으로 학습(교과서, 수업)을 통해 얻은 사실, 용어, 원리, 개념이 필요한 문제. 경험만으로는 알 수 없고, 누군가 가르쳐주거나 읽어서 알아야 하는 지식.

**판별 기준** (하나라도 해당하면 F):
- 과학 용어, 전문 개념, 인물명이 정답에 핵심적인가?
- 교과서를 읽지 않으면 답을 모르는가?
- 분류 체계, 정의, 법칙을 알아야 하는가?
- 과학사 인물(멘델, 다윈, 에디슨 등)이 관련된가?

**예시**:

| Q (요약) | Answer | Why F |
|----------|--------|-------|
| 유성생식 세포 | gametes | 생물학 용어 |
| 에디슨 발명 | light bulb | 과학사 사실 |
| 달 궤도 유지 | gravity | 물리 법칙 |
| 세포 물질 조절 구조 | cell membrane | 세포생물학 |
| 화석 포함 암석 | sedimentary | 지질학 분류 |
| 멘델 = 유전학 | Mendel | 과학사 |
| 광합성 에너지원 | sunlight | 생물학 개념 |
| 에너지 보존 법칙 | conservation | 물리 법칙 |
| 종의 정의 → 생식 가능한 자손 | species definition | 생물학 정의 |
| 동원체 역할 | centromere | 세포생물학 |
| 유전정보 → 유전자 | genes | 분자생물학 |
| photovoltaic → solar | solar energy | 전문 용어 |
| 같은 족 → 화학 반응성 유사 | periodic table | 화학 원리 |
| 적란운 날씨 | thunderstorms | 기상학 지식 |

**가장 흔한 유형**: ARC가 과학 QA이므로 대부분이 F. 확신이 없으면 F가 기본값.

---

### P — Procedural Knowledge (절차적 지식)

**정의**: 단계적 계산, 절차 수행, 도구 선택, 데이터 해석 방법 등 "어떻게 하는지(how-to)"가 핵심인 문제.

**판별 기준** (하나라도 해당하면 P):
- 숫자를 세거나 계산해야 하는가?
- 화학식에서 원자 수를 카운팅해야 하는가?
- 그래프/차트 유형을 선택해야 하는가?
- 측정 도구나 방법을 선택해야 하는가? (단, "현미경으로 뭘 보나" 같은 건 F)
- 벡터, 방향, 합력 등 공간적 추론이 필요한가?
- "어떤 순서로", "어떤 방법으로" 가 핵심인가?

**예시**:

| Q (요약) | Answer | Why P |
|----------|--------|-------|
| 시간 변화 표시 방법 | line graph | 그래프 유형 선택 |
| 개체수 변화 표현 | line graph | 데이터 표현 방법 |
| Mg(OH)₂ 원자 수 카운팅 | counting atoms | 화학식 계산 |
| 질량수 = 양성자+중성자 계산 | mass number | 산술 절차 |
| 보트+해류+바람 방향 벡터 추론 | vector reasoning | 공간 추론 |
| 연간 추세 → 적절한 그래프 선택 | graph selection | 방법 선택 |
| 끓는점 측정에 불필요한 도구 판별 | tool selection | 절차적 판단 |
| 망간 핵자수 계산 | nucleon counting | 계산 |
| 데이터 표현 방법 선택 | data representation | 방법론 |

**주의 — P와 F의 경계**:
- "부피 측정 도구 = graduated cylinder" → **F** (도구 이름을 아는 것은 사실 지식)
- "부피를 측정하려면 어떤 순서로 해야 하나" → **P** (절차)
- "질량 측정 도구 = balance" → **F**
- "질량수를 구하려면 양성자+중성자를 더해라" → **P**
- 핵심: **답에 도달하기 위해 "계산/절차 실행"이 필요한가?** Yes → P

---

## Decision Flowchart

```
문제를 읽는다
    │
    ├─ 답에 도달하기 위해 계산, 카운팅, 절차 실행,
    │  그래프/도구/방법 선택이 필요한가?
    │      YES → P
    │      NO ↓
    │
    ├─ 일상 경험/관찰/체험만으로 답을 알 수 있는가?
    │  (교과서 없이, 과학 용어 몰라도?)
    │      YES → S
    │      NO ↓
    │
    └─ F (기본값)
```

**우선순위: P > S > F**
- 계산이 필요하면 무조건 P (경험적이든 아니든)
- 계산은 아니지만 경험으로 풀 수 있으면 S
- 나머지는 F

---

## Validation (검증)

### Step 1: 126개 Agreement-Easy로 검증

이미 수동 코딩된 126개 Agreement-Easy 결과가 있다. 자동 코딩 실행 후 이 126개와 비교하여 일치율을 확인한다.

**수동 코딩 결과 (정답)**:
- S = 30개 (23.8%)
- F = 94개 (74.6%)
- P = 2개 (1.6%)

**허용 기준**: Cohen's κ ≥ 0.75 또는 accuracy ≥ 90%

불일치 사례가 있으면, 아래 경계 사례 가이드를 참고하여 재검토한다.

### Step 2: 불일치 분석 후 나머지 339개 코딩

검증 통과 시, 나머지 Human-Easy 예제 (465 - 126 = 339개)에 동일 기준 적용.

---

## 경계 사례 (Edge Cases)

### 1. 답이 과학 용어이지만 경험적으로 풀 수 있는 경우 → S
- "웅덩이가 왜 사라졌나?" → 답: evaporation → **S** (현상은 경험적, 용어는 부가적)
- "얼음은 무슨 상태?" → 답: solid → **S** (만져보면 안다)
- "열에너지를 만드는 힘?" → 답: friction → **S** (손 비비면 따뜻)

### 2. 관찰 가능하지만 원리를 알아야 하는 경우 → F
- "달이 왜 지구 주위를 도나?" → 답: gravity → **F** (떨어지는 건 보이지만 궤도는 개념)
- "피부가 왜 질병을 막나?" → 답: physical barrier → **F** (피부의 면역학적 역할은 학습 필요)

### 3. 도구 이름 vs 도구 사용법
- "박테리아 관찰 도구?" → 답: microscope → **F** (도구명 = 사실 지식)
- "끓는점 측정에 불필요한 도구?" → 답: (판별) → **P** (어떤 도구가 절차에 필요한지 추론)

### 4. 동물/식물 관찰
- "새 관찰 도구?" → 답: binoculars → **S** (일상 경험)
- "물고기 수중 호흡?" → 답: gills → **F** (아가미 = 생물학 용어)
- "물고기 비늘 기능?" → 답: protection → **S** (관찰 가능한 기능)

### 5. Grade level은 참고만
- 저학년(G3-G4)이라고 무조건 S가 아님. "G3 + 광합성" → F
- 고학년(G7-G8)이라고 무조건 F가 아님. "G8 + 고무줄 소리" → S

---

## Output Format

```json
{
  "id": "Mercury_SC_401281",
  "question": "A wasp uses poison in a stinger to",
  "choices": ["produce eggs.", "defend itself.", "build a nest.", "attract a mate."],
  "answer_key": "B",
  "grade": "Grade 05",
  "bloom": "Understanding",
  "difficulty": "Low",
  "knowledge_type": "S",
  "knowledge_type_reason": "일상에서 벌에 쏘여본 경험으로 방어용임을 알 수 있음"
}
```

각 예제에 `knowledge_type` ("S" / "F" / "P")과 `knowledge_type_reason` (한 줄 근거)을 추가한다.

---

## Summary Statistics (수동 코딩 참고값)

### Agreement-Easy (n=126, 수동 코딩 완료)
| Type | Count | % |
|------|-------|---|
| S | 30 | 23.8% |
| F | 94 | 74.6% |
| P | 2 | 1.6% |

### HE-MH (n=40, 수동 코딩 완료)
| Type | Count | % |
|------|-------|---|
| S | 13 | 32.5% |
| F | 20 | 50.0% |
| P | 7 | 17.5% |

전체 465개 코딩 후 이 분포와의 일관성을 확인한다.

---

## Reference

- de Jong, T., & Ferguson-Hessler, M. G. M. (1996). Types and qualities of knowledge. *Educational Psychologist*, 31(2), 105–113.
