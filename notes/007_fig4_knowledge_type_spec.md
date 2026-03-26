# Plan 007: Human vs Model Disagreement — Knowledge Type Analysis (§4.2)

## 목적

§4.2에서 사용할 분석 및 Figure. ARC 데이터셋에서 인간 난이도(Low/High)와 모델 confidence(t5-v1_1-xxl)의 불일치를 de Jong & Ferguson-Hessler (1996)의 지식 유형 분류로 분석.

**핵심 질문**: 인간이 쉽다고 본 예제 중, 모델도 쉽다고 동의한 것과 모델이 어려워한 것은 지식 유형에서 어떻게 다른가?

---

## 2×2 Quadrant 정의

|  | Model Easy (top-K conf) | Model Hard (bottom-K conf) |
|--|------------------------|---------------------------|
| **Human Easy (Low)** | Agreement-Easy (n=126) | **HE-MH (n=40)** |
| **Human Hard (High)** | **HH-ME (n=45)** | Agreement-Hard (n=54) |

## 지식 유형 분류 (de Jong & Ferguson-Hessler, 1996)

- **Situational Knowledge (S)**: 일상 경험/감각적 관찰로 알 수 있는 것. Formal education 불필요.
  - 예: "고무 손잡이는 열을 전달하지 않는다", "매끈한 바닥에서 공이 빨리 구른다"
- **Factual/Conceptual Knowledge (F)**: 명시적으로 학습된 사실, 용어, 원리가 필요한 것.
  - 예: "멘델 = 유전학", "유전자 → 단백질", "photovoltaic = solar"
- **Procedural Knowledge (P)**: 단계적 계산이나 절차 수행이 필요한 것.
  - 예: "Mg(OH)₂의 원자 수 세기", "질량수 = 양성자 + 중성자 계산"

---

## 코딩 결과

### 분석 1: HE-MH vs Agreement-Easy (모델 관점 — 본문 Figure 4)

**Base rate = Agreement-Easy (n=126)**: 인간이 쉽다고 본 예제 중 모델도 동의한 것

| Knowledge Type | Agreement-Easy (n=126) | HE-MH (n=40) | Δ (pp) |
|---|---|---|---|
| Situational | 30 (23.8%) | 13 (32.5%) | **+8.7** |
| Factual/Conceptual | 94 (74.6%) | 20 (50.0%) | **−24.6** |
| Procedural | 2 (1.6%) | 7 (17.5%) | **+15.9** |

**핵심 발견 (Agreement-Easy를 base rate로):**

1. **Factual/Conceptual이 급감** (74.6% → 50.0%, −24.6pp): 모델이 어려워하는 문제는 사실/개념 문제가 아닌 것들. 모델은 factual 문제에서 인간과 일치하지만, 다른 유형에서 불일치 발생.
2. **Procedural이 급증** (1.6% → 17.5%, +15.9pp): 모델이 체계적으로 약한 영역. 계산, 도구 선택, 벡터 추론 등.
3. **Situational도 증가** (23.8% → 32.5%, +8.7pp): 일상 경험/상식 문제에서도 모델이 상대적으로 약함.

→ **메시지**: "인간이 쉬운 문제 중 모델이 어려워하는 것은, 사실적 지식이 아닌 절차적 추론과 경험적 판단이 필요한 문제에 집중되어 있다."

### 분석 2: HE-MH vs HH-ME (양방향 비교 — 부록 보강)

| Knowledge Type | HE-MH (n=40) | HH-ME (n=45) |
|---|---|---|
| Situational | 13 (32.5%) | 13 (28.9%) |
| Factual/Conceptual | 20 (50.0%) | 32 (71.1%) |
| Procedural | 7 (17.5%) | 0 (0.0%) |

- HH-ME에서 Factual/Conceptual이 71.1%로 지배적 → 모델의 강점
- Procedural은 HE-MH에서만 존재 → 모델의 체계적 약점

---

## 시각화 요구사항

### 본문 Figure 4: HE-MH의 지식 유형 분포 (Agreement-Easy를 base rate로)

**형식: Bar chart (1-column width)**
- X축 또는 Y축: 3개 지식 유형
- Base rate (Agreement-Easy 분포)를 점선/회색 바로 표시
- HE-MH 분포를 색상 바로 표시
- enrichment pp를 라벨로 표시
- 색상: HE-MH = 주황(#FF9800 계열), base rate = 회색

**대안: Diverging bar chart**
- Y축: 3개 지식 유형
- X축: Agreement-Easy 대비 enrichment (pp)
- 양의 방향 = HE-MH에서 over-represented
- 음의 방향 = HE-MH에서 under-represented

### 부록 Figure: 4-quadrant 비교
- Agreement-Easy, HE-MH, HH-ME, Agreement-Hard 4개 그룹의 지식 유형 분포를 side-by-side bar chart로

### 스타일 가이드
- 폰트: 영문만 사용 (논문용)
- 논문 삽입용이므로 깔끔하고 과하지 않은 디자인
- matplotlib 또는 seaborn 기반

---

## 전체 465개 확장 계획 (자동 코딩)

### 목적
Agreement-Easy 126개와 HE-MH 40개는 수동 코딩 완료. 나머지 Human-Easy 예제(465 - 126 - 40 = 299개, 중간 confidence 영역)도 코딩하여 연속적 분석 가능.

### 자동 코딩 전략

**Step 1: 규칙 기반 Procedural 식별 (높은 정밀도)**
- 문제 텍스트에 계산/수치 키워드 포함: "how many", "calculate", "what is the mass", 화학식(Mg, Fe, 원소기호+숫자), "AU", 단위 변환
- Bloom = Applying이면서 수학적 요소 포함
- 예상: 전체 465개 중 약 20~30개

**Step 2: 규칙 기반 Situational 식별 (중간 정밀도)**
- 저학년(G3-G4) + Bloom = Understanding/Applying
- 문제에 일상 키워드: "kitchen", "cooking", "playground", "garden", "weather", "home", "everyday"
- 동물/식물의 관찰 가능한 행동 관련
- 수동 코딩과의 일치율 검증 후 적용

**Step 3: 나머지 = Factual/Conceptual (기본값)**
- ARC가 과학 QA이므로 대부분이 Factual/Conceptual
- Step 1, 2에서 식별되지 않은 나머지를 F로 분류

**Step 4: 검증**
- 수동 코딩된 166개(126+40)에 대해 자동 코딩을 적용하고, 수동 결과와의 일치율(Cohen's κ) 측정
- κ ≥ 0.7이면 나머지 299개에 적용

### 코드 위치
`outputs/plan7/code/auto_knowledge_coding.py`

---

## 개별 코딩 내역

### Agreement-Easy (126개) — confidence 높은 순

| # | type | Q (요약) | A (요약) |
|---|------|---------|---------|
| 1 | S | 말벌 독침 용도 | defend itself |
| 2 | F | 유성생식 세포 | gametes |
| 3 | S | 거북이 껍데기 기능 | protection |
| 4 | F | 백혈구 역할 | fight disease |
| 5 | F | 에디슨 발명 | light bulb |
| 6 | P | 시간 변화 표시 방법 | line graph |
| 7 | S | 달팽이 껍데기 | protection |
| 8 | F | 질병 유전 | inherited disease |
| 9 | S | 새 관찰 도구 | binoculars |
| 10 | F | 달 궤도 유지 | gravity |
| 11 | F | 수중 음파 탐지 | sonar |
| 12 | F | 세포 물질 조절 구조 | cell membrane |
| 13 | S | 물고기 비늘 | protection |
| 14 | S | 얼음 녹은 이유 | heat |
| 15 | F | 꽃가루 알레르기 → 면역계 | immune |
| 16 | F | 화성 궤도 유지 | gravity |
| 17 | F | 수정란 발달 결정 | genes |
| 18 | F | 뇌·척수·신경 기능 | conducting messages |
| 19 | F | 호흡기관 | lungs |
| 20 | F | 플라스틱 원료 | petroleum |
| 21 | F | 질병 방어 기관 | skin |
| 22 | F | 거울 이미지 | reflecting light |
| 23 | S | 화학물질 보호 장비 | safety goggles |
| 24 | F | 세포막 기능 | cell membrane |
| 25 | F | 병원체 파괴 | white blood cells |
| 26 | S | 빠른 원거리 이동 | airplane |
| 27 | F | 옹스트롬 측정 대상 | atoms |
| 28 | F | 태양 궤도 유지 | gravity |
| 29 | F | 화석 조개 포함 암석 | limestone |
| 30 | F | 약한 뼈 → 체계 | skeletal |
| 31 | F | 섬 동물 관찰 과학자 | Darwin |
| 32 | F | 혈중 산소 유입 기관 | lungs |
| 33 | F | 빛 반사 | reflected |
| 34 | F | 지구 온도 상승 원인 | carbon dioxide |
| 35 | F | 적혈구 역할 | transport oxygen |
| 36 | F | 전염성 질병 용어 | infectious |
| 37 | F | 노폐물 제거 체계 | excretory |
| 38 | P | 개체수 변화 표현 | line graph |
| 39 | F | 구름 형성 기체 | water vapor |
| 40 | S | 북극 생존 특성 | thick fur |
| 41 | F | 태양 핵융합 산물 | helium |
| 42 | F | 피부암 원인 | ultraviolet rays |
| 43 | S | 질병 연구 사회적 이익 | prevention/cure |
| 44 | F | 대기 최대 비율 기체 | nitrogen |
| 45 | S | 철가루와 모래 분리 | magnet |
| 46 | F | 가장 추운 지역 | tundra |
| 47 | F | 태양 위치 구조 | Milky Way |
| 48 | F | 소화→전신 영양분 운반 | circulatory |
| 49 | S | 증발 에너지원 | sun |
| 50 | S | 자석에 끌리는 것 | iron nail |
| 51 | F | 박테리아 관찰 도구 | microscope |
| 52 | F | 질량 측정 도구 | balance |
| 53 | S | 열에너지 생성하는 힘 | friction |
| 54 | F | SO₂ 발생원 | fossil fuels |
| 55 | S | 시계 태양전지 기능 | energy source |
| 56 | F | 영양분 이동 근육층 | digestive |
| 57 | F | 동화작용 예 | protein synthesis |
| 58 | F | 원소 최소 입자 | atom |
| 59 | F | 생물 분류 기준 | structure |
| 60 | F | 장수 식물 → 번식 전략 | reproduction |
| 61 | F | Mg 화학 기호 | magnesium |
| 62 | F | 렌즈 빛 집중 | refraction |
| 63 | F | 최고 밀도 천체 | neutron star |
| 64 | S | 눈 덮인 산 동물 특성 | thick fur |
| 65 | F | 재생 가능 자원 | wood |
| 66 | F | 식물 형질 규칙 과학자 | Mendel |
| 67 | F | 물고기 수중 호흡 | gills |
| 68 | F | 화석 포함 암석 | sedimentary |
| 69 | S | 하루에 생태계 변화 | wildfire |
| 70 | F | 풍화·침식 결과 | sediments |
| 71 | F | 비재생 자원 | oil |
| 72 | F | 배터리 에너지 형태 | chemical |
| 73 | F | 광합성 | photosynthesis |
| 74 | F | 신경계 주요 기관 | brain |
| 75 | F | 식도·위·장 → 체계 | digestive |
| 76 | F | 물 순환 에너지원 | Sun |
| 77 | F | 피부 세포 기능 | physical barrier |
| 78 | F | 오존층 파괴 원인 | fluorocarbons |
| 79 | S | 종이 재활용 이유 | reduces cutting trees |
| 80 | F | DNA 검사 정보원 | scientific journals |
| 81 | F | 항성 정의 | emits light |
| 82 | F | 농업·유전학 과학자 | Mendel |
| 83 | S | 호흡기 장애 증상 | difficulty breathing |
| 84 | F | 광합성 에너지원 | sunlight |
| 85 | S | 플루트 소리 원리 | vibrating |
| 86 | F | 발전기 원리 | magnetic field |
| 87 | F | 동굴 메아리 원리 | reflection of sound |
| 88 | S | 웅덩이 사라짐 | evaporation |
| 89 | S | 얼음 상태 | solid |
| 90 | F | 천왕성 궤도 유지 | gravity |
| 91 | F | 식물 화학에너지 원천 | solar energy |
| 92 | F | 광년 정의 | distance light travels |
| 93 | F | 세포 영양분 통과 구조 | membrane |
| 94 | F | 점진적 환경 변화 | succession |
| 95 | F | 포보스 분류 | moon |
| 96 | S | 길이 단위 | meters |
| 97 | F | 혈중 염분 조절 | feedback loop |
| 98 | F | Fe 화학 기호 | iron |
| 99 | S | 전구 도움 활동 | reading |
| 100 | F | 광물 물리적 성질 | solid |
| 101 | F | 유기 화합물 원소 | hydrogen |
| 102 | F | 유리한 변이 선택 | natural selection |
| 103 | F | 유기 퇴적물 성분 | carbon |
| 104 | S | 물이 바위 위 오래 흐르면 | erosion |
| 105 | F | 은하군 결합력 | gravity |
| 106 | F | 수증기 최다 대기층 | troposphere |
| 107 | F | 파동 반사 | reflection |
| 108 | F | 하천 암석 마모 | abrasion |
| 109 | F | 물고기 이상 원인 | environmental condition |
| 110 | F | 개인용 컴퓨터 발명 기여 | integrated circuit |
| 111 | F | 해변 형성 과정 | mechanical |
| 112 | F | 모든 생물 공통점 | cells |
| 113 | F | 별·가스·먼지 체계 | galaxy |
| 114 | F | 화학 변화 증거 | new substance |
| 115 | F | 인간이 감지하는 전자기파 | visible light |
| 116 | S | 특성·행동 기록 방법 | observing |
| 117 | F | 작은 부피 + 큰 질량 | high density |
| 118 | S | 재활용 예시 | aluminum cans |
| 119 | S | 설탕+소금+땅콩 혼합물 | mixture |
| 120 | S | 식물 성장 기록 | observation |
| 121 | F | 운동 관련 체계 | muscular and skeletal |
| 122 | F | 적란운 날씨 | thunderstorms |
| 123 | F | 부피 측정 도구 | graduated cylinder |
| 124 | S | 블록 느낌 기록 | observation |
| 125 | F | 광물 형성 과정 | magma cooling |
| 126 | F | 유전 형질 | height |

**Agreement-Easy 요약: S=30 (23.8%), F=94 (74.6%), P=2 (1.6%)**

### HE-MH (40개) — confidence 낮은 순

| # | conf | type | reason |
|---|------|------|--------|
| 1 | 0.151 | S | 계절 변화는 일상 경험 |
| 2 | 0.154 | F | 수증기가 최대 온실가스 — 과학적 사실 |
| 3 | 0.173 | P | Mg(OH)₂ 원자 수 카운팅 |
| 4 | 0.211 | S | 가뭄 저항 작물 — 실용적 상식 |
| 5 | 0.259 | S | 돋보기로 곤충 관찰 — 경험적 |
| 6 | 0.295 | F | 화성 AU 거리 — 천문학 수치 |
| 7 | 0.309 | F | 반사 망원경으로 볼 수 있는 것 |
| 8 | 0.336 | S | 레몬즙+우유 응고 — 주방 경험 |
| 9 | 0.338 | P | 질량수 = 양성자+중성자 계산 |
| 10 | 0.345 | F | 생산자→산소 — 생물학 개념 |
| 11 | 0.347 | S | 거미줄 짜기 = 본능 — 자연 관찰 |
| 12 | 0.381 | F | 같은 족→화학 반응성 유사 |
| 13 | 0.399 | S | 비가 얼면서 떨어짐 = sleet — 날씨 경험 |
| 14 | 0.399 | P | 연간 추세 → 적절한 그래프 선택 |
| 15 | 0.431 | S | 매끈한 바닥에서 공이 빨리 구름 |
| 16 | 0.438 | F | 에너지 보존 법칙 |
| 17 | 0.446 | F | 이란성 쌍둥이 → 독립 분리 |
| 18 | 0.448 | F | 종의 정의 → 생식 가능한 자손 |
| 19 | 0.450 | F | 분해자의 역할 — 생태학 |
| 20 | 0.466 | F | 짚신벌레+볼복스 공통점 — 생물학 |
| 21 | 0.478 | S | 큰 나무 쓰러지면 햇빛 확보 — 관찰 가능 |
| 22 | 0.485 | F | 태양형 주계열성 색깔 — 천문학 |
| 23 | 0.499 | F | 나무의 수분 순환 역할 |
| 24 | 0.508 | S | 얼음/돌/알루미늄 = 고체 — 일상 경험 |
| 25 | 0.514 | F | 사암→규암 변성 시 보존되는 성질 |
| 26 | 0.514 | F | 먹이 사슬 방향 — 생태학 개념 |
| 27 | 0.517 | P | 보트+해류+바람 방향 벡터 추론 |
| 28 | 0.519 | F | 효소 활성과 온도 관계 |
| 29 | 0.519 | F | 지각층 구조 — 지질학 |
| 30 | 0.520 | F | 동원체 역할 — 세포생물학 |
| 31 | 0.521 | S | 외행성은 태양에서 멀어서 춥다 |
| 32 | 0.522 | S | 고무줄 튕기면 소리 남 — 일상 경험 |
| 33 | 0.524 | F | 중력과 거리 관계 — 물리 원리 |
| 34 | 0.549 | P | 데이터 표현 방법 선택 |
| 35 | 0.552 | F | 은하 > 행성 > 위성 크기 — 천문학 |
| 36 | 0.572 | P | 끓는점 측정에 불필요한 도구 판별 |
| 37 | 0.584 | P | 망간 핵자수 계산 |
| 38 | 0.590 | S | 고무 손잡이 = 열 차단 — 주방 경험 |
| 39 | 0.590 | F | 중성자 특성 — 원자물리 |
| 40 | 0.594 | S | 무당벌레 세기 → 돋보기 — 실용 경험 |

### HH-ME (45개) — confidence 높은 순

| # | conf | type | reason |
|---|------|------|--------|
| 1 | 0.999 | F | 지구 내부 열 → 화산 — 지질학 |
| 2 | 0.999 | F | 지형도 — 전문 용어 |
| 3 | 0.999 | S | 자동차 엔진 열 → 얼음 녹임 — 일상 관찰 |
| 4 | 0.999 | F | 유전정보 → 유전자 — 생물학 정의 |
| 5 | 0.999 | S | 여름에 벼룩 많음 → 온도 — 계절 경험 |
| 6 | 0.999 | F | 멘델 + 완두콩 — 과학사 사실 |
| 7 | 0.999 | F | 영양분 운반 → 순환계 — 생물학 |
| 8 | 0.999 | F | 유전자 → 단백질 — 분자생물학 |
| 9 | 0.999 | F | photovoltaic → solar — 전문 용어 |
| 10 | 0.999 | F | SO₂ 증가 → 석탄 발전소 — 환경과학 |
| 11 | 0.999 | F | 플라스틱 = 절연체 — 개념+용어 |
| 12 | 0.999 | F | 석탄 기원 = 생물 — 지질학 |
| 13 | 0.999 | S | 체온 조절 → 땀 — 신체 경험 |
| 14 | 0.999 | S | 엔진 마찰 줄이기 → 오일 — 자동차 상식 |
| 15 | 0.999 | F | 사실 vs 의견 → 검증 가능한 데이터 |
| 16 | 0.999 | S | 다운 깃털 = 따뜻함 — 체감 경험 |
| 17 | 0.999 | F | 세포 주기 → 새 세포 형성 — 생물학 |
| 18 | 0.998 | F | 은하수 = 나선형 — 천문학 사실 |
| 19 | 0.998 | F | 태양전지 에너지 변환 — 물리학 |
| 20 | 0.998 | F | 식물 시듦 → 탈수 — 생물학 용어 |
| 21 | 0.998 | F | 에디슨 → 인공 조명 — 과학사 |
| 22 | 0.998 | F | 생태계 기반 → 생산자 — 생태학 |
| 23 | 0.998 | F | 재생 에너지 → 태양광 — 환경과학 |
| 24 | 0.998 | S | 카펫에서 차 느려짐 → 마찰 — 일상 경험 |
| 25 | 0.998 | S | 꽃 개폐 → 빛 — 자연 관찰 |
| 26 | 0.998 | F | 갈릴레오 → 망원경 — 과학사 |
| 27 | 0.998 | F | 판 충돌 → 단층과 습곡 — 지질학 |
| 28 | 0.998 | F | 꽃가루 → 번식 — 생물학 |
| 29 | 0.998 | F | 물의 용해력 — 화학 |
| 30 | 0.998 | S | 골판지 → 재활용 — 일상 환경 실천 |
| 31 | 0.998 | F | 촉매 변환기 → 스모그 감소 — 전문 용어 |
| 32 | 0.998 | F | 열대우림 파괴 → 탄소 방출 — 환경과학 |
| 33 | 0.998 | F | 풍화와 침식 — 지질학 |
| 34 | 0.998 | F | 기름+물 → 현탁액 — 화학 용어 |
| 35 | 0.998 | F | 납/수은 → 생명 위협 — 독성학 |
| 36 | 0.997 | F | 기온 상승 → 증발 증가 — 기후학 |
| 37 | 0.997 | S | 그늘 없는 마당에 잔디 빨리 자람 → 빛 |
| 38 | 0.997 | F | 수평 해양층 → 밀도 — 해양학 |
| 39 | 0.997 | F | 식초+물 → 용액 — 화학 용어 |
| 40 | 0.997 | S | 홍수 → 물고기 이동 — 자연 관찰 |
| 41 | 0.997 | F | 베타 물고기 반사 — 광학 용어 |
| 42 | 0.997 | S | 나무 제거 → 뿌리 없음 → 침식 — 직관적 |
| 43 | 0.997 | S | 냄비 손잡이 → 열 보호 — 주방 경험 |
| 44 | 0.997 | F | 최고 절연체 → 플라스틱 — 재료 지식 |
| 45 | 0.996 | S | 감염 → 목 부종 — 신체 경험 |

---

## 논문 서술 방향

### §4.2 핵심 메시지 (업데이트)

> 인간이 쉽다고 판단한 예제(Human-Easy) 중, 모델도 동의한 예제(Agreement-Easy)의 74.6%가 Factual/Conceptual 문제인 반면, 모델이 어려워한 예제(HE-MH)에서는 이 비율이 50.0%로 급감한다. 대신 Procedural 문제가 1.6%에서 17.5%로, Situational 문제가 23.8%에서 32.5%로 증가한다. 이는 **모델이 명시적으로 학습한 사실과 개념에서는 인간과 일치하지만, 절차적 추론과 경험적 판단이 필요한 문제에서 체계적으로 불일치**함을 보여준다.

### 가이드라인(§5)과의 연결

- Factual/Conceptual 문제에서 모델이 쉽다고 한 것 → 유의미한 학습 완료 신호일 수 있으나, 단순 패턴 매칭일 가능성도 있음 (HH-ME의 71.1%가 이 유형)
- Procedural/Situational 문제에서 인간이 쉬운데 모델이 어려워한 것 → 모델에게 유의미한 학습 신호를 제공할 가능성. 이런 예제를 학습에서 보존하는 전략 고려

---

## Reference

- de Jong, T., & Ferguson-Hessler, M. G. M. (1996). Types and qualities of knowledge. *Educational Psychologist*, 31(2), 105–113.
- Anderson, L. W., & Krathwohl, D. R. (2001). *A Taxonomy for Learning, Teaching, and Assessing*. Longman.
- Meta-Cognitive Analysis: Evaluating Declarative and Procedural Knowledge in Datasets and LLMs (ACL 2024). arXiv:2403.09750.
