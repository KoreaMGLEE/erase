---
name: "006_qualitative_analysis"
description: "Human vs Model difficulty 불일치 예제 정성 분석"
---

# Plan 006: Human-Easy vs Model-Easy 불일치 정성 분석

## 목적

인간이 쉽다고 판단한 예제(Low difficulty)와 모델이 쉽다고 판단한 예제(높은 confidence)가 왜 다른지 정성적으로 분석.
ARC 데이터셋에서 t5-v1_1-xxl (11B, ARC 최강 모델) 기준으로 수행.

## 분석 대상 그룹 (4 quadrant)

| | Model Easy (top-K conf) | Model Hard (bottom-K conf) |
|--|------------------------|---------------------------|
| **Human Easy (Low)** | Agreement-Easy | Human-Easy, Model-Hard |
| **Human Hard (High)** | Human-Hard, Model-Easy | Agreement-Hard |

- **Agreement-Easy**: 인간도 쉽고 모델도 쉬움 → 기본 사실 / 단순 추론
- **Agreement-Hard**: 인간도 어렵고 모델도 어려움 → 복잡한 추론 / 전문 지식
- **Human-Easy, Model-Hard**: 인간은 쉬운데 모델은 어려움 → 상식/경험 기반 문제?
- **Human-Hard, Model-Easy**: 인간은 어려운데 모델은 쉬움 → 패턴 매칭 / shortcut?

## 분석 관점

1. **주제/과목 분포**: grade, bloom skill 분포 차이
2. **문제 유형**: 사실 회상 vs 추론 vs 응용
3. **선택지 특성**: 오답 매력도, 선택지 유사성
4. **언어적 특성**: 문장 길이, 전문 용어 비율
5. **대표 예제 분석**: 각 그룹에서 5~10개 선택하여 왜 불일치가 발생하는지 해석

## 모델

t5-v1_1-xxl (11B, LoRA r=32, ARC val_acc=0.846)

---

## 결과

### 4 Quadrant 크기

| | Model Easy (top-465) | Model Hard (bottom-287) |
|--|---------------------|------------------------|
| **Human Easy (465)** | 126 (Agreement) | 40 (Disagreement) |
| **Human Hard (287)** | 45 (Disagreement) | 54 (Agreement) |

### 통계 요약

| Quadrant | N | Mean Conf | Q Length | Top Bloom |
|----------|---|-----------|----------|-----------|
| Agreement-Easy | 126 | 0.999 | 15.6 | Remembering (53%) |
| Agreement-Hard | 54 | 0.365 | 23.2 | Applying (37%) |
| HumanEasy-ModelHard | 40 | 0.436 | 18.1 | Applying (38%) |
| HumanHard-ModelEasy | 45 | 0.998 | 18.2 | Remembering (31%) |

### 핵심 발견

#### 1. Agreement-Easy: 단순 사실 회상
- **Remembering이 53%로 압도적** — "What is X?", "Which is Y?" 형태
- 질문이 짧고 (15.6 words), 정답이 명확
- 모델 confidence 0.999 → 거의 확신

#### 2. Agreement-Hard: 응용/분석 필요
- **Applying(37%) + Analyzing(24%)** 지배적
- 질문이 길고 (23.2 words), 맥락 이해 필요
- Grade 8이 48% — 고학년 문제
- Evaluating도 11%로 다른 그룹 대비 높음

#### 3. HumanEasy-ModelHard (인간 쉬움, 모델 어려움)
- **상식/경험 기반 문제가 많음**:
  - "계절은 약 3개월마다 바뀐다" (conf=0.15)
  - "수증기가 온실가스의 가장 큰 비율" (conf=0.15) — 모델은 CO2로 예측할 가능성
  - "Mg(OH)2의 원자 수 = 5" (conf=0.17) — 화학식 계산
- **Applying이 38%로 가장 높음** — 지식의 적용/계산 필요
- 모델이 **오답 매력도(distractor quality)에 취약**: 그럴듯한 오답이 있으면 혼란

#### 4. HumanHard-ModelEasy (인간 어려움, 모델 쉬움)
- **Remembering이 31%로 가장 높음** + Analyzing 24%
- 모델은 **사실적 지식(factual recall)에 강함**:
  - "멘델의 완두콩 실험" (conf=0.999)
  - "유전자는 단백질을 코딩" (conf=0.999)
  - "석탄 발전소가 SO2 증가" (conf=0.999)
- 인간에게는 **전문 용어/과학 지식이 어렵지만**, 모델은 학습 데이터에서 직접 습득
- Grade 분포가 넓음 (G3~G8) — 학년보다는 전문성이 난이도 결정

### 결론

| 불일치 유형 | 원인 | 특징 |
|-----------|------|------|
| **인간 쉬움 → 모델 어려움** | 상식/일상 경험 기반 추론, 계산, distractor 취약 | Applying 높음, 짧은 문제에서도 모델 혼란 |
| **인간 어려움 → 모델 쉬움** | 전문 지식의 직접 회상 (factual recall) | Remembering 높음, 과학 용어 포함 |

모델은 **사실 회상에 강하고 상식적 추론/계산에 약함**. 인간은 반대: **상식은 쉽고 전문 용어가 어려움**.

---

## 전체 비율 통계

전체 annotated 2,151개 중 Human Easy(Low)=465, Human Hard(High)=287.
Model Easy = confidence 상위 465개, Model Hard = confidence 하위 287개.

| Quadrant | 개수 | 전체 대비 비율 | 해당 Human 그룹 대비 |
|----------|------|-------------|-------------------|
| Agreement-Easy (HE ∩ ME) | 126 | 5.9% | 126/465 = 27.1% |
| HumanEasy-ModelHard (HE ∩ MH) | 40 | 1.9% | 40/465 = 8.6% |
| HumanHard-ModelEasy (HH ∩ ME) | 45 | 2.1% | 45/287 = 15.7% |
| Agreement-Hard (HH ∩ MH) | 54 | 2.5% | 54/287 = 18.8% |

- Human Easy 예제 중 **27.1%만 모델도 Easy**로 판단 (완전 일치는 낮음)
- Human Hard 예제 중 **15.7%는 모델이 Easy**로 판단 (모델에게는 쉬운 전문 지식)
- 나머지 대부분은 Human Medium 영역에 속함

---

## 전체 예제 목록

### HumanEasy-ModelHard (40개)

인간은 쉽다고 판단했지만 모델이 어려워한 예제. confidence 낮은 순 정렬.

| # | conf | grade | bloom | Q | 정답 |
|---|------|-------|-------|---|------|
| 1 | 0.151 | G5 | Remembering | Which of these events occurs about every three months? (A)high tide (B)new moon **(C)new season** (D)solar eclipse | C |
| 2 | 0.154 | G8 | Remembering | Which gas accounts for the largest percentage of greenhouse gases in the atmosphere? (A)CO (B)CO2 (C)N2O **(D)water vapor** | D |
| 3 | 0.173 | G8 | Applying | How many atoms are in one formula unit of Mg(OH)₂? (A)6 **(B)5** (C)4 (D)3 | B |
| 4 | 0.211 | G5 | Applying | Which activity conserves water? **(A)planting drought-resistant crops** (B)running water while brushing teeth (C)washing vehicles frequently (D)watering grass after rainfall | A |
| 5 | 0.259 | G3 | Analyzing | What about insects can best be seen with a magnifying lens? (A)colors (B)protection methods **(C)type of eyes** (D)size | C |
| 6 | 0.295 | G7 | Applying | About how many AU is Mars from the Sun? (A)0.4 (B)0.7 (C)1.0 **(D)1.5** | D |
| 7 | 0.309 | G6 | Understanding | Which is visible through a reflecting telescope? (A)planets around other stars (B)black holes **(C)moons around Jupiter** (D)surface of Saturn | C |
| 8 | 0.336 | G8 | Understanding | Which causes a chemical change? (A)sugar+soda (B)detergent+water **(C)lemon juice+milk** (D)salt+water | C |
| 9 | 0.338 | G8 | Remembering | Mass number of atom with 17p, 17e, 18n? (A)17 (B)34 **(C)35** (D)52 | C |
| 10 | 0.345 | Bio | Understanding | Which gas concentration results from producer organisms? (A)nitrogen **(B)oxygen** (C)water vapor (D)CO2 | B |
| 11 | 0.347 | G6 | Understanding | Which is an inherited behavior? (A)seal balancing ball (B)dog getting newspaper **(C)spider spinning web** (D)ape using sign language | C |
| 12 | 0.381 | G7 | Remembering | Two elements in same group are most similar in? (A)atomic mass (B)protons (C)atomic size **(D)chemical reactivity** | D |
| 13 | 0.399 | G8 | Remembering | What precipitation occurs when raindrops freeze while falling? (A)hail (B)frost **(C)sleet** (D)snow | C |
| 14 | 0.399 | G5 | Applying | Which graph best for yearly water usage? (A)bar **(B)line** (C)scatterplot (D)circle | B |
| 15 | 0.431 | G5 | Applying | Why did one identical ball roll faster down incline? (A)color (B)age (C)shininess **(D)sticky vs smooth** | D |
| 16 | 0.438 | G7 | Analyzing | Chemical→mechanical energy: what happens to total energy? (A)increases **(B)constant** (C)destroyed (D)chemical increases | B |
| 17 | 0.446 | Bio | Understanding | Fraternal twins differ due to? **(A)independent assortment** (B)polygenic inheritance (C)incomplete dominance (D)multiple alleles | A |
| 18 | 0.448 | G7 | Understanding | Same species must produce offspring that are? **(A)fertile** (B)adaptable (C)alive at birth (D)identical to parents | A |
| 19 | 0.450 | G7 | Understanding | Which organisms best recycle nutrients? **(A)decomposers** (B)predators (C)producers (D)scavengers | A |
| 20 | 0.466 | G7 | Analyzing | What does paramecium have in common with volvox? (A)gametes (B)photosynthesis **(C)organelle for movement** (D)colony | C |
| 21 | 0.478 | G5 | Remembering | If tall tree falls in crowded forest, which resource becomes available? (A)air (B)soil (C)water **(D)sunlight** | D |
| 22 | 0.485 | G8 | Applying | Most common color of Sun-like main-sequence stars? (A)blue (B)white (C)red **(D)yellow** | D |
| 23 | 0.499 | G7 | Applying | Trees change environment by? (A)releasing N (B)crowding species (C)adding CO2 **(D)removing water from soil → atmosphere** | D |
| 24 | 0.508 | G5 | Analyzing | What do ice, rock, aluminum have in common? **(A)all solids** (B)liquids (C)minerals (D)elements | A |
| 25 | 0.514 | G8 | Applying | Sandstone→quartzite: which property stays same? **(A)mass** (B)volume (C)crystal structure (D)particle shape | A |
| 26 | 0.514 | G4 | Applying | Which shows energy flow in food chain? **(A)Leaves→Caterpillar→Bird** (B)Tree→Bird→Cat (C)Leaves→Tree→Cat (D)Cat→Leaves→Bird | A |
| 27 | 0.517 | G8 | Analyzing | Boat travels NE with northward current. Wind direction? (A)west **(B)east** (C)north (D)south | B |
| 28 | 0.519 | Bio | Applying | What inhibits enzyme stain removal? (A)minerals (B)fiber type **(C)water temperature** (D)fragrances | C |
| 29 | 0.519 | G8 | Remembering | Solid layer moving over viscous layer? (A)core **(B)crust** (C)asthenosphere (D)atmosphere | B |
| 30 | 0.520 | Bio | Remembering | Role of centromere? (A)microtubule formation (B)nucleus location (C)chromosome alignment **(D)chromatid attachment** | D |
| 31 | 0.521 | G7 | Analyzing | Why is life on outermost planets less likely? (A)too little mass (B)too much mass (C)extremely hot **(D)extremely cold** | D |
| 32 | 0.522 | G5 | Understanding | Rubber band makes sound when? **(A)vibrated** (B)stretched (C)cut (D)shot | A |
| 33 | 0.524 | G8 | Applying | How to increase gravitational force between 2 objects 2m apart? (A)↓mass one (B)↓mass both **(C)move to 1m** (D)move to 3m | C |
| 34 | 0.549 | G5 | Applying | Clearest way to present daily F→C conversions? **(A)table** (B)formula (C)pie chart (D)line graph | A |
| 35 | 0.552 | G8 | Remembering | Which has greatest mass? (A)star (B)moon (C)planet **(D)galaxy** | D |
| 36 | 0.572 | G5 | Applying | To determine boiling point, which tool NOT needed? **(A)stopwatch** (B)heat source (C)goggles (D)thermometer | A |
| 37 | 0.584 | G8 | Applying | Manganese (Z=25, A=55): particles in nucleus? (A)25 (B)30 **(C)55** (D)80 | C |
| 38 | 0.590 | G5 | Analyzing | Why do cooking pans have rubber handles? (A)easy to hold **(B)good insulator** (C)keeps food hot (D)keeps metal cool | B |
| 39 | 0.590 | G8 | Understanding | Which best describes a neutron? (A)negative charge (B)moves around nucleus **(C)adds mass to nucleus** (D)positive charge | C |
| 40 | 0.594 | G5 | Applying | Which tool helps count ladybugs on a leaf? **(A)hand lens** (B)meter stick (C)microscope (D)graduated cylinder | A |

### HumanHard-ModelEasy (45개)

인간은 어렵다고 판단했지만 모델이 쉽게 맞춘 예제. confidence 높은 순 정렬.

| # | conf | grade | bloom | Q | 정답 |
|---|------|-------|-------|---|------|
| 1 | 0.999 | G3 | Analyzing | Which rapid changes are caused by heat from inside Earth? (A)landslides **(B)volcanoes** (C)avalanches (D)floods | B |
| 2 | 0.999 | G8 | Applying | Surveyors measure ground altitude at 1m intervals over 400m². Best representation? **(A)topographical map** (B)wind map (C)tectonic diagram (D)sedimentary chart | A |
| 3 | 0.999 | G5 | Applying | Car engine melts ice on hood. What energy form? (A)chemical (B)electrical **(C)heat** (D)light | C |
| 4 | 0.999 | G4 | Remembering | Hereditary info transmitted through? (A)cell division **(B)genes** (C)germination (D)metamorphosis | B |
| 5 | 0.999 | G7 | Applying | Dogs have more fleas in summer. Best question to investigate? (A)wind (B)diet (C)rain **(D)temperature** | D |
| 6 | 0.999 | G8 | Remembering | Which scientist: genetics + pea plants? (A)Darwin **(B)Mendel** (C)Linnaeus (D)Hooke | B |
| 7 | 0.999 | G5 | Applying | What system carries nutrients from digestive to body? **(A)circulatory** (B)nervous (C)respiratory (D)skeletal | A |
| 8 | 0.999 | Bio | Remembering | Genes code for? (A)acids (B)lipids (C)sugars **(D)proteins** | D |
| 9 | 0.999 | G8 | Applying | Photovoltaic cells use which renewable resource? (A)hydrothermal **(B)solar** (C)geothermal (D)nuclear | B |
| 10 | 0.999 | G8 | Analyzing | What increases SO₂ in air? (A)fertilizer **(B)coal power plants** (C)hot day (D)rain | B |
| 11 | 0.999 | G5 | Understanding | Plastic on wire protects because plastic is? (A)keeps cool (B)allows heat **(C)good insulator** (D)magnetic | C |
| 12 | 0.999 | G8 | Remembering | Source of coal and natural gas? **(A)once-living organisms** (B)cold oceans (C)volcanoes (D)forest fires | A |
| 13 | 0.999 | G4 | Remembering | What helps body cool down? (A)shivering **(B)sweating** (C)fever (D)deep breaths | B |
| 14 | 0.999 | G3 | Applying | What reduces friction heat in car engine? (A)fan **(B)oil** (C)gasoline (D)battery | B |
| 15 | 0.999 | G7 | Understanding | To distinguish fact from opinion, conclusions should be? (A)on computer (B)bar graphs **(C)based on verifiable data** (D)in table | C |
| 16 | 0.999 | G5 | Analyzing | Down feathers in sleeping bags because they are? (A)fire resistant (B)padding **(C)good insulators** (D)water resistant | C |
| 17 | 0.999 | G7 | Remembering | Cell cycle describes processes as cell? (A)absorbs nutrients (B)makes proteins (C)repairs cells **(D)forms new cells** | D |
| 18 | 0.998 | G8 | Remembering | What type of galaxy is Milky Way? **(A)spiral** (B)elliptical (C)irregular (D)oval | A |
| 19 | 0.998 | G7 | Understanding | Solar panels convert? (A)mechanical→nuclear (B)thermal→chemical (C)kinetic→potential **(D)radiant→electrical** | D |
| 20 | 0.998 | G8 | Remembering | Plants wilt on hot day due to? (A)geotropism (B)photosynthesis **(C)dehydration** (D)blooming | C |
| 21 | 0.998 | G4 | Understanding | Greatest impact on artificial light? (A)Darwin (B)Einstein **(C)Edison** (D)Franklin | C |
| 22 | 0.998 | G8 | Understanding | Base of all ecosystems? (A)scavengers **(B)producers** (C)consumers (D)decomposers | B |
| 23 | 0.998 | G8 | Remembering | Which is renewable? (A)coal (B)minerals (C)petroleum **(D)sunlight** | D |
| 24 | 0.998 | G4 | Applying | Car slows on carpet because? (A)inertia↓ (B)gravity↑ **(C)friction↑** (D)magnetism↓ | C |
| 25 | 0.998 | G7 | Evaluating | Flowers open 12h then close. Stimulus? **(A)light levels** (B)moon (C)seasonal temp (D)CO2 | A |
| 26 | 0.998 | G5 | Remembering | What enabled Galileo to see Jupiter's moons? (A)Jupiter close (B)heliocentric **(C)invented telescope** (D)others not interested | C |
| 27 | 0.998 | G8 | Understanding | How do colliding plates change rock layer order? (A)pollutants (B)droughts **(C)faults and folds** (D)river erosion | C |
| 28 | 0.998 | G5 | Understanding | Pollen is necessary for plant to? (A)grow (B)blossom (C)germinate **(D)reproduce** | D |
| 29 | 0.998 | G8 | Analyzing | Water property that transports materials? (A)expands on solidifying (B)transparent **(C)dissolves many substances** (D)compound | C |
| 30 | 0.998 | G8 | Analyzing | Best disposal for unused cardboard boxes? (A)burn monthly (B)dump remotely **(C)recycle** (D)dumpster | C |
| 31 | 0.998 | G8 | Understanding | Catalytic converter helps to? (A)↑ozone **(B)↓smog** (C)↑nitrogen (D)↓CO2 | B |
| 32 | 0.998 | G5 | Analyzing | How does rainforest destruction contribute to greenhouse effect? (A)threatens species (B)↑erosion (C)↑ocean levels **(D)releases carbon** | D |
| 33 | 0.998 | G7 | Understanding | Boulders become sediments by first being? (A)heated/melted **(B)weathered/eroded** (C)recrystallized (D)buried | B |
| 34 | 0.998 | G5 | Remembering | Oil+water mixed form? (A)gas (B)solid (C)compound **(D)suspension** | D |
| 35 | 0.998 | G8 | Remembering | Concern about lead/mercury because? **(A)threaten life** (B)↑resources (C)healthy mutations (D)environmental stability | A |
| 36 | 0.997 | G8 | Analyzing | Air temp↑ → hydrosphere? (A)↓earthquake (B)↓mountains (C)↑weathering **(D)↑evaporation** | D |
| 37 | 0.997 | G7 | Analyzing | Grass grows 2x faster in treeless yard. Factor? (A)animals **(B)light** (C)soil (D)rain | B |
| 38 | 0.997 | Earth | Understanding | Water property causing horizontal ocean layers? **(A)density** (B)viscosity (C)turbidity (D)acidity | A |
| 39 | 0.997 | G5 | Remembering | Vinegar+water = ? (A)gas (B)solid **(C)solution** (D)compound | C |
| 40 | 0.997 | G3 | Analyzing | What lets fish move from pond to river? (A)fire (B)drought (C)snowstorm **(D)flood** | D |
| 41 | 0.997 | G5 | Remembering | Beta fish sees reflection because of? (A)absorption (B)refraction **(C)reflection** (D)diffraction | C |
| 42 | 0.997 | G7 | Analyzing | Removing trees → erosion because? (A)less bedrock **(B)fewer roots** (C)↑insects (D)↑solar radiation | B |
| 43 | 0.997 | G4 | Applying | Potholders serve as? (A)conductors **(B)insulators** (C)reflectors (D)transmitters | B |
| 44 | 0.997 | G8 | Evaluating | Best electrical insulator? (A)copper wire (B)steel tubing **(C)plastic tape** (D)aluminum foil | C |
| 45 | 0.996 | G7 | Analyzing | Which condition results from infection? (A)high blood sugar **(B)swelling of throat** (C)numbness (D)blood vessel blockage | B |
