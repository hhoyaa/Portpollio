# AI 기반 개인 맞춤형 패션 에이전트 (Data Pipeline & SLM Fine-Tuning)

> **빅데이터 분석 및 도메인 특화 SLM(소형 언어 모델) 파인튜닝 프로젝트**
>
> 사용자의 비정형 자연어 요청과 취향·의류 데이터를 분석하여 TPO(시간·장소·상황)에 맞는 맞춤형 코디(고정 JSON 스키마)를 추천하고, 상황에 따라 필요한 조회 도구(Tool) 또는 직접 응답(`direct_answer`)을 분기하는 **Fashion AI Agent의 학습 데이터 파이프라인 구축 및 Qwen3.5-4B LoRA 파인튜닝(`Pilot` → `v0.1`)**을 수행한 프로젝트입니다.

* **Repository:** <https://github.com/2026-2-projects/ai-fashion-agent>

## 1. 프로젝트 개요

### 1.1. 문제 정의 (Problem Definition)

* **비정형 출력 및 스키마 불일치 문제:** 범용 소형 언어 모델(Base SLM)은 패션 추천 요청 시 `look_001`, `interview_classic_navy`와 같은 임의의 문자열 ID를 생성하거나 중첩 딕셔너리 형태로 응답하여, 후속 검증 로직(Validator) 및 DB/백엔드 시스템에서 자동 파싱이 불가능한 문제가 있었습니다.

* **불필요한 구조화 응답 및 분기 오류:** 단순 패션 상식 질문(예: *"검정 니트와 베이지 팬츠 색 조합 괜찮아?"*)처럼 도구 호출이나 5 LOOK 추천이 불필요한 질의에서도, Base Model은 불필요하게 거대한 5 LOOK JSON을 생성하는 과잉 응답 오류를 보였습니다.

### 1.2. 데이터 분석 및 해결 방안 (Data Analysis & Solution)

* **2단계 데이터 확장 및 이중 축(Dual-Axis) 데이터셋 설계:** 
  1. **1단계 (Pilot, 80건):** 팀 공동으로 구축한 기준 정답 데이터(Gold Data 80건: FR 50, TC 30)에 대해 자동 무결성 검증 스크립트를 적용하고, `16-bit LoRA` Pilot 학습으로 파이프라인 유효성을 검증했습니다.
  2. **2단계 (LoRA v0.1, 864건):** 서비스 요구사항 고도화(3 LOOK → **5 LOOK 출력 계약**)에 맞춰 학습 데이터를 **864건(FR 540건 + TC 324건)**으로 확장 구축하고, `FP16 LoRA` 통합 파인튜닝을 수행했습니다.

* **동일 Test Set 기반 정량 평가 및 오류 분석:** 학습에 일절 노출되지 않은 동일 Test Set(8건)을 기준으로 Base Model과 Fine-Tuned Model(`LoRA v0.1`)의 24개 세부 평가 항목을 정량 비교하고, 형식 통과 이후의 자연어 생성 품질 한계(표현 반복·오탈자)까지 분석하여 차기 버전(`v0.2`) 개선 과제를 도출했습니다.

### 1.3. 기대 효과

* 비정형 자연어 질의를 서비스 표준 JSON 출력 계약(`looks` 배열, `look_id: 1~5` 정수 통일, 역할 순서 준수)으로 100% 정형화하여 백엔드·DB 연동 안정성 확보

* 의도 분류(`Tool Calling` vs. `direct_answer`) 정확도 개선을 통해 불필요한 연산 및 JSON 생성 비용 절감

*(※ 본 포트폴리오 문서는 전체 팀 프로젝트 중 **김진호가 주도한 AI/ML 학습 데이터 파이프라인 구축, 데이터 자동 검증·분할, Qwen3.5-4B LoRA 파인튜닝(Pilot 및 v0.1)과 정량·정성 성능 평가**를 중심으로 작성되었습니다.)*

## 2. 기술 스택 (Tech Stack)

| 구분 | 사용 기술 및 도구 | 활용 목적 |
| :--- | :--- | :--- |
| **Language** | Python | JSONL 데이터 변환, 자동 무결성 검증(`validation_report`), 데이터 분할 및 평가 스크립트 구현 |
| **AI / LLM** | `unsloth/Qwen3.5-4B`, LoRA (FP16) | 경량 베이스 모델 선정 및 파라미터 효율적 파인튜닝 (`Pilot` → `LoRA v0.1`) |
| **Environment** | Google Colab (T4 GPU) | GPU 기반 모델 파인튜닝, Loss 모니터링 및 Base vs. Fine-Tuned 추론 비교 실험 |
| **Data / Validation** | JSONL, JSON Schema, Custom Validator | 대화형 학습 데이터 변환 및 5 LOOK 출력 계약(JSON 유효성·ID·역할 순서) 자동 검증 |
| **Backend / DB (연동 지원)** | FastAPI, PostgreSQL | Agent 출력 JSON과 DB 테이블 간 데이터 형식 정합성 검토 |
| **Collaboration** | Git, GitHub | 브랜치 전략(`main` - `develop` - `feature/*`) 기반 코드 및 실험 결과 버전 관리 |

## 3. 구현 과정 (Data & Modeling Pipeline)

데이터 스키마 설계 및 자동 무결성 검증부터 Pilot 실험을 통한 파이프라인 검증, 864건 대규모 데이터 확장, `LoRA v0.1` 본 학습 및 정량·정성 평가까지 단계적(Iterative)으로 고도화했습니다.

```text
[1단계: Gold Data 80건 구축 및 자동 무결성 검증 (Pilot)]
  └─ FR 50 / TC 30 검증 → Train 64 / Val 8 / Test 8 분할 → 3 LOOK Pilot 학습 및 한계 도출
       ↓
[2단계: 5 LOOK 출력 계약 반영 및 학습 데이터 864건 확장]
  └─ FR 540건 + TC 324건 통합 데이터셋 구축 (Validation 8건 / Test 8건 별도 격리)
       ↓
[3단계: Qwen3.5-4B LoRA v0.1 파인튜닝 (FP16, Colab T4)]
  └─ 1 Epoch 학습 수행 및 Train/Validation Loss 수렴 안정성 모니터링
       ↓
[4단계: 동일 Test Set(8건, 24개 항목) 전후 정량 비교 및 오류 분석]
```

### 3.1. 학습 데이터 구축 및 자동 무결성 검증

* **Pilot 데이터 검증 (`validation_report.json`):** 초기 Gold Data 80건(팀원 4인 각 20건 작성)에 대해 고유 ID 중복 여부, 유형 분포(FR 50 / TC 30), Tool 호출 분포(`get_weather` 14건, `get_calendar_events` 11건, `get_user_context` 8건, `get_wardrobe` 8건)를 자동 검수하여 데이터 편향과 결측을 사전에 차단했습니다.
* **v0.1 데이터 확장 (864건):** Pilot 단계에서 확인된 소규모 데이터(80건)의 어휘·표현 한계를 극복하고 서비스 출력 규격(5 LOOK)을 반영하기 위해 학습 데이터를 **Train 864건(FR 540건 + TC 324건)** 규모로 확장했습니다.
* **데이터 누수(Data Leakage) 방지:** `Validation 8건`과 `Test 8건`을 학습 데이터(864건)와 완전히 분리하여 공정한 일반화 성능 측정 환경을 구축했습니다.

### 3.2. 모델 파인튜닝 설정 (`Pilot` vs. `LoRA v0.1`)

| 구분 | 1단계: LoRA Pilot (파이프라인 검증) | 2단계: LoRA v0.1 (통합용 체크포인트) |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/Qwen3.5-4B` | `unsloth/Qwen3.5-4B` |
| **학습 데이터 규모** | Train 64건 (FR 50 / TC 30 기반) | **Train 864건 (FR 540 / TC 324)** |
| **출력 계약 규격** | 3 LOOK JSON 스키마 | **5 LOOK JSON 스키마 + 행동 분기(`direct_answer`)** |
| **학습 설정** | 16-bit LoRA (`r=16, alpha=16`), 3 Epochs, `lr=2e-4` | **LoRA · FP16, 1 Epoch, Max Seq Length 4096** |
| **학습 환경** | Google Colab T4 GPU | Google Colab T4 GPU |

## 4. 결과 (Results)

### 4.1. 학습 손실(Loss) 수렴 분석 (`LoRA v0.1`)

* 864건 확장 데이터로 1 Epoch 학습을 수행한 결과, 학습 초반 Loss가 급감한 뒤 안정적으로 수렴했습니다.
* **최종 손실 지표:** 평균 **Train Loss `0.4376`**, **Validation Loss `0.4976`**을 기록했으며, 학습 과정 중 발산(Divergence) 징후 없이 안정적인 학습 상태를 확인했습니다.

### 4.2. Base Model vs. LoRA v0.1 정량 평가 비교 (Test 8건 기준)

동일한 Test Set 8건(총 24개 세부 평가 항목)에 대해 Base Model과 Fine-Tuned Model(`LoRA v0.1`)을 비교한 결과, **자동 평가 통과율이 87.5%(21/24)에서 100.0%(24/24)로 향상**되며 모든 구조적 목표를 충족했습니다.

| 평가 지표 | Base Model (`Qwen3.5-4B`) | Fine-Tuned (`LoRA v0.1`) | 개선 폭 및 주요 변화 |
| :--- | :---: | :---: | :--- |
| **LOOK ID 정확도** | 기준선 대비 -40%p | **100.0%** | **+40.0%p 개선** (`look_id: 1~5` 정수 통일) |
| **첫 행동(Action) 정확도** | 기준선 대비 -33.3%p | **100.0%** | **+33.3%p 개선** (`Tool` 호출 vs. `direct_answer` 분기) |
| **JSON 유효성 / 5 LOOK / 역할 순서** | 유지 | **100.0%** | 서비스 출력 계약 3대 형식 규격 완벽 준수 |
| **자동 평가 총 통과 항목** | **21 / 24 (87.5%)** | **24 / 24 (100.0%)** | **+12.5%p 향상 (전 항목 통과)** |

*(※ 참고: 1단계 80건 Pilot 실험에서도 전체 단순 평가 통과율이 `3/8 (37.5%) → 7/8 (87.5%)`로 개선되었으며, `FR-015`에서 발생했던 JSON 문법 파싱 오류(`-items":[`)가 864건 확장 학습을 거치며 `100%` 해소되었습니다.)*

### 4.3. 실제 테스트 케이스 출력 변화 사례

* **사례 1 — ID 및 출력 계약 정형화 (`TEST-001`, `TEST-005`):**
  * **Base Model:** `"look_001"`, `"interview_classic_navy"` 등 일관성 없는 문자열 ID를 생성하여 시스템 매핑 실패
  * **Fine-Tuned (`LoRA v0.1`):** `"look_id": 1, 2, 3, 4, 5` 정수 형식과 지정된 역할(Role) 순서를 일관되게 준수

* **사례 2 — 불필요한 JSON 생성 억제 및 의도 분기 (`TEST-007`):**
  * **질의 유형:** Tool 조회가 필요 없는 단순 패션 색 조합 질문
  * **Base Model:** 단순 질문임에도 불필요하게 거대한 5 LOOK JSON 구조를 생성
  * **Fine-Tuned (`LoRA v0.1`):** `"type": "direct_answer"`를 정확히 선택하고 간결한 자연어 답변만 반환

### 4.4. 오류 분석 및 향후 고도화 방향 (`LoRA v0.2` 계획)

자동 평가 100% 달성이 **'구조적 형식 통과'**를 의미함을 명확히 인지하고, 생성된 텍스트의 정성적 품질을 점검하여 다음 개선 과제를 도출했습니다.

1. **자연어 표현 품질 이슈:** 일부 응답에서 `"차분한 차분한"`과 같은 어휘 반복 및 `"소개팅"` 등 간헐적 오탈자·비문 현상이 확인되었습니다.
2. **평가 표본 확대 필요성:** 현재 Test Set(8건)은 파이프라인 및 형식 준수 검증에는 유효하나, 다양한 실사용 시나리오에 대한 일반화 성능을 단정하기에는 표본 수가 제한적입니다.
3. **차기 고도화 계획 (`LoRA v0.2`):**
   * 현재 체크포인트를 앱 통합용 **`LoRA v0.1`**로 확정하여 백엔드(Tool·Validator) 연동에 투입
   * 향후 서비스 연동 과정에서 수집되는 실제 실패 로그를 기반으로 **HITL(Human-in-the-Loop) 정제 데이터**를 구축하고, 이를 반영해 **`LoRA v0.2`** 전체 재학습 수행

## 5. 개인 기여도 (Personal Contribution — 김진호)

팀 내 **AI/ML 및 학습 데이터 파이프라인 구축을 전담·주도**했으며, 초기 Gold Data 설계 및 자동 검증부터 864건 데이터 확장, 2차례의 LoRA 파인튜닝(`Pilot` → `v0.1`), 정량·정성 성능 평가 및 차기 버전(`v0.2`) 개선안 도출까지 전 과정을 수행했습니다. (Backend 및 DB 연동은 출력 JSON 스키마 정합성 검토 수준의 보조 역할로 참여했습니다.)

1. **학습 데이터 스키마 설계 및 자동 무결성 검증 (데이터 분석/엔지니어링)**
   * 패션 추천(FR)과 도구 호출·직접 응답 분기(TC)를 아우르는 이중 축 학습 데이터 포맷 및 JSON 출력 계약(3 LOOK → 5 LOOK) 설계
   * 초기 Gold Data 80건의 분포 편향과 고유 ID 무결성을 검수하는 자동 검증 리포트(`validation_report.json`) 파이프라인 구축
2. **단계적 데이터 확장 및 누수 방지 분할 (데이터 파이프라인)**
   * Pilot(80건) 한계 분석을 바탕으로 학습 데이터를 **864건(FR 540 + TC 324)** 규모로 확장 구축
   * `Validation 8건`과 `Test 8건`을 학습 데이터와 엄격히 격리하여 객관적인 모델 비교 기준 마련
3. **Qwen3.5-4B LoRA 파인튜닝 및 다각도 성능 평가 (모델링/평가)**
   * Google Colab T4 GPU 환경에서 `unsloth/Qwen3.5-4B` 기반 LoRA 학습(`Pilot` 및 `FP16 LoRA v0.1`) 수행 및 Loss 수렴 안정성 검증
   * 동일 Test Set(8건, 24개 항목) 전후 비교를 통해 자동 평가 통과율 **100%(24/24)**, LOOK ID 정확도 **+40%p**, 첫 행동 정확도 **+33.3%p** 개선 입증
   * 형식 통과 이후 잔존하는 어휘 반복·오탈자 등 정성적 품질 한계를 분석하여 HITL 기반 `LoRA v0.2` 재학습 로드맵 수립
4. **Agent 아키텍처 공동 설계 및 출력 형식 연동 지원 (협업)**
   * Router + Tool Calling 기반 Agent 흐름 및 4대 핵심 Tool 명세 공동 설계, 백엔드·DB 스키마 호환성 검토 지원

## 6. 실행 방법 (How to Run)

*(※ 클라우드 환경 내 작업으로 실행 방법 기재 어려움.)*

### 6.1. 저장소 클론 및 환경 설정

```bash
# 1. 저장소 클론
git clone https://github.com/2026-2-projects/ai-fashion-agent.git
cd ai-fashion-agent

# 2. 가상환경 생성 및 의존성 패키지 설치
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 6.2. 데이터 검증 및 모델 평가 실행

```bash
# 1. 학습 데이터 스키마 무결성 검증 및 리포트 생성
python src/validate_dataset.py

# 2. Base Model vs. LoRA v0.1 동일 Test Set(8건) 평가 실행
python src/evaluate_model.py
```

### 6.3. 주의사항

* 현재 저장소에 기록된 `LoRA v0.1`은 **형식 준수 및 Tool Routing 검증을 완료한 앱 통합용 체크포인트**이며, 문장 품질 고도화는 추후 실패 로그 기반 HITL 검수 및 `LoRA v0.2` 재학습을 통해 진행됩니다.
* 대용량 모델 가중치(Adapter) 및 API 인증 정보(`.env`)는 보안과 용량 관리를 위해 Git 저장소에 직접 포함하지 않았습니다.
