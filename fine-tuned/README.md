# AI 기반 개인 맞춤형 패션 에이전트 (Data Pipeline & SLM Fine-Tuning)

> **빅데이터 분석 및 도메인 특화 SLM(소형 언어 모델) 파인튜닝 프로젝트**
>
> 사용자의 비정형 자연어 요청과 취향·의류 데이터를 분석하여 TPO(시간·장소·상황)에 맞는 3가지 맞춤형 코디(고정 JSON 데이터)를 추천하는 AI Agent의 핵심 학습 데이터 파이프라인 구축 및 **Qwen3.5-4B LoRA Pilot 파인튜닝 검증**을 수행한 프로젝트입니다.

* **Repository:** <https://github.com/2026-2-projects/ai-fashion-agent>

## 1. 프로젝트 개요

### 1.1. 문제 정의

* **비정형 출력 및 스키마 불일치 문제:** 기존 범용 소형 언어 모델(Base SLM)은 패션 추천 요청 시 임의의 JSON 구조(`{"answer": {"look_1": ...}}`)로 응답하여, 후속 검증 로직(Validator) 및 DB/백엔드 시스템에서 자동 파싱이 불가능한 문제가 있었습니다.
* **도메인 특화 데이터 및 의사결정 기준 부족:** *"내일 저녁 데이트인데 내 옷으로 깔끔하게 입고 싶어"*와 같은 복합 자연어 질의에서 TPO와 스타일 조건을 해석하고, 필요한 데이터 조회 도구(Tool)를 선택하는 도메인 학습 데이터가 부재했습니다.

### 1.2. 데이터 분석 및 해결 방안

* **이중 축(Dual-Axis) Gold Data 설계 및 구축:** 모델이 수행해야 할 핵심 과업을 **패션 추천(FR: Fashion Recommendation)**과 **도구 호출 판단(TC: Tool Calling)**으로 구조화하여 총 80건의 고품질 기준 정답 데이터(Gold Data)를 구축했습니다.
* **경량 모델 파인튜닝 파이프라인 검증 (LoRA Pilot):** 제한된 GPU 자원(Google Colab T4) 환경에서 학습 파이프라인의 유효성과 스키마 준수율 개선 폭을 사전 검증하기 위해 **Qwen3.5-4B** 모델 기반의 **16-bit LoRA Pilot 파인튜닝**을 수행했습니다.
* **정량적 전후 비교 및 오류 분석:** 학습에 사용하지 않은 동일한 Test Set(8건)을 기준으로 Base Model과 Fine-Tuned Model의 출력 구조를 비교하고, 실패 사례의 원인을 분석하여 후속 데이터 증강(Synthetic Data) 및 자동 재생성 로직 설계의 근거를 도출했습니다.

### 1.3. 기대 효과

* 비정형 자연어 질의를 서비스 표준 JSON 스키마(`looks` 배열 및 필수 필드 구조)로 정형화하여 데이터 파이프라인의 자동 처리 안정성 확보
* 소규모 Gold Data(80건)만으로도 모델의 출력 형식 준수율이 대폭 개선됨을 입증하여, 향후 합성 데이터(500~1,000건) 확장 학습의 타당성 검증

*(※ 본 포트폴리오 문서는 전체 프로젝트 중 **1단계 핵심 구현 범위인 Gold Data 구축, 데이터 분할 및 자동 검증, Qwen3.5-4B LoRA Pilot 파인튜닝 및 성능 평가**를 중심으로 작성되었습니다.)*

## 2. 기술 스택

| 구분 | 사용 기술 및 도구 | 활용 목적 |
| :--- | :--- | :--- |
| **Language** | Python | JSONL 데이터 변환, 자동 검증·분할 스크립트 및 모델 평가 구현 |
| **AI / LLM** | `unsloth/Qwen3.5-4B`, LoRA (16-bit) | 베이스 모델 선정 및 파라미터 효율적 파인튜닝 (`r=16`, `alpha=16`) |
| **Environment** | Google Colab (T4 GPU) | GPU 기반 모델 파인튜닝 및 Base vs. Fine-Tuned 추론 비교 실험 |
| **Data / Validation** | JSONL, JSON Schema, Custom Validator | 대화형 학습 데이터 포맷 변환 및 3 LOOK 출력 구조 무결성 검증 |
| **Backend / DB (연동 지원)** | FastAPI, PostgreSQL | Agent 출력 JSON과 DB 테이블 간 데이터 형식 정합성 검토 |
| **Collaboration** | Git, GitHub | 브랜치 전략(`main` - `develop` - `feature/*`) 기반 코드 및 실험 결과 관리 |

## 3. 구현 과정

데이터 설계부터 JSONL 포맷 변환, 데이터 누수 방지를 위한 분할, Pilot 파인튜닝, 동일 테스트셋 기반 정량 평가 및 오류 분석에 이르는 체계적인 데이터 분석·모델링 절차를 수행했습니다.

```text
[공개 데이터 조사 + Gold Data 80건 구축 (FR 50 / TC 30)]
       ↓
[대화형 JSONL 포맷 변환 및 데이터 자동 검증]
       ↓
[데이터셋 분할 (Train 64 / Validation 8 / Test 8)]
       ↓
[Base Model 사전 평가 (Test 8건 기준선 확보)]
       ↓
[Qwen3.5-4B 16-bit LoRA Pilot 파인튜닝 (3 Epochs)]
       ↓
[동일 Test Set 전후 정량 비교 및 오류 원인 분석]
```

### 3.1. 학습 데이터 구성 및 전처리

모델의 추천 역량과 에이전트 도구 선택 역량을 동시에 학습하기 위해 두 가지 유형의 Gold Data(총 80건)를 구축하고 대화형 `JSONL` 포맷으로 변환했습니다.

* **FR (Fashion Recommendation, 50건):** 계절, 기온, TPO, 선호 스타일·색상·핏 조건에 맞춰 3가지 코디(`look_id: 1~3`)와 추천 사유를 고정 JSON 구조로 출력하는 학습 데이터
* **TC (Tool Calling, 30건):** 사용자 발화 의도를 파악해 `get_user_context`, `get_wardrobe`, `get_weather`, `get_calendar_events` 중 필요한 Tool과 인자(Argument)를 호출하는 학습 데이터

### 3.2. 데이터 누수 방지를 위한 데이터셋 분할

* 전체 80건의 데이터를 **Train 64건(80%) / Validation 8건(10%) / Test 8건(10%)** 비율로 엄격히 분리했습니다.
* **Test Set 구성 (8건):** FR 5건(`FR-006`, `FR-015`, `FR-018`, `FR-031`, `FR-039`)과 TC 3건(`TC-006`, `TC-017`, `TC-027`)으로 구성하여 학습 과정에 일절 노출되지 않도록 격리했습니다.

### 3.3. LoRA Pilot 파인튜닝 설정

Google Colab T4 GPU 환경에서 아래와 같은 하이퍼파라미터 구성으로 학습을 진행했습니다.

* **Base Model:** `unsloth/Qwen3.5-4B`
* **Method:** 16-bit LoRA (`r=16`, `lora_alpha=16`)
* **Training Hyperparameters:** `3 epochs`, `learning_rate=2e-4`, `max_sequence_length=4096`, `batch_size=1` (`gradient_accumulation_steps=4`)

### 3.4. 출력 스키마 정합성 검증 (DB/Backend 연동 지원)

* 모델이 생성한 JSON 응답이 PostgreSQL DB 스키마 및 Backend(FastAPI) 검증 로직에서 오류 없이 파싱될 수 있도록 필수 키(`looks`, `look_id`, `role`, `item_id` 등)와 자료형 규격을 조율했습니다.

## 4. 결과

### 4.1. Base Model vs. LoRA Pilot 정량 평가 비교

학습에 사용하지 않은 동일한 **Test Set 8건(FR 5건 + TC 3건)**을 대상으로 파인튜닝 전후 성능을 비교한 결과, **전체 단순 평가 통과율이 37.5%(3/8)에서 87.5%(7/8)로 크게 향상**되었습니다. *(※ 객관적 평가를 위해 생성 오류를 수동 보정하지 않고 실제 실패로 기록)*

| 평가 항목 | 표본 수 | Base Model (`Qwen3.5-4B`) | LoRA Pilot (Fine-Tuned) | 변화 및 개선 폭 |
| :--- | :---: | :---: | :---: | :---: |
| **FR 3 LOOK 스키마 준수** | 5건 | 0 / 5 (0.0%) | **4 / 5 (80.0%)** | **+80.0%p** (구조 준수 성공) |
| **TC 첫 Tool 선택 정확도** | 3건 | 3 / 3 (100.0%) | **3 / 3 (100.0%)** | 0.0%p (기존 성능 안정 유지) |
| **전체 단순 평가 통과율** | **8건** | **3 / 8 (37.5%)** | **7 / 8 (87.5%)** | **+50.0%p 향상** |

* **Tool Calling 유지 검증:** `TC-006`(`get_calendar_events`), `TC-017`(`get_wardrobe`), `TC-027`(`get_calendar_events`) 모두 파인튜닝 이후에도 첫 Tool 선택 정확도 100%를 유지했습니다.

### 4.2. 출력 데이터 구조 변화 (FR-006 사례 비교)
(※ 출력 결과 파일을 찾지 못해서 json 문서의 형식만 맞춰 작성함.)

파인튜닝 전에는 임의의 중첩 딕셔너리 형태로 출력되어 시스템 연동이 불가능했으나, 학습 후에는 서비스가 정의한 배열 구조(`looks`)와 `look_id`·`role` 순서, 일반 추천 시 `item_id=null` 규칙을 정확히 준수했습니다.

```json
// [Before] Base Model 출력 (스키마 불일치 → 3 LOOK 판정 실패)
{
  "answer": {
    "look_1": { ... }
  }
}

// [After] LoRA Pilot 출력 (서비스 표준 JSON 스키마 준수)
{
  "looks": [
    { "look_id": 1, ... },
    { "look_id": 2, ... },
    { "look_id": 3, ... }
  ]
}
```

### 4.3. 오류 분석 및 한계점 도출

정량 평가 과정에서 발견된 실패 사례(`FR-015`)와 텍스트 품질 이슈를 분석하여 향후 고도화 과제를 도출했습니다.

1. **JSON 문법 파싱 오류 (`FR-015`):** 3개의 LOOK과 옷장 ID는 정상 추론했으나, `"style":"스포트리블",-items":[`와 같이 불필요한 하이픈(`-`) 토큰이 삽입되어 자동 JSON 파싱에 실패했습니다.
2. **소규모 데이터(80건) 학습에 따른 어휘 노이즈:** Pilot 단계의 제한된 학습 데이터로 인해 일부 오탈자 및 비표준 어휘(`러닝 셔틀츠`, `하프 지프`, `니론 팬츠`, `스포트리블`)가 관찰되었습니다.
3. **향후 개선 방향:**
   * Gold Data 기반 **합성 데이터 500~1,000건 확장** 및 **20~30% HITL 교차 검수**를 통한 어휘·추론 품질 강화
   * JSON Validator 파싱 실패 시 오류 로그를 반영해 자동 재생성하는 예외 처리 파이프라인 적용

## 5. 개인 기여도

팀 내 **AI/ML 및 학습 데이터 파이프라인 구축을 주도**했으며, Gold Data 설계부터 전처리·분할, LoRA Pilot 파인튜닝, 전후 정량 평가 및 오류 분석까지 전 과정을 수행했습니다.

1. **데이터 요구사항 분석 및 입·출력 스키마 설계 (분석/설계)**
   * 패션 추천(FR)과 도구 호출(TC) 판단을 위한 학습 데이터 포맷 및 고정 JSON 출력 구조(`looks` 배열 스키마) 정의
   * 공개 패션 데이터셋의 라이선스 및 활용 가능성 조사·검토
2. **Gold Data 구축 및 학습 파이프라인 전처리 (데이터 엔지니어링)**
   * 도메인 기준 정답 데이터(Gold Data) 80건(FR 50건, TC 30건) 구축 및 대화형 `JSONL` 포맷 변환
   * 평가 신뢰도 확보를 위한 데이터 자동 검증 및 `Train(64) / Validation(8) / Test(8)` 분할 수행
3. **SLM Pilot 파인튜닝 및 정량 비교 실험 수행 (모델링/평가)**
   * Google Colab T4 환경에서 `unsloth/Qwen3.5-4B` 기반 16-bit LoRA Pilot 학습 파이프라인 구현
   * 동일한 Test Set(8건)을 활용해 Base Model과 Fine-Tuned Model의 스키마 준수율(0/5 → 4/5) 및 Tool 선택 정확도(3/3) 정량 비교
   * `FR-015` JSON 파싱 오류 및 어휘 노이즈 원인을 분석하여 합성 데이터 확장 및 자동 재생성 개선안 도출
4. **Agent 아키텍처 공동 설계 및 출력 형식 연동 지원 (협업/검증)**
   * Router + Tool Calling 기반 Agent 흐름 및 4대 핵심 Tool 명세 공동 설계
   * Backend 및 DB 스키마와 호환되도록 모델의 최종 JSON 출력 형식 정합성 검증 지원

## 6. 실행 방법

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

### 6.2. 학습 및 평가 실행 흐름

```bash
# 1. Gold Data 전처리 및 Train / Val / Test 분할
python src/prepare_dataset.py

# 2. Base Model vs. LoRA Pilot 동일 Test Set 평가 실행
python src/evaluate_pilot.py
```

### 6.3. 주의사항

* 현재 반영된 실험 수치는 파이프라인 검증을 위한 **Gold Data 80건 기반 Pilot 파인튜닝 결과**이며, 최종 서비스 모델 성능은 합성 데이터(500~1,000건) 확장 및 HITL 검수 완료 후 재학습을 통해 산출됩니다.
* 대용량 모델 가중치(LoRA Adapter) 및 민감 인증 정보(`.env`)는 보안 및 용량 관리를 위해 저장소에 직접 포함하지 않았습니다.
