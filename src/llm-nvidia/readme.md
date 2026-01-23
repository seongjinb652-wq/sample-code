# nvidea_core_llm Examples

## 📚 개요
이 저장소는 NVIDIA 샘플 코드를 기반으로 작성된 LLM 클라이언트 예제 모음입니다.  
교육용으로 시작했지만, 실무에서도 활용할 수 있도록 구조화되어 있습니다.  
각 파일은 **단독 실행 가능**하며, 필요에 따라 **조합 실행**도 권장됩니다.

---

## 📂 파일 목록 및 추천도

### 1. `nvidea_core_llm_client_request.py` ★
- 서버 연결 확인 및 기본 GET 요청 예제
- 교육용: REST 호출 구조 학습
- 실무 활용성: 낮음 (단순 연결 확인용)

---

### 2. `nvidea_core_llm_model_list.py` ★★
- `/v1/models` 엔드포인트에서 모델 목록 조회
- 교육용: 모델 탐색 및 응답 구조 학습
- 실무 활용성: 중간 (모델 관리/탐색에 유용)

---

### 3. `nvidea_core_llm_chat_request.py` ★★
- 특정 모델에 단일 요청을 보내고 응답 수신
- 교육용: 기본 대화 요청 구조 학습
- 실무 활용성: 중간 (단일 요청 테스트에 적합)

---

### 4. `nvidea_core_llm_chat_completions.py` ★★
- `chat/completions` 엔드포인트 활용 예제
- 교육용: OpenAI/NVIDIA API 구조 학습
- 실무 활용성: 중간 (대화형 응답 처리에 적합)

---

### 5. `nvidea_core_llm_chat_stream.py` ★★★
- 스트리밍 응답을 토큰 단위로 실시간 출력
- 교육용: 스트리밍 구조 학습
- 실무 활용성: 높음 (실시간 응답 처리에 필수)

---

### 6. `nvidea_core_llm_openai_client.py` ★★
- OpenAI Python Client를 활용한 NVIDIA LLM 호출
- 교육용: 클라이언트 라이브러리 활용법 학습
- 실무 활용성: 중간 (OpenAI/NVIDIA API 통합 테스트에 적합)

---

### 7. `nvidea_core_llm_langchain_client.py` ★★
- LangChain NVIDIA ChatNVIDIA 클라이언트 활용 예제
- 교육용: LangChain과 NVIDIA API 연동 학습
- 실무 활용성: 중간 (LangChain 기반 프로젝트에 적합)

---

### 8. `nvidea_core_llm_model_trials.py` ★★★
- 여러 모델을 순회하며 스트리밍 응답 테스트
- 교육용: 모델 비교 및 필터링 학습
- 실무 활용성: 높음 (모델 성능/응답 비교에 유용)

---

### 9. `nvidea_core_llm_early_stopping.py` ★★★
- PyTorch 학습 루프에 Early Stopping 기능 추가
- 교육용: 학습 중단 조건 구현 학습
- **실무 활용성: 매우 높음 (GPU 자원 절약, 과적합 방지에 필수)**

---

## 📂 응답 JSON 예시

```json
{
    "id": "d34d436a-c28b-4451-aa9c-02eed2141ed3",
    "choices": [{
        "index": 0,
        "message": { "role": "assistant", "content": "Bonjour! ..." },
        "finish_reason": "stop"
    }],
    "usage": {
        "completion_tokens": 450,
        "prompt_tokens": 152,
        "total_tokens": 602
    }
}

## 실행 흐름 예시
모델 확인

bash
python nvidea_core_llm_model_list.py
단일 요청 테스트

bash
python nvidea_core_llm_chat_request.py
스트리밍 응답 테스트

bash
python nvidea_core_llm_chat_stream.py
LangChain 기반 모델 비교

bash
python nvidea_core_llm_model_trials.py
Early Stopping 적용 학습

bash
python nvidea_core_llm_early_stopping.py
