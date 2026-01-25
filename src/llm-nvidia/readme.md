# NVIDIA Core LLM Examples

이 저장소는 **LangChain + NVIDIA AI Endpoints**를 활용한 다양한 실습 예제를 포함합니다.  
문서 처리, 대화형 체인, 벡터스토어, Gradio 챗봇 등 여러 기능을 단계별로 구현했습니다.  

---

## 📌 주요 파일 설명

### 대화 및 체인
- `nvidea_core_llm_chat_request.py` → 단일 대화 요청 (비스트리밍)
- `nvidea_core_llm_chat_stream.py` → 스트리밍 대화 요청
- `nvidea_core_llm_auto_conversation_store.py` → 대화 맥락 자동 저장
- `nvidea_core_llm_context_chain.py` → ChatPromptTemplate 기반 질의응답 체인
- `nvidea_core_llm_retrieval_chain.py` → history/context 기반 Retrieval Chain

### 문서 처리
- `nvidea_core_llm_loading_chunking_docs.py` → 논문 로딩 및 청크 분할
- `nvidea_core_llm_doc_summary_util.py` → 텍스트 요약 체인
- `nvidea_core_llm_doc_embedding_story.py` → 문서 임베딩 및 유사도 시각화

### 벡터스토어
- `nvidea_core_llm_construct_vectorstores.py` → 여러 Vector Store 병합
- `nvidea_core_llm_conversation_vectorstore.py` → 대화용 Vector Store 생성
- `nvidea_core_llm_save_index.py` → 인덱스 저장 및 불러오기
- `nvidea_core_llm_vectorstores_setup.py` → 콘솔 스타일 설정

### Gradio 챗봇
- `nvidea_core_llm_gradio_chatbot.py` → Gradio 기반 챗봇 인터페이스
- `nvidea_core_llm_guardrail_chat.py` → Guardrail 챗봇 시뮬레이션

### Guardrailing & Embedding
- `nvidea_core_llm_guardrail_embedding.py` → 좋은/나쁜 응답 임베딩 비교
- `nvidea_core_llm_guardrail_training.py` → PCA/t-SNE + 분류 모델 학습
- `nvidea_core_llm_embeddings_instruct_setup.py` → Embeddings 모델 설정

### Knowledge Base
- `knowledge_base_flight_simple_example.py` → 항공편 조회 (LLM 연결 없음)
- `nvidea_core_llm_knowledge_base_flight_chain_kor.py` → 한국어 항공편 조회 체인
- `nvidea_core_llm_knowledge_base_update_chain.py` → KnowledgeBase 업데이트 테스트

---

## 📌 추천 실행 환경
- Python **3.10 ~ 3.11** 권장
- 필수 라이브러리: `langchain`, `faiss`, `gradio`, `langchain_nvidia_ai_endpoints`

---

## 📌 활용 팁
- 실습 후 `docstore.save_local()`로 인덱스를 저장해두면, 최종 평가나 다른 프로젝트에서 재사용 가능  
- Gradio 챗봇 예제를 실행하면 웹 브라우저에서 직접 대화형 테스트 가능  
- Guardrailing 예제는 모델 응답 품질 평가 및 필터링에 활용 가능  

---
