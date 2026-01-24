# README: Embedding 실험 및 Guardrailing

## 📌 개요
이 프로젝트는 **NVIDIA 기반 LangChain Embeddings**를 활용하여 질의(Query)와 문서(Document)를 벡터화하고,  
유사도 분석 및 Guardrailing(의미적 안전장치) 실험을 수행하는 예제 코드 모음입니다.  

## 🛠️ 주요 기능
- **문서 임베딩**: `NVIDIAEmbeddings` 모델을 사용하여 질의와 문서를 벡터화
- **유사도 분석**: Cosine Similarity 기반으로 Query-Document 간 관계 시각화
- **Guardrailing 실험**: 좋은 응답과 나쁜 응답을 분류하여 안전성 검증
- **시각화**: PCA 및 t-SNE를 활용한 임베딩 분포 시각화
- **모델 학습**:  
  - 얕은 신경망(Neural Network)  
  - 로지스틱 회귀(Logistic Regression)  
  두 가지 접근으로 Guardrailing 분류 성능 비교

## 📂 파일 구조
- `nvidea_core_llm_doc_setup_util.py` : 문서 로딩 및 환경 셋업 유틸리티
- `nvidea_core_llm_doc_summary_util.py` : 문서 요약 체인 구성
- `nvidea_core_llm_doc_embedding_story.py` : 질의/문서 임베딩 및 스토리 확장
- `nvidea_core_llm_guardrail_embedding.py` : 좋은/나쁜 응답 임베딩 비교 및 시각화
- `nvidea_core_llm_guardrail_training.py` : Guardrailing 분류 모델 학습 (NN, Logistic Regression)
- `readme_embedding.md` : 본 문서

## 🚀 실행 방법
1. NVIDIA API 키 설정  
   ```python
   import os
   os.environ["NVIDIA_API_KEY"] = "nvapi-..."
