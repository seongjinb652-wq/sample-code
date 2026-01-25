# ============================================================
# NVIDIA Core LLM - Embeddings & Instruct LLM Setup
# ------------------------------------------------------------
# 이 스크립트는 NVIDIA Embeddings 및 Instruct LLM을 초기화하는
# 예제 코드입니다.
#
# 주요 기능:
#  - Embeddings 모델 불러오기 및 설정
#  - Instruct LLM 모델 불러오기 및 설정
# ============================================================
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Embeddings 모델 목록 확인
# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# Instruct LLM 모델 목록 확인
# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
