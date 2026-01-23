# ============================================================
# File: nvidea_core_llm_langchain_client.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   LangChain NVIDIA AI Endpoints의 ChatNVIDIA를 활용한 예제.
#   환경변수 기반 API Key 사용, 기본 호출, 마지막 입력/응답 확인까지 포함.
#
# Usage:
#   - 단독 실행 가능
#   - 조합 가능 (nvidea_core_llm_model_list.py, nvidea_core_llm_chat_request.py 와 조합 권장)
# ============================================================

from langchain_nvidia_ai_endpoints import ChatNVIDIA

## NVIDIA_API_KEY는 환경변수에서 자동으로 불러옵니다.
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

# 필요 시 직접 base_url 지정 가능 (예: 로컬 서버 테스트)
# llm = ChatNVIDIA(
#     model="mistralai/mixtral-8x7b-instruct-v0.1",
#     mode="open",
#     base_url="http://llm_client:9000/v1"
# )

## 간단한 요청 실행: "Hello World" 메시지를 모델에 전달
llm.invoke("Hello World")

## 마지막 입력과 응답 객체를 확인할 수 있습니다.
# - last_inputs: 직전 요청에 사용된 입력 데이터
# - last_response: 직전 요청에 대한 서버 응답
print(llm._client.last_inputs)
print(llm._client.last_response)

## 응답을 JSON 형태로 변환하여 구조 확인
print(llm._client.last_response.json())
