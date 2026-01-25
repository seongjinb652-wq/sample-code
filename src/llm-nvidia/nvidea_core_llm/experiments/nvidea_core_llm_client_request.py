# ============================================================
# File: nvidea_core_llm_client_request.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   LLM 서버의 기본 상태/엔드포인트를 GET으로 확인하는 간단 요청 예제.
#   연결 확인 및 응답 JSON 출력 흐름을 포함.
#
# Usage:
#   - 단독 실행 가능
#   - 조합 가능 (nvidea_core_llm_model_list.py 와 조합 권장)
# ============================================================


import requests

# 서버 호출 URL
invoke_url = "http://llm_client:9000"

# 요청 헤더
headers = {"content-type": "application/json"}

# GET 요청 및 응답 JSON 출력
response = requests.get(invoke_url, headers=headers, stream=False).json()
print(response)
