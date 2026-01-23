# ============================================================
# File: nvidea_core_llm_client_request.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   NVIDIA   샘플 코드를 기반으로 작성된 클라이언트 요청 예제.
#   REST API 호출을 통해 LLM 서버와 통신하는 기본 구조를 포함.
# ============================================================

import requests

# 서버 호출 URL
invoke_url = "http://llm_client:9000"

# 요청 헤더
headers = {"content-type": "application/json"}

# GET 요청 및 응답 JSON 출력
response = requests.get(invoke_url, headers=headers, stream=False).json()
print(response)
