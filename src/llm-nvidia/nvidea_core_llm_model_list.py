# ============================================================
# File: nvidea_core_llm_model_list.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   서버에 등록된 모델 목록을 조회하고 ID를 출력하는 예제.
#   /v1/models 엔드포인트 응답을 순회하여 가독성 있게 표시.
#
# Usage:
#   - 단독 실행 가능
#   - 조합 가능 (nvidea_core_llm_chat_request.py, nvidea_core_llm_chat_stream.py 와 조합 권장)
# ============================================================


import requests

# 기본 호출 URL (LLM 서버)
invoke_url = "http://llm_client:9000/v1/models"
# 다른 예시 URL들:
# invoke_url = "https://api.openai.com/v1/models"
# invoke_url = "https://integrate.api.nvidia.com/v1"
# invoke_url = "http://llm_client:9000/v1/models/mistralai/mixtral-8x7b-instruct-v0.1"
# invoke_url = "http://llm_client:9000/v1/models/mistralaimixtral-8x7b-instruct-v0.1"

headers = {
    "content-type": "application/json",
    # 필요 시 인증키 추가:
    # "Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY')}",
    # "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
}

print("Available Models:")
response = requests.get(invoke_url, headers=headers, stream=False)
# print(response.json())  ## <- Raw Response. Very Verbose
for model_entry in response.json().get("data", []):
    print(" -", model_entry.get("id"))

print("\nExample Entry:")
invoke_url = "http://llm_client:9000/v1/models/mistralai/mixtral-8x7b-instruct-v0.1"
example_response = requests.get(invoke_url, headers=headers, stream=False).json()
print(example_response)
