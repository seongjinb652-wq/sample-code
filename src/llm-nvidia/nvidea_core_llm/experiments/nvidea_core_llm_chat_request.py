# ============================================================
# File: nvidea_core_llm_chat_request.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   ★ 단일 대화 요청을 보내고 비스트리밍 응답을 받는 기본 예제.
#   헤더/페이로드 구성과 POST 호출 흐름을 명확히 제시.
#
# Usage: ★
#   - 단독 실행 가능
#   - 조합 가능 (nvidea_core_llm_model_list.py 와 조합 권장)
# ============================================================


from getpass import getpass
import os
import requests

## Where are you sending your requests?
invoke_url = "http://llm_client:9000/v1/chat/completions"

## If you wanted to use your own API Key, it's very similar
# if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
#     os.environ["NVIDIA_API_KEY"] = getpass("NVIDIA_API_KEY: ")
# invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

## If you wanted to use OpenAI, it's very similar
# if not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
#     os.environ["OPENAI_API_KEY"] = getpass("OPENAI_API_KEY: ")
# invoke_url = "https://api.openai.com/v1/models"

## Meta communication-level info about who you are, what you want, etc.
headers = {
    "accept": "text/event-stream",
    "content-type": "application/json",
    # "Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY')}",
    # "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
}

## Arguments to your server function
payload = {
    "model": "mistralai/mixtral-8x7b-instruct-v0.1",
    "messages": [{"role": "user", "content": "Tell me hello in French"}],
    "temperature": 0.5,
    "top_p": 1,
    "max_tokens": 1024,
    "stream": True
}
