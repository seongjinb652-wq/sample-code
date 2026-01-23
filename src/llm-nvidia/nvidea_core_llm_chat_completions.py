# ============================================================
# File: nvidea_core_llm_chat_request.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   NVIDIA 샘플 코드를 기반으로 작성된 클라이언트 채팅 요청 예제.
#   REST API 호출을 통해 LLM 서버와 대화형 응답을 주고받는 기본 구조를 포함.
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
