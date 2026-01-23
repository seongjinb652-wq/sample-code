# ============================================================
# File: nvidea_core_llm_chat_stream.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   NVIDIA 샘플 코드를 기반으로 작성된 클라이언트 채팅 스트리밍 예제.
#   REST API 호출을 통해 LLM 서버와 대화하며,
#   응답을 JSON 파싱 후 토큰 단위로 실시간 출력하는 구조를 포함.
#
# Usage:
#   - 단독 실행 가능
#   - 조합 가능 (nvidea_core_llm_model_list.py, nvidea_core_llm_chat_request.py 와 조합 권장)
# ============================================================

import requests
import json

## Use requests.post to send the header (streaming meta-info) the payload to the endpoint
## Make sure streaming is enabled, and expect the response to have an iter_lines response.
response = requests.post(invoke_url, headers=headers, json=payload, stream=True)

## If your response is an error message, this will raise an exception in Python
try: 
    response.raise_for_status()  ## If not 200 or similar, this will raise an exception
except Exception as e:
    # print(response.json())
    print(response.json())
    raise e

## Custom utility to make live a bit easier
def get_stream_token(entry: bytes):
    """Utility: Coerces out ['choices'][0]['delta'][content] from the bytestream"""
    if not entry: return ""
    entry = entry.decode('utf-8')
    if entry.startswith('data: '):
        try: entry = json.loads(entry[5:])
        except ValueError: return ""
    return entry.get('choices', [{}])[0].get('delta', {}).get('content') or ""

## If the post request is honored, you should be able to iterate over 
for line in response.iter_lines():
    
    ## Without Processing: data: {"id":"...", ... "choices":[{"index":0,"delta":{"role":"assistant","content":""}...}...
    # if line: print(line.decode("utf-8"))

    ## With Processing: An actual stream of tokens printed one-after-the-other as they come in
    print(get_stream_token(line), end="")
