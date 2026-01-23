# ============================================================
# File: nvidea_core_llm_openai_client.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   OpenAI Python Client를 활용하여 NVIDIA LLM 서버와 통신하는 예제.
#   스트리밍/비스트리밍 응답을 모두 처리하는 구조를 포함.
#
# Usage:
#   - 단독 실행 가능
#   - 조합 가능 (nvidea_core_llm_model_list.py, nvidea_core_llm_chat_stream.py 와 조합 권장)
# ============================================================
## Using General OpenAI Client
from openai import OpenAI

# client = OpenAI()  ## Assumes OPENAI_API_KEY is set

# client = OpenAI(
#     base_url = "https://integrate.api.nvidia.com/v1",
#     api_key = os.environ.get("NVIDIA_API_KEY", "")
# )

client = OpenAI(
    base_url = "http://llm_client:9000/v1",
    api_key = "I don't have one"
)

completion = client.chat.completions.create(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    # model="gpt-4-turbo-2024-04-09",
    messages=[{"role":"user","content":"Hello World"}],
    temperature=1,
    top_p=1,
    max_tokens=1024,
    stream=True,
)
# 첫 번째 completion (stream=True) 
# for chunk in completion: 루프를 돌면서 토큰 단위로 응답을 실시간 출력합니다.
# 즉, 모델이 생성하는 답변을 한 글자/한 단어씩 바로바로 확인할 수 있습니다.

## Streaming with Generator: Results come out as they're generated
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")


## Non-Streaming: Results come from server when they're all ready
completion = client.chat.completions.create(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    # model="gpt-4-turbo-2024-04-09",
    messages=[{"role":"user","content":"Hello World"}],
    temperature=1,
    top_p=1,
    max_tokens=1024,
    stream=False,
)
# 두 번째 completion (stream=False) 전체 응답이 서버에서 다 준비된 뒤 한 번에 반환됩니다.
# 스트리밍과 달리, 결과를 기다렸다가 최종 완성된 답변을 받는 구조입니다.
completion
