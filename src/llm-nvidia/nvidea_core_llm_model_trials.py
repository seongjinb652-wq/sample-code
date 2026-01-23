# ============================================================
# File: nvidea_core_llm_model_trials.py
# Author: 성진
# Date: 2026-01-23
# Description:
#   LangChain NVIDIA ChatNVIDIA를 활용하여 여러 모델을 순회하며
#   스트리밍 응답을 테스트하는 예제.
#   환경변수 확인, 모델 필터링, 스트리밍 출력, 예외 처리까지 포함.
#
# Usage:
#   - 단독 실행 가능
#   - 조합 가능 (nvidea_core_llm_langchain_client.py, nvidea_core_llm_model_list.py 와 조합 권장)
# ============================================================
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

## [셀1] 환경변수 확인
## NVIDIA 관련 환경변수들을 확인합니다.
print({k: v for k, v in os.environ.items() if k.startswith("NVIDIA_")})

## Translation: base_url을 llm_client:9000 으로 설정하고,
## "open" API-spec 접근을 통해 모델 검색 및 URL 포맷을 사용할 수 있습니다.

# ------------------------------------------------------------

## [셀2] 모델 목록 조회 및 스트리밍 테스트
model_list = ChatNVIDIA.get_available_models()

for model_card in model_list:
    model_name = model_card.id

    ## 특정 키워드가 포함된 모델만 필터링 (meta/llama 등)
    if not any([keyword in model_name for keyword in ["meta/llama"]]):
        continue
    if "405b" in model_name:
        continue
    if "embed" in model_name:
        continue

    ## 모델별 스트리밍 요청 실행
    llm = ChatNVIDIA(model=model_name)
    print(f"TRIAL: {model_name}")

    try:
        for token in llm.stream("Tell me about yourself! 2 sentences.", max_tokens=100):
            print(token.content, end="")
    except Exception as e:
        print(f"EXCEPTION: {e}")    ## 일부 모델 실패 시 다른 모델 사용 가능
    except KeyboardInterrupt:
        print(f"Stopped manually")  ## 실행 중 수동 중단 가능
        break

    print("\n\n" + "=" * 84)
