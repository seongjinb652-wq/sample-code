# ==================================================
# File: nvidea_core_llm_doc_setup_util.py
# Author: 성진
# Date: 2026-01-24
# Description: NVIDIA 기반 LLM 환경 셋업 유틸리티 코드.
#              Colab 환경에서는 필요한 패키지 설치를 포함하며,
#              LangChain 및 NVIDIA AI Endpoints 연동을 위한 초기 설정을 제공한다.
# Usage:★★ - 단독 실행 가능
# ==================================================

# ---- Colab 환경 패키지 설치 (코스 환경에서는 불필요) ----
# %pip install -qq langchain langchain-nvidia-ai-endpoints gradio
# %pip install -qq arxiv pymupdf

# ---- NVIDIA API 키 환경 변수 설정 ----
# import os
# os.environ["NVIDIA_API_KEY"] = "nvapi-..."

# ---- 콘솔 출력 스타일 설정 ----
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

# ---- NVIDIA LLM 모델 확인 ----
from langchain_nvidia_ai_endpoints import ChatNVIDIA
ChatNVIDIA.get_available_models()  # 중간 상태 확인용 유틸리티 메서드

# ---- LangChain RunnableLambda 활용 ----
from langchain_core.runnables import RunnableLambda
from functools import partial

# ---- 상태 출력 함수 (일반 print) ----
def RPrint(preface="State: "):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

# ---- 상태 출력 함수 (rich pprint) ----
def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))
