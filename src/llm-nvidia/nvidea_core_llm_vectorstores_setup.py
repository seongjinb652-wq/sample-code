# ============================================================
# NVIDIA Core LLM - Vector Stores Setup
# ------------------------------------------------------------
# 이 스크립트는 LangChain 및 관련 라이브러리를 활용하여
# Vector Store 환경을 설정하고 초기화하는 예제입니다.
#
# 주요 기능:
#  - 필요한 패키지 설치 (Colab 환경에서만 필요)
#  - Rich 콘솔 스타일 정의
#  - 출력 함수(pprint) 설정
# ============================================================
# %%capture
## ^^ 설치 과정 로그를 보고 싶다면 주석 처리하세요

## Colab 환경에서만 필요, 일반 환경에서는 불필요
# %pip install -q langchain langchain-nvidia-ai-endpoints gradio rich
# %pip install -q arxiv pymupdf faiss-cpu

## typing-extensions 관련 오류 발생 시 런타임 재시작 후 재설치
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# ChatNVIDIA.get_available_models()

from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# Rich 콘솔 스타일 정의
console = Console()
base_style = Style(color="#76B900", bold=True)

# pprint 함수: NVIDIA 그린 색상으로 출력
pprint = partial(console.print, style=base_style)
