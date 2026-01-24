# ============================================================
# File: nvidea_core_llm_langchain_rich_utils.py
# Author: 성진
# Date: 2026-01-24
# Description:
#   LangChain NVIDIA ChatNVIDIA 클라이언트와 Rich 콘솔 출력을 결합한 예제.
#   RPrint, PPrint 유틸을 통해 상태를 직관적으로 출력할 수 있음.
#
# Usage:★★
#   - 단독 실행 가능 (LangChain 클라이언트 및 출력 유틸 테스트)
#   - 조합 가능 (nvidea_core_llm_langchain_client.py, nvidea_core_llm_chat_stream.py 와 함께 사용 권장)
# ============================================================
## Cell 1: 설치 및 환경 변수 설정 (Colab 전용)
# %pip install -q langchain langchain-nvidia-ai-endpoints gradio

# import os
# os.environ["NVIDIA_API_KEY"] = "nvapi-..."

## Cell 2: Rich 콘솔 및 기본 스타일 정의
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

## Cell 3: LangChain NVIDIA ChatNVIDIA 및 출력 유틸 정의
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# ChatNVIDIA.get_available_models() ## Useful utility method for printing intermediate states
from langchain_core.runnables import RunnableLambda
from functools import partial

def RPrint(preface="State: "):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

#📚 Rich 라이브러리 요약
# Rich는 Python에서 터미널 출력을 강화하는 라이브러리로, 단순한 print() 대신 컬러, 스타일, 표, 트리, 프로그레스바 등을 지원해준다.
# 🔑 주요 기능
# 🎨 텍스트 스타일링: 색상, 굵기, 배경색 지정 가능
# 📊 표/트리 출력: 데이터 구조를 직관적으로 시각화
# ⏳ 프로그레스바: 작업 진행 상황 표시
# 🖼 마크다운 렌더링: 콘솔에서 바로 마크다운 문법 지원
# 📝 로깅 강화: 로그 메시지를 색상과 구조로 구분
# 📚 Rich 라이브러리 사용법은 nvidea_core_llm_langchain_rich_utils.py 파일 상단 요약 참고

# 💡 사용 예시
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# table = Table(title="Model Performance")
# table.add_column("Model", style="cyan")
# table.add_column("Accuracy", style="green")
#
# table.add_row("Mixtral-8x7B", "92%")
# table.add_row("GPT-4 Turbo", "95%")

# console.print(table)
