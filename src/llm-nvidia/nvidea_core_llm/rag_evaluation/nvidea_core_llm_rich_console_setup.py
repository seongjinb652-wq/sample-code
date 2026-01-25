# ============================================================
# NVIDIA Core LLM - Rich Console Setup & LLM 초기화
# ------------------------------------------------------------
# 이 스크립트는 Rich 콘솔 스타일을 정의하고,
# NVIDIA Embeddings 및 ChatNVIDIA 모델을 초기화하는 예제입니다.
#
# 주요 기능:
#  - Rich 콘솔 스타일 정의 (색상, Bold 등)
#  - Embeddings 모델 불러오기
#  - ChatNVIDIA 모델 불러오기
# ============================================================
# 필요한 라이브러리 설치 (주석 처리된 pip 명령어)
# %pip install -q langchain langchain-nvidia-ai-endpoints gradio rich
# %pip install -q arxiv pymupdf faiss-cpu ragas

## typing-extensions 관련 오류가 발생하면 런타임을 재시작 후 다시 시도하세요
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# ChatNVIDIA.get_available_models()

from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# Rich 콘솔 객체 생성
console = Console()

# 콘솔 스타일 정의
base_style = Style(color="#76B900", bold=True)   # NVIDIA 그린 색상 + Bold
norm_style = Style(bold=True)                    # 기본 Bold 스타일

# 출력 함수 정의
pprint = partial(console.print, style=base_style)  # 기본 스타일 출력
pprint2 = partial(console.print, style=norm_style) # 보조 스타일 출력

# NVIDIA AI Endpoints 불러오기
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Embeddings 모델 초기화
# 사용 가능한 모델 확인: NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# ChatNVIDIA 모델 초기화
# 사용 가능한 모델 확인: ChatNVIDIA.get_available_models(base_url="http://llm_client:9000/v1")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
