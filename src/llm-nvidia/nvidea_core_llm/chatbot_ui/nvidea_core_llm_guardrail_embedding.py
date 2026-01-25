# ==================================================
# File: nvidea_core_llm_guardrail_embedding.py
# Author: 성진
# Date: 2026-01-24
# Description: NVIDIA 기반 LangChain Guardrailing 임베딩 실험 코드.
#              좋은/나쁜 질의 응답을 구분하여 임베딩 후 시각화(PCA, t-SNE)로 비교한다.
# Usage:★★ - 단독 실행 가능
# ==================================================

## ---- Colab 환경 패키지 설치 (코스 환경에서는 불필요) ----
# %pip install -qq langchain langchain-nvidia-ai-endpoints gradio

# import os
# os.environ["NVIDIA_API_KEY"] = "nvapi-..."

## ---- 콘솔 출력 스타일 설정 ----
from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

## ---- NVIDIA Embeddings 및 LLM 모델 불러오기 ----
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

NVIDIAEmbeddings.get_available_models()

## ---- 출력 파서 및 유틸 정의 ----
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ChatMessage
from operator import itemgetter

def EnumParser(*idxs):
    '''Mistral 모델이 번호 매긴 출력값을 파싱하는 유틸'''
    idxs = idxs or [slice(0, None, 1)]
    entry_parser = lambda v: v if ('. ' not in v) else v[v.index('. ')+2:]
    out_lambda = lambda x: [entry_parser(v).strip() for v in x.split("\n")]
    return StrOutputParser() | RunnableLambda(lambda x: itemgetter(*idxs)(out_lambda(x)))

instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1") | EnumParser()

## ---- 프롬프트 정의 ----
gen_prompt = {'input' : lambda x:x} | ChatPromptTemplate.from_template(
    "20개의 대표적인 대화 예시를 {input}에 맞게 생성하세요."
    " 질문은 모두 다른 표현과 내용으로 작성하세요."
    " 질문에 답하지 말고, 번호를 붙여 나열하세요."
    " 예시: 1. <질문>\n2. <질문>\n3. <질문>\n..."
)

## ---- 응답 생성 ----
responses_1 = (gen_prompt | instruct_llm).invoke(
    "NVIDIA
