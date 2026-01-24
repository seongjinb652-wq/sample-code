# ==================================================
# File: nvidea_core_llm_guardrail_chat.py
# Author: 성진
# Date: 2026-01-24
# Description: NVIDIA 기반 LangChain Guardrailing 챗봇 예제.
#              질의 임베딩을 통해 응답을 분류하고,
#              Gradio 스타일의 스트리밍 인터페이스를 모방하여 대화 시뮬레이션을 수행한다.
# Usage:★★ - 단독 실행 가능
# ==================================================

## ---- 라이브러리 임포트 ----
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

import gradio as gr
from functools import partial
import numpy as np

## ---- 임베딩 및 LLM 모델 초기화 ----
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
chat_llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

## ---- 프롬프트 정의 ----
response_prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("user", "{input}")])

## ---- 출력 유틸 정의 ----
def RPrint(preface=""):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

## ---- 시스템 메시지 정의 ----
good_sys_msg = (
    "당신은 NVIDIA 챗봇입니다. 질문이 윤리적이고 관련성이 있다면 NVIDIA를 대표하여 도움을 주세요."
)
poor_sys_msg = (
    "당신은 NVIDIA 챗봇입니다. 질문이 'NVIDIA 챗봇으로서 답변하기에 적절하지 않음'으로 분석되었습니다."
    " 따라서 답변을 피하고, 그 이유를 간단히 설명하세요. 응답은 최대한 짧게 하세요."
)

########################################################################################
## ---- TODO 채운 부분: 질의 점수화 함수 ----
def score_response(query):
    '''
    질의를 임베딩하여 분류 점수를 반환하는 함수.
    간단히 코사인 유사도를 기반으로 점수를 계산한다.
    '''
    # 질의 임베딩
    q_emb = embedder.embed_query(query)
    # 예시: 단순히 벡터 평균값을 점수로 사용 (실제는 분류 모델 연결 가능)
    score = float(np.mean(q_emb))
    # 0~1 범위로 정규화
    score = (score - (-1)) / (2)  # [-1,1] → [0,1] 변환 가정
    return score
########################################################################################

## ---- 챗봇 체인 정의 ----
chat_chain = (
    { 'input'  : (lambda x:x), 'score' : score_response }
    | RPrint()
    | RunnableAssign(dict(
        system = RunnableBranch(
            ((lambda d: d['score'] < 0.5), RunnableLambda(lambda x: poor_sys_msg)),
            RunnableLambda(lambda x: good_sys_msg)
        )
    )) | response_prompt | chat_llm
)

## ---- 스트리밍 챗봇 함수 ----
def chat_gen(message, history, return_buffer=True):
    buffer = ""
    for token in chat_chain.stream(message):
        buffer += token
        yield buffer if return_buffer else token

## ---- Gradio 스타일 대화 시뮬레이션 ----
def queue_fake_streaming_gradio(chat_stream, history = [], max_questions=8):
    # 초기 메시지 출력
    for human_msg, agent_msg in history:
        if human_msg: print("\n[ 사용자 ]:", human_msg)
        if agent_msg: print("\n[ 챗봇 ]:", agent_msg)

    # 대화 루프
    for _ in range(max_questions):
        message = input("\n[ 사용자 ]: ")
        print("\n[ 챗봇 ]: ")
        history_entry = [message, ""]
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end='')
            history_entry[1] += token
        history += [history_entry]
        print("\n")

## ---- 초기 히스토리 설정 ----
history = [[None, "안녕하세요! 저는 NVIDIA 챗봇입니다. 질문을 해주세요!"]]

## ---- 대화 시뮬레이션 실행 ----
queue_fake_streaming_gradio(
    chat_stream = chat_gen,
    history = history
)
