# ============================================================
# NVIDIA Core LLM - Retrieval Chain with Conversation Memory
# ------------------------------------------------------------
# 이 스크립트는 LangChain을 활용하여 문서 기반 대화형 챗봇을
# 구성하는 예제입니다.
#
# 주요 기능:
#  - NVIDIA Embeddings 및 LLM 초기화
#  - 대화 기록(convstore) 저장 및 불러오기
#  - ChatPromptTemplate을 통한 질의응답 프롬프트 구성
#  - Retrieval Chain 구현 (TODO 부분에서 history/context 연결 필요)
#  - Gradio 인터페이스를 통한 대화 스트리밍
# ============================================================
from langchain.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import gradio as gr
from functools import partial
from operator import itemgetter

# NVIDIA Embeddings 모델 초기화
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# LLM 모델 초기화
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
# instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

# 대화 저장소 초기화
convstore = default_FAISS()

def save_memory_and_get_output(d, vstore):
    """사용자 입력/출력을 convstore에 저장"""
    vstore.add_texts([
        f"사용자: {d.get('input')}",
        f"에이전트: {d.get('output')}"
    ])
    return d.get('output')

# 초기 메시지
initial_msg = (
    "안녕하세요! 저는 문서 기반 챗봇 에이전트입니다."
    f" 현재 접근 가능한 문서 목록: {doc_string}\n\n무엇을 도와드릴까요?"
)

# 프롬프트 템플릿 정의
chat_prompt = ChatPromptTemplate.from_messages([("system",
    "당신은 문서 챗봇입니다. 사용자가 문서에 대해 질문하면 도와주세요."
    " 사용자 입력: {input}\n\n"
    " 검색된 대화 기록:\n{history}\n\n"
    " 검색된 문서 내용:\n{context}\n\n"
    " (검색된 내용만 활용하여 답변하세요. 사용된 출처만 인용하고, 대화체로 답변하세요.)"
), ('user', '{input}')])

# 스트리밍 체인 정의
stream_chain = chat_prompt | RPrint() | instruct_llm | StrOutputParser()

################################################################################################
## TODO: Retrieval Chain 구현
retrieval_chain = (
    {'input': (lambda x: x)}
    | RunnableAssign({'history': itemgetter('input') | convstore.as_retriever() | docs2str})
    | RunnableAssign({'context': itemgetter('input') | docstore.as_retriever() | long_reorder | docs2str})
)

# retrieval_chain = (
#    {'input' : (lambda x: x)}
#    ## TODO: convstore에서 history, docstore에서 context를 가져오도록 수정 필요
#    ## HINT: RunnableAssign, itemgetter, long_reorder, docs2str 활용
#    | RunnableAssign({'history' : lambda d: None})
#    | RunnableAssign({'context' : lambda d: None})
#)
################################################################################################

def chat_gen(message, history=[], return_buffer=True):
    buffer = ""
    ## 1. 입력 메시지를 기반으로 retrieval 수행
    retrieval = retrieval_chain.invoke(message)

    ## 2. 스트리밍 체인 실행
    for token in stream_chain.stream(retrieval):
        buffer += token
        yield buffer if return_buffer else token

    ## 3. 대화 기록 저장
    save_memory_and_get_output({'input':  message, 'output': buffer}, convstore)

# 테스트 실행
test_question = "RAG에 대해 알려주세요!"  ## <- 원하는 질문으로 변경 가능
for response in chat_gen(test_question, return_buffer=False):
    print(response, end='')
