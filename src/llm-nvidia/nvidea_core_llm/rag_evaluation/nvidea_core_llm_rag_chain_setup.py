# ============================================================
# NVIDIA Core LLM - RAG Chain Setup
# ------------------------------------------------------------
# 이 스크립트는 LangChain Core 모듈을 활용하여
# RAG(Retrieval-Augmented Generation) 체인을 구성하는 예제입니다.
#
# 주요 기능:
#  - Embeddings 및 ChatNVIDIA 모델 초기화
#  - 문서 청크를 문자열로 변환하는 유틸리티 함수 정의
#  - ChatPromptTemplate을 통한 질의응답 프롬프트 구성
#  - Retrieval Chain과 Generator Chain 연결
#  - RAG Chain 실행 및 스트리밍 출력
# 완성본(Retriever + LLM 연결) → nvidea_core_llm_rag_chain_complete.py 확인할 것.
# ============================================================
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain.document_transformers import LongContextReorder

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

#####################################################################
# Embeddings 및 LLM 초기화
#####################################################################

embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
llm = instruct_llm | StrOutputParser()

#####################################################################
# 저장된 docstore 불러오기 (사전에 docstore_index.tgz 압축 해제 필요)
#####################################################################

docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)
retriever = docstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 문서 검색

#####################################################################
# 문서 청크를 문자열로 변환하는 유틸리티 함수
#####################################################################

def docs2str(docs, title="Document"):
    """문서 청크를 문자열로 변환하여 context로 활용"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name: out_str += f"[출처: {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

#####################################################################
# ChatPromptTemplate 정의
#####################################################################

chat_prompt = ChatPromptTemplate.from_template(
    " 당신은 문서 기반 챗봇입니다. 사용자가 문서에 대해 질문하면 도와주세요."
    " 사용자 질문: {input}\n\n"
    " 다음 정보가 응답에 유용할 수 있습니다: "
    " 문서 검색 결과:\n{context}\n\n"
    " (검색 결과만 활용하여 답변하세요. 실제 사용한 출처만 인용하세요. 답변은 대화체로 작성)"
    "\n\n사용자 질문: {input}"
)

#####################################################################
# 출력 처리 함수
#####################################################################

def output_puller(inputs):
    """체인 결과에서 'output' 키를 추출하여 출력"""
    if isinstance(inputs, dict):
        inputs = [inputs]
    for token in inputs:
        if token.get('output'):
            yield token.get('output')

#####################################################################
# RAG 체인 구성
#####################################################################

# Chain 1: Retrieval Chain
long_reorder = RunnableLambda(LongContextReorder().transform_documents)
context_getter = RunnableLambda(lambda x: docs2str(retriever.get_relevant_documents(x["input"])))
retrieval_chain = {"input": (lambda x: x)} | RunnableAssign({"context": context_getter})

# Chain 2: Generator Chain
generator_chain = chat_prompt | llm
generator_chain = {"output": generator_chain} | RunnableLambda(output_puller)

# 최종 RAG 체인
rag_chain = retrieval_chain | generator_chain

#####################################################################
# 실행 예시
#####################################################################

for token in rag_chain.stream({"input": "흥미로운 사실을 알려줘!"}):
    print(token, end="")
