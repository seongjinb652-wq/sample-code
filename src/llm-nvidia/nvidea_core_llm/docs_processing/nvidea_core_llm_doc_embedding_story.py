# ==================================================
# File: nvidea_core_llm_doc_embedding_story.py
# Author: 성진
# Date: 2026-01-24
# Description: NVIDIA 기반 LangChain 임베딩 및 스토리 확장 유틸리티.
#              질의와 문서를 임베딩하여 유사도 행렬을 시각화하고,
#              LLM을 활용해 질의 기반 확장 문서를 생성한다.
# Usage:★★ - 단독 실행 가능
# ==================================================

## ---- Colab 환경 패키지 설치 (코스 환경에서는 불필요) ----
# %pip install -qq langchain langchain-nvidia-ai-endpoints gradio
# %pip install -qq arxiv pymupdf

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

[m for m in NVIDIAEmbeddings.get_available_models() if "embed" in m.id]

# NVIDIAEmbeddings.get_available_models()
# embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2")
# embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
# embedder = NVIDIAEmbeddings(model="nvidia/embed-qa-4")
# embedder = NVIDIAEmbeddings(model="snowflake/arctic-embed-l")
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

# ChatNVIDIA.get_available_models()
# instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
# instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
# instruct_llm = ChatNVIDIA(model="meta/llama-3-70b-instruct")
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

## ---- 예시 질의 및 문서 (한글 변환) ----
queries = [
    "록키산맥의 날씨는 어떤가요?",
    "이탈리아는 어떤 음식으로 유명한가요?",
    "내 이름이 뭐였지? 기억 못하겠죠...",
    "삶의 의미는 무엇일까요?",
    "삶의 의미는 즐거움을 찾는 거예요 :D"
]

documents = [
    "캄차카의 날씨는 춥고, 긴 혹독한 겨울이 특징입니다.",
    "이탈리아는 파스타, 피자, 젤라토, 에스프레소로 유명합니다.",
    "개인 이름은 기억할 수 없고, 정보만 제공합니다.",
    "삶의 목적은 다양하며, 개인적 성취로 여겨지기도 합니다.",
    "삶의 순간을 즐기는 것은 정말 멋진 접근입니다."
]

## ---- 질의 및 문서 임베딩 ----
%%time
q_embeddings = [embedder.embed_query(query) for query in queries]
d_embeddings = embedder.embed_documents(documents)

## ---- 유사도 행렬 시각화 ----
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def plot_cross_similarity_matrix(emb1, emb2):
    cross_similarity_matrix = cosine_similarity(np.array(emb1), np.array(emb2))
    plt.imshow(cross_similarity_matrix, cmap='Greens', interpolation='nearest')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title("Cross-Similarity Matrix")
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_cross_similarity_matrix(q_embeddings, d_embeddings)
plt.xlabel("Query Embeddings")
plt.ylabel("Document Embeddings")
plt.show()

plt.figure(figsize=(8, 6))
plot_cross_similarity_matrix(
    q_embeddings,
    [embedder.embed_query(doc) for doc in documents]
)
plt.xlabel("Query Embeddings (of queries)")
plt.ylabel("Query Embeddings (of documents)")
plt.show()

## ---- LLM 기반 스토리 확장 ----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

expound_prompt = ChatPromptTemplate.from_template(
    "다음 질문들을 모두 포함할 수 있는 긴 이야기의 일부를 생성하세요: {questions}\n"
    " 단, 아래 질문에 대해서만 구체적으로 답하세요: {q1}."
    " 특이한 포맷을 주고, 다른 질문들은 직접적으로 답하지 마세요."
    " '여기에 답변이 있습니다' 같은 코멘트는 포함하지 마세요."
)

###############################################################################################
## ---- TODO 채운 부분 ----
expound_chain = expound_prompt | instruct_llm | StrOutputParser()

longer_docs = []
for i, q in enumerate(queries):
    longer_doc = expound_chain.invoke({"questions": queries, "q1": q})
    pprint(f"\n\n[Query {i+1}]")
    print(q)
    pprint(f"\n\n[Document {i+1}]")
    print(longer_doc)
    pprint("-"*64)
    longer_docs += [longer_doc]
###############################################################################################

## ---- 임베딩 모델 토큰 제한 고려 ----
longer_docs_cut = [doc[:2048] for doc in longer_docs]

q_long_embs = [embedder._embed([doc], model_type='query')[0] for doc in longer_docs_cut]
d_long_embs = [embedder._embed([doc], model_type='passage')[0] for doc in longer_docs_cut]

## ---- 유사도 비교 시각화 ----
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_cross_similarity_matrix(q_embeddings, q_long_embs)
plt.xlabel("Query Embeddings (of queries)")
plt.ylabel("Query Embeddings (of long documents)")

plt.subplot(1, 2, 2)
plot_cross_similarity_matrix(q_embeddings, d_long_embs)
plt.xlabel("Query Embeddings (of queries)")
plt.ylabel("Document Embeddings (of long documents)")
plt.show()
