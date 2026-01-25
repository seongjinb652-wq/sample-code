# ============================================================
# NVIDIA Core LLM - Loading and Chunking Latest Documents
# ------------------------------------------------------------
# 이 스크립트는 Arxiv 논문을 불러와서 문서 내용을 분할(chunking)하고
# Vector Store에 활용할 수 있도록 준비하는 예제입니다.
#
# 주요 기능:
#  - Arxiv 논문 로딩 (ArxivLoader, ArxivRetriever)
#  - References 이후 내용 제거
#  - RecursiveCharacterTextSplitter로 문서 분할
#  - 최근 논문(< 1개월 이내) 추가 로딩
#  - 2024–2025 기준 최신, 한국 연구자/한글 포함
# ============================================================

from langchain.document_loaders import ArxivLoader
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ArxivRetriever

# 문서 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " "],
)    

print("문서 불러오는 중...")

# 기존 논문 목록
docs = [doc[0] for doc in ArxivRetriever().batch([
    "1706.03762",     ## Attention Is All You Need
    "1810.04805",     ## BERT
    "2005.11401",     ## RAG
    "2310.06825",     ## Mistral
    "2306.05685",     ## LLM-as-a-Judge
])] 

# ✅ 최신 논문 추가 (예: Retrieval-Augmented Generation 관련 최근 논문 2편)
docs += ArxivLoader(query="Retrieval-Augmented Generation", load_max_docs=2).load()
# ✅ 최신 논문 추가 (예: Large Language Models 관련 최근 논문 2편)
docs += ArxivLoader(query="Large Language Models", load_max_docs=2).load()
# ✅ 한글 최신 논문 추가
# Thunder-LLM: Efficiently Adapting LLMs to Korean with Minimal Resources	
## Jinpyo Kim 외	한국어 LLM 적응	2025-06-18
# Prioritizing Informative Features and Examples for Deep Learning from Noisy Data	
## 박동민 (KAIST)	딥러닝/데이터사이언스	2024-12-11
# Hanprome: Modified Hangeul for Expression of Foreign Language Pronunciation	
## 김원찬 외	한글 음성/언어학	2024-12-20
docs += ArxivRetriever().batch([
    "2506.21595",   ## Thunder-LLM (한국어 LLM 적응)
    "2205.00445",   ## Prioritizing Informative Features (KAIST 박사학위 논문)
])
docs += ArxivLoader(query="Hanprome Hangeul", load_max_docs=1).load()

# Open Ko-LLM Leaderboard: Evaluating Large Language Models in Korean with Ko-H5 Benchmark	
## 2405.20574	2024-05-31	Ko-H5 벤치마크를 활용한 한국어 LLM 평가 프레임워크 구축
# Open Ko-LLM Leaderboard2: Bridging Foundational and Practical Evaluation for Korean LLMs	
## 2410.12445	2025-03-04	기존 리더보드를 개선하여 실제 활용에 가까운 한국어 LLM 평가
# ₩on: Establishing Best Practices for Korean Financial NLP	
## 2503.17963	2025-03-23	금융 분야 한국어 LLM 평가 및 오픈 데이터셋 구축
# HRET: A Self-Evolving LLM Evaluation Toolkit for Korean	
## 2503.xxxx (정식 ID: 2503.???)	
##  with Korean Educational Standards (KoNET Benchmark)	2502.15422	2025-02-21	한국 교육 시험을 활용한 멀티모달 생성형 AI 성능 평가
docs += ArxivRetriever().batch([
    "2506.21595",   ## Thunder-LLM (한국어 LLM 적응)
    "2205.00445",   ## Prioritizing Informative Features (KAIST 박사학위 논문)
])
docs += ArxivLoader(query="Hanprome Hangeul", load_max_docs=1).load()

# References 이후 내용 제거
for doc in docs:
    content = json.dumps(doc.page_content)
    if "References" in content:
        doc[0].page_content = content[:content.index("References")]

# 문서 분할 및 짧은 청크 제거
print("문서 청크 분할 중...")
docs_chunks = [text_splitter.split_documents([doc]) for doc in docs]
docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]

# 문서 메타데이터 요약
doc_string = "사용 가능한 문서 목록:"
doc_metadata = []
for chunks in docs_chunks:
    metadata = getattr(chunks[0], 'metadata', {})
    doc_string += "\n - " + metadata.get('Title')
    doc_metadata += [str(metadata)]

extra_chunks = [doc_string] + doc_metadata

# 요약 정보 출력
print(doc_string, '\n')
for i, chunks in enumerate(docs_chunks):
    print(f"문서 {i}")
    print(f" - 청크 개수: {len(chunks)}")
    print(f" - 메타데이터: ")
    print(chunks[0].metadata)
    print()
