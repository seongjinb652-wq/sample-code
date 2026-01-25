# ============================================================
# NVIDIA Core LLM - Constructing Aggregate Vector Stores
# ------------------------------------------------------------
# 이 스크립트는 여러 개의 FAISS Vector Store를 생성하고,
# 이를 하나의 통합 docstore로 병합하는 예제입니다.
#
# 주요 기능:
#  - FAISS Vector Store 초기화 및 생성
#  - 빈 FAISS 인덱스 생성 유틸리티 함수 정의
#  - 여러 Vector Store 병합 (aggregate_vstores)
#  - 최종 docstore 청크 개수 출력
# ============================================================
%%time
print("Vector Store 생성 중...")

# extra_chunks와 docs_chunks를 기반으로 Vector Store 생성
vecstores = [FAISS.from_texts(extra_chunks, embedder)]
vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]

from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

# 임베딩 차원 계산
embed_dims = len(embedder.embed_query("test"))

def default_FAISS():
    '''빈 FAISS Vector Store를 생성하는 유틸리티 함수'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores):
    """여러 개의 Vector Store를 하나로 병합"""
    agg_vstore = default_FAISS()
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

# 최종 docstore 병합
docstore = aggregate_vstores(vecstores)

print(f"통합 docstore가 {len(docstore.docstore._dict)} 개의 청크를 포함합니다.")
