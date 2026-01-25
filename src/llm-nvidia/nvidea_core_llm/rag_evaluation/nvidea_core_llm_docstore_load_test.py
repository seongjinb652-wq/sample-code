# ============================================================
# NVIDIA Core LLM - Docstore 불러오기 및 확인
# ------------------------------------------------------------
# 이 스크립트는 이전에 저장한 FAISS 기반 docstore_index를
# 로컬에서 불러와 정상적으로 로드되는지 확인하는 예제입니다.
#
# 주요 기능:
#  - docstore_index.tgz 압축 해제
#  - FAISS VectorStore 로드
#  - 저장된 문서 청크 개수 확인
#  - 샘플 문서 청크 출력
# ============================================================
## 작업 디렉토리에 docstore_index.tgz 파일이 있는지 확인하세요
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

# Embeddings 모델 초기화 (필요 시 다른 모델로 교체 가능)
# embedder = NVIDIAEmbeddings(model="nvidia/embed-qa-4", truncate="END")

# tgz 파일 압축 해제
!tar xzvf docstore_index.tgz

# 저장된 docstore 불러오기
docstore = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)

# docstore 내부 문서 목록 가져오기
docs = list(docstore.docstore._dict.values())

# 문서 청크 포맷 함수 정의
def format_chunk(doc):
    return (
        f"논문 제목: {doc.metadata.get('Title', 'unknown')}"
        f"\n\n요약: {doc.metadata.get('Summary', 'unknown')}"
        f"\n\n본문 내용: {doc.page_content}"
    )

## 저장소가 정상적으로 불러와졌는지 확인 출력
pprint(f"불러온 docstore에는 총 {len(docstore.docstore._dict)} 개의 청크가 있습니다.")
pprint(f"샘플 청크 출력:")
print(format_chunk(docs[len(docs)//2]))
