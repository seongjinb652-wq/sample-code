# ============================================================
# NVIDIA Core LLM - Save and Load Vector Store Index
# ------------------------------------------------------------
# 이 스크립트는 RAG 체인에서 생성된 FAISS 벡터스토어를
# 로컬에 저장하고, 압축한 뒤 다시 불러오는 예제입니다.
#
# 주요 기능:
#  - docstore 저장 (save_local)
#  - tgz 압축 및 삭제
#  - tgz 파일에서 다시 로드 (load_local)
#  - 인덱스 검색 테스트
# ============================================================
# 인덱스 저장 및 압축
docstore.save_local("docstore_index")   # 벡터스토어를 로컬 디렉토리에 저장
# docstore.save_local("문서저장소")
# 운영체제/파일시스템 호환성:
# Windows, macOS, Linux 모두 한글 디렉토리명을 지원하지만, 일부 환경
# (특히 Docker, WSL, 또는 외부 서버)에서는 인코딩 문제로 깨질 수 있습니다.
#####################################################################
# 재사용 시 일관성:
# 나중에 불러올 때도 동일한 이름을 써야 합니다.
#####################################################################
!tar czvf docstore_index.tgz docstore_index   # tgz 파일로 압축
!rm -rf docstore_index   # 원본 디렉토리 삭제 (압축본만 남김)

# 인덱스 불러오기
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS

# NVIDIA 임베딩 모델 초기화
# embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# tgz 파일 압축 해제
!tar xzvf docstore_index.tgz

# 저장된 인덱스 로드
new_db = FAISS.load_local("docstore_index", embedder, allow_dangerous_deserialization=True)

# 인덱스 검색 테스트
docs = new_db.similarity_search("인덱스 테스트")
print(docs[0].page_content[:1000])   # 검색된 문서의 앞 1000자 출력
