# ============================================================
# NVIDIA Core LLM - Conversation Embedding with FAISS
# ------------------------------------------------------------
# 이 스크립트는 대화(conversation) 데이터를 NVIDIA Embeddings을
# 활용하여 벡터화하고, FAISS Vector Store에 저장하는 예제입니다.
#
# 주요 기능:
#  - 대화 텍스트 리스트를 Embedding 처리
#  - FAISS Vector Store 초기화 및 Retriever 생성
#  - 실행 시간 측정 (%%time)
# ============================================================
conversation = [ ## AI가 생성한 대화를 일부 수정하여 사용
    "[User] 안녕! 내 이름은 베라스야, 나는 큰 파란 곰이야! 록키산맥에 대해 알려줄래?",
    "[Agent] 록키산맥은 북미 대륙을 가로지르는 아름답고 웅장한 산맥이야.",
    "[Beras] 와, 정말 멋지다! 나는 아직 록키산맥에 가본 적은 없지만, 좋은 이야기를 많이 들었어.",
    "[Agent] 언젠가 꼭 방문해 보길 바라, 베라스! 너에게 멋진 모험이 될 거야.",
    "[Beras] 제안해 줘서 고마워! 앞으로 꼭 기억해 둘게.",
    "[Agent] 그동안은 인터넷에서 자료를 찾아보거나 다큐멘터리를 보면서 더 알아볼 수 있어.",
    "[Beras] 나는 북극에 살아서 따뜻한 기후에는 익숙하지 않아. 그냥 궁금했을 뿐이야!",
    "[Agent] 물론이지! 대화를 이어가면서 록키산맥과 그 의미에 대해 더 알아보자!"
]
%%time
## ^^ 이 셀은 대화 임베딩 처리 시간을 측정합니다

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS

## 대화 텍스트 리스트를 Embedding 처리 후 FAISS Vector Store 생성
convstore = FAISS.from_texts(conversation, embedding=embedder)

## Retriever 객체 생성 (검색용)
retriever = convstore.as_retriever()

# 대화형 질의 (한국어 버전)
pprint(retriever.invoke("당신의 이름은 무엇인가요?"))
pprint(retriever.invoke("록키산맥은 어디에 있나요?"))

