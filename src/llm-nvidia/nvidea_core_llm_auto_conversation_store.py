# ============================================================
# NVIDIA Core LLM - Automatic Conversation Storage
# ------------------------------------------------------------
# 이 스크립트는 LangChain을 활용하여 대화 내용을 자동으로
# Vector Store(FAISS)에 저장하는 예제입니다.
#
# 주요 기능:
#  - 사용자 입력과 LLM 출력 자동 저장
#  - 저장된 대화를 Retriever로 불러와 맥락 유지
#  - 자연스러운 대화 흐름을 위한 Prompt 체인 구성
# ============================================================
# 🔑 간단 설명 (차이점)
# 이전 체인: 단순히 Retriever → Prompt → LLM → Parser 흐름으로 답변만 생성. 대화 내용은 따로 저장되지 않음.
# 지금 체인: save_memory_and_get_output 함수를 통해 사용자 입력과 LLM 출력을 convstore에 자동으로 추가 저장.
# 즉, "User said ..." / "Agent said ..." 형태로 벡터스토어에 기록됨.
# 이후 질의 시, 이 저장된 대화 맥락이 검색되어 더 자연스럽고 일관된 대화가 가능해짐.
# 👉 요약하면: 아까는 단순 질의응답 체인, 지금은 대화가 자동으로 메모리에 축적되는 체인입니다.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

########################################################################
## 대화 저장소 초기화 및 메시지 추가 정의
convstore = FAISS.from_texts(conversation, embedding=embedder)

def save_memory_and_get_output(d, vstore):
    """'input'/'output' 딕셔너리를 받아 convstore에 저장"""
    vstore.add_texts([f"사용자: {d.get('input')}", f"에이전트: {d.get('output')}"])
    return d.get('output')

########################################################################

# instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

chat_prompt = ChatPromptTemplate.from_template(
    "질문에 답할 때는 반드시 주어진 문맥만 사용하세요."
    "\n\n검색된 문맥: {context}"
    "\n\n사용자 질문: {input}"
    "\n대화체로 답변하세요. 대화 흐름이 자연스럽게 이어지도록 하세요.\n"
    "[Agent]"
)

conv_chain = (
    {
        'context': convstore.as_retriever() | long_reorder | docs2str,
        'input': (lambda x:x)
    }
    | RunnableAssign({'output' : chat_prompt | instruct_llm | StrOutputParser()})
    | partial(save_memory_and_get_output, vstore=convstore)
)

# 대화 실행 예시
pprint(conv_chain.invoke("당신이 동의해줘서 기뻐요! 거기서 아이스크림을 먹을 날이 기다려져요! 정말 맛있는 음식이죠!"))
print()
pprint(conv_chain.invoke("제 가장 좋아하는 음식이 무엇인지 맞출 수 있나요?"))
print()
pprint(conv_chain.invoke("사실 제가 가장 좋아하는 건 꿀이에요! 왜 그렇게 생각했는지 모르겠네요."))
print()
pprint(conv_chain.invoke("알겠어요! 괜찮아요! 이제 제 가장 좋아하는 음식이 뭔지 아시겠죠?"))
