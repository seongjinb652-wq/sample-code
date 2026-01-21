from typing import TypedDict, Annotated
from langgraph.graph import StateGraph , START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# LLM 선택
USE_OLLAMA = False # True 시 ollama 사용
if USE_OLLAMA:
    from langchain_ollama import OllamaLLM
    llm = OllamaLLM(
                base_url='http://192.168.1.18:11434',
                model='llama3.1'  # 한국어 학습 잘된 모델
    )
else:
    import os
    from dotenv import load_dotenv

    load_dotenv()
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# 상태 클래스 정의
class DraftMessageState(TypedDict):
    # 대화 메시지를 누적(리듀서 활용)
    messages : Annotated[list, add_messages]

# 노드 수행 함수 정의
def Write_draft(state: DraftMessageState):
    """사용자 요구를 바탕으로 문서 '초안'을 작성하는 노드 """
    prompt = [
        SystemMessage(
            content=(
                "너는 전문 문서 작성 보조자야. 아래 사용자 요청을 바탕으로 깔끔한 문서 '초안'을 작성해."
                "불필요한 수사는 줄이고, 구조(제목, 소제목, 글머리표 등)를 명확히 해."
            )
        ),
        *state["messages"]  # 메시지 언패킹
    ]
    draft = llm.invoke(prompt)  # AI message
    return {"messages": [draft]}  # add_messages 로 메시지 누적

def Human_Feedback_dumy(state: DraftMessageState):
    """사람 검토 단계(여기에서 인터럽트 발생), 이 노드는 실제로 아무 일도 하지 않음."""
    # 중단점에서 사람이 피드백을 넣을 것이므로 여기서는 pass
    pass

def Final_draft_write(state: DraftMessageState):
    """사람 피드백(인터럽트 이후 HumanMessage)을 반영해 초안을 수정"""
    # 전체 히스토리를 기준으로 '가장 최근 HumanMessage'를 피드백으로 간주
    # (초안은 직전 AIMessage로 존재함)
    prompt = [
        SystemMessage(
            content=(
                "너는 전문 편집자야. 대화 맥락의 마지막 HumanMessage를 '피드백'으로 간주하고, "
                "직전에 생성된 문서 초안을 핵심 유지/불필요 제거/정확성 보강 원칙으로 수정해. "
                "변경사항은 자연스럽게 반영하고, 최종본만 출력해."
            )
        ),
        *state["messages"]
    ]
    final_draft = llm.invoke(prompt)
    return {"messages": [final_draft]}



# 그래프 구성
graph = StateGraph(DraftMessageState)

graph.add_node("Write_draft", Write_draft)
graph.add_node("Human_Feedback_dumy", Human_Feedback_dumy)  # 인터럽트 전용 더미 노드
graph.add_node("Final_draft_write", Final_draft_write)
# 엣지 연결 , 흐름 제어
graph.add_edge(START, "Write_draft")
graph.add_edge("Write_draft","Human_Feedback_dumy")
graph.add_edge("Human_Feedback_dumy", "Final_draft_write")
graph.add_edge("Final_draft_write",END)

#  체크포인터 & 컴파일(Human_Feedback_dumy 직전에 인터럽트)
memory = MemorySaver()
draft_app = graph.compile(interrupt_before=["Human_Feedback_dumy"], checkpointer=memory)

if __name__ == "__main__":
    # 사용자의 문서 작성 요청(예시)
    initial_request = HumanMessage(content=
                                   "주제: 딥러닝과 머신러닝의 차이\n"
                                   "- 대상: 파이썬 중급 개발자\n"
                                   "- 분량: 800~1200자\n"
                                   "- 포함: 개념 요약, 실무 팁 3가지\n"
                                   "- 톤: 간결하고 실용적으로"
                                   )

    state = DraftMessageState({'messages':[initial_request]})

    config = RunnableConfig(
        recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
        configurable={"thread_id": "doc_1"},  # 스레드 ID 설정
    )

    # 스트리밍 실행 : 인터럽트 지점 감지

    for event in draft_app.stream(state, config=config):
        if event == {'__interrupt__': ()}:
            print("\n===== [중단점 도달: Human_Feedback_dumy] =====")

            # 중단 상태 확인
            paused = draft_app.get_state(config)
            print("중단 상태의 messages :", paused.values["messages"])
            print("다음 실행 예정 노드:", paused.next)
            print("\n===== [문서 작성 초안] =====")
            paused.values["messages"][-1].pretty_print()

            # 사람 검토/피드백 입력
            input("계속하려면 Enter 키를 누르세요..")
            user_feedback = input("피드백을 입력하세요 (예: 2개 예제 코드, 소제목 추가, 실무 팁 설명 보강 등)\n")

            # Human_Feedback_dumy 노드로 상태 업데이트(메시지 추가)
            draft_app.update_state(
                config,
                {"messages":[HumanMessage(content=f"피드백: {user_feedback}")]},
                as_node="Human_Feedback_dumy"
            )


            # 재개 실행만 하고 출력은 하지 않음
            # 단, 재개 후 업데이트된 피드백 메시지를 활용해 최종 문서 작성 됨
            for _ in draft_app.stream(None, config, stream_mode="values"):
                pass

            # 최종 상태에서 마지막 메시지만 출력
            final_state = draft_app.get_state(config).values
            final_msg = final_state["messages"][-1]
            print("\n------ [최종 문서] ------")
            try:
                print(final_msg.content)
            except Exception:
                pass

            break # 종료

        else:
            pass



