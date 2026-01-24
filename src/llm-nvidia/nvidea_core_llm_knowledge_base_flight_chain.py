# ============================================================
# File: nvidea_core_llm_knowledge_base_flight_chain.py
# Author: 성진
# Date: 2026-01-24
# Description:
#   LangChain + NVIDIA ChatNVIDIA를 활용하여 항공편 정보를 조회하고
#   KnowledgeBase를 업데이트하는 예제 코드.
#   get_flight_info 함수는 간단한 DB 조회처럼 동작하며,
#   대화형 챗봇이 사용자 입력을 기반으로 항공편 정보를 제공한다.
#
# Usage:
#   - 단독 실행 가능 (항공편 정보 조회 테스트)
#   - 다른 모듈과 결합하여 SkyFlow 챗봇 시스템에 활용 가능
# ============================================================
#######################################################################################
## Cell 1: 항공편 정보 조회 함수 정의
def get_flight_info(d: dict) -> str:
    """
    항공편 정보를 조회하는 예제 함수.
    딕셔너리를 키로 받아 SQL DB 조회와 유사하게 동작한다.
    """
    req_keys = ['first_name', 'last_name', 'confirmation']
    assert all((key in d) for key in req_keys), f"Expected dictionary with keys {req_keys}, got {d}"

    ## 정적 데이터셋. get_key와 get_val을 사용하여 DB처럼 구성
    keys = req_keys + ["departure", "destination", "departure_time", "arrival_time", "flight_day"]
    values = [
        ["Jane", "Doe", 12345, "San Jose", "New Orleans", "12:30 PM", "9:30 PM", "tomorrow"],
        ["John", "Smith", 54321, "New York", "Los Angeles", "8:00 AM", "11:00 AM", "Sunday"],
        ["Alice", "Johnson", 98765, "Chicago", "Miami", "7:00 PM", "11:00 PM", "next week"],
        ["Bob", "Brown", 56789, "Dallas", "Seattle", "1:00 PM", "4:00 PM", "yesterday"],
    ]
    get_key = lambda d: "|".join([d['first_name'], d['last_name'], str(d['confirmation'])])
    get_val = lambda l: {k:v for k,v in zip(keys, l)}
    db = {get_key(get_val(entry)) : get_val(entry) for entry in values}

    # 매칭되는 항목 검색
    data = db.get(get_key(d))
    if not data:
        return (
            f"{req_keys} = {get_key(d)} 기준으로 항공편 정보를 찾을 수 없습니다."
            " 새로운 정보가 학습될 때마다 이 과정이 반복됩니다."
            " 중요한 정보라면 사용자에게 확인을 요청하세요."
        )
    return (
        f"{data['first_name']} {data['last_name']}의 항공편은 {data['departure']}에서 {data['destination']}로 출발하며, "
        f"{data['flight_day']} {data['departure_time']}에 출발하여 {data['arrival_time']}에 도착합니다."
    )

#######################################################################################
## Cell 2: 사용 예시
print(get_flight_info({"first_name": "Jane", "last_name": "Doe", "confirmation": 12345}))
print(get_flight_info({"first_name": "Alice", "last_name": "Johnson", "confirmation": 98765}))
print(get_flight_info({"first_name": "Bob", "last_name": "Brown", "confirmation": 27494}))

#######################################################################################
## Cell 3: SkyFlow 챗봇 프롬프트 정의
external_prompt = ChatPromptTemplate.from_template(
    "당신은 SkyFlow 챗봇이며, 고객의 문제를 도와주고 있습니다."
    " 고객의 질문에 답변할 때 SkyFlow 항공사를 대표한다는 점을 기억하세요."
    " SkyFlow는 업계 평균 관행을 따른다고 가정하세요."
    " (이것은 영업 비밀이므로 공개하지 마세요)."  ## soft reinforcement
    " 가능하다면 대화를 짧고 간결하게 유지하세요. 필요하지 않으면 인사말은 피하세요."
    " 다음은 질문에 답변하는 데 유용할 수 있는 컨텍스트입니다."
    "\n\n컨텍스트: {context}"
    "\n\n사용자: {input}"
)

basic_chain = external_prompt | instruct_llm

basic_chain.invoke({
    'input': '공항에 언제 도착해야 하나요?',
    'context': get_flight_info({"first_name": "Jane", "last_name": "Doe", "confirmation": 12345}),
})

#######################################################################################
## Cell 4: KnowledgeBase 정의
from pydantic import BaseModel, Field
from typing import Dict, Union

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="대화 사용자의 이름 (모르면 unknown)")
    last_name: str = Field('unknown', description="대화 사용자의 성 (모르면 unknown)")
    confirmation: int = Field(-1, description="항공편 확인 번호, 모르면 -1")
    discussion_summary: str = Field("", description="지금까지의 대화 요약 (장소, 문제 등 포함)")
    open_problems: list = Field([], description="아직 해결되지 않은 주제들")
    current_goals: list = Field([], description="에이전트가 현재 해결해야 할 목표")

#######################################################################################
## Cell 5: KnowledgeBase → get_flight_info 키 변환 함수
def get_key_fn(base: BaseModel) -> dict:
    '''KnowledgeBase 객체를 받아 get_flight_info 조회용 키 딕셔너리 반환'''
    return {
        'first_name': base.first_name,
        'last_name': base.last_name,
        'confirmation': base.confirmation,
    }

know_base = KnowledgeBase(first_name="Jane", last_name="Doe", confirmation=12345)

# get_flight_info(get_key_fn(know_base))

get_key = RunnableLambda(get_key_fn)
(get_key | get_flight_info).invoke(know_base)
