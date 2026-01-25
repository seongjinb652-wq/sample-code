# ============================================================
# File: knowledge_base_flight_simple_example.py
# Author: 성진
# Date: 2026-01-24
# Description:
#   SkyFlow Airlines 챗봇 단순화 버전.
#   LLM 연결 없이 항공편 정보를 조회하고 KnowledgeBase를 사용하는 예제.
# ============================================================

from typing import Optional
from pprint import pprint
from pydantic import BaseModel, Field

# ✅ 항공편 정보 조회 함수
def get_flight_info(d: dict) -> str:
    req_keys = ['first_name', 'last_name', 'confirmation']
    assert all((key in d) for key in req_keys)

    keys = req_keys + ["departure", "destination", "departure_time", "arrival_time", "flight_day"]
    values = [
        ["이순신", "장군", 12345, "인천공항", "김해공항", "12:30 PM", "2:30 PM", "내일"],
        ["강감찬", "장군", 54321, "김포공항", "제주공항", "8:00 AM", "9:30 AM", "일요일"],
        ["홍길동", "의적", 98765, "김해공항", "인천공항", "7:00 PM", "9:00 PM", "다음주"],
        ["유관순", "열사", 56789, "제주공항", "김포공항", "1:00 PM", "3:00 PM", "어제"],
    ]
    get_key = lambda d: "|".join([d['first_name'], d['last_name'], str(d['confirmation'])])
    get_val = lambda l: {k:v for k,v in zip(keys, l)}
    db = {get_key(get_val(entry)): get_val(entry) for entry in values}

    data = db.get(get_key(d))
    if not data:
        return "해당 항공편 정보를 찾을 수 없습니다."
    return (
        f"{data['first_name']} {data['last_name']}의 항공편은 "
        f"{data['departure']}에서 {data['destination']}로 출발하며, "
        f"{data['flight_day']} {data['departure_time']}에 출발하여 "
        f"{data['arrival_time']}에 도착합니다."
    )

# ✅ KnowledgeBase 정의
class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="사용자의 이름")
    last_name: str = Field('unknown', description="사용자의 성")
    confirmation: Optional[int] = Field(None, description="항공편 확인 번호")
    discussion_summary: str = Field("", description="대화 요약")
    open_problems: str = Field("", description="아직 해결되지 않은 문제")
    current_goals: str = Field("", description="현재 목표")

# ✅ 실행 예시
if __name__ == "__main__":
    know_base = KnowledgeBase(first_name="이순신", last_name="장군", confirmation=12345)
    result = get_flight_info(know_base.dict())
    pprint(result)

    know_base2 = KnowledgeBase(first_name="홍길동", last_name="의적", confirmation=98765)
    result2 = get_flight_info(know_base2.dict())
    pprint(result2)

    know_base3 = KnowledgeBase(first_name="유관순", last_name="열사", confirmation=27494)
    result3 = get_flight_info(know_base3.dict())
    pprint(result3)
