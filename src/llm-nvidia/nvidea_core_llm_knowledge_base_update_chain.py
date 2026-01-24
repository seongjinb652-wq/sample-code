nvidea_core_llm_chat_request.py
# ============================================================
# File: nvidea_core_llm_knowledge_base_update_chain.py
# Author: 성진
# Date: 2026-01-24
# Description:
#   LangChain + NVIDIA ChatNVIDIA를 활용하여 대화 중 KnowledgeBase를
#   지속적으로 업데이트하는 예제 코드.
#   사용자의 입력 메시지를 기반으로 KnowledgeBase를 갱신하고,
#   응답(response)을 생성하여 대화 흐름을 이어간다.
#
# Usage:
#   - 단독 실행 가능 (KnowledgeBase 업데이트 테스트)
#   - 다른 모듈과 결합하여 대화형 챗봇 시스템에 활용 가능
# ============================================================
## Cell 1: 라이브러리 및 데이터 구조 정의
from pydantic import BaseModel, Field

class KnowledgeBase(BaseModel):
    firstname: str = Field('unknown', description="대화 사용자의 이름 (모르면 unknown)")
    lastname: str = Field('unknown', description="대화 사용자의 성 (모르면 unknown)")
    location: str = Field('unknown', description="사용자가 위치한 장소")
    summary: str = Field('unknown', description="대화 요약. 새로운 입력으로 갱신됨")
    response: str = Field('unknown', description="사용자의 새 메시지에 기반한 이상적인 응답")

## Cell 2: 프롬프트 템플릿 정의
from langchain.prompts import ChatPromptTemplate

parser_prompt = ChatPromptTemplate.from_template(
    "당신은 사용자와 대화 중입니다. 사용자가 방금 응답했습니다 ('input'). "
    "KnowledgeBase를 업데이트하세요. "
    "대화를 이어가기 위해 'response' 태그에 응답을 기록하세요. "
    "세부사항을 지어내지 말고, KnowledgeBase가 중복되지 않도록 하세요. "
    "대화 흐름에 맞게 항목을 자주 갱신하세요."
    "\n{format_instructions}"
    "\n\n이전 KnowledgeBase: {know_base}"
    "\n\n새 메시지: {input}"
    "\n\n새 KnowledgeBase:"
)

## Cell 3: LLM 모델 정의
from langchain.output_parsers import StrOutputParser

## 더 강력한 모델로 전환
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

## Cell 4: RExtract 모듈 연결
extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
info_update = RunnableAssign({'know_base': extractor})

## Cell 5: KnowledgeBase 초기화 및 갱신 테스트
from pprint import pprint

# 초기 상태 정의
state = {'know_base': KnowledgeBase()}

# 첫 번째 입력
state['input'] = "내 이름은 Carmen Sandiego야! 내가 어디 있는지 맞춰봐! 힌트: 미국 어딘가야."
state = info_update.invoke(state)
pprint(state)

# 두 번째 입력
state['input'] = "나는 재즈의 발상지로 여겨지는 곳에 있어."
state = info_update.invoke(state)
pprint(state)

# 세 번째 입력
state['input'] = "맞아, 난 뉴올리언스에 있어... 어떻게 알았어?"
state = info_update.invoke(state)
pprint(state)
