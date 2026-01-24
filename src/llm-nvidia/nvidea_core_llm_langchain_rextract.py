# ============================================================
# File: nvidea_core_llm_langchain_rextract.py
# Author: 성진
# Date: 2026-01-24
# Description:
#   LangChain NVIDIA ChatNVIDIA와 Pydantic을 활용한 KnowledgeBase 추출 예제.
#   RExtract 모듈을 통해 입력 메시지에서 상태 정보를 추출하여
#   구조화된 KnowledgeBase 객체로 변환하는 방법을 보여준다.
#
# Usage:★★
#   - 단독 실행 가능 (KnowledgeBase 추출 테스트)
#   - 조합 가능 (nvidea_core_llm_langchain_client.py, nvidea_core_llm_chat_stream.py 와 함께 사용 권장)
# ============================================================
## Cell 1: 라이브러리 임포트
from pydantic import BaseModel, Field
from typing import Any, Dict, Union, Optional

## Cell 2: NVIDIA ChatNVIDIA 모델 정의
instruct_chat = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3")
## 선택적 강화: 일부 모델은 서버 측에서 제약된 디코딩을 통해 출력을 제한할 수 있음
## (outlines, xgrammar 참고)
instruct_chat = instruct_chat.with_structured_output(KnowledgeBase)

## Cell 3: KnowledgeBase 데이터 구조 정의
class KnowledgeBase(BaseModel):
    ## BaseModel의 필드 정의 (KnowledgeBase가 생성될 때 검증/할당됨)
    topic: str = Field('general', description="현재 대화 주제")
    user_preferences: Dict[str, Union[str, Any]] = Field({}, description="사용자 선호 및 선택")
    session_notes: list = Field([], description="진행 중인 세션에 대한 메모")
    unresolved_queries: list = Field([], description="아직 해결되지 않은 사용자 질문")
    action_items: list = Field([], description="대화 중 식별된 실행 항목")

print(repr(KnowledgeBase(topic="Travel")))

## Cell 4: PydanticOutputParser 사용 예시
from langchain.output_parsers import PydanticOutputParser

instruct_string = PydanticOutputParser(pydantic_object=KnowledgeBase).get_format_instructions()
pprint(instruct_string)

################################################################################
## Cell 5: RExtract 모듈 정의
def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction 모듈
    입력 메시지에서 정보를 추출하여 KnowledgeBase 딕셔너리로 반환
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions': lambda x: parser.get_format_instructions()})
    
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        # print(string)  ## 진단용 출력
        return string
    
    return instruct_merge | prompt | llm | preparse | parser

################################################################################
## Cell 6: RExtract 실제 사용 예시
parser_prompt = ChatPromptTemplate.from_template(
    "지식 베이스를 업데이트하세요: {format_instructions}. 입력에서만 정보를 사용하세요."
    "\n\n새 메시지: {input}"
)

extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)

knowledge = extractor.invoke({'input': "나는 꽃을 정말 좋아해! 난초가 너무 멋져! 나에게 좀 사줄래?"})
pprint(knowledge)
