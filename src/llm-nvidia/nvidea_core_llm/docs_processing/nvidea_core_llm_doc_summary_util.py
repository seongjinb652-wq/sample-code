# ==================================================
# File: nvidea_core_llm_doc_summary_util.py
# Author: 성진
# Date: 2026-01-24
# Description: NVIDIA 기반 LangChain 문서 요약 유틸리티.
#              Arxiv 논문 로더를 통해 문서를 불러오고,
#              텍스트 분할 및 Pydantic 기반 요약 체인을 구성하여
#              기술 사용자에게 유용한 문서 요약을 생성한다.
# Usage:★★ - 단독 실행 가능
# ==================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field
from typing import List
from IPython.display import clear_output

# ---- 요약 데이터 구조 정의 ----
class DocumentSummaryBase(BaseModel):
    running_summary: str = Field("", description="문서의 진행 요약. 덮어쓰지 말고 업데이트만 수행.")
    main_ideas: List[str] = Field([], description="문서의 핵심 정보 (최대 3개)")
    loose_ends: List[str] = Field([], description="아직 알려지지 않은 열린 질문 (최대 3개)")

# ---- 요약 프롬프트 (한글 버전) ----
summary_prompt = ChatPromptTemplate.from_template(
    "당신은 문서의 진행 요약을 생성하는 역할을 맡고 있습니다. "
    "기술 사용자가 읽을 수 있도록 작성하세요. "
    "이후 기존 지식 베이스는 새로운 것으로 교체됩니다. "
    "독자가 모든 내용을 이해할 수 있도록 하세요. "
    "짧지만 밀도 있고 유용하게 작성해야 합니다. "
    "정보는 문서 조각에서 (열린 질문 또는 핵심 아이디어)로, 그리고 running_summary로 흐르도록 하세요. "
    "업데이트된 지식 베이스는 여기의 running_summary 정보를 모두 유지해야 합니다: {info_base}. "
    "\n\n{format_instructions}. 반드시 형식을 정확히 따르세요. 따옴표와 쉼표 포함. "
    "\n\n정보를 잃지 않고 다음 내용을 반영하여 지식 베이스를 업데이트하세요: {input}"
)

# ---- 추출 모듈 정의 ----
def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction 모듈
    Pydantic 기반 슬롯 채우기 방식으로 지식 딕셔너리 반환
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        return string
    return instruct_merge | prompt | llm | preparse | parser

latest_summary = ""

# ---- 요약 체인 정의 (TODO 채움) ----
def RSummarizer(knowledge, llm, prompt, verbose=False):
    '''
    문서 요약 체인 생성
    '''
    def summarize_docs(docs):        
        # RExtract를 활용하여 파싱 체인 구성
        parse_chain = RExtract(knowledge.__class__, llm, prompt)
        
        # 초기 상태 설정
        state = {"info_base": knowledge.json()}
        
        global latest_summary
        
        for i, doc in enumerate(docs):
            # 문서 조각을 체인에 전달하여 상태 업데이트
            state = parse_chain.invoke({"info_base": state["info_base"], "input": doc.page_content})
            
            if verbose:
                print(f"Considered {i+1} documents")
                pprint(state["running_summary"])
                latest_summary = state["running_summary"]
                clear_output(wait=True)

        return state["running_summary"]
        
    return RunnableLambda(summarize_docs)

# ---- NVIDIA LLM 모델 초기화 ----
instruct_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1").bind(max_tokens=4096)
instruct_llm = instruct_model | StrOutputParser()

# ---- 문서 요약 실행 ----
summarizer = RSummarizer(DocumentSummaryBase(), instruct_llm, summary_prompt, verbose=True)
summary = summarizer.invoke(docs_split[:15])
pprint(latest_summary)
