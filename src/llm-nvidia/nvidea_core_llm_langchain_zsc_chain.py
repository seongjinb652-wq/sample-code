# ============================================================
# File: nvidea_core_llm_langchain_zsc_chain.py
# Author: 성진
# Date: 2026-01-24
# Description:
#   LangChain NVIDIA ChatNVIDIA를 활용한 Zero-shot 분류 및 생성 체인 예제.
#   단일 단어 분류(zsc_call), 주제 기반 문장 생성(gen_chain),
#   Branch 체인을 통한 입력/출력 결합(big_chain) 구조를 포함.
#
# Usage:★★
#   - 단독 실행 가능 (Zero-shot 분류 및 생성 테스트)
#   - 조합 가능 (nvidea_core_llm_langchain_client.py, nvidea_core_llm_langchain_rich_utils.py 와 함께 사용 권장)
# ============================================================
# %%time
## 이 노트북은 실행 시간을 측정하여 전체 소요 시간을 출력합니다.

from langchain_core.runnables import RunnableLambda
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Union
from operator import itemgetter

## Cell 1: Zero-shot 분류 프롬프트 및 체인 정의
sys_msg = (
    "문장을 문맥으로 하여 가장 가능성이 높은 주제 분류를 선택하세요."
    " 단 하나의 단어만 출력하고, 설명은 하지 마세요.\n[옵션 : {options}]"
)

zsc_prompt = ChatPromptTemplate.from_template(
    f"{sys_msg}\n\n"
    "[[The sea is awesome]][/INST]boat</s><s>[INST]"
    "[[{input}]]"
)

## Cell 2: 간단한 instruct 모델 정의
instruct_chat = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.3")
instruct_llm = instruct_chat | StrOutputParser()
one_word_llm = instruct_chat.bind(stop=[" ", "\n"]) | StrOutputParser()

zsc_chain = zsc_prompt | one_word_llm

## Cell 3: Zero-shot 호출 함수 정의
def zsc_call(input, options=["car", "boat", "airplane", "bike"]):
    return zsc_chain.invoke({"input" : input, "options" : options}).split()[0]

print("-" * 80)
print(zsc_call("다음 출구에서 나가야 할까, 아니면 다음까지 계속 가야 할까?"))
print("-" * 80)
print(zsc_call("나는 배멀미가 심해서 여행은 피해야 할 것 같아"))
print("-" * 80)
print(zsc_call("나는 고소공포증이 있어서 비행기는 무서워"))
print("-" * 80)
print(zsc_call("주말에 자전거를 타고 한강을 달리면 기분이 상쾌해."))
print("-" * 80)
print(zsc_call("자동차로 출근하는데 매일 교통 체증 때문에 힘들다."))
print("-" * 80)
print(zsc_call("여름 휴가에는 배를 타고 섬으로 여행 가고 싶어."))
print("-" * 80)
print(zsc_call("비행기를 타면 창밖으로 구름이 펼쳐져서 항상 감탄하게 돼."))
print("-" * 80)
print(zsc_call("자전거는 환경에도 좋고 건강에도 좋아서 자주 이용한다."))

# %%time
## 이 노트북은 실행 시간을 측정하여 전체 소요 시간을 출력합니다.

## Cell 4: 주제 기반 문장 생성 체인 정의
gen_prompt = ChatPromptTemplate.from_template(
    "다음 주제에 대해 새로운 문장을 만들어 주세요: {topic}. 창의적으로 작성하세요!"
)

gen_chain = gen_prompt | instruct_llm

input_msg = "I get seasick, so I think I'll pass on the trip"
options = ["car", "boat", "airplane", "bike"]

chain = (
    ## -> {"input", "options"}
    {'topic' : zsc_chain}
    | PPrint()
    ## -> {**, "topic"}
    | gen_chain
    ## -> 문자열 출력
)

chain.invoke({"input" : input_msg, "options" : options})

# %%time
## 이 노트북은 실행 시간을 측정하여 전체 소요 시간을 출력합니다.

from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from langchain.schema.runnable.passthrough import RunnableAssign
from functools import partial

## Cell 5: Branch 체인 정의 (입력/생성 결합)
big_chain = (
    PPrint()
    ## 수동 매핑: 입력과 주제를 직접 매핑
    | {'input' : lambda d: d.get('input'), 'topic' : zsc_chain}
    | PPrint()
    ## RunnableAssign: 기본 상태 체인 실행에 적합
    | RunnableAssign({'generation' : gen_chain})
    | PPrint()
    ## 입력과 생성 결과를 결합하여 새로운 문장 생성
    | RunnableAssign({'combination' : (
        ChatPromptTemplate.from_template(
            "다음 두 문장을 고려하세요:"
            "\nP1: {input}"
            "\nP2: {generation}"
            "\n\n두 문장의 아이디어를 하나의 간단한 문장으로 결합하세요."
        )
        | instruct_llm
    )})
)

output = big_chain.invoke({
    "input" : "I get seasick, so I think I'll pass on the trip",
    "options" : ["car", "boat", "airplane", "bike", "unknown"]
})
pprint("최종 출력: ", output)
