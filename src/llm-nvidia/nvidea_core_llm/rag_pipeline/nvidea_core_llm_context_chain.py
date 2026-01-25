# ============================================================
# NVIDIA Core LLM - Context Chain with FAISS Retriever
# ------------------------------------------------------------
# 이 스크립트는 LangChain을 활용하여 대화 데이터를 기반으로
# Retriever → Prompt → LLM → Output Parser 체인을 구성하는 예제입니다.
#
# 주요 기능:
#  - 문서 재배치(LongContextReorder) 적용
#  - Retriever에서 문서 검색 후 문자열 변환
#  - ChatPromptTemplate을 통한 질의응답 체인 구성
#  - instruct_llm을 사용하여 대화형 답변 생성
# ============================================================
from langchain.document_transformers import LongContextReorder
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from functools import partial
from operator import itemgetter

########################################################################
## 유틸리티 함수 정의
def RPrint(preface=""):
    """간단한 체인: 출력 후 반환"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        pprint(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """문서 리스트를 문자열로 변환하여 context로 활용"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
        #    out_str += f"[Quote from {doc_name}] "
        #out_str += getattr(doc, 'page_content', str(doc)) + "\n"
        ## "Quote from" → "출처" 로 변경
        #     out_str += f"[출처: {doc_name}] "
        #out_str += getattr(doc, 'page_content', str(doc)) + "\n"

        # 또는 "인용" 으로 표현 가능
             out_str += f"[인용: {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"

    return out_str

## 긴 문서를 중앙에 배치하도록 재정렬
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

## 프롬프트 템플릿 정의
# context_prompt = ChatPromptTemplate.from_template(
    # "Answer the question using only the context"
    # "\n\nRetrieved Context: {context}"
    # "\n\nUser Question: {question}"
    # "\nAnswer the user conversationally. User is not aware of context."
# )
context_prompt = ChatPromptTemplate.from_template(
    "질문에 답할 때는 반드시 주어진 문맥만 사용하세요."
    "\n\n검색된 문맥: {context}"
    "\n\n사용자 질문: {question}"
    "\n대화체로 답변하세요. 사용자는 문맥이 있다는 사실을 모릅니다."
)


## 체인 구성: Retriever → Reorder → docs2str → Prompt → LLM → Parser
# chain = (
#    {
#        'context': convstore.as_retriever() | long_reorder | docs2str,
#        'question': (lambda x:x)
#    }
#    | context_prompt
#    # | RPrint()
#    | instruct_llm
#    | StrOutputParser()
# )
chain = (
    {
        '문맥': convstore.as_retriever() | long_reorder | docs2str,
        '질문': (lambda x:x)
    }
    | ChatPromptTemplate.from_template(
        "질문에 답할 때는 반드시 주어진 문맥만 사용하세요."
        "\n\n검색된 문맥: {문맥}"
        "\n\n사용자 질문: {질문}"
        "\n대화체로 답변하세요. 사용자는 문맥이 있다는 사실을 모릅니다."
    )
    | instruct_llm
    | StrOutputParser()
)

## 질의 실행 (영문 질문 → 한국어 번역 가능)
# pprint(chain.invoke("Where does Beras live?"))  
# pprint(chain.invoke("Where are the Rocky Mountains?"))  
# pprint(chain.invoke("Where are the Rocky Mountains? Are they close to California?"))  
# pprint(chain.invoke("How far away is Beras from the Rocky Mountains?"))
pprint(chain.invoke("베라스는 어디에 살고 있나요?"))  
pprint(chain.invoke("록키산맥은 어디에 있나요?"))  
pprint(chain.invoke("록키산맥은 어디에 있나요? 캘리포니아와 가까운가요?"))  
pprint(chain.invoke("베라스는 록키산맥에서 얼마나 떨어져 있나요?"))

