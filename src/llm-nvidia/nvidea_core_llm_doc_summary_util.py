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

# ---- 문서 로더 임포트 ----
%%time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import ArxivLoader

# ---- 문서 불러오기 ----
## UnstructuredFileLoader: 범용 로더 (적당히 충분)
# documents = UnstructuredFileLoader("llama2_paper.pdf").load()

## ArxivLoader: 특정 논문 로더 (더 나은 결과 가능)
documents = ArxivLoader(query="2404.16130").load()  # GraphRAG
# documents = ArxivLoader(query="2404.03622").load()  # Visualization-of-Thought
# documents = ArxivLoader(query="2404.19756").load()  # KAN: Kolmogorov-Arnold Networks
# documents = ArxivLoader(query="2404.07143").load()  # Infini-Attention
# documents = ArxivLoader(query="2210.03629").load()  # ReAct

# ---- 문서 샘플 출력 ----
print("Number of Documents Retrieved:", len(documents))
print(f"Sample of Document 1 Content (Total Length: {len(documents[0].page_content)}):")
print(documents[0].page_content[:1000])
pprint(documents[0].metadata)

# ---- 텍스트 분할기 설정 ----
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

docs_split = text_splitter.split_documents(documents)

print(len(docs_split))
for i in (0, 1, 2, 15, -1):
    pprint(f"[Document {i}]")
    print(docs_split[i].page_content)
    pprint("="*64)

# ---- LangChain 유틸 임포트 ----
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
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

# ---- 요약 프롬프트 정의 ----
summary_prompt = ChatPromptTemplate.from_template(
    "You are generating a running summary of the document. Make it readable by a technical user."
    " After this, the old knowledge base will be replaced by the new one. Make sure a reader can still understand everything."
    " Keep it short, but as dense and useful as possible! The information should flow from chunk to (loose ends or main ideas) to running_summary."
    " The updated knowledge base keep all of the information from running_summary here: {info_base}."
    "\n\n{format_instructions}. Follow the format precisely, including quotations and commas"
    "\n\nWithout losing any of the info, update the knowledge base with the following: {input}"
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
            .replace("\]

", "]")
            .replace("

\[", "[")
        )
        return string
    return instruct_merge | prompt | llm | preparse | parser

latest_summary = ""

# ---- 요약 체인 정의 ----
def RSummarizer(knowledge, llm, prompt, verbose=False):
    '''
    문서 요약 체인 생성
    '''
    def summarize_docs(docs):        
        parse_chain = RunnableAssign({'info_base' : (lambda x: None)})
        state = {}

        global latest_summary
        
        for i, doc in enumerate(docs):
            ## TODO: parse_chain
