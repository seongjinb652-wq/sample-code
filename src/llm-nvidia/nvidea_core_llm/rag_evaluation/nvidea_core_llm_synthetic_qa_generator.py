# ============================================================
# NVIDIA Core LLM - Synthetic QA Generator
# ------------------------------------------------------------
# 이 스크립트는 문서(docstore)에서 무작위로 두 개의 청크를 선택하여
# LLM을 활용해 질문-답변(QA) 쌍을 자동으로 생성하는 예제입니다.
#
# 주요 기능:
#  - 문서 청크 무작위 샘플링
#  - ChatPromptTemplate을 통한 QA 생성 프롬프트 구성
#  - LLM 호출로 질문과 답변 생성
#  - 생성된 QA Pair 출력
# ============================================================import random

# 생성할 질문-답변 쌍 개수
num_questions = 3
synth_questions = []
synth_answers = []

# 간단한 프롬프트 템플릿 정의
simple_prompt = ChatPromptTemplate.from_messages([
    ('system', '{system}'),
    ('user', '입력: {input}')   # INPUT → 한글화
])

# QA Pair 생성 루프
for i in range(num_questions):
    # 문서에서 무작위로 두 개 선택
    doc1, doc2 = random.sample(docs, 2)

    # 시스템 메시지 (LLM 지시사항)
    sys_msg = (
        "사용자가 제공한 문서를 활용하여 흥미로운 질문-답변 쌍을 생성하세요."
        " 가능하다면 두 문서를 모두 활용하고, 요약보다는 본문 내용을 더 많이 참조하세요."
        " 출력 형식:\nQuestion: (좋은 질문, 1-3문장, 구체적)\n\nAnswer: (문서에서 도출된 답변)"
        " '여기 흥미로운 질문 쌍이 있습니다' 같은 문구는 쓰지 마세요. 반드시 형식을 따르세요!"
    )

    # 사용자 메시지 (문서 내용 전달)
    usr_msg = (
        f"문서1: {format_chunk(doc1)}\n\n"
        f"문서2: {format_chunk(doc2)}"
    )

    # LLM 호출하여 QA Pair 생성
    qa_pair = (simple_prompt | llm).invoke({'system': sys_msg, 'input': usr_msg})

    # 질문과 답변 분리 저장
    synth_questions += [qa_pair.split('\n\n')[0]]
    synth_answers += [qa_pair.split('\n\n')[1]]

    # 결과 출력
    pprint2(f"QA Pair {i+1}")
    pprint2(synth_questions[-1])   # 질문 출력
    pprint(synth_answers[-1])      # 답변 출력
    print()


